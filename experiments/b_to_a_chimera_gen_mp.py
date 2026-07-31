"""Greedy samples from the current B-to-A Pythia-prefix/RWKV-suffix chimera."""

from __future__ import annotations

import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ninaxander.adapter import LatentAdapter
from ninaxander.result_io import write_csv
from ninaxander.b_to_a_chimera import (
    BToAChimera,
    detect_residual_offsets,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="artifacts/checkpoints/z4096_L32/SNAP_L32_final.pt",
    )
    parser.add_argument("--rwkv", default="models/rwkv-raven-7b-hf-fp16")
    parser.add_argument("--pythia", default="models/tulu-pythia69-fp16")
    parser.add_argument("--tok", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--hid", type=int, default=4096)
    parser.add_argument("--switch", type=int, default=4)
    parser.add_argument("--ntok", type=int, default=40)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Reproducibility contract: only 0 is supported (deterministic argmax).",
    )
    parser.add_argument("--dev_a", default="cuda:0")
    parser.add_argument("--dev_b", default="cuda:1")
    parser.add_argument(
        "--out",
        default="artifacts/metrics/raw/b_to_a_generation_samples.csv",
    )
    args = parser.parse_args()
    if args.temperature != 0.0:
        raise ValueError(
            "This reported generation uses temperature=0 only "
            "(deterministic greedy argmax)."
        )
    dev_a, dev_b = args.dev_a, args.dev_b
    layer = args.switch
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.tok)
    rwkv = (
        AutoModelForCausalLM.from_pretrained(
            args.rwkv,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .to(dev_a)
        .eval()
    )
    pythia = (
        AutoModelForCausalLM.from_pretrained(
            args.pythia,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .to(dev_b)
        .eval()
    )
    for parameter in list(rwkv.parameters()) + list(pythia.parameters()):
        parameter.requires_grad_(False)

    probe = tokenizer("The capital of France is", return_tensors="pt").input_ids
    offsets = detect_residual_offsets(rwkv, pythia, probe, dev_a, dev_b)
    prompts = [
        "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
        "The three primary colors are",
        "Question: Water is made of hydrogen and\nAnswer:",
        (
            "### Instruction:\nExplain what a black hole is in one sentence."
            "\n\n### Response:\n"
        ),
    ]

    @torch.no_grad()
    def greedy(logits_function, prompt):
        token_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt_length = token_ids.shape[1]
        for _ in range(args.ntok):
            next_token = (
                logits_function(token_ids)[:, -1].argmax(dim=-1, keepdim=True).cpu()
            )
            token_ids = torch.cat([token_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        return tokenizer.decode(
            token_ids[0, prompt_length:],
            skip_special_tokens=True,
        )

    parent_outputs: dict[str, list[str]] = {"pure-Pythia": [], "pure-RWKV": []}
    for prompt in prompts:
        parent_outputs["pure-Pythia"].append(
            greedy(lambda ids: pythia(ids.to(dev_b), use_cache=False).logits, prompt)
        )
        parent_outputs["pure-RWKV"].append(
            greedy(lambda ids: rwkv(ids.to(dev_a), use_cache=False).logits, prompt)
        )

    def generate_checkpoint(path: str, label: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["cfg"]
        adapter = (
            LatentAdapter(
                config["dA"],
                config["dB"],
                config["z"],
                blocks=args.blocks,
                hidden=args.hid,
                arch="resnet",
            )
            .to(dev_a)
            .eval()
        )
        adapter.load_state_dict(checkpoint["model"])
        # Generation only executes the B->A half of the bidirectional bridge.
        adapter.eA.to("cpu")
        adapter.dB.to("cpu")
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        stats = {
            int(index): [tensor.to(dev_a) for tensor in values]
            for index, values in checkpoint["stats"].items()
        }
        b_to_a = checkpoint["eval"]["B->A"]
        checkpoint_step = checkpoint.get("step", "")
        del checkpoint
        runtime = BToAChimera(
            rwkv,
            pythia,
            adapter,
            stats,
            offsets,
            dev_a,
            dev_b,
        )
        errors = runtime.gate_parent_reproduction(probe, [layer])
        direct_errors = runtime.gate_direct_suffix(probe, [layer])
        print(
            f"{label} {os.path.basename(path)} B->A={b_to_a:+.4f} "
            f"hook_gate={errors[layer]:.1e} direct_gate={direct_errors[layer]:.1e}",
            flush=True,
        )

        @torch.no_grad()
        def b_to_a_logits(token_ids):
            source = pythia(
                token_ids.to(dev_b),
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_b = source.hidden_states[layer + offsets.pythia]
            hidden_a = runtime.translate_adapter(hidden_b, layer)
            return runtime.run_rwkv_suffix_direct(
                hidden_a,
                layer,
            )

        outputs = [greedy(b_to_a_logits, prompt) for prompt in prompts]
        metadata = {
            "label": label,
            "b_to_a": b_to_a,
            "checkpoint_step": checkpoint_step,
            "checkpoint": os.path.basename(path),
            "gate_relative_error": errors[layer],
            "direct_gate_relative_error": direct_errors[layer],
        }
        return outputs, metadata

    b_to_a_outputs, b_to_a_metadata = generate_checkpoint(
        args.adapter,
        f"B_to_A@{layer}",
    )
    gc.collect()
    torch.cuda.empty_cache()

    generation_metadata = {
        "decoding": "greedy_argmax",
        "temperature": args.temperature,
        "max_new_tokens": args.ntok,
        "seed": 0,
    }
    rows = []
    for prompt_id, prompt in enumerate(prompts):
        candidates = (
            (
                "pure-Pythia",
                parent_outputs["pure-Pythia"][prompt_id],
                None,
            ),
            (
                "pure-RWKV",
                parent_outputs["pure-RWKV"][prompt_id],
                None,
            ),
            (
                f"B_to_A@{layer}",
                b_to_a_outputs[prompt_id],
                b_to_a_metadata,
            ),
        )
        print(f"PROMPT {prompt!r}")
        for name, output, metadata in candidates:
            print(f"  [{name}] {output!r}")
            execution_path = {
                "pure-Pythia": "B_parent",
                "pure-RWKV": "A_parent",
            }.get(name, "B_to_A")
            rows.append(
                {
                    "direction": execution_path,
                    "source_model": "Tulu-Pythia-6.9B",
                    "target_model": "RWKV-Raven-7B",
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "config": name,
                    "switch_layer": layer if metadata else "",
                    "adapter_cross_r2": metadata["b_to_a"] if metadata else "",
                    "checkpoint_step": metadata["checkpoint_step"] if metadata else "",
                    "parent_gate_relative_error": (
                        metadata["gate_relative_error"] if metadata else ""
                    ),
                    "direct_gate_relative_error": (
                        metadata["direct_gate_relative_error"] if metadata else ""
                    ),
                    **generation_metadata,
                    "output": output,
                }
            )
        print(flush=True)

    output_path = write_csv(args.out, rows)
    print(f"wrote {output_path}\nB_TO_A_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
