"""Actually pruned serving benchmark for Pythia-front -> RWKV-back.

For switch L the retained blocks are Pythia 0..L and RWKV L+1..31.  The
Pythia LM head/final norm and the RWKV embedding/prefix are freed.  Three gates
compare (1) truncated Pythia's residual, (2) the manual RWKV suffix, and (3)
the complete pruned chimera against their unpruned references.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ninaxander.adapter import LatentAdapter
from ninaxander.result_io import write_bundle
from ninaxander.b_to_a_chimera import (
    BToAChimera,
    detect_residual_offsets,
    relative_error,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

MIB = 1024 * 1024
GATE_TOLERANCE = 1e-4


def weight_bytes(module, device):
    target = torch.device(device)
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
        if parameter.device == target
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", default="artifacts/checkpoints/z4096_L32/SNAP_L32_final.pt"
    )
    parser.add_argument("--rwkv", default="models/rwkv-raven-7b-hf-fp16")
    parser.add_argument("--pythia", default="models/tulu-pythia69-fp16")
    parser.add_argument("--tok", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--hid", type=int, default=4096)
    parser.add_argument("--switches", default="4,8,16,24")
    parser.add_argument("--ctx", default="512,2048")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--pruned-only",
        action="store_true",
        help="skip parent and early-exit profiles (useful for a fast wiring smoke test)",
    )
    parser.add_argument("--dev_a", default="cuda:0")
    parser.add_argument("--dev_b", default="cuda:1")
    parser.add_argument(
        "--out",
        default="artifacts/metrics/raw/b_to_a_serving_pruned.json",
    )
    args = parser.parse_args()
    dev_a, dev_b = args.dev_a, args.dev_b
    switches = [int(value) for value in args.switches.split(",")]
    contexts = [int(value) for value in args.ctx.split(",")]
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.tok)
    checkpoint = torch.load(args.adapter, map_location="cpu", weights_only=False)
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
    # A deployable B-to-A-only path retains only E_B and D_A.
    adapter.eA.to("cpu")
    adapter.dB.to("cpu")
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    stats = {
        int(layer): [tensor.to(dev_a) for tensor in values]
        for layer, values in checkpoint["stats"].items()
    }

    def load_pair():
        model_a = (
            AutoModelForCausalLM.from_pretrained(
                args.rwkv,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            .to(dev_a)
            .eval()
        )
        model_b = (
            AutoModelForCausalLM.from_pretrained(
                args.pythia,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            .to(dev_b)
            .eval()
        )
        for parameter in list(model_a.parameters()) + list(model_b.parameters()):
            parameter.requires_grad_(False)
        return model_a, model_b

    def synchronize():
        torch.cuda.synchronize(dev_a)
        torch.cuda.synchronize(dev_b)

    def profile(forward, profile_contexts):
        values = {}
        for context in profile_contexts:
            token_ids = tokenizer(
                "A " * (context + 8),
                return_tensors="pt",
            ).input_ids[:, :context]
            try:
                with torch.no_grad():
                    forward(token_ids)
                    synchronize()
                torch.cuda.reset_peak_memory_stats(dev_a)
                torch.cuda.reset_peak_memory_stats(dev_b)
                durations = []
                for _ in range(args.reps):
                    synchronize()
                    start = time.perf_counter()
                    with torch.no_grad():
                        forward(token_ids)
                    synchronize()
                    durations.append((time.perf_counter() - start) * 1e3)
                durations.sort()
                values[context] = {
                    "ms": durations[len(durations) // 2],
                    "peak_devA_MiB": (torch.cuda.max_memory_allocated(dev_a) / MIB),
                    "peak_devB_MiB": (torch.cuda.max_memory_allocated(dev_b) / MIB),
                }
            except RuntimeError as exception:
                torch.cuda.empty_cache()
                values[context] = {
                    "error": (
                        "OOM"
                        if "out of memory" in str(exception).lower()
                        else str(exception)[:80]
                    )
                }
        return values

    probe = tokenizer("The capital of France is", return_tensors="pt").input_ids
    results = {
        "schema_version": 1,
        "direction": "B_to_A",
        "source_model": "Tulu-Pythia-6.9B",
        "target_model": "RWKV-Raven-7B",
        "implementation": "physically_pruned",
        "adapter": os.path.basename(args.adapter),
        "checkpoint_step": checkpoint.get("step", ""),
        "checkpoint_b_to_a_r2": checkpoint["eval"]["B->A"],
        "switches": switches,
        "ctx": contexts,
        "repetitions": args.reps,
        "gate_tolerance": GATE_TOLERANCE,
        "parent_reproduction_relative_error": {},
        "configs": {},
    }

    if not args.pruned_only:
        # Parent baselines.
        parent_rwkv, parent_pythia = load_pair()
        detect_residual_offsets(
            parent_rwkv,
            parent_pythia,
            probe,
            dev_a,
            dev_b,
        )
        weight_a = weight_bytes(parent_rwkv, dev_a) / MIB
        weight_b = weight_bytes(parent_pythia, dev_b) / MIB
        results["configs"]["pure-Pythia"] = {
            "pythia_blocks": 32,
            "rwkv_blocks": 0,
            "weight_MiB_devA": 0.0,
            "weight_MiB_devB": weight_b,
            "prefill": profile(
                lambda ids, current=parent_pythia: (
                    current(
                        ids.to(dev_b),
                        use_cache=False,
                    ).logits
                ),
                contexts,
            ),
        }
        results["configs"]["pure-RWKV"] = {
            "pythia_blocks": 0,
            "rwkv_blocks": 32,
            "weight_MiB_devA": weight_a,
            "weight_MiB_devB": 0.0,
            "prefill": profile(
                lambda ids, current=parent_rwkv: (
                    current(
                        ids.to(dev_a),
                        use_cache=False,
                    ).logits
                ),
                contexts,
            ),
        }
        print("pure parents profiled", flush=True)
        del parent_rwkv, parent_pythia
        gc.collect()
        torch.cuda.empty_cache()

        # Equal-KV early-exit Pythia: the B-to-A chimera retains L+1
        # Transformer blocks, so the matched model uses exactly K=L+1.
        for layer in switches:
            block_count = layer + 1
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.pythia,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
                .to(dev_b)
                .eval()
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.gpt_neox.layers = nn.ModuleList(
                list(model.gpt_neox.layers)[:block_count]
            )
            gc.collect()
            torch.cuda.empty_cache()
            name = f"B_early_exit@{block_count}"
            results["configs"][name] = {
                "pythia_blocks": block_count,
                "rwkv_blocks": 0,
                "weight_MiB_devA": 0.0,
                "weight_MiB_devB": weight_bytes(model, dev_b) / MIB,
                "prefill": profile(
                    lambda ids, current=model: (
                        current(
                            ids.to(dev_b),
                            use_cache=False,
                        ).logits
                    ),
                    contexts,
                ),
            }
            print(f"{name} profiled", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Destructive truncation requires a fresh pair for each switch.
    for layer in switches:
        rwkv, pythia = load_pair()
        offsets = detect_residual_offsets(rwkv, pythia, probe, dev_a, dev_b)
        runtime = BToAChimera(
            rwkv,
            pythia,
            adapter,
            stats,
            offsets,
            dev_a,
            dev_b,
        )
        parent_gate = runtime.gate_parent_reproduction(probe, [layer])[layer]
        results["parent_reproduction_relative_error"][str(layer)] = parent_gate
        with torch.no_grad():
            full_a = rwkv(
                probe.to(dev_a),
                output_hidden_states=True,
                use_cache=False,
            )
            full_b = pythia(
                probe.to(dev_b),
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_a_true = full_a.hidden_states[layer + offsets.rwkv]
            hidden_b_true = full_b.hidden_states[layer + offsets.pythia]
            translated = runtime.translate_adapter(hidden_b_true, layer)
            unpruned_reference = runtime.run_rwkv_suffix(
                probe.to(dev_a),
                translated,
                layer,
            ).logits

        # All RWKV blocks have already been inference-rescaled by the forward
        # above.  Retain the original suffix modules and preserve original
        # block indices in the explicit /2 schedule below.
        rwkv.rwkv.blocks = nn.ModuleList(list(rwkv.rwkv.blocks)[layer + 1 :])
        rwkv.rwkv.embeddings = nn.Identity()
        pythia.gpt_neox.layers = nn.ModuleList(
            list(pythia.gpt_neox.layers)[: layer + 1]
        )
        pythia.gpt_neox.final_layer_norm = nn.Identity()
        pythia.embed_out = nn.Identity()
        gc.collect()
        torch.cuda.empty_cache()

        @torch.no_grad()
        def source_front(token_ids, current_pythia=pythia):
            return current_pythia.gpt_neox(
                input_ids=token_ids.to(dev_b),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state

        @torch.no_grad()
        def target_suffix(hidden, current_rwkv=rwkv, current_layer=layer):
            value = hidden.to(dev_a)
            for original_index, block in enumerate(
                current_rwkv.rwkv.blocks,
                start=current_layer + 1,
            ):
                value, _, _ = block(
                    value,
                    state=None,
                    use_cache=False,
                    output_attentions=False,
                )
                if (
                    current_rwkv.rwkv.layers_are_rescaled
                    and current_rwkv.config.rescale_every > 0
                    and (original_index + 1) % current_rwkv.config.rescale_every == 0
                ):
                    value = value / 2
            return current_rwkv.head(current_rwkv.rwkv.ln_out(value))

        @torch.no_grad()
        def pruned_chimera(
            token_ids,
            current_runtime=runtime,
            current_layer=layer,
            current_source=source_front,
            current_target=target_suffix,
        ):
            hidden_b = current_source(token_ids)
            hidden_a = current_runtime.translate_adapter(
                hidden_b,
                current_layer,
            )
            return current_target(hidden_a)

        with torch.no_grad():
            source_pruned = source_front(probe)
            source_gate = relative_error(source_pruned, hidden_b_true)
            target_reproduced = target_suffix(hidden_a_true)
            target_gate = relative_error(target_reproduced, full_a.logits)
            pruned_output = pruned_chimera(probe)
            pruned_gate = relative_error(pruned_output, unpruned_reference)
        # Keep every deployable-path gate at the same threshold as the
        # canonical parent-reproduction check.  A looser pruning-only
        # tolerance can hide a boundary or rescaling regression.
        tolerance = GATE_TOLERANCE
        if source_gate >= tolerance:
            raise AssertionError(
                f"source-front gate failed L={layer}: {source_gate:.3e}"
            )
        if target_gate >= tolerance:
            raise AssertionError(
                f"target-suffix gate failed L={layer}: {target_gate:.3e}"
            )
        if pruned_gate >= tolerance:
            raise AssertionError(
                f"pruned B-to-A gate failed L={layer}: {pruned_gate:.3e}"
            )

        weight_dev_a = (weight_bytes(rwkv, dev_a) + weight_bytes(adapter, dev_a)) / MIB
        weight_dev_b = weight_bytes(pythia, dev_b) / MIB
        name = f"pruned-B_to_A@{layer}"
        results["configs"][name] = {
            "pythia_blocks": layer + 1,
            "rwkv_blocks": 31 - layer,
            "weight_MiB_devA": weight_dev_a,
            "weight_MiB_devB": weight_dev_b,
            "weight_MiB_total": weight_dev_a + weight_dev_b,
            "parent_gate_rel": parent_gate,
            "source_front_gate_rel": source_gate,
            "target_suffix_gate_rel": target_gate,
            "gate_rel": pruned_gate,
            "prefill": profile(pruned_chimera, contexts),
        }
        print(
            f"{name}: blocks B/A={layer + 1}/{31 - layer} "
            f"weights={weight_dev_b:.0f}+{weight_dev_a:.0f} MiB "
            f"gates parent/source/target/pruned={parent_gate:.1e}/"
            f"{source_gate:.1e}/{target_gate:.1e}/{pruned_gate:.1e}",
            flush=True,
        )
        del source_front, target_suffix, pruned_chimera
        del runtime, rwkv, pythia
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for name, values in results["configs"].items():
        base = {
            "direction": "B_to_A",
            "benchmark": "pruned",
            "phase": "prefill",
            "config": name,
            "pythia_blocks": values.get("pythia_blocks", 0),
            "rwkv_blocks": values.get("rwkv_blocks", 0),
            "weight_mib_dev_a": values.get("weight_MiB_devA", 0),
            "weight_mib_dev_b": values.get("weight_MiB_devB", 0),
            "weight_mib_total": values.get(
                "weight_MiB_total",
                values.get("weight_MiB_devA", 0) + values.get("weight_MiB_devB", 0),
            ),
            "parent_gate_relative_error": values.get("parent_gate_rel", ""),
            "source_front_gate_relative_error": values.get(
                "source_front_gate_rel",
                "",
            ),
            "target_suffix_gate_relative_error": values.get(
                "target_suffix_gate_rel",
                "",
            ),
            "gate_relative_error": values.get("gate_rel", ""),
        }
        for context, profile_values in values["prefill"].items():
            rows.append(
                {
                    **base,
                    "context_tokens": context,
                    **profile_values,
                }
            )
    json_path, csv_path = write_bundle(args.out, results, rows)
    print(
        f"wrote {json_path} + {csv_path}\nB_TO_A_SERVING_PRUNED_DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
