"""Unpruned serving-cost benchmark for the B-to-A execution path."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ninaxander.adapter import LatentAdapter
from ninaxander.result_io import write_bundle
from ninaxander.b_to_a_chimera import (
    BToAChimera,
    detect_residual_offsets,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="artifacts/checkpoints/z4096_L32/SNAP_L32_final.pt")
    parser.add_argument("--rwkv", default="models/rwkv-raven-7b-hf-fp16")
    parser.add_argument("--pythia", default="models/tulu-pythia69-fp16")
    parser.add_argument("--tok", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--hid", type=int, default=4096)
    parser.add_argument("--switches", default="4,8,16,24")
    parser.add_argument("--ctx", default="512,2048")
    parser.add_argument("--decode_ctx", type=int, default=512)
    parser.add_argument("--decode_tok", type=int, default=16)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--dev_a", default="cuda:0")
    parser.add_argument("--dev_b", default="cuda:1")
    parser.add_argument(
        "--out",
        default="artifacts/metrics/raw/b_to_a_serving_bench.json",
    )
    args = parser.parse_args()
    dev_a, dev_b = args.dev_a, args.dev_b
    switches = [int(value) for value in args.switches.split(",")]
    contexts = [int(value) for value in args.ctx.split(",")]
    mebibyte = 1024 * 1024
    torch.manual_seed(0)

    checkpoint = torch.load(args.adapter, map_location="cpu", weights_only=False)
    config = checkpoint["cfg"]
    adapter = LatentAdapter(
        config["dA"],
        config["dB"],
        config["z"],
        blocks=args.blocks,
        hidden=args.hid,
        arch="resnet",
    ).to(dev_a).eval()
    adapter.load_state_dict(checkpoint["model"])
    adapter.eA.to("cpu")
    adapter.dB.to("cpu")
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    stats = {
        int(layer): [tensor.to(dev_a) for tensor in values]
        for layer, values in checkpoint["stats"].items()
    }

    tokenizer = AutoTokenizer.from_pretrained(args.tok)
    rwkv = AutoModelForCausalLM.from_pretrained(
        args.rwkv,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(dev_a).eval()
    pythia = AutoModelForCausalLM.from_pretrained(
        args.pythia,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(dev_b).eval()
    for parameter in list(rwkv.parameters()) + list(pythia.parameters()):
        parameter.requires_grad_(False)
    weight_a = torch.cuda.memory_allocated(dev_a) / mebibyte
    weight_b = torch.cuda.memory_allocated(dev_b) / mebibyte

    probe = tokenizer("The capital of France is", return_tensors="pt").input_ids
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
    gate_errors = runtime.gate_parent_reproduction(probe, switches)
    print(
        f"weights: RWKV+adapter={weight_a:.0f} MiB {dev_a}; "
        f"Pythia={weight_b:.0f} MiB {dev_b}",
        flush=True,
    )
    print(
        "gate "
        + " ".join(f"L{layer}={error:.1e}" for layer, error in gate_errors.items()),
        flush=True,
    )

    @torch.no_grad()
    def source_residual(token_ids, layer):
        captured = {}

        def capture(module, inputs, output):
            del module, inputs
            captured["hidden"] = output[0] if isinstance(output, tuple) else output

        handle = pythia.gpt_neox.layers[layer].register_forward_hook(capture)
        try:
            pythia(token_ids.to(dev_b), use_cache=False)
        finally:
            handle.remove()
        return captured["hidden"]

    @torch.no_grad()
    def b_to_a_chimera(token_ids, layer):
        hidden_b = source_residual(token_ids, layer)
        hidden_a = runtime.translate_adapter(hidden_b, layer)
        return runtime.run_rwkv_suffix(
            token_ids.to(dev_a),
            hidden_a,
            layer,
        ).logits

    configurations = (
        [("pure-Pythia", None), ("pure-RWKV", None)]
        + [(f"B_to_A@{layer}", layer) for layer in switches]
    )

    @torch.no_grad()
    def run(name, layer, token_ids):
        if name == "pure-Pythia":
            return pythia(token_ids.to(dev_b), use_cache=False).logits
        if name == "pure-RWKV":
            return rwkv(token_ids.to(dev_a), use_cache=False).logits
        return b_to_a_chimera(token_ids, layer)

    def synchronize():
        torch.cuda.synchronize(dev_a)
        torch.cuda.synchronize(dev_b)

    results = {
        "schema_version": 1,
        "direction": "B_to_A",
        "source_model": "Tulu-Pythia-6.9B",
        "target_model": "RWKV-Raven-7B",
        "implementation": "unpruned_full_parent_reforward",
        "adapter": os.path.basename(args.adapter),
        "checkpoint_step": checkpoint.get("step", ""),
        "checkpoint_b_to_a_r2": checkpoint["eval"]["B->A"],
        "switches": switches,
        "contexts": contexts,
        "decode_context": args.decode_ctx,
        "decode_tokens": args.decode_tok,
        "repetitions": args.reps,
        "parent_reproduction_relative_error": {
            str(layer): error for layer, error in gate_errors.items()
        },
        "weights_MiB": {
            "rwkv_adapter_devA": weight_a,
            "pythia_devB": weight_b,
        },
        "prefill": {},
        "decode": {},
    }

    for context in contexts:
        token_ids = tokenizer(
            "A " * (context + 8),
            return_tensors="pt",
        ).input_ids[:, :context]
        results["prefill"][context] = {}
        print(f"=== B-to-A prefill ctx={context} ===", flush=True)
        for name, layer in configurations:
            try:
                with torch.no_grad():
                    run(name, layer, token_ids)
                    synchronize()
                torch.cuda.reset_peak_memory_stats(dev_a)
                torch.cuda.reset_peak_memory_stats(dev_b)
                durations = []
                for _ in range(args.reps):
                    synchronize()
                    start = time.perf_counter()
                    with torch.no_grad():
                        run(name, layer, token_ids)
                    synchronize()
                    durations.append((time.perf_counter() - start) * 1e3)
                durations.sort()
                values = {
                    "ms": durations[len(durations) // 2],
                    "peak_devA_MiB": (
                        torch.cuda.max_memory_allocated(dev_a) / mebibyte
                    ),
                    "peak_devB_MiB": (
                        torch.cuda.max_memory_allocated(dev_b) / mebibyte
                    ),
                }
                results["prefill"][context][name] = values
                print(
                    f"  {name:20s} {values['ms']:.1f} ms "
                    f"peak={values['peak_devA_MiB']:.0f}/"
                    f"{values['peak_devB_MiB']:.0f} MiB",
                    flush=True,
                )
            except RuntimeError as exception:
                torch.cuda.empty_cache()
                message = (
                    "OOM"
                    if "out of memory" in str(exception).lower()
                    else f"ERR:{str(exception)[:80]}"
                )
                results["prefill"][context][name] = {"error": message}
                print(f"  {name:20s} {message}", flush=True)

    context = args.decode_ctx
    print(
        f"=== B-to-A decode ctx={context} tokens={args.decode_tok} ===",
        flush=True,
    )
    for name, layer in configurations:
        token_ids = tokenizer(
            "A " * (context + 8),
            return_tensors="pt",
        ).input_ids[:, :context]
        try:
            with torch.no_grad():
                run(name, layer, token_ids)
                synchronize()
            synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(args.decode_tok):
                    next_token = (
                        run(name, layer, token_ids)[:, -1]
                        .argmax(dim=-1, keepdim=True)
                        .cpu()
                    )
                    token_ids = torch.cat([token_ids, next_token], dim=1)
            synchronize()
            elapsed = time.perf_counter() - start
            values = {
                "tok_per_s": args.decode_tok / elapsed,
                "ms_per_token": elapsed / args.decode_tok * 1e3,
            }
            results["decode"][name] = values
            print(
                f"  {name:20s} {values['tok_per_s']:.2f} tok/s",
                flush=True,
            )
        except RuntimeError as exception:
            torch.cuda.empty_cache()
            message = (
                "OOM"
                if "out of memory" in str(exception).lower()
                else f"ERR:{str(exception)[:80]}"
            )
            results["decode"][name] = {"error": message}
            print(f"  {name:20s} {message}", flush=True)

    rows = []
    for context, values_by_config in results["prefill"].items():
        for name, values in values_by_config.items():
            rows.append(
                {
                    "direction": "B_to_A",
                    "benchmark": "unpruned",
                    "phase": "prefill",
                    "context_tokens": context,
                    "config": name,
                    **values,
                }
            )
    for name, values in results["decode"].items():
        rows.append(
            {
                "direction": "B_to_A",
                "benchmark": "unpruned",
                "phase": "decode",
                "context_tokens": args.decode_ctx,
                "config": name,
                **values,
            }
        )
    json_path, csv_path = write_bundle(args.out, results, rows)
    print(f"wrote {json_path} + {csv_path}\nB_TO_A_SERVING_DONE", flush=True)


if __name__ == "__main__":
    main()
