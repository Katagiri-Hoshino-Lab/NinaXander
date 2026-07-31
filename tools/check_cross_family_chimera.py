#!/usr/bin/env python3
"""Two-GPU smoke and boundary gates for both AB and BA chimera runtimes."""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ninaxander.adapter import LatentAdapter  # noqa: E402
from ninaxander.b_to_a_chimera import (  # noqa: E402
    BToAChimera,
    detect_residual_offsets,
    relative_error,
)
from ninaxander.result_io import write_bundle  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


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
    parser.add_argument("--dev_a", default="cuda:0")
    parser.add_argument("--dev_b", default="cuda:1")
    parser.add_argument(
        "--out",
        default="artifacts/metrics/raw/cross_family_runtime_smoke.json",
    )
    args = parser.parse_args()
    switches = [int(value) for value in args.switches.split(",")]

    checkpoint = torch.load(args.adapter, map_location="cpu", weights_only=False)
    config = checkpoint["cfg"]
    adapter = LatentAdapter(
        config["dA"],
        config["dB"],
        config["z"],
        blocks=args.blocks,
        hidden=args.hid,
        arch="resnet",
    ).eval()
    adapter.load_state_dict(checkpoint["model"])
    # Only the two modules used by each direction occupy that direction's
    # suffix device. LayerNorm is parameter-free and follows its input.
    adapter.eA.to(args.dev_b)
    adapter.dB.to(args.dev_b)
    adapter.eB.to(args.dev_a)
    adapter.dA.to(args.dev_a)
    stats_a = {
        int(layer): [tensor.to(args.dev_a) for tensor in values]
        for layer, values in checkpoint["stats"].items()
    }
    stats_b = {
        int(layer): [tensor.to(args.dev_b) for tensor in values]
        for layer, values in checkpoint["stats"].items()
    }

    tokenizer = AutoTokenizer.from_pretrained(args.tok)
    rwkv = (
        AutoModelForCausalLM.from_pretrained(
            args.rwkv,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .to(args.dev_a)
        .eval()
    )
    pythia = (
        AutoModelForCausalLM.from_pretrained(
            args.pythia,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .to(args.dev_b)
        .eval()
    )
    for parameter in (
        list(adapter.parameters()) + list(rwkv.parameters()) + list(pythia.parameters())
    ):
        parameter.requires_grad_(False)

    token_ids = tokenizer(
        "The capital of France is",
        return_tensors="pt",
    ).input_ids
    offsets = detect_residual_offsets(
        rwkv,
        pythia,
        token_ids,
        args.dev_a,
        args.dev_b,
    )
    rows: list[dict[str, object]] = []

    # AB: RWKV post-block L -> adapter -> Pythia pre-block L+1.
    injected_b: dict[str, torch.Tensor | None] = {"value": None}

    def pythia_prehook(module, positional, keyword):
        del module
        if injected_b["value"] is None:
            return None
        if positional:
            return (injected_b["value"],) + positional[1:], keyword
        replacement = dict(keyword)
        replacement["hidden_states"] = injected_b["value"]
        return positional, replacement

    @torch.no_grad()
    def run_pythia_suffix(
        ids: torch.Tensor,
        hidden_b: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        block_index = layer + offsets.pythia
        if not 0 <= block_index < len(pythia.gpt_neox.layers):
            raise ValueError(f"A-to-B switch L={layer} has no Pythia suffix")
        handle = pythia.gpt_neox.layers[block_index].register_forward_pre_hook(
            pythia_prehook,
            with_kwargs=True,
        )
        injected_b["value"] = hidden_b.to(args.dev_b)
        try:
            return pythia(ids.to(args.dev_b), use_cache=False).logits
        finally:
            injected_b["value"] = None
            handle.remove()

    @torch.no_grad()
    def translate_a_to_b(hidden_a: torch.Tensor, layer: int) -> torch.Tensor:
        mean_a, std_a, mean_b, std_b = stats_b[layer]
        standardized = (hidden_a.to(args.dev_b).float() - mean_a) / std_a
        decoded = adapter.dB(adapter.enc_a(standardized))
        return (decoded * std_b + mean_b).to(pythia.dtype)

    with torch.no_grad():
        source_a = rwkv(
            token_ids.to(args.dev_a),
            output_hidden_states=True,
            use_cache=False,
        )
        parent_b = pythia(
            token_ids.to(args.dev_b),
            output_hidden_states=True,
            use_cache=False,
        )
    for layer in switches:
        parent_hidden = parent_b.hidden_states[layer + offsets.pythia]
        parent_logits = run_pythia_suffix(token_ids, parent_hidden, layer)
        parent_error = relative_error(parent_logits, parent_b.logits)
        if parent_error >= 1e-4:
            raise AssertionError(
                f"A-to-B parent-reproduction gate failed at L={layer}: "
                f"{parent_error:.3e}"
            )
        hidden_a = source_a.hidden_states[layer + offsets.rwkv]
        translated = translate_a_to_b(hidden_a, layer)
        logits = run_pythia_suffix(token_ids, translated, layer)
        finite = bool(torch.isfinite(logits).all().item())
        if not finite:
            raise AssertionError(f"non-finite A-to-B logits at L={layer}")
        row: dict[str, object] = {
            "direction": "A_to_B",
            "switch_layer": layer,
            "parent_gate_relative_error": parent_error,
            "direct_gate_relative_error": None,
            "direct_vs_hook_relative_error": None,
            "concurrent_vs_sequential_relative_error": None,
            "fused_batch_vs_batch1_relative_error": None,
            "fused_batch_accepted_at_1e-4": None,
            "translated_residual_l2": translated.float().norm().item(),
            "logits_finite": finite,
            "last_token_argmax": int(logits[0, -1].argmax().item()),
        }
        rows.append(row)
        print(
            f"AB L={layer} gate={parent_error:.3e} "
            f"translated_norm={row['translated_residual_l2']:.3f} "
            f"argmax={row['last_token_argmax']}",
            flush=True,
        )

    # BA: retain the extra direct-suffix and CUDA-stream diagnostics needed by
    # RWKV's batch-sensitive suffix implementation.
    runtime = BToAChimera(
        rwkv,
        pythia,
        adapter,
        stats_a,
        offsets,
        args.dev_a,
        args.dev_b,
    )
    gate_errors = runtime.gate_parent_reproduction(token_ids, switches)
    direct_gate_errors = runtime.gate_direct_suffix(token_ids, switches)
    for layer in switches:
        hidden_b = parent_b.hidden_states[layer + offsets.pythia]
        translated = runtime.translate_adapter(hidden_b, layer)
        hook_logits = runtime.run_rwkv_suffix(
            token_ids,
            translated,
            layer,
        ).logits
        logits = runtime.run_rwkv_suffix_direct(translated, layer)
        path_error = relative_error(logits, hook_logits)
        if path_error >= 1e-4:
            raise AssertionError(
                f"direct/hook B-to-A disagreement at L={layer}: {path_error:.3e}"
            )
        concurrent_logits = runtime.run_rwkv_suffixes_concurrent(
            [translated, translated, translated],
            layer,
        )
        concurrent_error = max(
            relative_error(candidate, logits) for candidate in concurrent_logits
        )
        if concurrent_error >= 1e-4:
            raise AssertionError(
                f"concurrent/sequential B-to-A disagreement at L={layer}: "
                f"{concurrent_error:.3e}"
            )
        fused_logits = runtime.run_rwkv_suffix_direct(
            torch.cat([translated, translated, translated], dim=0),
            layer,
        )
        fused_error = max(
            relative_error(fused_logits[index : index + 1], logits)
            for index in range(fused_logits.shape[0])
        )
        finite = bool(torch.isfinite(logits).all().item())
        if not finite:
            raise AssertionError(f"non-finite B-to-A logits at L={layer}")
        row = {
            "direction": "B_to_A",
            "switch_layer": layer,
            "parent_gate_relative_error": gate_errors[layer],
            "direct_gate_relative_error": direct_gate_errors[layer],
            "direct_vs_hook_relative_error": path_error,
            "concurrent_vs_sequential_relative_error": concurrent_error,
            "fused_batch_vs_batch1_relative_error": fused_error,
            "fused_batch_accepted_at_1e-4": fused_error < 1e-4,
            "translated_residual_l2": translated.float().norm().item(),
            "logits_finite": finite,
            "last_token_argmax": int(logits[0, -1].argmax().item()),
        }
        rows.append(row)
        print(
            f"BA L={layer} gate={gate_errors[layer]:.3e} "
            f"direct={direct_gate_errors[layer]:.3e} path={path_error:.3e} "
            f"streams={concurrent_error:.3e} "
            f"fused_batch3={fused_error:.3e} "
            f"translated_norm={row['translated_residual_l2']:.3f} "
            f"argmax={row['last_token_argmax']}",
            flush=True,
        )

    payload = {
        "schema_version": 2,
        "directions": ["A_to_B", "B_to_A"],
        "boundaries": {
            "A_to_B": "RWKV post-block L -> Pythia pre-block L+1",
            "B_to_A": "Pythia post-block L -> RWKV pre-block L+1",
        },
        "switches": switches,
        "offsets": {
            "rwkv": offsets.rwkv,
            "pythia": offsets.pythia,
        },
        "checkpoint_r2": {
            "A_to_B": checkpoint["eval"]["A->B"],
            "B_to_A": checkpoint["eval"]["B->A"],
        },
        "canonical_option_batch": 1,
        "rwkv_fused_batch_role": "diagnostic only; never used for reported QA",
        "rows": rows,
    }
    json_path, csv_path = write_bundle(args.out, payload, rows)
    print(
        f"wrote {json_path} + {csv_path}\nCROSS_FAMILY_RUNTIME_AB_BA_VALID",
        flush=True,
    )


if __name__ == "__main__":
    main()
