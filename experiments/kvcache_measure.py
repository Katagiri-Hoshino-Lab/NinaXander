"""Measure the actual Tulu-Pythia KV cache and emit the paper's memory rows."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from ninaxander.result_io import write_csv
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/tulu-pythia69-fp16")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--context-for-table", type=int, default=4096)
    parser.add_argument("--switches", default="4,8,16,24")
    parser.add_argument(
        "--direction",
        choices=["A_to_B", "B_to_A"],
        default="A_to_B",
        help="A_to_B keeps the Pythia suffix; B_to_A keeps the Pythia prefix",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="artifacts/metrics/raw/a_to_b_kv_cache.csv")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    del tokenizer  # Loading it is part of the reproducibility gate; the probe uses random valid ids.
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .to(args.device)
        .eval()
    )
    token_ids = torch.randint(
        0,
        min(50_000, model.config.vocab_size),
        (1, args.sequence_length),
        device=args.device,
    )
    with torch.no_grad():
        output = model(token_ids, use_cache=True)
    layers = [(layer.keys, layer.values) for layer in output.past_key_values.layers]
    total_bytes = sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in layers
    )
    key0, _ = layers[0]
    layer_count = len(layers)
    kib_per_token = total_bytes / args.sequence_length / 1024
    kib_per_token_layer = kib_per_token / layer_count
    formula_kib = 2 * layer_count * model.config.hidden_size * 2 / 1024
    rwkv_state_kib_layer = 5 * model.config.hidden_size * 2 / 1024

    print(
        f"MEASURED: layers={layer_count}  K shape={tuple(key0.shape)} dtype={key0.dtype}"
    )
    print(
        f"  total KV bytes for T={args.sequence_length}: {total_bytes:,}  ->  {kib_per_token:.1f} KiB per token  "
        f"({kib_per_token_layer:.1f} KiB per token per layer)"
    )
    print(
        f"FORMULA 2*L*d*2B = {formula_kib:.1f} KiB per token   "
        f"[{'MATCH' if abs(kib_per_token - formula_kib) < 1 else 'MISMATCH'}]"
    )
    print(
        f"RWKV state: {rwkv_state_kib_layer:.0f} KiB per layer, CONSTANT in sequence length "
        "(vs KV which grows linearly)"
    )

    configurations = [("pure-Pythia", None, layer_count)]
    if args.direction == "A_to_B":
        configurations.extend(
            (f"A_to_B@{switch}", switch, layer_count - (switch + 1))
            for switch in (int(value) for value in args.switches.split(","))
        )
    else:
        configurations.extend(
            (f"B_to_A@{switch}", switch, switch + 1)
            for switch in (int(value) for value in args.switches.split(","))
        )
    configurations.append(("pure-RWKV", None, 0))
    rows = []
    source_model, target_model = (
        ("RWKV-Raven-7B", "Tulu-Pythia-6.9B")
        if args.direction == "A_to_B"
        else ("Tulu-Pythia-6.9B", "RWKV-Raven-7B")
    )
    for config, switch, transformer_layers in configurations:
        config_kib_per_token = kib_per_token_layer * transformer_layers
        rows.append(
            {
                "direction": args.direction,
                "source_model": source_model,
                "target_model": target_model,
                "config": config,
                "switch_layer": "" if switch is None else switch,
                "transformer_layers": transformer_layers,
                "kv_kib_per_token": config_kib_per_token,
                "kv_gib_at_context": config_kib_per_token
                * args.context_for_table
                / 1024
                / 1024,
                "context_tokens": args.context_for_table,
                "kv_reduction_fraction": 1 - transformer_layers / layer_count,
                "measured_kib_per_token_per_layer": kib_per_token_layer,
                "rwkv_state_kib_per_layer": rwkv_state_kib_layer,
                "probe_sequence_tokens": args.sequence_length,
            }
        )
    output_path = write_csv(args.out, rows)
    print(f"wrote {output_path}\nKV_DONE")


if __name__ == "__main__":
    main()
