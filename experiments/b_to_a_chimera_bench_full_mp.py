"""Full-set QA for the B-to-A Pythia-prefix -> adapter -> RWKV-suffix chimera.

This is the direction-symmetric counterpart of
``a_to_b_chimera_bench_full_mp.py``.
For every switch L it evaluates:

* ``B_to_A@L``: D_A(E_B(Pythia residual)) followed by RWKV blocks L+1..31;
* ``B_to_A_alpha0@L``: the same path with every learned residual-MLP branch
  scaled to zero, a same-adapter causal intervention;
* ``B_to_A_affine@L``: a separately fitted affine standardized B-to-A map;
* ``A_to_A@L``: D_A(E_A(RWKV residual)) injected into RWKV, isolating A-decoder loss;
* both unmodified parents.

The JSON contains item-aligned correctness for paired tests.  A result is never
emitted unless injecting the true post-L RWKV residual reproduces pure RWKV at
every requested switch.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ninaxander import adapter as adapter_module
from ninaxander.adapter import LatentAdapter
from ninaxander.result_io import write_bundle
from ninaxander.b_to_a_chimera import (
    BToAChimera,
    detect_residual_offsets,
    fit_b_to_a_affine_maps,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


_ALPHA = {"value": 1.0}


def _scaled_resblock_forward(module, value):
    return value + _ALPHA["value"] * module.fc2(
        F.gelu(module.fc1(module.norm(value)))
    )


adapter_module.ResBlock.forward = _scaled_resblock_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="artifacts/checkpoints/z4096_L32/SNAP_L32_final.pt")
    parser.add_argument("--rwkv", default="models/rwkv-raven-7b-hf-fp16")
    parser.add_argument("--pythia", default="models/tulu-pythia69-fp16")
    parser.add_argument("--tok", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--hid", type=int, default=4096)
    parser.add_argument("--arch", default="resnet")
    parser.add_argument("--task", default="arc_easy", choices=["arc_easy", "sciq"])
    parser.add_argument("--n", type=int, default=0, help="0 = full test set")
    parser.add_argument("--switches", default="4,8,16,24")
    parser.add_argument("--dev_a", default="cuda:0")
    parser.add_argument("--dev_b", default="cuda:1")
    parser.add_argument("--fit_rows", type=int, default=40000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--option_batch",
        type=int,
        default=1,
        help="answer options per GPU microbatch; must remain 1 for canonical numerics",
    )
    parser.add_argument(
        "--concurrent_arm_max_tokens",
        type=int,
        default=96,
        help="run batch-1 adapter/linear/self arms on separate CUDA streams up to this length",
    )
    parser.add_argument("--dump", default="")
    args = parser.parse_args()
    if not args.dump:
        task_name = "arc" if args.task == "arc_easy" else args.task
        args.dump = f"artifacts/metrics/raw/b_to_a_qa_items_{task_name}.json"
    if args.option_batch != 1:
        raise ValueError(
            "B-to-A QA requires --option_batch 1: RWKV changes numerically "
            "with batch size, so answer options must use the canonical batch-1 path"
        )

    dev_a, dev_b = args.dev_a, args.dev_b
    switches = [int(value) for value in args.switches.split(",")]
    torch.manual_seed(0)

    checkpoint = torch.load(args.adapter, map_location="cpu", weights_only=False)
    config = checkpoint["cfg"]
    adapter = LatentAdapter(
        config["dA"],
        config["dB"],
        config["z"],
        blocks=args.blocks,
        hidden=args.hid,
        arch=args.arch,
    ).to(dev_a).eval()
    adapter.load_state_dict(checkpoint["model"])
    # B->A QA needs E_B/D_A and the A->A control needs E_A/D_A; D_B is
    # unreachable in every arm and remains on CPU to preserve GPU headroom.
    adapter.dB.to("cpu")
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    stats = {
        int(layer): [tensor.to(dev_a) for tensor in values]
        for layer, values in checkpoint["stats"].items()
    }
    print(
        f"adapter {os.path.basename(args.adapter)} B->A={checkpoint['eval']['B->A']:+.4f} "
        f"direction=Pythia->RWKV devA={dev_a} devB={dev_b}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tok)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    print(f"off: RWKV={offsets.rwkv} Pythia={offsets.pythia}", flush=True)

    # Fit the B-to-A affine baseline on the same held-out Alpaca region and
    # number of residual rows used by the A-to-B benchmark.
    from datasets import load_dataset

    calibration_text = "\n\n".join(
        load_dataset("tatsu-lab/alpaca", split="train")["text"]
    )
    window = 112
    windows = args.fit_rows // window + 1
    fit_ids = tokenizer(
        calibration_text[
            2_000_000 : 2_000_000 + windows * window * 8
        ],
        return_tensors="pt",
    ).input_ids[0][: windows * window].reshape(windows, window)
    linear_maps = fit_b_to_a_affine_maps(
        runtime,
        fit_ids,
        switches,
        ridge=args.ridge,
    )
    torch.cuda.empty_cache()
    print(f"B-to-A affine maps fit for {switches}", flush=True)

    gate_errors = runtime.gate_parent_reproduction(probe, switches)
    direct_gate_errors = runtime.gate_direct_suffix(probe, switches)
    print(
        "B-TO-A GATE passed: true RWKV residual reproduces pure RWKV "
        + " ".join(f"L{layer}={error:.1e}" for layer, error in gate_errors.items()),
        flush=True,
    )
    print(
        "DIRECT SUFFIX GATE passed "
        + " ".join(
            f"L{layer}={error:.1e}"
            for layer, error in direct_gate_errors.items()
        ),
        flush=True,
    )

    # Gate independent batch-1 CUDA streams against sequential batch-1 calls.
    # Concatenating arms along the batch axis is deliberately forbidden because
    # RWKV's numerical path changes measurably with batch size.
    with torch.no_grad():
        probe_a = rwkv(
            probe.to(dev_a),
            output_hidden_states=True,
            use_cache=False,
        )
        probe_b = pythia(
            probe.to(dev_b),
            output_hidden_states=True,
            use_cache=False,
        )
        concurrent_arm_errors = {}
        for layer in switches:
            _ALPHA["value"] = 1.0
            probe_adapter = runtime.translate_adapter(
                probe_b.hidden_states[layer + offsets.pythia],
                layer,
            )
            _ALPHA["value"] = 0.0
            probe_alpha0 = runtime.translate_adapter(
                probe_b.hidden_states[layer + offsets.pythia],
                layer,
            )
            _ALPHA["value"] = 1.0
            probe_arms = (
                probe_adapter,
                probe_alpha0,
                runtime.translate_linear(
                    probe_b.hidden_states[layer + offsets.pythia],
                    layer,
                    linear_maps[layer],
                ),
                runtime.translate_self_a(
                    probe_a.hidden_states[layer + offsets.rwkv],
                    layer,
                ),
            )
            sequential_logits = [
                runtime.run_rwkv_suffix_direct(translated, layer)
                for translated in probe_arms
            ]
            concurrent_logits = runtime.run_rwkv_suffixes_concurrent(
                probe_arms,
                layer,
            )
            layer_errors = []
            for arm_index, (concurrent, sequential) in enumerate(
                zip(concurrent_logits, sequential_logits)
            ):
                error = (
                    (concurrent - sequential)
                    .float()
                    .norm()
                    .item()
                    / (sequential.float().norm().item() + 1e-9)
                )
                if error >= 1e-4:
                    raise AssertionError(
                        f"concurrent-arm gate failed L={layer} arm={arm_index}: "
                        f"{error:.3e}"
                    )
                layer_errors.append(error)
            concurrent_arm_errors[layer] = max(layer_errors)
    print(
        "CONCURRENT BATCH-1 ARM GATE passed "
        + " ".join(
            f"L{layer}={error:.1e}"
            for layer, error in concurrent_arm_errors.items()
        ),
        flush=True,
    )

    items: list[tuple[str, list[str], int]] = []
    if args.task == "arc_easy":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        for row in dataset:
            options = row["choices"]["text"]
            labels = row["choices"]["label"]
            if row["answerKey"] not in labels:
                continue
            items.append(
                (
                    f"Question: {row['question']}\nAnswer:",
                    [" " + option for option in options],
                    labels.index(row["answerKey"]),
                )
            )
    else:
        dataset = load_dataset("allenai/sciq", split="test")
        for row in dataset:
            options = [
                row["correct_answer"],
                row["distractor1"],
                row["distractor2"],
                row["distractor3"],
            ]
            items.append(
                (
                    f"Question: {row['question']}\nAnswer:",
                    [" " + option for option in options],
                    0,
                )
            )
    if args.n > 0:
        items = items[: args.n]
    print(f"task={args.task} items={len(items)}", flush=True)

    configs = (
        ["pythia", "rwkv"]
        + [f"B_to_A@{layer}" for layer in switches]
        + [f"B_to_A_alpha0@{layer}" for layer in switches]
        + [f"B_to_A_affine@{layer}" for layer in switches]
        + [f"A_to_A@{layer}" for layer in switches]
    )
    hit_sum = {name: 0 for name in configs}
    hit_norm = {name: 0 for name in configs}
    per_item = {name: [] for name in configs}

    @torch.no_grad()
    def option_log_probability(
        logits,
        token_ids,
        prompt_tokens,
        sequence_tokens,
        batch_index=0,
    ):
        log_probabilities = torch.log_softmax(
            logits[batch_index, : sequence_tokens - 1].float(),
            dim=-1,
        )
        targets = token_ids[
            batch_index,
            1:sequence_tokens,
        ].to(log_probabilities.device)
        selected = log_probabilities[
            torch.arange(len(targets), device=log_probabilities.device),
            targets,
        ][prompt_tokens - 1 :]
        return selected.sum().item(), selected.mean().item()

    for item_index, (prompt_text, options, gold) in enumerate(items):
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]
        scores_sum = {name: [] for name in configs}
        scores_norm = {name: [] for name in configs}
        for option_start in range(0, len(options), args.option_batch):
            option_batch = options[
                option_start : option_start + args.option_batch
            ]
            encoded = tokenizer(
                [prompt_text + option for option in option_batch],
                return_tensors="pt",
                padding=True,
            )
            token_ids = encoded.input_ids
            sequence_lengths = encoded.attention_mask.sum(dim=1).tolist()
            ids_a = token_ids.to(dev_a)
            ids_b = token_ids.to(dev_b)
            mask_b = encoded.attention_mask.to(dev_b)
            with torch.no_grad():
                output_a = rwkv(
                    ids_a,
                    output_hidden_states=True,
                    use_cache=False,
                )
                output_b = pythia(
                    ids_b,
                    attention_mask=mask_b,
                    output_hidden_states=True,
                    use_cache=False,
                )
                for option_index, sequence_tokens in enumerate(sequence_lengths):
                    total, normalized = option_log_probability(
                        output_a.logits,
                        ids_a,
                        prompt_tokens,
                        sequence_tokens,
                        batch_index=option_index,
                    )
                    scores_sum["rwkv"].append(total)
                    scores_norm["rwkv"].append(normalized)
                    total, normalized = option_log_probability(
                        output_b.logits,
                        ids_b,
                        prompt_tokens,
                        sequence_tokens,
                        batch_index=option_index,
                    )
                    scores_sum["pythia"].append(total)
                    scores_norm["pythia"].append(normalized)

                for layer in switches:
                    hidden_b = output_b.hidden_states[layer + offsets.pythia]
                    hidden_a = output_a.hidden_states[layer + offsets.rwkv]
                    _ALPHA["value"] = 1.0
                    translated_adapter = runtime.translate_adapter(hidden_b, layer)
                    _ALPHA["value"] = 0.0
                    translated_alpha0 = runtime.translate_adapter(hidden_b, layer)
                    _ALPHA["value"] = 1.0
                    arms = (
                        (
                            f"B_to_A@{layer}",
                            translated_adapter,
                        ),
                        (
                            f"B_to_A_alpha0@{layer}",
                            translated_alpha0,
                        ),
                        (
                            f"B_to_A_affine@{layer}",
                            runtime.translate_linear(
                                hidden_b,
                                layer,
                                linear_maps[layer],
                            ),
                        ),
                        (
                            f"A_to_A@{layer}",
                            runtime.translate_self_a(hidden_a, layer),
                        ),
                    )
                    if max(sequence_lengths) <= args.concurrent_arm_max_tokens:
                        arm_logits = runtime.run_rwkv_suffixes_concurrent(
                            [translated for _, translated in arms],
                            layer,
                        )
                        for (name, _), logits in zip(arms, arm_logits):
                            for option_index, sequence_tokens in enumerate(
                                sequence_lengths
                            ):
                                total, normalized = option_log_probability(
                                    logits,
                                    ids_a,
                                    prompt_tokens,
                                    sequence_tokens,
                                    batch_index=option_index,
                                )
                                scores_sum[name].append(total)
                                scores_norm[name].append(normalized)
                    else:
                        for name, translated in arms:
                            logits = runtime.run_rwkv_suffix_direct(
                                translated,
                                layer,
                            )
                            for option_index, sequence_tokens in enumerate(
                                sequence_lengths
                            ):
                                total, normalized = option_log_probability(
                                    logits,
                                    ids_a,
                                    prompt_tokens,
                                    sequence_tokens,
                                    batch_index=option_index,
                                )
                                scores_sum[name].append(total)
                                scores_norm[name].append(normalized)

        for name in configs:
            correct_sum = int(
                max(range(len(options)), key=lambda index: scores_sum[name][index])
                == gold
            )
            correct_norm = int(
                max(range(len(options)), key=lambda index: scores_norm[name][index])
                == gold
            )
            hit_sum[name] += correct_sum
            hit_norm[name] += correct_norm
            per_item[name].append(correct_norm)
        if (item_index + 1) % 100 == 0:
            shown = ["pythia", "rwkv"] + [
                f"B_to_A@{layer}" for layer in switches
            ]
            print(
                f"[{item_index + 1}/{len(items)}] "
                + " ".join(
                    f"{name}={100 * hit_norm[name] / (item_index + 1):.0f}"
                    for name in shown
                ),
                flush=True,
            )

    item_count = len(items)
    print(f"\n== B-to-A {args.task} N={item_count} accuracy (%) ==")
    for name in configs:
        print(
            f"{name:14s} sum={100 * hit_sum[name] / item_count:5.1f} "
            f"norm={100 * hit_norm[name] / item_count:5.1f}"
        )

    payload = {
        "schema_version": 2,
        "direction": "B_to_A",
        "source_model": "Tulu-Pythia-6.9B",
        "target_model": "RWKV-Raven-7B",
        "boundary": "Pythia post-block L -> RWKV pre-block L+1",
        "task": args.task,
        "N": item_count,
        "cfgs": configs,
        "switches": switches,
        "gold_chance": 1.0 / len(items[0][1]),
        "parent_reproduction_relative_error": {
            str(layer): error for layer, error in gate_errors.items()
        },
        "direct_suffix_relative_error": {
            str(layer): error for layer, error in direct_gate_errors.items()
        },
        "concurrent_arm_relative_error": {
            str(layer): error for layer, error in concurrent_arm_errors.items()
        },
        "option_batch": args.option_batch,
        "concurrent_arm_max_tokens": args.concurrent_arm_max_tokens,
        "checkpoint": os.path.basename(args.adapter),
        "checkpoint_step": checkpoint.get("step", ""),
        "checkpoint_b_to_a_r2": checkpoint["eval"]["B->A"],
        "linear_fit_rows_requested": args.fit_rows,
        "linear_ridge": args.ridge,
        "same_adapter_intervention": {
            "branch": "ResBlock.fc2(GELU(fc1(LayerNorm(x))))",
            "alphas": [0.0, 1.0],
            "alpha0_config_prefix": "B_to_A_alpha0",
        },
        "per_item_norm": per_item,
        "acc_norm": {
            name: hit_norm[name] / item_count
            for name in configs
        },
        "acc_sum": {
            name: hit_sum[name] / item_count
            for name in configs
        },
    }
    rows = [
        {
            "direction": "B_to_A",
            "source_model": payload["source_model"],
            "target_model": payload["target_model"],
            "task": args.task,
            "config": name,
            "switch_layer": name.rsplit("@", 1)[1] if "@" in name else "",
            "accuracy_sum": payload["acc_sum"][name],
            "accuracy_norm": payload["acc_norm"][name],
            "n_items": item_count,
            "chance": payload["gold_chance"],
        }
        for name in configs
    ]
    json_path, csv_path = write_bundle(args.dump, payload, rows)
    print(f"wrote {json_path} + {csv_path}\nB_TO_A_BENCHFULL_DONE", flush=True)


if __name__ == "__main__":
    main()
