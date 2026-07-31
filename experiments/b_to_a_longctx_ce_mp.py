"""Long-context CE for the B-to-A Pythia-prefix -> RWKV-suffix path."""

from __future__ import annotations

import argparse
import math
import os
import sys

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
    fit_b_to_a_affine_maps,
    relative_error,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    parser.add_argument("--ctx", default="512:48,1024:32,2048:20")
    parser.add_argument("--dev_a", default="cuda:0")
    parser.add_argument("--dev_b", default="cuda:1")
    parser.add_argument("--fit_rows", type=int, default=40000)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--concurrent_arm_max_tokens",
        type=int,
        default=512,
        help="run batch-1 adapter/linear arms on separate CUDA streams up to this length",
    )
    parser.add_argument("--text", default="wikitext", choices=["wikitext", "alpaca"])
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if not args.out:
        args.out = f"artifacts/metrics/raw/b_to_a_longctx_{args.text}.json"

    dev_a, dev_b = args.dev_a, args.dev_b
    switches = [int(value) for value in args.switches.split(",")]
    plan = [
        (int(specification.split(":")[0]), int(specification.split(":")[1]))
        for specification in args.ctx.split(",")
    ]
    torch.manual_seed(0)

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
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    adapter = adapter.half()
    # Only E_B and D_A participate in B-to-A CE.
    adapter.eA.to("cpu")
    adapter.dB.to("cpu")
    stats = {
        int(layer): [tensor.to(dev_a) for tensor in values]
        for layer, values in checkpoint["stats"].items()
    }
    print(
        f"adapter {os.path.basename(args.adapter)} B->A={checkpoint['eval']['B->A']:+.4f} "
        f"direction=Pythia->RWKV plan={plan}",
        flush=True,
    )

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

    from datasets import load_dataset

    if args.text == "wikitext":
        text = "".join(
            load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split="test",
            )["text"]
        )
        all_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    else:
        text = "\n\n".join(load_dataset("tatsu-lab/alpaca", split="train")["text"])
        all_ids = tokenizer(
            text[2_000_000:2_800_000],
            return_tensors="pt",
        ).input_ids[0]
    fit_token_ids = all_ids[: args.fit_rows]
    required_evaluation = max(windows * context for context, windows in plan)
    evaluation_ids = all_ids[args.fit_rows : args.fit_rows + required_evaluation + 10]
    if evaluation_ids.shape[0] < required_evaluation:
        raise ValueError(
            f"not enough {args.text} tokens: have={evaluation_ids.shape[0]} "
            f"need={required_evaluation}"
        )
    print(
        f"text={args.text} tokens={all_ids.shape[0]} fit={fit_token_ids.shape[0]} "
        f"eval_pool={evaluation_ids.shape[0]} (disjoint)",
        flush=True,
    )

    fit_window = 128
    fit_windows = fit_token_ids.shape[0] // fit_window
    fit_matrix = fit_token_ids[: fit_windows * fit_window].reshape(
        fit_windows,
        fit_window,
    )
    linear_maps = fit_b_to_a_affine_maps(
        runtime,
        fit_matrix,
        switches,
        ridge=args.ridge,
    )
    torch.cuda.empty_cache()
    print(f"B-to-A affine maps fit for {switches}", flush=True)

    gate_errors = runtime.gate_parent_reproduction(probe, switches)
    direct_gate_errors = runtime.gate_direct_suffix(probe, switches)
    print(
        "parent-reproduction gate passed "
        + " ".join(f"L{layer}={error:.1e}" for layer, error in gate_errors.items()),
        flush=True,
    )
    print(
        "direct-suffix gate passed "
        + " ".join(
            f"L{layer}={error:.1e}" for layer, error in direct_gate_errors.items()
        ),
        flush=True,
    )

    # The two controls remain separate batch-1 computations. Gate stream
    # concurrency before using it; changing the RWKV batch dimension is not
    # numerically equivalent and is never used.
    with torch.no_grad():
        source_probe = pythia(
            probe.to(dev_b),
            output_hidden_states=True,
            use_cache=False,
        )
        concurrent_arm_errors = {}
        for layer in switches:
            hidden_b = source_probe.hidden_states[layer + offsets.pythia]
            probe_arms = (
                runtime.translate_adapter(hidden_b, layer),
                runtime.translate_linear(
                    hidden_b,
                    layer,
                    linear_maps[layer],
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
            errors = [
                relative_error(concurrent, sequential)
                for concurrent, sequential in zip(
                    concurrent_logits,
                    sequential_logits,
                )
            ]
            if max(errors) >= 1e-4:
                raise AssertionError(
                    f"concurrent-arm gate failed at L={layer}: {max(errors):.3e}"
                )
            concurrent_arm_errors[layer] = max(errors)
    print(
        "concurrent batch-1 arm gate passed "
        + " ".join(
            f"L{layer}={error:.1e}" for layer, error in concurrent_arm_errors.items()
        ),
        flush=True,
    )

    # Capture only the four post-block Pythia residuals needed for switching.
    captured: dict[int, torch.Tensor] = {}
    wanted = set(switches)

    def make_capture(block_index):
        def capture(module, inputs, output):
            del module, inputs
            if block_index in wanted:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[block_index + offsets.pythia] = hidden

        return capture

    handles = [
        pythia.gpt_neox.layers[index].register_forward_hook(make_capture(index))
        for index in switches
    ]
    with torch.no_grad():
        captured.clear()
        gate_source = pythia(
            probe.to(dev_b),
            output_hidden_states=True,
            use_cache=False,
        )
        for layer in switches:
            index = layer + offsets.pythia
            error = relative_error(captured[index], gate_source.hidden_states[index])
            if error >= 1e-4:
                raise AssertionError(
                    f"Pythia capture gate failed at L={layer}: {error:.3e}"
                )
    print("Pythia residual-hook capture gate passed", flush=True)

    cross_entropy = torch.nn.functional.cross_entropy

    @torch.no_grad()
    def window_ce(logits, ids, chunk=512):
        values = logits[0, :-1]
        targets = ids[0, 1:].to(values.device)
        total = 0.0
        for start in range(0, values.shape[0], chunk):
            total += cross_entropy(
                values[start : start + chunk].float(),
                targets[start : start + chunk],
                reduction="sum",
            ).item()
        return total / values.shape[0]

    configs = (
        ["pure-Pythia", "pure-RWKV"]
        + [f"B_to_A@{layer}" for layer in switches]
        + [f"B_to_A_affine@{layer}" for layer in switches]
    )
    output = {
        "schema_version": 1,
        "direction": "B_to_A",
        "source_model": "Tulu-Pythia-6.9B",
        "target_model": "RWKV-Raven-7B",
        "boundary": "Pythia post-block L -> RWKV pre-block L+1",
        "adapter": os.path.basename(args.adapter),
        "adapter_dtype": str(next(adapter.parameters()).dtype),
        "checkpoint_step": checkpoint.get("step", ""),
        "checkpoint_b_to_a_r2": checkpoint["eval"]["B->A"],
        "text": args.text,
        "switches": switches,
        "fit_rows_requested": args.fit_rows,
        "fit_rows_actual": int(fit_windows * fit_window),
        "linear_ridge": args.ridge,
        "evaluation_windows": {str(context): windows for context, windows in plan},
        "parent_reproduction_relative_error": {
            str(layer): error for layer, error in gate_errors.items()
        },
        "direct_suffix_relative_error": {
            str(layer): error for layer, error in direct_gate_errors.items()
        },
        "concurrent_arm_relative_error": {
            str(layer): error for layer, error in concurrent_arm_errors.items()
        },
        "concurrent_arm_max_tokens": args.concurrent_arm_max_tokens,
        "ctx": {},
    }

    for context, window_count in plan:
        windows = evaluation_ids[: window_count * context].reshape(
            window_count,
            context,
        )
        observations = {name: [] for name in configs}
        for window_index in range(window_count):
            ids_a = windows[window_index : window_index + 1].to(dev_a)
            ids_b = windows[window_index : window_index + 1].to(dev_b)
            with torch.no_grad():
                captured.clear()
                output_b = pythia(ids_b, use_cache=False)
                output_a = rwkv(ids_a, use_cache=False)
                observations["pure-Pythia"].append(window_ce(output_b.logits, ids_b))
                observations["pure-RWKV"].append(window_ce(output_a.logits, ids_a))
                for layer in switches:
                    hidden_b = captured[layer + offsets.pythia]
                    translated_adapter = runtime.translate_adapter(hidden_b, layer)
                    translated_linear = runtime.translate_linear(
                        hidden_b,
                        layer,
                        linear_maps[layer],
                    )
                    if context <= args.concurrent_arm_max_tokens:
                        adapter_logits, linear_logits = (
                            runtime.run_rwkv_suffixes_concurrent(
                                [translated_adapter, translated_linear],
                                layer,
                            )
                        )
                    else:
                        adapter_logits = runtime.run_rwkv_suffix_direct(
                            translated_adapter,
                            layer,
                        )
                        linear_logits = runtime.run_rwkv_suffix_direct(
                            translated_linear,
                            layer,
                        )
                    observations[f"B_to_A@{layer}"].append(
                        window_ce(adapter_logits, ids_a)
                    )
                    observations[f"B_to_A_affine@{layer}"].append(
                        window_ce(linear_logits, ids_a)
                    )
            if (window_index + 1) % 4 == 0:
                print(
                    f"ctx={context} [{window_index + 1}/{window_count}]",
                    flush=True,
                )

        def statistics(values):
            count = len(values)
            mean = sum(values) / count
            standard_deviation = (
                (sum((value - mean) ** 2 for value in values) / (count - 1)) ** 0.5
                if count > 1
                else 0.0
            )
            return mean, standard_deviation / count**0.5

        output["ctx"][context] = {
            name: {
                "ce": statistics(observations[name])[0],
                "se": statistics(observations[name])[1],
            }
            for name in configs
        }
        print(f"== B-to-A {args.text} ctx={context} windows={window_count} ==")
        for name in configs:
            mean, standard_error = statistics(observations[name])
            print(
                f"  {name:14s} CE={mean:.4f} +-{standard_error:.4f} "
                f"ppl={math.exp(mean):.2f}",
                flush=True,
            )

    rows = [
        {
            "direction": "B_to_A",
            "source_model": output["source_model"],
            "target_model": output["target_model"],
            "domain": args.text,
            "context_tokens": int(context),
            "windows": dict(plan)[int(context)],
            "config": config_name,
            "config_type": config_name.rsplit("@", 1)[0],
            "switch_layer": (
                config_name.rsplit("@", 1)[1] if "@" in config_name else ""
            ),
            "cross_entropy": values["ce"],
            "standard_error": values["se"],
            "perplexity": math.exp(values["ce"]),
        }
        for context, configuration_values in output["ctx"].items()
        for config_name, values in configuration_values.items()
    ]
    json_path, csv_path = write_bundle(args.out, output, rows)
    for handle in handles:
        handle.remove()
    print(f"wrote {json_path} + {csv_path}\nB_TO_A_LONGCTX_DONE", flush=True)


if __name__ == "__main__":
    main()
