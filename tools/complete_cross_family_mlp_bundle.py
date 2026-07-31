#!/usr/bin/env python3
"""Complete the matched AB/BA same-adapter intervention bundles.

The representation sweep is produced before the two downstream evaluators
finish. This post-processing step joins the item-aligned AB and BA alpha=0/1
QA outcomes into the same task-level JSON/CSV bundle, so a file named
``cross_family`` never contains only one downstream direction.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ninaxander.result_io import write_bundle  # noqa: E402


TASKS = {
    "arc": ("arc_easy", 2376),
    "sciq": ("sciq", 1000),
}
LAYERS = (4, 8, 16, 24)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_qa(
    data: dict[str, Any],
    direction: str,
    count: int,
    source_file: str,
) -> dict[str, Any]:
    configs = [
        *(f"{direction}@{layer}" for layer in LAYERS),
        *(f"{direction}_alpha0@{layer}" for layer in LAYERS),
    ]
    per_item = data.get("per_item_norm", {})
    accuracies = data.get("acc_norm", {})
    for config in configs:
        outcomes = per_item.get(config)
        if not isinstance(outcomes, list) or len(outcomes) != count:
            raise ValueError(
                f"{source_file}: per_item_norm[{config}] does not have {count} rows"
            )
        if any(value not in (0, 1) for value in outcomes):
            raise ValueError(f"{source_file}: non-binary outcomes for {config}")
        observed = sum(outcomes) / count
        if abs(float(accuracies[config]) - observed) > 1e-12:
            raise ValueError(
                f"{source_file}: acc_norm[{config}] disagrees with per-item outcomes"
            )
    return {
        "N": count,
        "configs": configs,
        "acc_norm": {config: accuracies[config] for config in configs},
        "per_item_norm": {config: per_item[config] for config in configs},
        "source_file": source_file,
    }


def rows_for_bundle(
    data: dict[str, Any],
    task: str,
    source_file: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction, alpha_values in data["r2_by_direction_alpha_layer"].items():
        for alpha, layer_values in alpha_values.items():
            for layer, value in layer_values.items():
                rows.append(
                    {
                        "result_type": "representation_r2",
                        "direction": direction,
                        "task": "held_out_residuals",
                        "alpha": float(alpha),
                        "layer": int(layer),
                        "config": "",
                        "value": value,
                        "n_items": "",
                        "source_file": source_file,
                    }
                )
    for direction, qa in data["qa_by_direction"].items():
        for config in qa["configs"]:
            layer = int(config.rsplit("@", 1)[1])
            rows.append(
                {
                    "result_type": "qa_accuracy_norm",
                    "direction": direction,
                    "task": task,
                    "alpha": 0.0 if "_alpha0@" in config else 1.0,
                    "layer": layer,
                    "config": config,
                    "value": qa["acc_norm"][config],
                    "n_items": qa["N"],
                    "source_file": source_file,
                }
            )
    return rows


def representation_rows(
    data: dict[str, Any],
    source_file: str,
) -> list[dict[str, Any]]:
    return [
        {
            "result_type": "representation_r2",
            "direction": direction,
            "task": "held_out_residuals",
            "alpha": float(alpha),
            "layer": int(layer),
            "config": "",
            "value": value,
            "n_items": "",
            "source_file": source_file,
        }
        for direction, alpha_values in data["r2_by_direction_alpha_layer"].items()
        for alpha, layer_values in alpha_values.items()
        for layer, value in layer_values.items()
    ]


def load_representation_intervention(raw_dir: Path) -> dict[str, Any]:
    """Load the canonical sweep, migrating one old task bundle when necessary."""

    path = raw_dir / "cross_family_mlp_representation.json"
    if path.is_file():
        return load(path)

    candidates = (
        raw_dir / "cross_family_mlp_ablation_sciq.json",
        raw_dir / "cross_family_mlp_ablation_arc.json",
    )
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        raise FileNotFoundError(path)
    old = load(source)
    keys = (
        "adapter",
        "checkpoint_step",
        "directions",
        "alphas",
        "r2_layers",
        "evaluation_windows",
        "window_tokens",
        "r2_by_direction_alpha_mean",
        "r2_by_direction_alpha_layer",
    )
    data = {
        "schema_version": 2,
        **{key: old[key] for key in keys if key in old},
        "migration": {
            "kind": "lossless_projection",
            "source_file": source.name,
        },
    }
    json_path, csv_path = write_bundle(
        path,
        data,
        representation_rows(data, path.name),
    )
    print(f"{json_path}: migrated matched AB/BA sweep; {csv_path}: 50 rows")
    return data


def qa_dump_rows(
    data: dict[str, Any],
    direction: str,
) -> list[dict[str, Any]]:
    source_model, target_model = (
        (
            data.get("source_model", "RWKV-Raven-7B"),
            data.get("target_model", "Tulu-Pythia-6.9B"),
        )
        if direction == "A_to_B"
        else (
            data.get("source_model", "Tulu-Pythia-6.9B"),
            data.get("target_model", "RWKV-Raven-7B"),
        )
    )
    return [
        {
            "direction": direction,
            "source_model": source_model,
            "target_model": target_model,
            "task": data["task"],
            "config": config,
            "switch_layer": config.rsplit("@", 1)[1] if "@" in config else "",
            "accuracy_sum": data.get("acc_sum", {}).get(config, ""),
            "accuracy_norm": data["acc_norm"][config],
            "n_items": data["N"],
            "chance": data["gold_chance"],
        }
        for config in data["cfgs"]
    ]


def fold_a_to_b_intervention(
    raw_dir: Path,
    suffix: str,
    task: str,
    count: int,
) -> dict[str, Any]:
    """Move legacy AB alpha-zero outcomes into the canonical 18-arm QA dump."""

    canonical_path = raw_dir / f"a_to_b_qa_items_{suffix}.json"
    if not canonical_path.is_file():
        raise FileNotFoundError(canonical_path)
    data = load(canonical_path)
    alpha_configs = [f"A_to_B_alpha0@{layer}" for layer in LAYERS]
    present = [config in data.get("cfgs", []) for config in alpha_configs]
    if all(present):
        return data
    if any(present):
        raise ValueError(f"{canonical_path.name}: partial A-to-B alpha-zero arms")

    intermediate_path = raw_dir / f"mlp_items_{suffix}.json"
    if not intermediate_path.is_file():
        raise FileNotFoundError(
            f"{canonical_path.name} lacks alpha-zero arms and "
            f"{intermediate_path.name} is absent"
        )
    intervention = load(intermediate_path)
    selected = selected_qa(
        intervention,
        "A_to_B",
        count,
        intermediate_path.name,
    )
    if data.get("task") != task or data.get("N") != count:
        raise ValueError(f"{canonical_path.name}: task/count mismatch")
    for config in (f"A_to_B@{layer}" for layer in LAYERS):
        if (
            data.get("per_item_norm", {}).get(config)
            != selected["per_item_norm"][config]
        ):
            raise ValueError(
                f"{canonical_path.name} and {intermediate_path.name} disagree "
                f"on {config}"
            )

    configs = list(data["cfgs"])
    insertion = next(
        (
            index
            for index, config in enumerate(configs)
            if config.startswith("A_to_B_affine@")
        ),
        len(configs),
    )
    configs[insertion:insertion] = alpha_configs
    data["cfgs"] = configs
    data.setdefault("per_item_norm", {})
    data.setdefault("acc_norm", {})
    data.setdefault("acc_sum", {})
    for config in alpha_configs:
        data["per_item_norm"][config] = selected["per_item_norm"][config]
        data["acc_norm"][config] = selected["acc_norm"][config]
        # The historical intervention recorded normalized-choice outcomes only.
        # Fresh canonical runs emit both accuracy variants directly.
        data["acc_sum"][config] = ""

    data.update(
        {
            "schema_version": 2,
            "direction": "A_to_B",
            "source_model": "RWKV-Raven-7B",
            "target_model": "Tulu-Pythia-6.9B",
            "boundary": "RWKV post-block L -> Pythia pre-block L+1",
            "switches": list(LAYERS),
            "checkpoint_step": data.get("checkpoint_step", 415_000),
            "linear_fit_rows_requested": data.get(
                "linear_fit_rows_requested",
                40_000,
            ),
            "linear_ridge": data.get("linear_ridge", 1e-3),
            "same_adapter_intervention": {
                "branch": "ResBlock.fc2(GELU(fc1(LayerNorm(x))))",
                "alphas": [0.0, 1.0],
                "alpha0_config_prefix": "A_to_B_alpha0",
                "source_file": intermediate_path.name,
                "base_outcomes_reused": True,
            },
            "migration": {
                "kind": "item_order_preserving_fold",
                "source_file": intermediate_path.name,
                "preserved_configs": len(configs) - len(alpha_configs),
                "added_configs": len(alpha_configs),
            },
        }
    )
    json_path, csv_path = write_bundle(
        canonical_path,
        data,
        qa_dump_rows(data, "A_to_B"),
    )
    print(
        f"{json_path}: folded {len(alpha_configs)} A-to-B intervention arms; "
        f"{csv_path}: {len(configs)} rows"
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "artifacts" / "metrics" / "raw",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="retain folded legacy per-item files for debugging",
    )
    args = parser.parse_args()
    representation = load_representation_intervention(args.raw_dir)
    if set(representation.get("directions", [])) != {"A_to_B", "B_to_A"}:
        raise ValueError("cross_family_mlp_representation.json lacks AB/BA")
    if set(representation.get("r2_by_direction_alpha_layer", {})) != {
        "A_to_B",
        "B_to_A",
    }:
        raise ValueError("cross_family_mlp_representation.json is incomplete")

    for suffix, (task, count) in TASKS.items():
        output = args.raw_dir / f"cross_family_mlp_intervention_{suffix}.json"
        ba_items = args.raw_dir / f"b_to_a_qa_items_{suffix}.json"
        for path in (ba_items,):
            if not path.is_file():
                raise FileNotFoundError(path)
        data = copy.deepcopy(representation)

        a_to_b = fold_a_to_b_intervention(
            args.raw_dir,
            suffix,
            task,
            count,
        )
        qa_by_direction = {
            "A_to_B": selected_qa(
                a_to_b,
                "A_to_B",
                count,
                f"a_to_b_qa_items_{suffix}.json",
            ),
            "B_to_A": selected_qa(
                load(ba_items),
                "B_to_A",
                count,
                ba_items.name,
            ),
        }
        data["schema_version"] = 2
        data["task"] = task
        data["qa_by_direction"] = qa_by_direction
        # Retain the original summary key for compatibility with old readers,
        # but make the symmetric structure authoritative.
        rows = rows_for_bundle(data, task, output.name)
        json_path, csv_path = write_bundle(output, data, rows)
        print(f"{json_path}: AB/BA QA {count} items each; {csv_path}: {len(rows)} rows")

    if not args.keep_intermediates:
        retired = [
            *(
                args.raw_dir / f"mlp_items_{suffix}.{extension}"
                for suffix in TASKS
                for extension in ("json", "csv")
            ),
            args.raw_dir / "paired_statistics_mlp.csv",
            *(
                args.raw_dir / f"cross_family_mlp_ablation_{suffix}.{extension}"
                for suffix in TASKS
                for extension in ("json", "csv")
            ),
        ]
        for path in retired:
            if path.is_file():
                path.unlink()
                print(f"removed retired intermediate {path}")

    print("CROSS_FAMILY_MLP_BUNDLES_COMPLETE")


if __name__ == "__main__":
    main()
