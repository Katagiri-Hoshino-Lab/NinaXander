#!/usr/bin/env python3
"""Build normalized, paper-facing CSV tables from raw experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SUPPORTED_SWITCH_LAYERS = (4, 8, 16, 24)
EXECUTION_PATH_ORDER = {
    "A_parent": 0,
    "A_to_A": 1,
    "A_to_B": 2,
    "B_parent": 3,
    "B_to_B": 4,
    "B_to_A": 5,
    "B_early_exit": 6,
    "cross_path": 7,
    "": 8,
}

from ninaxander.paired import (  # noqa: E402
    mcnemar_exact,
    paired_bootstrap_ci as bootstrap_ci,
)
from ninaxander.result_io import write_csv  # noqa: E402


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_parts(config: str) -> tuple[str, int | None]:
    if "@" in config:
        kind, layer = config.split("@", 1)
        return kind, int(layer)
    return config, None


def execution_path_for_config(
    config: str,
    fallback: str = "",
) -> str:
    """Classify a result by the path it actually executes, not its producer."""

    normalized = config.strip().removeprefix("pruned-")
    if normalized in {"rwkv", "pure-RWKV", "A_parent"}:
        return "A_parent"
    if normalized in {"pythia", "pure-Pythia", "B_parent"}:
        return "B_parent"
    if "earlyexit" in normalized or "early_exit" in normalized:
        # A truncated pure Pythia baseline is not the BB adapter self-map.
        return "B_early_exit"
    for path in ("A_to_A", "A_to_B", "B_to_B", "B_to_A"):
        if normalized.startswith(path):
            return path
    return fallback


def with_execution_paths(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = []
    for source in rows:
        row = dict(source)
        config = str(row.get("config", ""))
        producer_path = str(row.pop("direction", ""))
        fallback = producer_path
        if fallback == "BA_vs_AB":
            fallback = "cross_path"
        enriched.append(
            {
                "execution_path": execution_path_for_config(config, fallback),
                "producer_path": producer_path,
                **row,
            }
        )
    return enriched


def path_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    layer = row.get("switch_layer", row.get("layer", ""))
    try:
        layer_key = int(layer)
    except (TypeError, ValueError):
        layer_key = -1
    return (
        str(row.get("task", row.get("domain", ""))),
        str(row.get("benchmark", "")),
        str(row.get("phase", "")),
        int(row.get("context_tokens") or 0),
        int(row.get("prompt_id") or 0),
        layer_key,
        EXECUTION_PATH_ORDER.get(str(row.get("execution_path", "")), 99),
        str(row.get("config", "")),
        str(row.get("producer_path", "")),
    )


def canonical_path_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(with_execution_paths(rows), key=path_sort_key)


def a_to_b_qa_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(raw_dir.glob("a_to_b_qa_items_*.json")):
        data = load_json(path)
        for config in data["cfgs"]:
            kind, layer = config_parts(config)
            rows.append(
                {
                    "direction": "A_to_B",
                    "source_model": "RWKV-Raven-7B",
                    "target_model": "Tulu-Pythia-6.9B",
                    "task": data["task"],
                    "config": config,
                    "config_type": kind,
                    "switch_layer": "" if layer is None else layer,
                    "accuracy_sum": data.get("acc_sum", {}).get(config, ""),
                    "accuracy_norm": data["acc_norm"][config],
                    "n_items": data["N"],
                    "chance": data["gold_chance"],
                    "source_file": path.name,
                }
            )
        write_csv(
            path.with_suffix(".csv"),
            [dict(row) for row in rows if row["source_file"] == path.name],
        )

    for path in sorted(raw_dir.glob("a_to_b_exit_items_*.json")):
        data = load_json(path)
        layer_by_blocks = {
            int(blocks): int(layer) for layer, blocks in data["Ks"].items()
        }
        local_rows = []
        for config in data["cfgs"]:
            _, blocks = config_parts(config)
            row = {
                "direction": "A_to_B",
                "source_model": "RWKV-Raven-7B",
                "target_model": "Tulu-Pythia-6.9B",
                "task": data["task"],
                "config": config,
                "config_type": "early_exit",
                "switch_layer": layer_by_blocks[blocks],
                "transformer_blocks": blocks,
                "accuracy_sum": "",
                "accuracy_norm": data["acc_norm"][config],
                "n_items": data["N"],
                "chance": 0.25,
                "source_file": path.name,
            }
            rows.append(row)
            local_rows.append(row)
        write_csv(path.with_suffix(".csv"), local_rows)

    return rows


def b_to_a_qa_rows(raw_dir: Path) -> list[dict[str, Any]]:
    """Normalize B-to-A QA and its equal-KV shallow-Pythia controls."""

    rows = []
    for path in sorted(raw_dir.glob("b_to_a_qa_items_*.json")):
        data = load_json(path)
        local_rows = []
        for config in data["cfgs"]:
            kind, layer = config_parts(config)
            row = {
                "direction": "B_to_A",
                "source_model": data.get("source_model", "Tulu-Pythia-6.9B"),
                "target_model": data.get("target_model", "RWKV-Raven-7B"),
                "task": data["task"],
                "config": config,
                "config_type": kind,
                "switch_layer": "" if layer is None else layer,
                "accuracy_sum": data.get("acc_sum", {}).get(config, ""),
                "accuracy_norm": data["acc_norm"][config],
                "n_items": data["N"],
                "chance": data["gold_chance"],
                "source_file": path.name,
            }
            rows.append(row)
            local_rows.append(row)
        write_csv(path.with_suffix(".csv"), local_rows)

    for path in sorted(raw_dir.glob("b_to_a_exit_items_*.json")):
        data = load_json(path)
        layer_by_blocks = {
            int(blocks): int(layer) for layer, blocks in data["Ks"].items()
        }
        local_rows = []
        for config in data["cfgs"]:
            _, blocks = config_parts(config)
            row = {
                "direction": "B_to_A",
                "source_model": "Tulu-Pythia-6.9B",
                "target_model": "RWKV-Raven-7B",
                "task": data["task"],
                "config": config,
                "config_type": "early_exit",
                "switch_layer": layer_by_blocks[blocks],
                "transformer_blocks": blocks,
                "accuracy_sum": "",
                "accuracy_norm": data["acc_norm"][config],
                "n_items": data["N"],
                "chance": 0.25,
                "source_file": path.name,
            }
            rows.append(row)
            local_rows.append(row)
        write_csv(path.with_suffix(".csv"), local_rows)
    return rows


def execution_path_qa_rows(
    a_to_b_rows: list[dict[str, Any]],
    b_to_a_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the symmetric 2x2 AA/AB/BB/BA execution-path table."""

    by_key = {(row["task"], row["config"]): row for row in [*a_to_b_rows, *b_to_a_rows]}
    paths = (
        ("A_to_A", "A", "A", "same_family_control"),
        ("A_to_B", "A", "B", "cross_family_chimera"),
        ("B_to_B", "B", "B", "same_family_control"),
        ("B_to_A", "B", "A", "cross_family_chimera"),
    )
    rows = []
    for task in ("arc_easy", "sciq"):
        for layer in SUPPORTED_SWITCH_LAYERS:
            for path_name, prefix, suffix, role in paths:
                source = by_key[(task, f"{path_name}@{layer}")]
                rows.append(
                    {
                        "task": task,
                        "switch_layer": layer,
                        "execution_path": path_name,
                        "model_name": f"NinaXander-{prefix}{suffix}@{layer}",
                        "prefix_family": prefix,
                        "suffix_family": suffix,
                        "prefix_model": (
                            "RWKV-Raven-7B" if prefix == "A" else "Tulu-Pythia-6.9B"
                        ),
                        "suffix_model": (
                            "RWKV-Raven-7B" if suffix == "A" else "Tulu-Pythia-6.9B"
                        ),
                        "path_role": role,
                        "accuracy_sum": source["accuracy_sum"],
                        "accuracy_norm": source["accuracy_norm"],
                        "n_items": source["n_items"],
                        "chance": source["chance"],
                        "source_file": source["source_file"],
                    }
                )
    return rows


def representation_rows(release_config: Path) -> list[dict[str, Any]]:
    data = load_json(release_config)
    metric_order = ("A->A", "A->B", "B->B", "B->A", "rho_ctr", "rho_raw", "f")
    labels = {
        "A->B": "cross_readout_a_to_b_r2",
        "B->A": "cross_readout_b_to_a_r2",
        "A->A": "self_reconstruction_a_r2",
        "B->B": "self_reconstruction_b_r2",
        "rho_ctr": "latent_correlation_centered",
        "rho_raw": "latent_correlation_raw",
        "f": "latent_alignment_f",
    }
    return [
        {
            "metric": labels.get(metric, metric),
            "checkpoint_key": metric,
            "execution_path": metric.replace("->", "_to_")
            if metric in {"A->A", "A->B", "B->B", "B->A"}
            else "",
            "value": data["evaluation"][metric],
            "checkpoint_step": data["training"]["step"],
            "latent_size": data["adapter"]["latent_size"],
            "layers": len(data["adapter"]["layer_mapping"]),
            "source_file": str(release_config.relative_to(ROOT)),
        }
        for metric in metric_order
    ]


def linearity_rows(raw_dir: Path) -> list[dict[str, Any]]:
    path = raw_dir / "cross_family_linearity_L32.json"
    data = load_json(path)
    rows = [
        {
            "scope": "layer",
            "direction": direction,
            "layer": int(layer),
            **values,
            "evaluation_rows": data["rows_per_layer"] - data["fit_rows"],
            "fit_rows": data["fit_rows"],
            "checkpoint_step": data["step"],
            "source_file": path.name,
        }
        for direction, layer_values in data["rows"].items()
        for layer, values in layer_values.items()
    ]
    rows.extend(
        {
            "scope": "mean",
            "direction": direction,
            "layer": "",
            **values,
            "evaluation_rows": (data["rows_per_layer"] - data["fit_rows"])
            * len(data["layers"]),
            "fit_rows": data["fit_rows"] * len(data["layers"]),
            "checkpoint_step": data["step"],
            "source_file": path.name,
        }
        for direction, values in data["mean"].items()
    )
    write_csv(path.with_suffix(".csv"), rows)
    return rows


def cross_family_long_context_rows(raw_dir: Path) -> list[dict[str, Any]]:
    """Normalize both cross-family paths through one schema and code path."""

    rows = []
    for stem in ("a_to_b", "b_to_a"):
        for path in sorted(raw_dir.glob(f"{stem}_longctx_*.json")):
            data = load_json(path)
            local = []
            for context, configs in data["ctx"].items():
                for config, values in configs.items():
                    kind, layer = config_parts(config)
                    row = {
                        "direction": data["direction"],
                        "source_model": data["source_model"],
                        "target_model": data["target_model"],
                        "domain": data["text"],
                        "context_tokens": int(context),
                        "windows": data["evaluation_windows"][str(context)],
                        "config": config,
                        "config_type": kind,
                        "switch_layer": "" if layer is None else layer,
                        "cross_entropy": values["ce"],
                        "standard_error": values["se"],
                        "perplexity": math.exp(values["ce"]),
                        "source_file": path.name,
                    }
                    rows.append(row)
                    local.append(row)
            write_csv(path.with_suffix(".csv"), local)
    return rows


def layer_rows(raw_dir: Path) -> list[dict[str, Any]]:
    path = raw_dir / "four_path_layer_metrics.json"
    data = load_json(path)
    metrics = [
        "A->A",
        "A->B",
        "B->B",
        "B->A",
        "rho_ctr",
        "A->B_shuf",
        "B->A_shuf",
        "rho_ctr_shuf",
    ]
    rows = [
        {
            "row_type": "layer",
            "layer": int(layer),
            "statistics": values["stats"],
            **{metric: values[metric] for metric in metrics},
            "n_windows": data["nwin"],
            "window_tokens": data["win"],
            "seed": data["seed"],
            "source_file": path.name,
        }
        for layer, values in data["rows"].items()
    ]
    for scope in ("mean_1_31", "mean_0_31"):
        rows.append(
            {
                "row_type": scope,
                "layer": "",
                "statistics": "mixed",
                **data[scope],
                "n_windows": data["nwin"],
                "window_tokens": data["win"],
                "seed": data["seed"],
                "source_file": path.name,
            }
        )
    write_csv(path.with_suffix(".csv"), rows)
    return rows


def correspondence_rows(raw_dir: Path) -> list[dict[str, Any]]:
    path = raw_dir / "all_layer_corr.json"
    data = load_json(path)
    rows = [
        {
            "source_layer": source,
            "target_layer": target,
            "linear_cka": data["cka_matrix"][source][target],
            "best_linear_r2": data["linr2_matrix"][source][target],
            "is_diagonal": source == target,
            "n_windows": data["nwin"],
            "window_tokens": data["win"],
            "sample_rows": data["rows"],
            "source_file": path.name,
        }
        for source in range(len(data["cka_matrix"]))
        for target in range(len(data["cka_matrix"][source]))
    ]
    write_csv(path.with_suffix(".csv"), rows)
    return rows


def mlp_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows = []
    representation_written = False
    symmetric_qa_tasks: set[str] = set()
    for path in sorted(raw_dir.glob("cross_family_mlp_intervention_*.json")):
        data = load_json(path)
        task_keys = [key for key in data if key.startswith("qa_")]
        local = []
        for direction, alpha_values in data["r2_by_direction_alpha_layer"].items():
            for alpha, layer_values in alpha_values.items():
                for layer, value in layer_values.items():
                    row = {
                        "result_type": "representation_r2",
                        "direction": direction,
                        "task": "held_out_residuals",
                        "alpha": float(alpha),
                        "layer": int(layer),
                        "config": "",
                        "value": value,
                        "n_items": "",
                        "source_file": path.name,
                    }
                    local.append(row)
                    if not representation_written:
                        rows.append(row)
        if not representation_written:
            representation_written = True
        qa_by_direction = data.get("qa_by_direction")
        if isinstance(qa_by_direction, dict):
            task = data["task"]
            symmetric_qa_tasks.add(task)
            for direction, qa in qa_by_direction.items():
                for config in qa["configs"]:
                    kind, layer = config_parts(config)
                    row = {
                        "result_type": "qa_accuracy_norm",
                        "direction": direction,
                        "task": task,
                        "alpha": 0.0 if kind.endswith("_alpha0") else 1.0,
                        "layer": layer,
                        "config": config,
                        "value": qa["acc_norm"][config],
                        "n_items": qa["N"],
                        "source_file": path.name,
                    }
                    rows.append(row)
                    local.append(row)
        else:
            for task_key in task_keys:
                qa = data[task_key]
                for config, value in qa["acc_norm"].items():
                    kind, layer = config_parts(config)
                    row = {
                        "result_type": "qa_accuracy_norm",
                        "direction": "A_to_B",
                        "task": task_key[3:],
                        "alpha": 0.0 if kind.endswith("_alpha0") else 1.0,
                        "layer": layer,
                        "config": config,
                        "value": value,
                        "n_items": qa["N"],
                        "source_file": path.name,
                    }
                    rows.append(row)
                    local.append(row)
        write_csv(path.with_suffix(".csv"), local)

    for path in sorted(raw_dir.glob("b_to_a_qa_items_*.json")):
        data = load_json(path)
        if data["task"] in symmetric_qa_tasks:
            continue
        for config in data["cfgs"]:
            if not (
                config.startswith("B_to_A@") or config.startswith("B_to_A_alpha0@")
            ):
                continue
            kind, layer = config_parts(config)
            rows.append(
                {
                    "result_type": "qa_accuracy_norm",
                    "direction": "B_to_A",
                    "task": data["task"],
                    "alpha": 0.0 if kind.endswith("_alpha0") else 1.0,
                    "layer": layer,
                    "config": config,
                    "value": data["acc_norm"][config],
                    "n_items": data["N"],
                    "source_file": path.name,
                }
            )
    return rows


def a_to_b_paired_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    symmetric_tasks = {
        data["task"]
        for path in raw_dir.glob("cross_family_mlp_intervention_*.json")
        if isinstance(
            (data := load_json(path)).get("qa_by_direction"),
            dict,
        )
    }
    for path in (
        raw_dir / "a_to_b_paired_statistics.csv",
        raw_dir / "paired_statistics_mlp.csv",
    ):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if "_mlp" in path.stem and row.get("task") in symmetric_tasks:
                    continue
                if (
                    path.name == "a_to_b_paired_statistics.csv"
                    and row.get("task") in symmetric_tasks
                    and "_alpha0@" in row.get("config_b", "")
                ):
                    # The symmetric task bundle below is authoritative for
                    # both AB and BA intervention comparisons. Avoid keeping
                    # an extra AB-only copy from the general QA statistics.
                    continue
                rows.append(
                    {
                        "comparison_family": "mlp_intervention"
                        if "_mlp" in path.stem
                        else "qa",
                        **row,
                        "direction": row.get("direction", "A_to_B"),
                        "source_file": path.name,
                    }
                )

    # Equal-memory results have index-aligned per-item records; derive the same paired statistics.
    for exit_path in sorted(raw_dir.glob("a_to_b_exit_items_*.json")):
        task = exit_path.stem.removeprefix("a_to_b_exit_items_")
        qa_path = raw_dir / f"a_to_b_qa_items_{task}.json"
        if not qa_path.exists():
            continue
        exits = load_json(exit_path)
        qa = load_json(qa_path)
        if exits["N"] != qa["N"]:
            continue
        for switch_text, blocks in exits["Ks"].items():
            switch = int(switch_text)
            config_a = f"B_early_exit@{blocks}"
            config_b = f"A_to_B@{switch}"
            x = exits["per_item_norm"][config_a]
            y = qa["per_item_norm"][config_b]
            b, c, p = mcnemar_exact(x, y)
            lo, hi = bootstrap_ci(x, y)
            accuracy_a = sum(x) / len(x)
            accuracy_b = sum(y) / len(y)
            rows.append(
                {
                    "comparison_family": "equal_memory",
                    "task": qa["task"],
                    "direction": "A_to_B",
                    "comparison": f"A_to_B early exit vs chimera @{switch}",
                    "config_a": config_a,
                    "config_b": config_b,
                    "n_items": len(x),
                    "accuracy_a": accuracy_a,
                    "accuracy_b": accuracy_b,
                    "accuracy_difference": accuracy_a - accuracy_b,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "mcnemar_b": b,
                    "mcnemar_c": c,
                    "mcnemar_p": p,
                    "significant_0_05": p < 0.05,
                    "source_file": f"{exit_path.name};{qa_path.name}",
                }
            )
    return rows


def b_to_a_paired_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    symmetric_tasks = {
        data["task"]
        for path in raw_dir.glob("cross_family_mlp_intervention_*.json")
        if isinstance(
            (data := load_json(path)).get("qa_by_direction"),
            dict,
        )
    }
    direct = raw_dir / "b_to_a_paired_statistics.csv"
    if direct.exists():
        with direct.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("task") in symmetric_tasks and "_alpha0@" in row.get(
                    "config_b", ""
                ):
                    continue
                rows.append(
                    {
                        "comparison_family": "qa",
                        **row,
                        "source_file": direct.name,
                    }
                )

    for exit_path in sorted(raw_dir.glob("b_to_a_exit_items_*.json")):
        task = exit_path.stem.removeprefix("b_to_a_exit_items_")
        qa_path = raw_dir / f"b_to_a_qa_items_{task}.json"
        if not qa_path.exists():
            continue
        exits = load_json(exit_path)
        qa = load_json(qa_path)
        if exits["N"] != qa["N"]:
            continue
        for switch_text, blocks in exits["Ks"].items():
            switch = int(switch_text)
            config_a = f"B_early_exit@{blocks}"
            config_b = f"B_to_A@{switch}"
            x = exits["per_item_norm"][config_a]
            y = qa["per_item_norm"][config_b]
            b, c, p = mcnemar_exact(x, y)
            lo, hi = bootstrap_ci(x, y)
            accuracy_a = sum(x) / len(x)
            accuracy_b = sum(y) / len(y)
            rows.append(
                {
                    "comparison_family": "equal_memory",
                    "direction": "B_to_A",
                    "task": qa["task"],
                    "comparison": f"B_to_A early exit vs chimera @{switch}",
                    "config_a": config_a,
                    "config_b": config_b,
                    "n_items": len(x),
                    "accuracy_a": accuracy_a,
                    "accuracy_b": accuracy_b,
                    "accuracy_difference": accuracy_a - accuracy_b,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "mcnemar_b": b,
                    "mcnemar_c": c,
                    "mcnemar_p": p,
                    "significant_0_05": p < 0.05,
                    "source_file": f"{exit_path.name};{qa_path.name}",
                }
            )

    for b_to_a_path in sorted(raw_dir.glob("b_to_a_qa_items_*.json")):
        a_to_b_path = raw_dir / b_to_a_path.name.replace(
            "b_to_a_qa_items_",
            "a_to_b_qa_items_",
            1,
        )
        if not a_to_b_path.exists():
            continue
        b_to_a = load_json(b_to_a_path)
        a_to_b = load_json(a_to_b_path)
        if b_to_a["N"] != a_to_b["N"] or b_to_a["task"] != a_to_b["task"]:
            continue
        for layer in SUPPORTED_SWITCH_LAYERS:
            config_a = f"B_to_A@{layer}"
            config_b = f"A_to_B@{layer}"
            x = b_to_a["per_item_norm"][config_a]
            y = a_to_b["per_item_norm"][config_b]
            b, c, p = mcnemar_exact(x, y)
            lo, hi = bootstrap_ci(x, y)
            accuracy_a = sum(x) / len(x)
            accuracy_b = sum(y) / len(y)
            rows.append(
                {
                    "comparison_family": "cross_path",
                    "direction": "BA_vs_AB",
                    "task": b_to_a["task"],
                    "comparison": f"BA vs AB @{layer}",
                    "config_a": config_a,
                    "config_b": config_b,
                    "n_items": len(x),
                    "accuracy_a": accuracy_a,
                    "accuracy_b": accuracy_b,
                    "accuracy_difference": accuracy_a - accuracy_b,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "mcnemar_b": b,
                    "mcnemar_c": c,
                    "mcnemar_p": p,
                    "significant_0_05": p < 0.05,
                    "source_file": f"{b_to_a_path.name};{a_to_b_path.name}",
                }
            )
    return rows


def cross_family_mlp_paired_rows(raw_dir: Path) -> list[dict[str, Any]]:
    """Derive matched alpha=1 versus alpha=0 statistics for both directions."""

    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("cross_family_mlp_intervention_*.json")):
        data = load_json(path)
        qa_by_direction = data.get("qa_by_direction")
        if not isinstance(qa_by_direction, dict):
            continue
        for direction, qa in qa_by_direction.items():
            for layer in SUPPORTED_SWITCH_LAYERS:
                config_a = f"{direction}@{layer}"
                config_b = f"{direction}_alpha0@{layer}"
                x = qa["per_item_norm"][config_a]
                y = qa["per_item_norm"][config_b]
                b, c, p = mcnemar_exact(x, y)
                lo, hi = bootstrap_ci(x, y)
                accuracy_a = sum(x) / len(x)
                accuracy_b = sum(y) / len(y)
                rows.append(
                    {
                        "comparison_family": "mlp_intervention",
                        "direction": direction,
                        "task": data["task"],
                        "comparison": f"MLP-branch @{layer}",
                        "config_a": config_a,
                        "config_b": config_b,
                        "n_items": len(x),
                        "accuracy_a": accuracy_a,
                        "accuracy_b": accuracy_b,
                        "accuracy_difference": accuracy_a - accuracy_b,
                        "bootstrap_ci_low": lo,
                        "bootstrap_ci_high": hi,
                        "mcnemar_b": b,
                        "mcnemar_c": c,
                        "mcnemar_p": p,
                        "significant_0_05": p < 0.05,
                        "source_file": path.name,
                    }
                )
    return rows


def cross_family_serving_rows(raw_dir: Path) -> list[dict[str, Any]]:
    """Normalize AB and BA serving measurements without a preferred path."""

    rows = []
    for stem in ("a_to_b", "b_to_a"):
        path = raw_dir / f"{stem}_serving_bench.json"
        data = load_json(path)
        local = []
        for context, configs in data["prefill"].items():
            for config, values in configs.items():
                row = {
                    "direction": data["direction"],
                    "benchmark": "unpruned",
                    "phase": "prefill",
                    "context_tokens": int(context),
                    "config": config,
                    **values,
                    "source_file": path.name,
                }
                rows.append(row)
                local.append(row)
        for config, values in data["decode"].items():
            row = {
                "direction": data["direction"],
                "benchmark": "unpruned",
                "phase": "decode",
                "context_tokens": "",
                "config": config,
                **values,
                "source_file": path.name,
            }
            rows.append(row)
            local.append(row)
        write_csv(path.with_suffix(".csv"), local)

        path = raw_dir / f"{stem}_serving_pruned.json"
        data = load_json(path)
        local = []
        for config, values in data["configs"].items():
            base = {
                "direction": data["direction"],
                "benchmark": "pruned",
                "phase": "prefill",
                "config": config,
                "rwkv_blocks": values.get("rwkv_blocks", 0),
                "pythia_blocks": values.get(
                    "pythia_blocks",
                    values.get(
                        "blocks",
                        32 if config == "pure-Pythia" else 0,
                    ),
                ),
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
                "source_file": path.name,
            }
            for context, profile in values["prefill"].items():
                row = {**base, "context_tokens": int(context), **profile}
                rows.append(row)
                local.append(row)
        write_csv(path.with_suffix(".csv"), local)
    return rows


TRAIN_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+lr=(?P<lr>[-+0-9.eE]+)\s+loss=(?P<loss>[-+0-9.eE]+).*?"
    r"A->B\(tr\)=(?P<tr>[-+0-9.]+)\s+A->B\(ev\)=(?P<ab>[-+0-9.]+).*?"
    r"B->A=(?P<ba>[-+0-9.]+)\s+A->A=(?P<aa>[-+0-9.]+)\s+B->B=(?P<bb>[-+0-9.]+).*?"
    r"rho_ctr=(?P<rho_ctr>[-+0-9.]+)\s+f=(?P<f>[-+0-9.]+)\s+\(rho_raw=(?P<rho_raw>[-+0-9.]+)\)"
)


def training_rows(raw_dir: Path, log_dir: Path) -> list[dict[str, Any]]:
    direct = raw_dir / "training_curve.csv"
    if direct.exists():
        with direct.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    by_step: dict[int, dict[str, Any]] = {}
    for path in sorted(log_dir.glob("slurm-L32*.out")):
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        for match in TRAIN_RE.finditer(text):
            values = match.groupdict()
            step = int(values["step"])
            by_step[step] = {
                "step": step,
                "learning_rate": float(values["lr"]),
                "loss": float(values["loss"]),
                "train_a_to_b_r2": float(values["tr"]),
                "eval_a_to_b_r2": float(values["ab"]),
                "eval_b_to_a_r2": float(values["ba"]),
                "eval_a_to_a_r2": float(values["aa"]),
                "eval_b_to_b_r2": float(values["bb"]),
                "latent_rho_centered": float(values["rho_ctr"]),
                "latent_rho_raw": float(values["rho_raw"]),
                "latent_f": float(values["f"]),
                "elapsed_seconds": "",
                "seed": 0,
                "tokens_per_layer": 228_000,
                "latent_size": 4096,
            }
    rows = [by_step[step] for step in sorted(by_step)]
    if rows:
        write_csv(direct, rows)
    return rows


def cross_family_generation_rows(raw_dir: Path) -> list[dict[str, Any]]:
    """Normalize legacy and current AB/BA sample files to one raw schema."""

    specifications = (
        ("a_to_b", "RWKV-Raven-7B", "Tulu-Pythia-6.9B"),
        ("b_to_a", "Tulu-Pythia-6.9B", "RWKV-Raven-7B"),
    )
    rows = []
    for stem, source_model, target_model in specifications:
        path = raw_dir / f"{stem}_generation_samples.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        local = []
        for source in source_rows:
            adapter_cross_r2 = source.get(
                "adapter_cross_r2",
                source.get("adapter_a_to_b", source.get("adapter_b_to_a", "")),
            )
            row = {
                "direction": source["direction"],
                "source_model": source.get("source_model") or source_model,
                "target_model": source.get("target_model") or target_model,
                "prompt_id": source["prompt_id"],
                "prompt": source["prompt"],
                "config": source["config"],
                "switch_layer": source.get("switch_layer", ""),
                "adapter_cross_r2": adapter_cross_r2,
                "checkpoint_step": source.get("checkpoint_step", ""),
                "parent_gate_relative_error": source.get(
                    "parent_gate_relative_error",
                    "",
                ),
                "direct_gate_relative_error": source.get(
                    "direct_gate_relative_error",
                    "",
                ),
                "decoding": source.get("decoding", "greedy_argmax"),
                "temperature": source.get("temperature", "0"),
                "max_new_tokens": source.get("max_new_tokens", "40"),
                "seed": source.get("seed", "0"),
                "output": source["output"],
            }
            local.append(row)
            rows.append({**row, "source_file": path.name})
        write_csv(path, local)
    return rows


def cross_family_memory_rows(
    raw_dir: Path,
    qa_by_direction: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Join measured AB/BA KV rows with QA through one normalized schema."""

    specifications = (
        ("a_to_b", "A_to_B", "RWKV-Raven-7B", "Tulu-Pythia-6.9B"),
        ("b_to_a", "B_to_A", "Tulu-Pythia-6.9B", "RWKV-Raven-7B"),
    )
    aliases = {"pure-Pythia": "pythia", "pure-RWKV": "rwkv"}
    combined = []
    for stem, direction, source_model, target_model in specifications:
        measured = raw_dir / f"{stem}_kv_cache.csv"
        with measured.open(encoding="utf-8", newline="") as handle:
            source_rows = [dict(row) for row in csv.DictReader(handle)]
        local = []
        for source in source_rows:
            row = {
                "direction": direction,
                "source_model": source.get("source_model") or source_model,
                "target_model": source.get("target_model") or target_model,
                "config": source["config"],
                "switch_layer": source.get("switch_layer", ""),
                "transformer_layers": source["transformer_layers"],
                "kv_kib_per_token": source["kv_kib_per_token"],
                "kv_gib_at_context": source["kv_gib_at_context"],
                "context_tokens": source["context_tokens"],
                "kv_reduction_fraction": source["kv_reduction_fraction"],
                "measured_kib_per_token_per_layer": source[
                    "measured_kib_per_token_per_layer"
                ],
                "rwkv_state_kib_per_layer": source["rwkv_state_kib_per_layer"],
                "probe_sequence_tokens": source["probe_sequence_tokens"],
            }
            local.append(row)
        write_csv(measured, local)

        lookup = {
            (row["task"], row["config"]): row["accuracy_norm"]
            for row in qa_by_direction[direction]
        }
        for row in local:
            qa_config = aliases.get(row["config"], row["config"])
            combined.append(
                {
                    **row,
                    "arc_easy_accuracy_norm": lookup.get(
                        ("arc_easy", qa_config),
                        "",
                    ),
                    "sciq_accuracy_norm": lookup.get(
                        ("sciq", qa_config),
                        "",
                    ),
                    "source_file": measured.name,
                }
            )
    return combined


def is_parent_row(row: dict[str, Any]) -> bool:
    return execution_path_for_config(str(row.get("config", ""))) in {
        "A_parent",
        "B_parent",
    }


def canonical_generation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep current-checkpoint AB/BA samples and one copy of each parent."""

    selected: list[dict[str, Any]] = []
    seen_parents: set[tuple[str, str]] = set()
    for source in rows:
        if is_parent_row(source):
            parent_key = (
                str(source.get("prompt_id", "")),
                execution_path_for_config(str(source.get("config", ""))),
            )
            if parent_key in seen_parents:
                continue
            seen_parents.add(parent_key)
        selected.append(dict(source))
    return canonical_path_rows(selected)


def comparison_path_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = []
    for source in rows:
        row = dict(source)
        producer_path = str(row.pop("direction", ""))
        execution_path = "cross_path" if producer_path == "BA_vs_AB" else producer_path
        row["path_a"] = execution_path_for_config(str(row.get("config_a", "")))
        row["path_b"] = execution_path_for_config(str(row.get("config_b", "")))
        enriched.append(
            {
                "execution_path": execution_path,
                "producer_path": producer_path,
                **row,
            }
        )
    return sorted(
        enriched,
        key=lambda row: (
            str(row.get("task", "")),
            int(
                str(row.get("config_a", "@-1")).rsplit("@", 1)[-1]
                if "@" in str(row.get("config_a", ""))
                else -1
            ),
            EXECUTION_PATH_ORDER.get(str(row.get("path_a", "")), 99),
            str(row.get("comparison_family", "")),
            str(row.get("comparison", "")),
        ),
    )


def result_coverage_rows(
    tables: dict[str, tuple[list[dict[str, Any]], str]],
) -> list[dict[str, Any]]:
    """Expose path balance explicitly instead of relying on filename intuition."""

    four_path_tables = {
        "paper_four_path_qa_accuracy.csv",
        "paper_cross_family_qa_accuracy.csv",
        "paper_representation.csv",
        "paper_layer_metrics.csv",
        "paper_training_curve.csv",
    }
    rows = []
    for filename, (table_rows, _) in tables.items():
        counts = {path: 0 for path in EXECUTION_PATH_ORDER}
        if filename == "paper_representation.csv":
            for row in table_rows:
                path = str(row.get("checkpoint_key", "")).replace("->", "_to_")
                if path in counts:
                    counts[path] += 1
        elif filename == "paper_layer_metrics.csv":
            for path in ("A_to_A", "A_to_B", "B_to_B", "B_to_A"):
                counts[path] = len(table_rows)
        elif filename == "paper_training_curve.csv":
            for path in ("A_to_A", "A_to_B", "B_to_B", "B_to_A"):
                counts[path] = len(table_rows)
        else:
            for row in table_rows:
                path = str(row.get("execution_path", ""))
                if path and path in counts:
                    counts[path] += 1
        if not any(counts.values()):
            continue
        four_path_applicable = filename in four_path_tables
        rows.append(
            {
                "table": filename,
                "count_unit": (
                    "metric_cells"
                    if filename
                    in {"paper_layer_metrics.csv", "paper_training_curve.csv"}
                    else "rows"
                ),
                "A_parent": counts["A_parent"],
                "A_to_A": counts["A_to_A"],
                "A_to_B": counts["A_to_B"],
                "B_parent": counts["B_parent"],
                "B_to_B": counts["B_to_B"],
                "B_to_A": counts["B_to_A"],
                "B_early_exit": counts["B_early_exit"],
                "cross_path": counts["cross_path"],
                "ab_ba_matched": counts["A_to_B"] == counts["B_to_A"],
                "four_path_applicable": four_path_applicable,
                "four_path_complete": (
                    all(
                        counts[path] > 0
                        for path in ("A_to_A", "A_to_B", "B_to_B", "B_to_A")
                    )
                    if four_path_applicable
                    else ""
                ),
            }
        )
    return rows


def write_manifest(table_dir: Path, descriptions: dict[str, str]) -> None:
    rows = []
    for filename, description in descriptions.items():
        path = table_dir / filename
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            count = sum(1 for _ in reader)
        rows.append(
            {
                "table": filename,
                "rows": count,
                "columns": len(header),
                "column_names": header,
                "description": description,
            }
        )
    write_csv(table_dir / "MANIFEST.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / "artifacts" / "metrics" / "raw"
    )
    parser.add_argument(
        "--table-dir", type=Path, default=ROOT / "artifacts" / "metrics" / "tables"
    )
    parser.add_argument(
        "--log-dir", type=Path, default=ROOT / "artifacts" / "logs" / "slurm"
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=ROOT
        / "release"
        / "huggingface"
        / "ninaxander-raven7b-tulu69-adapter"
        / "config.json",
    )
    args = parser.parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, tuple[list[dict[str, Any]], str]] = {}
    a_to_b_qa = a_to_b_qa_rows(args.raw_dir)
    b_to_a_qa = b_to_a_qa_rows(args.raw_dir)
    long_context = cross_family_long_context_rows(args.raw_dir)
    serving = cross_family_serving_rows(args.raw_dir)
    generation = cross_family_generation_rows(args.raw_dir)
    memory = cross_family_memory_rows(
        args.raw_dir,
        {"A_to_B": a_to_b_qa, "B_to_A": b_to_a_qa},
    )
    tables["paper_four_path_qa_accuracy.csv"] = (
        execution_path_qa_rows(a_to_b_qa, b_to_a_qa),
        "Symmetric AA/AB/BB/BA full-set QA matrix for every switch layer.",
    )
    tables["paper_cross_family_qa_accuracy.csv"] = (
        canonical_path_rows(
            [
                *a_to_b_qa,
                *(row for row in b_to_a_qa if not is_parent_row(row)),
            ]
        ),
        "Canonical full-set QA with one copy of each parent and matched "
        "AA/AB/BB/BA, affine, intervention, and equal-memory rows.",
    )
    if args.release_config.is_file():
        representation = representation_rows(args.release_config)
    else:
        # release/ is gitignored: a fresh public checkout rebuilds every raw-derived
        # table but cannot re-project this summary from the staged package, so the
        # shipped table is carried through unchanged.
        shipped = args.table_dir / "paper_representation.csv"
        if not shipped.is_file():
            raise SystemExit(
                "paper_representation.csv cannot be rebuilt: neither the release config "
                f"({args.release_config}) nor the shipped table ({shipped}) exists. "
                "Stage the release package with tools/export_hf_adapter.py or restore "
                "the shipped table from git."
            )
        with shipped.open(encoding="utf-8", newline="") as handle:
            representation = list(csv.DictReader(handle))
        if not representation:
            raise SystemExit(f"{shipped} is empty; restore the shipped table from git.")
        print("NOTE: release config absent; kept existing paper_representation.csv rows.")
    tables["paper_representation.csv"] = (
        representation,
        "Final-checkpoint representation alignment and reconstruction metrics.",
    )
    tables["paper_cross_family_linearity.csv"] = (
        canonical_path_rows(linearity_rows(args.raw_dir)),
        "Matched AB/BA adapter linearity and direct affine baselines.",
    )
    tables["paper_cross_family_domain_shift.csv"] = (
        canonical_path_rows(
            [
                row
                for row in long_context
                if row["direction"] == "A_to_B" or not is_parent_row(row)
            ]
        ),
        "Combined AB/BA Alpaca/WikiText cross-entropy by context; affine controls "
        "are refit on a disjoint prefix of each domain.",
    )
    tables["paper_layer_metrics.csv"] = (
        layer_rows(args.raw_dir),
        "Per-layer bidirectional reconstruction, alignment, and shuffled controls.",
    )
    tables["paper_layer_correspondence.csv"] = (
        correspondence_rows(args.raw_dir),
        "All 32x32 source/target layer CKA and direct-linear correspondence.",
    )
    tables["paper_cross_family_mlp_intervention.csv"] = (
        canonical_path_rows(mlp_rows(args.raw_dir)),
        "Matched AB/BA same-adapter residual-MLP branch intervention.",
    )
    tables["paper_cross_family_paired_statistics.csv"] = (
        comparison_path_rows(
            [
                *a_to_b_paired_rows(args.raw_dir),
                *b_to_a_paired_rows(args.raw_dir),
                *cross_family_mlp_paired_rows(args.raw_dir),
            ]
        ),
        "Combined AB/BA McNemar tests and paired-bootstrap intervals for QA, "
        "interventions, equal-memory, and same-layer AB/BA comparisons.",
    )
    tables["paper_cross_family_serving.csv"] = (
        canonical_path_rows(serving),
        "Combined AB/BA unpruned and physically pruned serving measurements.",
    )
    tables["paper_training_curve.csv"] = (
        training_rows(args.raw_dir, args.log_dir),
        "Structured adapter training trajectory emitted directly or migrated once from authoritative logs.",
    )
    tables["paper_cross_family_generation_samples.csv"] = (
        canonical_generation_rows(generation),
        "Current-checkpoint AB/BA qualitative samples with one copy of each parent.",
    )
    tables["paper_cross_family_memory_accuracy.csv"] = (
        canonical_path_rows(
            [
                row
                for row in memory
                if row["direction"] == "A_to_B" or not is_parent_row(row)
            ]
        ),
        "Canonical parent/AB/BA measured KV-cache cost joined with full-set QA.",
    )
    tables["paper_result_coverage.csv"] = (
        result_coverage_rows(tables),
        "Per-table AA/AB/BB/BA coverage audit; parents and Pythia early exits are separate controls.",
    )

    descriptions = {}
    current_names = set(tables)
    for stale in args.table_dir.glob("paper_*.csv"):
        if stale.name not in current_names:
            stale.unlink()
    for filename, (rows, description) in tables.items():
        output = write_csv(args.table_dir / filename, rows)
        descriptions[filename] = description
        print(f"{display(output)}: {len(rows)} rows")
    write_manifest(args.table_dir, descriptions)
    print(display(args.table_dir / "MANIFEST.csv"))
    print("RESULT_TABLES_DONE")


if __name__ == "__main__":
    main()
