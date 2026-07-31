#!/usr/bin/env python3
"""Validate normalized result tables and their links to raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "artifacts" / "metrics" / "raw"
DEFAULT_TABLES = ROOT / "artifacts" / "metrics" / "tables"

EXPECTED_TABLES = {
    "paper_four_path_qa_accuracy.csv": 32,
    "paper_cross_family_qa_accuracy.csv": 84,
    "paper_representation.csv": 7,
    "paper_cross_family_linearity.csv": 12,
    "paper_cross_family_domain_shift.csv": 108,
    "paper_layer_metrics.csv": 34,
    "paper_layer_correspondence.csv": 1024,
    "paper_cross_family_mlp_intervention.csv": 82,
    "paper_cross_family_paired_statistics.csv": 120,
    "paper_cross_family_serving.csv": 76,
    "paper_training_curve.csv": 417,
    "paper_cross_family_generation_samples.csv": 16,
    "paper_cross_family_memory_accuracy.csv": 10,
    "paper_result_coverage.csv": 12,
}

EXECUTION_PATH_TABLE_COUNTS = {
    "paper_four_path_qa_accuracy.csv": {
        "A_to_A": 8,
        "A_to_B": 8,
        "B_to_B": 8,
        "B_to_A": 8,
    },
    "paper_cross_family_qa_accuracy.csv": {
        "A_parent": 2,
        "A_to_A": 8,
        "A_to_B": 24,
        "B_parent": 2,
        "B_to_B": 8,
        "B_to_A": 24,
        "B_early_exit": 16,
    },
    "paper_cross_family_linearity.csv": {"A_to_B": 6, "B_to_A": 6},
    "paper_cross_family_mlp_intervention.csv": {"A_to_B": 41, "B_to_A": 41},
    "paper_cross_family_paired_statistics.csv": {
        "A_to_B": 56,
        "B_to_A": 56,
        "cross_path": 8,
    },
    "paper_cross_family_domain_shift.csv": {
        "A_parent": 6,
        "A_to_B": 48,
        "B_parent": 6,
        "B_to_A": 48,
    },
    "paper_cross_family_serving.csv": {
        "A_parent": 10,
        "A_to_B": 20,
        "B_parent": 10,
        "B_to_A": 20,
        "B_early_exit": 16,
    },
    "paper_cross_family_generation_samples.csv": {
        "A_parent": 4,
        "A_to_B": 4,
        "B_parent": 4,
        "B_to_A": 4,
    },
    "paper_cross_family_memory_accuracy.csv": {
        "A_parent": 1,
        "A_to_B": 4,
        "B_parent": 1,
        "B_to_A": 4,
    },
}

RAW_BUNDLES = (
    "all_layer_corr",
    "four_path_layer_metrics",
    "cross_family_linearity_L32",
    "cross_family_mlp_representation",
    "cross_family_mlp_intervention_arc",
    "cross_family_mlp_intervention_sciq",
    "a_to_b_exit_items_arc",
    "a_to_b_exit_items_sciq",
    "a_to_b_longctx_alpaca",
    "a_to_b_longctx_wikitext",
    "a_to_b_qa_items_arc",
    "a_to_b_qa_items_sciq",
    "a_to_b_serving_bench",
    "a_to_b_serving_pruned",
    "b_to_a_exit_items_arc",
    "b_to_a_exit_items_sciq",
    "b_to_a_longctx_alpaca",
    "b_to_a_longctx_wikitext",
    "b_to_a_qa_items_arc",
    "b_to_a_qa_items_sciq",
    "b_to_a_serving_bench",
    "b_to_a_serving_pruned",
    "cross_family_runtime_smoke",
)

RETIRED_RAW = (
    "linearity_L32",
    "mlp_ablation_arc",
    "mlp_ablation_sciq",
    "mlp_items_arc",
    "mlp_items_sciq",
    "paired_statistics_mlp",
    "layer_bta_shuf",
    "cross_family_mlp_ablation_arc",
    "cross_family_mlp_ablation_sciq",
    "b_to_a_runtime_smoke",
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(
            set(reader.fieldnames)
        ):
            raise ValueError(f"invalid or duplicate header: {path}")
        return reader.fieldnames, list(reader)


def validate_generation_contract(
    rows: list[dict[str, str]],
    source: Path,
    errors: list[str],
) -> None:
    """Require the reported deterministic 40-token generation protocol."""

    observed_decoding = {row.get("decoding", "") for row in rows}
    if observed_decoding != {"greedy_argmax"}:
        errors.append(
            f"{display(source)} has decoding={sorted(observed_decoding)!r}; "
            "expected 'greedy_argmax' on every row"
        )

    numeric_expected = {
        "temperature": 0.0,
        "max_new_tokens": 40.0,
        "seed": 0.0,
    }
    for field, expected in numeric_expected.items():
        observed = {row.get(field, "") for row in rows}
        try:
            numeric = {float(value) for value in observed}
        except ValueError:
            numeric = set()
        if numeric != {expected}:
            errors.append(
                f"{display(source)} has {field}={sorted(observed)!r}; "
                f"expected numeric {expected:g} on every row"
            )


def display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json_checked(path: Path, errors: list[str]) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {display(path)}: {exc}")
        return None


def validate_error_map(
    data: dict,
    field: str,
    tolerance: float,
    source: Path,
    errors: list[str],
) -> None:
    values = data.get(field)
    if not isinstance(values, dict) or not values:
        errors.append(f"{display(source)} lacks non-empty {field}")
        return
    for layer, value in values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"{display(source)} has invalid {field}[{layer}]={value!r}")
            continue
        if numeric >= tolerance:
            errors.append(
                f"{display(source)} {field}[{layer}]={numeric:.3e} "
                f"exceeds tolerance {tolerance:.3e}"
            )


def validate_metric_record(
    record: object,
    required_fields: tuple[str, ...],
    source: Path,
    label: str,
    errors: list[str],
) -> None:
    """Require finite positive metrics, or a structured runtime error."""

    if not isinstance(record, dict):
        errors.append(f"{display(source)} has invalid metric record for {label}")
        return
    if "error" in record:
        if not isinstance(record["error"], str) or not record["error"]:
            errors.append(f"{display(source)} has an empty error for {label}")
        return
    for field in required_fields:
        try:
            value = float(record[field])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{display(source)} has invalid {field} for {label}")
            continue
        if not math.isfinite(value) or value <= 0:
            errors.append(
                f"{display(source)} has out-of-range {field}={value!r} for {label}"
            )


def validate_a_to_b_semantics(raw_dir: Path, errors: list[str]) -> None:
    """Require canonical 18-arm AB/BB QA bundles."""

    expected_configs = (
        {"pythia", "rwkv"}
        | {f"A_to_B@{layer}" for layer in (4, 8, 16, 24)}
        | {f"A_to_B_alpha0@{layer}" for layer in (4, 8, 16, 24)}
        | {f"A_to_B_affine@{layer}" for layer in (4, 8, 16, 24)}
        | {f"B_to_B@{layer}" for layer in (4, 8, 16, 24)}
    )
    for filename, task, count in (
        ("a_to_b_qa_items_arc.json", "arc_easy", 2376),
        ("a_to_b_qa_items_sciq.json", "sciq", 1000),
    ):
        path = raw_dir / filename
        if not path.is_file():
            continue
        data = load_json_checked(path, errors)
        if data is None:
            continue
        if data.get("direction") != "A_to_B" or data.get("task") != task:
            errors.append(f"{display(path)} has inconsistent direction/task metadata")
        if data.get("boundary") != "RWKV post-block L -> Pythia pre-block L+1":
            errors.append(f"{display(path)} has the wrong residual boundary")
        if data.get("N") != count:
            errors.append(f"{display(path)} has N={data.get('N')}; expected {count}")
        if set(data.get("cfgs", [])) != expected_configs:
            errors.append(f"{display(path)} does not contain the 18 canonical configs")
        if (
            data.get("switches") != [4, 8, 16, 24]
            or data.get("checkpoint_step") != 415_000
            or data.get("linear_fit_rows_requested") != 40_000
            or data.get("linear_ridge") != 1e-3
        ):
            errors.append(f"{display(path)} has noncanonical QA run settings")
        if "parent_reproduction_relative_error" in data:
            validate_error_map(
                data,
                "parent_reproduction_relative_error",
                1e-4,
                path,
                errors,
            )
        elif data.get("migration", {}).get("kind") != "item_order_preserving_fold":
            errors.append(f"{display(path)} lacks parent-reproduction gate metadata")

        per_item = data.get("per_item_norm", {})
        accuracies = data.get("acc_norm", {})
        for config in expected_configs:
            values = per_item.get(config)
            if not isinstance(values, list) or len(values) != count:
                errors.append(
                    f"{display(path)} per_item_norm[{config}] does not have "
                    f"{count} rows"
                )
                continue
            if any(value not in (0, 1) for value in values):
                errors.append(f"{display(path)} has non-binary outcomes for {config}")
            try:
                reported = float(accuracies[config])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{display(path)} has invalid acc_norm[{config}]")
                continue
            observed = sum(values) / count
            if not math.isclose(reported, observed, rel_tol=0, abs_tol=1e-12):
                errors.append(
                    f"{display(path)} acc_norm[{config}]={reported} "
                    f"does not match per-item mean {observed}"
                )


def validate_cross_path_semantics(raw_dir: Path, errors: list[str]) -> None:
    """Require canonical cross-path QA, controls, and matched AB/BA bundles."""

    runtime_path = raw_dir / "cross_family_runtime_smoke.json"
    if runtime_path.is_file():
        runtime = load_json_checked(runtime_path, errors)
        if runtime is not None:
            if runtime.get("directions") != ["A_to_B", "B_to_A"]:
                errors.append(f"{display(runtime_path)} lacks both directions")
            if runtime.get("boundaries") != {
                "A_to_B": "RWKV post-block L -> Pythia pre-block L+1",
                "B_to_A": "Pythia post-block L -> RWKV pre-block L+1",
            }:
                errors.append(
                    f"{display(runtime_path)} has the wrong residual boundaries"
                )
            if runtime.get("offsets") != {"rwkv": 0, "pythia": 1}:
                errors.append(f"{display(runtime_path)} has incorrect HF offsets")
            if runtime.get("canonical_option_batch") not in (None, 1):
                errors.append(
                    f"{display(runtime_path)} has a noncanonical option batch"
                )
            if (
                "rwkv_fused_batch_role" in runtime
                and runtime.get("rwkv_fused_batch_role")
                != "diagnostic only; never used for reported QA"
            ):
                errors.append(
                    f"{display(runtime_path)} has an invalid fused-batch role"
                )
            rows = runtime.get("rows", [])
            expected_runtime_rows = {
                (direction, layer)
                for direction in ("A_to_B", "B_to_A")
                for layer in (4, 8, 16, 24)
            }
            actual_runtime_rows = {
                (row.get("direction"), row.get("switch_layer")) for row in rows
            }
            if actual_runtime_rows != expected_runtime_rows:
                errors.append(
                    f"{display(runtime_path)} lacks matched AB/BA runtime rows"
                )
            for row in rows:
                layer = row.get("switch_layer", "?")
                direction = row.get("direction")
                fields = ["parent_gate_relative_error"]
                if direction == "B_to_A":
                    fields.extend(
                        (
                            "direct_gate_relative_error",
                            "direct_vs_hook_relative_error",
                            "concurrent_vs_sequential_relative_error",
                        )
                    )
                for field in fields:
                    try:
                        value = float(row[field])
                    except (KeyError, TypeError, ValueError):
                        errors.append(
                            f"{display(runtime_path)} has invalid {field} at L={layer}"
                        )
                        continue
                    if value >= 1e-4:
                        errors.append(
                            f"{display(runtime_path)} {field} at L={layer} "
                            f"is {value:.3e}, expected <1e-4"
                        )
                if row.get("logits_finite") is not True:
                    errors.append(
                        f"{display(runtime_path)} has non-finite logits at L={layer}"
                    )
                if direction == "B_to_A":
                    try:
                        fused_error = float(row["fused_batch_vs_batch1_relative_error"])
                    except (TypeError, ValueError):
                        errors.append(
                            f"{display(runtime_path)} has invalid fused-batch "
                            f"diagnostic at L={layer}"
                        )
                    else:
                        accepted = fused_error < 1e-4
                        if (
                            not math.isfinite(fused_error)
                            or fused_error < 0
                            or row.get("fused_batch_accepted_at_1e-4") is not accepted
                        ):
                            errors.append(
                                f"{display(runtime_path)} has inconsistent "
                                f"fused-batch diagnostic at L={layer}"
                            )

    expected_qa = {
        "b_to_a_qa_items_arc.json": ("arc_easy", 2376),
        "b_to_a_qa_items_sciq.json": ("sciq", 1000),
    }
    expected_configs = (
        {"pythia", "rwkv"}
        | {f"B_to_A@{layer}" for layer in (4, 8, 16, 24)}
        | {f"B_to_A_alpha0@{layer}" for layer in (4, 8, 16, 24)}
        | {f"B_to_A_affine@{layer}" for layer in (4, 8, 16, 24)}
        | {f"A_to_A@{layer}" for layer in (4, 8, 16, 24)}
    )
    for filename, (task, count) in expected_qa.items():
        path = raw_dir / filename
        if not path.is_file():
            continue
        data = load_json_checked(path, errors)
        if data is None:
            continue
        if data.get("direction") != "B_to_A" or data.get("task") != task:
            errors.append(f"{display(path)} has inconsistent direction/task metadata")
        if data.get("boundary") != "Pythia post-block L -> RWKV pre-block L+1":
            errors.append(f"{display(path)} has the wrong residual boundary")
        if data.get("N") != count:
            errors.append(f"{display(path)} has N={data.get('N')}; expected {count}")
        if set(data.get("cfgs", [])) != expected_configs:
            errors.append(f"{display(path)} does not contain the 18 canonical configs")
        if data.get("option_batch") != 1:
            errors.append(
                f"{display(path)} was not evaluated with canonical option_batch=1"
            )
        if (
            data.get("switches") != [4, 8, 16, 24]
            or data.get("checkpoint_step") != 415_000
            or data.get("linear_fit_rows_requested") != 40_000
            or data.get("linear_ridge") != 1e-3
            or data.get("concurrent_arm_max_tokens") != 96
        ):
            errors.append(f"{display(path)} has noncanonical QA run settings")
        for field in (
            "parent_reproduction_relative_error",
            "direct_suffix_relative_error",
            "concurrent_arm_relative_error",
        ):
            validate_error_map(data, field, 1e-4, path, errors)
        per_item = data.get("per_item_norm", {})
        accuracies = data.get("acc_norm", {})
        for config in expected_configs:
            values = per_item.get(config)
            if not isinstance(values, list) or len(values) != count:
                errors.append(
                    f"{display(path)} per_item_norm[{config}] does not have {count} rows"
                )
                continue
            if any(value not in (0, 1) for value in values):
                errors.append(f"{display(path)} has non-binary outcomes for {config}")
            try:
                reported = float(accuracies[config])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{display(path)} has invalid acc_norm[{config}]")
                continue
            observed = sum(values) / count
            if not math.isclose(reported, observed, rel_tol=0, abs_tol=1e-12):
                errors.append(
                    f"{display(path)} acc_norm[{config}]={reported} "
                    f"does not match per-item mean {observed}"
                )

        a_to_b_name = filename.removeprefix("b_to_a_")
        a_to_b_path = raw_dir / f"a_to_b_{a_to_b_name}"
        if a_to_b_path.is_file():
            a_to_b = load_json_checked(a_to_b_path, errors)
            if a_to_b is not None:
                a_to_b_items = a_to_b.get("per_item_norm", {})
                for parent in ("pythia", "rwkv"):
                    if per_item.get(parent) != a_to_b_items.get(parent):
                        errors.append(
                            f"{display(path)} parent item outcomes for {parent} "
                            f"do not match {display(a_to_b_path)}"
                        )

    for stem, direction, expected_depths in (
        (
            "a_to_b",
            "A_to_B",
            {"4": 27, "8": 23, "16": 15, "24": 7},
        ),
        (
            "b_to_a",
            "B_to_A",
            {"4": 5, "8": 9, "16": 17, "24": 25},
        ),
    ):
        expected_exit_configs = {
            f"B_early_exit@{blocks}" for blocks in expected_depths.values()
        }
        for suffix, task, count in (
            ("arc", "arc_easy", 2376),
            ("sciq", "sciq", 1000),
        ):
            path = raw_dir / f"{stem}_exit_items_{suffix}.json"
            if not path.is_file():
                continue
            data = load_json_checked(path, errors)
            if data is None:
                continue
            if data.get("direction") != direction or data.get("task") != task:
                errors.append(
                    f"{display(path)} has inconsistent direction/task metadata"
                )
            if data.get("N") != count:
                errors.append(
                    f"{display(path)} has N={data.get('N')}; expected {count}"
                )
            if data.get("Ks") != expected_depths:
                errors.append(f"{display(path)} has incorrect equal-memory depths")
            if set(data.get("cfgs", [])) != expected_exit_configs:
                errors.append(f"{display(path)} lacks the four canonical B early exits")
            per_item = data.get("per_item_norm", {})
            accuracies = data.get("acc_norm", {})
            for config in expected_exit_configs:
                values = per_item.get(config)
                if not isinstance(values, list) or len(values) != count:
                    errors.append(
                        f"{display(path)} per_item_norm[{config}] does not have "
                        f"{count} rows"
                    )
                    continue
                if any(value not in (0, 1) for value in values):
                    errors.append(
                        f"{display(path)} has non-binary outcomes for {config}"
                    )
                try:
                    reported = float(accuracies[config])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{display(path)} has invalid acc_norm[{config}]")
                    continue
                if not math.isclose(
                    reported,
                    sum(values) / count,
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f"{display(path)} acc_norm[{config}] does not match "
                        "its outcomes"
                    )
            comparisons = data.get("paired_equal_memory")
            if not isinstance(comparisons, list) or len(comparisons) != 4:
                errors.append(f"{display(path)} lacks four equal-memory comparisons")
            else:
                expected_comparisons = {
                    (
                        int(layer),
                        f"B_early_exit@{blocks}",
                        f"{direction}@{layer}",
                    )
                    for layer, blocks in expected_depths.items()
                }
                actual_comparisons = {
                    (
                        row.get("switch_layer"),
                        row.get("config_a"),
                        row.get("config_b"),
                    )
                    for row in comparisons
                }
                if actual_comparisons != expected_comparisons:
                    errors.append(
                        f"{display(path)} has inconsistent equal-memory comparisons"
                    )

    directions = {
        "A_to_B": {
            "stem": "a_to_b",
            "source_model": "RWKV-Raven-7B",
            "target_model": "Tulu-Pythia-6.9B",
            "boundary": "RWKV post-block L -> Pythia pre-block L+1",
            "checkpoint_field": "checkpoint_a_to_b_r2",
            "checkpoint_r2": 0.5587504604392047,
            "long_gates": (
                "parent_reproduction_relative_error",
                "source_capture_relative_error",
            ),
            "early_depths": (7, 15, 23, 27),
            "pruned_gates": ("parent_gate_rel", "gate_rel"),
        },
        "B_to_A": {
            "stem": "b_to_a",
            "source_model": "Tulu-Pythia-6.9B",
            "target_model": "RWKV-Raven-7B",
            "boundary": "Pythia post-block L -> RWKV pre-block L+1",
            "checkpoint_field": "checkpoint_b_to_a_r2",
            "checkpoint_r2": 0.5900067890580303,
            "long_gates": (
                "parent_reproduction_relative_error",
                "direct_suffix_relative_error",
                "concurrent_arm_relative_error",
            ),
            "early_depths": (5, 9, 17, 25),
            "pruned_gates": (
                "parent_gate_rel",
                "source_front_gate_rel",
                "target_suffix_gate_rel",
                "gate_rel",
            ),
        },
    }

    def legacy_omits(data: dict, field: str) -> bool:
        provenance = data.get("measurement_provenance", {})
        return (
            provenance.get("kind") == "legacy_metadata_enrichment"
            and provenance.get("metrics_recomputed") is False
            and provenance.get("metrics_unchanged") is True
            and field in provenance.get("unpersisted_runtime_checks", [])
        )

    def validate_direction_metadata(
        data: dict,
        path: Path,
        direction: str,
        specification: dict,
    ) -> None:
        if (
            data.get("schema_version") != 1
            or data.get("direction") != direction
            or data.get("source_model") != specification["source_model"]
            or data.get("target_model") != specification["target_model"]
            or data.get("checkpoint_step") != 415_000
        ):
            errors.append(f"{display(path)} has inconsistent directional metadata")
        try:
            checkpoint_r2 = float(data[specification["checkpoint_field"]])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{display(path)} lacks the checkpoint cross-readout R2")
        else:
            if not math.isclose(
                checkpoint_r2,
                specification["checkpoint_r2"],
                rel_tol=0,
                abs_tol=1e-12,
            ):
                errors.append(
                    f"{display(path)} has the wrong checkpoint cross-readout R2"
                )

    def validate_gate_or_legacy(
        data: dict,
        field: str,
        path: Path,
    ) -> None:
        if field not in data and legacy_omits(data, field):
            return
        validate_error_map(data, field, 1e-4, path, errors)
        values = data.get(field)
        if isinstance(values, dict):
            try:
                layers = {int(layer) for layer in values}
            except (TypeError, ValueError):
                layers = set()
            if layers != {4, 8, 16, 24}:
                errors.append(f"{display(path)} {field} lacks canonical switches")

    for direction, specification in directions.items():
        stem = specification["stem"]
        for domain in ("alpaca", "wikitext"):
            path = raw_dir / f"{stem}_longctx_{domain}.json"
            if not path.is_file():
                continue
            data = load_json_checked(path, errors)
            if data is None:
                continue
            validate_direction_metadata(data, path, direction, specification)
            if data.get("text") != domain:
                errors.append(f"{display(path)} has inconsistent domain metadata")
            if data.get("boundary") != specification["boundary"]:
                errors.append(f"{display(path)} has the wrong residual boundary")
            if {int(value) for value in data.get("ctx", {})} != {512, 1024, 2048}:
                errors.append(f"{display(path)} lacks the canonical context lengths")
            if data.get("evaluation_windows") != {
                "512": 48,
                "1024": 32,
                "2048": 20,
            }:
                errors.append(f"{display(path)} has incorrect evaluation window counts")
            settings_are_canonical = (
                data.get("switches") == [4, 8, 16, 24]
                and data.get("checkpoint_step") == 415_000
                and data.get("fit_rows_requested") == 40_000
                and data.get("fit_rows_actual") == 39_936
                and data.get("linear_ridge") == 1e-3
            )
            if direction == "B_to_A":
                settings_are_canonical = (
                    settings_are_canonical
                    and data.get("concurrent_arm_max_tokens") == 512
                )
            if not settings_are_canonical:
                errors.append(f"{display(path)} has noncanonical long-context settings")
            expected_long_configs = (
                {"pure-Pythia", "pure-RWKV"}
                | {f"{direction}@{layer}" for layer in (4, 8, 16, 24)}
                | {f"{direction}_affine@{layer}" for layer in (4, 8, 16, 24)}
            )
            for context, configurations in data.get("ctx", {}).items():
                if set(configurations) != expected_long_configs:
                    errors.append(
                        f"{display(path)} ctx={context} does not contain "
                        "10 canonical configs"
                    )
                for config, values in configurations.items():
                    try:
                        cross_entropy = float(values["ce"])
                        standard_error = float(values["se"])
                    except (KeyError, TypeError, ValueError):
                        errors.append(
                            f"{display(path)} has invalid CE/SE for {config} "
                            f"at ctx={context}"
                        )
                        continue
                    if not (
                        math.isfinite(cross_entropy)
                        and math.isfinite(standard_error)
                        and cross_entropy > 0
                        and standard_error >= 0
                    ):
                        errors.append(
                            f"{display(path)} has out-of-range CE/SE for {config} "
                            f"at ctx={context}"
                        )
            for field in specification["long_gates"]:
                validate_gate_or_legacy(data, field, path)

        expected_serving_configs = {
            "pure-Pythia",
            "pure-RWKV",
            *(f"{direction}@{layer}" for layer in (4, 8, 16, 24)),
        }
        serving_path = raw_dir / f"{stem}_serving_bench.json"
        if serving_path.is_file():
            data = load_json_checked(serving_path, errors)
            if data is not None:
                validate_direction_metadata(
                    data,
                    serving_path,
                    direction,
                    specification,
                )
                if (
                    data.get("implementation") != "unpruned_full_parent_reforward"
                    or data.get("switches") != [4, 8, 16, 24]
                    or data.get("contexts") != [512, 2048]
                    or data.get("decode_context") != 512
                    or data.get("decode_tokens") != 16
                    or data.get("repetitions") != 3
                ):
                    errors.append(
                        f"{display(serving_path)} has noncanonical benchmark settings"
                    )
                validate_gate_or_legacy(
                    data,
                    "parent_reproduction_relative_error",
                    serving_path,
                )
                prefill = data.get("prefill", {})
                if {int(context) for context in prefill} != {512, 2048}:
                    errors.append(
                        f"{display(serving_path)} has incorrect prefill contexts"
                    )
                for context, configurations in prefill.items():
                    if set(configurations) != expected_serving_configs:
                        errors.append(
                            f"{display(serving_path)} ctx={context} "
                            "lacks canonical configs"
                        )
                    for config, record in configurations.items():
                        validate_metric_record(
                            record,
                            ("ms", "peak_devA_MiB", "peak_devB_MiB"),
                            serving_path,
                            f"prefill/{context}/{config}",
                            errors,
                        )
                decode = data.get("decode", {})
                if set(decode) != expected_serving_configs:
                    errors.append(
                        f"{display(serving_path)} lacks canonical decode configs"
                    )
                for config, record in decode.items():
                    validate_metric_record(
                        record,
                        ("tok_per_s", "ms_per_token"),
                        serving_path,
                        f"decode/{config}",
                        errors,
                    )

        pruned_path = raw_dir / f"{stem}_serving_pruned.json"
        if pruned_path.is_file():
            data = load_json_checked(pruned_path, errors)
            if data is None:
                continue
            validate_direction_metadata(data, pruned_path, direction, specification)
            historical_a_to_b = (
                direction == "A_to_B"
                and data.get("measurement_provenance", {}).get("kind")
                == "legacy_metadata_enrichment"
            )
            tolerance_is_canonical = data.get("gate_tolerance") == 1e-4
            if historical_a_to_b:
                provenance = data["measurement_provenance"]
                tolerance_is_canonical = (
                    provenance.get("parent_gate_tolerance_at_measurement") == 1e-4
                    and provenance.get("pruned_gate_tolerance_at_measurement") == 1e-3
                )
            if (
                data.get("implementation") != "physically_pruned"
                or data.get("switches") != [4, 8, 16, 24]
                or data.get("ctx") != [512, 2048]
                or data.get("repetitions") != 3
                or not tolerance_is_canonical
            ):
                errors.append(
                    f"{display(pruned_path)} has noncanonical benchmark settings"
                )
            if "parent_reproduction_relative_error" in data or not legacy_omits(
                data, "parent_reproduction_relative_error"
            ):
                validate_gate_or_legacy(
                    data,
                    "parent_reproduction_relative_error",
                    pruned_path,
                )
            configurations = data.get("configs", {})
            expected_pruned_configs = (
                {"pure-Pythia", "pure-RWKV"}
                | {f"B_early_exit@{depth}" for depth in specification["early_depths"]}
                | {f"pruned-{direction}@{layer}" for layer in (4, 8, 16, 24)}
            )
            if set(configurations) != expected_pruned_configs:
                errors.append(f"{display(pruned_path)} lacks canonical configurations")
            for config, values in configurations.items():
                profiles = values.get("prefill", {})
                if {int(context) for context in profiles} != {512, 2048}:
                    errors.append(
                        f"{display(pruned_path)} has incorrect contexts for {config}"
                    )
                for context, record in profiles.items():
                    validate_metric_record(
                        record,
                        ("ms", "peak_devA_MiB", "peak_devB_MiB"),
                        pruned_path,
                        f"{config}/{context}",
                        errors,
                    )
            for layer in (4, 8, 16, 24):
                name = f"pruned-{direction}@{layer}"
                values = configurations.get(name)
                if not isinstance(values, dict):
                    errors.append(f"{display(pruned_path)} lacks {name}")
                    continue
                expected_pythia = 31 - layer if direction == "A_to_B" else layer + 1
                expected_rwkv = layer + 1 if direction == "A_to_B" else 31 - layer
                if values.get("pythia_blocks") != expected_pythia:
                    errors.append(
                        f"{display(pruned_path)} has wrong Pythia depth for {name}"
                    )
                if values.get("rwkv_blocks") != expected_rwkv:
                    errors.append(
                        f"{display(pruned_path)} has wrong RWKV depth for {name}"
                    )
                for field in specification["pruned_gates"]:
                    if (
                        field == "parent_gate_rel"
                        and field not in values
                        and legacy_omits(
                            data,
                            "parent_reproduction_relative_error",
                        )
                    ):
                        continue
                    try:
                        value = float(values[field])
                    except (KeyError, TypeError, ValueError):
                        errors.append(
                            f"{display(pruned_path)} has invalid {field} for {name}"
                        )
                        continue
                    if value >= 1e-4:
                        errors.append(
                            f"{display(pruned_path)} {field}={value:.3e} "
                            f"for {name}, expected <1e-4"
                        )

    for stem, direction, source_model, target_model, cross_layers in (
        (
            "a_to_b",
            "A_to_B",
            "RWKV-Raven-7B",
            "Tulu-Pythia-6.9B",
            {4: 27, 8: 23, 16: 15, 24: 7},
        ),
        (
            "b_to_a",
            "B_to_A",
            "Tulu-Pythia-6.9B",
            "RWKV-Raven-7B",
            {4: 5, 8: 9, 16: 17, 24: 25},
        ),
    ):
        kv_path = raw_dir / f"{stem}_kv_cache.csv"
        if not kv_path.is_file():
            continue
        try:
            _, rows = read_csv(kv_path)
            by_config = {row["config"]: row for row in rows}
            expected_layers = {
                "pure-Pythia": 32,
                **{
                    f"{direction}@{layer}": transformer_layers
                    for layer, transformer_layers in cross_layers.items()
                },
                "pure-RWKV": 0,
            }
            if set(by_config) != set(expected_layers):
                errors.append(f"{display(kv_path)} lacks the six canonical configs")
            for config, expected in expected_layers.items():
                if config not in by_config:
                    continue
                row = by_config[config]
                if (
                    row.get("direction") != direction
                    or row.get("source_model") != source_model
                    or row.get("target_model") != target_model
                ):
                    errors.append(
                        f"{display(kv_path)} has wrong path metadata for {config}"
                    )
                if (
                    row.get("context_tokens") != "4096"
                    or row.get("probe_sequence_tokens") != "128"
                ):
                    errors.append(
                        f"{display(kv_path)} has noncanonical probe settings "
                        f"for {config}"
                    )
                try:
                    actual = int(row["transformer_layers"])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{display(kv_path)} has invalid depth for {config}")
                    continue
                if actual != expected:
                    errors.append(
                        f"{display(kv_path)} has {actual} Transformer layers "
                        f"for {config}; expected {expected}"
                    )
                try:
                    reduction = float(row["kv_reduction_fraction"])
                    kib = float(row["kv_kib_per_token"])
                    per_layer = float(row["measured_kib_per_token_per_layer"])
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"{display(kv_path)} has invalid memory metrics for {config}"
                    )
                    continue
                expected_reduction = 1 - expected / 32
                if not math.isclose(
                    reduction,
                    expected_reduction,
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f"{display(kv_path)} has wrong KV reduction for {config}"
                    )
                if not math.isclose(
                    kib,
                    per_layer * expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f"{display(kv_path)} has inconsistent KV bytes for {config}"
                    )
        except Exception as exc:
            errors.append(str(exc))

    a_to_b_generation_path = raw_dir / "a_to_b_generation_samples.csv"
    if a_to_b_generation_path.is_file():
        try:
            _, rows = read_csv(a_to_b_generation_path)
            validate_generation_contract(
                rows,
                a_to_b_generation_path,
                errors,
            )
            if len(rows) != 12:
                errors.append(
                    f"{display(a_to_b_generation_path)} has {len(rows)} rows; expected 12"
                )
            if {row.get("direction") for row in rows} != {
                "A_parent",
                "B_parent",
                "A_to_B",
            }:
                errors.append(
                    f"{display(a_to_b_generation_path)} has inconsistent path labels"
                )
            if {row.get("config") for row in rows} != {
                "pure-Pythia",
                "pure-RWKV",
                "A_to_B@4",
            }:
                errors.append(
                    f"{display(a_to_b_generation_path)} lacks canonical configs"
                )
            if {row.get("prompt_id") for row in rows} != {"0", "1", "2", "3"}:
                errors.append(
                    f"{display(a_to_b_generation_path)} lacks the four prompts"
                )
            for row in rows:
                if row.get("config") != "A_to_B@4":
                    continue
                if row.get("switch_layer") != "4":
                    errors.append(
                        f"{display(a_to_b_generation_path)} has wrong generation switch"
                    )
                if row.get("checkpoint_step") != "415000":
                    errors.append(
                        f"{display(a_to_b_generation_path)} has wrong checkpoint"
                    )
                try:
                    gate = float(row["parent_gate_relative_error"])
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"{display(a_to_b_generation_path)} has invalid parent gate"
                    )
                    continue
                if gate >= 1e-4:
                    errors.append(
                        f"{display(a_to_b_generation_path)} parent gate={gate:.3e}"
                    )
        except Exception as exc:
            errors.append(str(exc))

    generation_path = raw_dir / "b_to_a_generation_samples.csv"
    if generation_path.is_file():
        try:
            _, rows = read_csv(generation_path)
            validate_generation_contract(rows, generation_path, errors)
            if len(rows) != 12:
                errors.append(
                    f"{display(generation_path)} has {len(rows)} rows; expected 12"
                )
            if {row.get("direction") for row in rows} != {
                "A_parent",
                "B_parent",
                "B_to_A",
            }:
                errors.append(
                    f"{display(generation_path)} has inconsistent direction labels"
                )
            expected_generation_configs = {
                "pure-Pythia",
                "pure-RWKV",
                "B_to_A@4",
            }
            if {row.get("config") for row in rows} != expected_generation_configs:
                errors.append(f"{display(generation_path)} lacks canonical configs")
            if {row.get("prompt_id") for row in rows} != {"0", "1", "2", "3"}:
                errors.append(f"{display(generation_path)} lacks the four prompts")
            for row in rows:
                if row.get("config") == "B_to_A@4":
                    if row.get("switch_layer") != "4":
                        errors.append(
                            f"{display(generation_path)} has wrong generation switch"
                        )
                    for field in (
                        "parent_gate_relative_error",
                        "direct_gate_relative_error",
                    ):
                        try:
                            value = float(row[field])
                        except (KeyError, TypeError, ValueError):
                            errors.append(
                                f"{display(generation_path)} has invalid {field}"
                            )
                            continue
                        if value >= 1e-4:
                            errors.append(
                                f"{display(generation_path)} {field}={value:.3e}"
                            )
                    if row.get("checkpoint_step") != "415000":
                        errors.append(
                            f"{display(generation_path)} has wrong checkpoint "
                            f"for {row.get('config')}"
                        )
            if a_to_b_generation_path.is_file():
                _, a_to_b_rows = read_csv(a_to_b_generation_path)
                a_to_b_parents = {
                    (row.get("prompt_id"), row.get("config")): row.get("output")
                    for row in a_to_b_rows
                    if row.get("config") in {"pure-Pythia", "pure-RWKV"}
                }
                b_to_a_parents = {
                    (row.get("prompt_id"), row.get("config")): row.get("output")
                    for row in rows
                    if row.get("config") in {"pure-Pythia", "pure-RWKV"}
                }
                if a_to_b_parents != b_to_a_parents:
                    errors.append("AB/BA generation parents differ on matched prompts")
        except Exception as exc:
            errors.append(str(exc))

    for stem, direction in (
        ("a_to_b", "A_to_B"),
        ("b_to_a", "B_to_A"),
    ):
        paired_path = raw_dir / f"{stem}_paired_statistics.csv"
        if not paired_path.is_file():
            continue
        try:
            _, rows = read_csv(paired_path)
            if len(rows) != 48:
                errors.append(
                    f"{display(paired_path)} has {len(rows)} rows; expected 48"
                )
            if {row.get("direction") for row in rows} != {direction}:
                errors.append(
                    f"{display(paired_path)} has inconsistent direction labels"
                )
            for row in rows:
                try:
                    discordant = int(row["mcnemar_b"]) + int(row["mcnemar_c"])
                    p_value = float(row["mcnemar_p"])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{display(paired_path)} has invalid McNemar fields")
                    continue
                if not 0.0 <= p_value <= 1.0:
                    errors.append(
                        f"{display(paired_path)} has out-of-range McNemar p-value"
                    )
                if discordant and p_value == 0.0:
                    errors.append(
                        f"{display(paired_path)} contains an underflowed "
                        "McNemar p-value"
                    )
            expected_parent_pairs = {
                (task, f"{direction}@{layer}", parent)
                for task in ("arc_easy", "sciq")
                for layer in (4, 8, 16, 24)
                for parent in ("pythia", "rwkv")
            }
            actual_parent_pairs = {
                (row.get("task"), row.get("config_a"), row.get("config_b"))
                for row in rows
                if row.get("config_b") in {"pythia", "rwkv"}
            }
            if actual_parent_pairs != expected_parent_pairs:
                errors.append(
                    f"{display(paired_path)} lacks all 16 {direction} "
                    "parent comparisons"
                )
            expected_interventions = {
                (
                    task,
                    f"{direction}@{layer}",
                    f"{direction}_alpha0@{layer}",
                )
                for task in ("arc_easy", "sciq")
                for layer in (4, 8, 16, 24)
            }
            actual_interventions = {
                (row.get("task"), row.get("config_a"), row.get("config_b"))
                for row in rows
                if "_alpha0@" in row.get("config_b", "")
            }
            if actual_interventions != expected_interventions:
                errors.append(
                    f"{display(paired_path)} lacks all eight {direction} "
                    "intervention comparisons"
                )
        except Exception as exc:
            errors.append(str(exc))


def validate_execution_path_table(table_dir: Path, errors: list[str]) -> None:
    """Require a complete, uniquely named 2x2 execution-path matrix."""

    path = table_dir / "paper_four_path_qa_accuracy.csv"
    if not path.is_file():
        return
    try:
        _, rows = read_csv(path)
    except Exception as exc:
        errors.append(str(exc))
        return

    path_specs = {
        "A_to_A": ("A", "A", "same_family_control"),
        "A_to_B": ("A", "B", "cross_family_chimera"),
        "B_to_B": ("B", "B", "same_family_control"),
        "B_to_A": ("B", "A", "cross_family_chimera"),
    }
    expected_keys = {
        (task, str(layer), execution_path)
        for task in ("arc_easy", "sciq")
        for layer in (4, 8, 16, 24)
        for execution_path in path_specs
    }
    actual_keys = {
        (row.get("task"), row.get("switch_layer"), row.get("execution_path"))
        for row in rows
    }
    if len(rows) != len(expected_keys) or actual_keys != expected_keys:
        errors.append(
            f"{display(path)} is not the complete 2 tasks x 4 layers x 4 paths matrix"
        )
    if len(actual_keys) != len(rows):
        errors.append(f"{display(path)} contains duplicate task/layer/path rows")

    for row in rows:
        execution_path = row.get("execution_path", "")
        if execution_path not in path_specs:
            continue
        prefix, suffix, role = path_specs[execution_path]
        layer = row.get("switch_layer")
        expected_name = f"NinaXander-{prefix}{suffix}@{layer}"
        if (
            row.get("model_name") != expected_name
            or row.get("prefix_family") != prefix
            or row.get("suffix_family") != suffix
            or row.get("path_role") != role
        ):
            errors.append(
                f"{display(path)} has inconsistent four-path naming for "
                f"{row.get('task')}/{execution_path}@{layer}"
            )
        try:
            accuracy = float(row["accuracy_norm"])
        except (KeyError, TypeError, ValueError):
            errors.append(
                f"{display(path)} has invalid accuracy for "
                f"{row.get('task')}/{execution_path}@{layer}"
            )
        else:
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                errors.append(
                    f"{display(path)} has out-of-range accuracy for "
                    f"{row.get('task')}/{execution_path}@{layer}"
                )


def validate_cross_family_mlp_bundles(
    raw_dir: Path,
    errors: list[str],
) -> None:
    """Require each intervention bundle to contain matched AB and BA outcomes."""

    representation_path = raw_dir / "cross_family_mlp_representation.json"
    if representation_path.is_file():
        representation = load_json_checked(representation_path, errors)
        if representation is not None:
            directions = representation.get("r2_by_direction_alpha_layer", {})
            expected_alphas = {"0.0", "0.25", "0.5", "0.75", "1.0"}
            expected_layers = {"4", "10", "16", "22", "28"}
            if set(directions) != {"A_to_B", "B_to_A"}:
                errors.append(
                    f"{display(representation_path)} does not contain AB and BA"
                )
            else:
                for direction, alpha_values in directions.items():
                    if set(alpha_values) != expected_alphas or any(
                        set(layer_values) != expected_layers
                        for layer_values in alpha_values.values()
                    ):
                        errors.append(
                            f"{display(representation_path)} has incomplete "
                            f"{direction} alpha/layer coverage"
                        )
            csv_path = representation_path.with_suffix(".csv")
            if csv_path.is_file():
                try:
                    _, rows = read_csv(csv_path)
                    counts = {
                        direction: sum(
                            row.get("direction") == direction for row in rows
                        )
                        for direction in ("A_to_B", "B_to_A")
                    }
                    if counts != {"A_to_B": 25, "B_to_A": 25}:
                        errors.append(f"{display(csv_path)} is not balanced: {counts}")
                except Exception as exc:
                    errors.append(str(exc))

    for suffix, task, count in (
        ("arc", "arc_easy", 2376),
        ("sciq", "sciq", 1000),
    ):
        path = raw_dir / f"cross_family_mlp_intervention_{suffix}.json"
        if not path.is_file():
            continue
        data = load_json_checked(path, errors)
        if data is None:
            continue
        qa_by_direction = data.get("qa_by_direction")
        if data.get("task") != task or not isinstance(qa_by_direction, dict):
            errors.append(f"{display(path)} lacks symmetric task-level QA")
            continue
        if set(qa_by_direction) != {"A_to_B", "B_to_A"}:
            errors.append(f"{display(path)} does not contain both AB and BA QA")
            continue
        for direction, qa in qa_by_direction.items():
            expected_configs = {
                *(f"{direction}@{layer}" for layer in (4, 8, 16, 24)),
                *(f"{direction}_alpha0@{layer}" for layer in (4, 8, 16, 24)),
            }
            if qa.get("N") != count or set(qa.get("configs", [])) != expected_configs:
                errors.append(
                    f"{display(path)} has incomplete {direction} intervention metadata"
                )
                continue
            outcomes = qa.get("per_item_norm", {})
            accuracies = qa.get("acc_norm", {})
            for config in expected_configs:
                values = outcomes.get(config)
                if not isinstance(values, list) or len(values) != count:
                    errors.append(
                        f"{display(path)} {direction}/{config} does not have "
                        f"{count} item outcomes"
                    )
                    continue
                if any(value not in (0, 1) for value in values):
                    errors.append(
                        f"{display(path)} has non-binary outcomes for {config}"
                    )
                try:
                    reported = float(accuracies[config])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{display(path)} lacks acc_norm[{config}]")
                    continue
                if not math.isclose(
                    reported,
                    sum(values) / count,
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f"{display(path)} acc_norm[{config}] disagrees with "
                        "its per-item outcomes"
                    )
        csv_path = path.with_suffix(".csv")
        if csv_path.is_file():
            try:
                _, rows = read_csv(csv_path)
                counts = {
                    direction: sum(row.get("direction") == direction for row in rows)
                    for direction in ("A_to_B", "B_to_A")
                }
                if counts != {"A_to_B": 33, "B_to_A": 33}:
                    errors.append(
                        f"{display(csv_path)} is not balanced across AB/BA: {counts}"
                    )
            except Exception as exc:
                errors.append(str(exc))


def validate_directional_raw_pairs(raw_dir: Path, errors: list[str]) -> None:
    """Reject any directional raw artifact that lacks its opposite-path peer."""

    for suffix in ("json", "csv"):
        a_to_b = {
            path.name.removeprefix("a_to_b_")
            for path in raw_dir.glob(f"a_to_b_*.{suffix}")
        }
        b_to_a = {
            path.name.removeprefix("b_to_a_")
            for path in raw_dir.glob(f"b_to_a_*.{suffix}")
        }
        missing = sorted(a_to_b - b_to_a)
        if missing:
            errors.append(
                f"A-to-B-only raw {suffix} artifacts lack B-to-A pairs: {missing}"
            )
        unexpected_b_only = sorted(b_to_a - a_to_b)
        if unexpected_b_only:
            errors.append(
                f"unclassified B-to-A-only raw {suffix} artifacts: {unexpected_b_only}"
            )
        if suffix == "csv":
            for relative in sorted(a_to_b & b_to_a):
                a_path = raw_dir / f"a_to_b_{relative}"
                b_path = raw_dir / f"b_to_a_{relative}"
                try:
                    a_header, _ = read_csv(a_path)
                    b_header, _ = read_csv(b_path)
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                if a_header != b_header:
                    errors.append(
                        "directional raw CSV schemas differ for "
                        f"{relative}: AB={a_header}, BA={b_header}"
                    )


def validate_four_path_layer_metrics(
    raw_dir: Path,
    errors: list[str],
) -> None:
    """Require one complete AA/AB/BB/BA record at every adapter layer."""

    path = raw_dir / "four_path_layer_metrics.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return  # The generic raw-bundle validation reports the concrete error.
    rows = data.get("rows")
    if not isinstance(rows, dict):
        errors.append(f"{display(path)} lacks per-layer rows")
        return
    try:
        layers = {int(layer) for layer in rows}
    except (TypeError, ValueError):
        errors.append(f"{display(path)} has non-integer layer keys")
        return
    if layers != set(range(32)):
        errors.append(f"{display(path)} does not cover layers 0..31 exactly")
    required = {
        "A->A",
        "A->B",
        "B->B",
        "B->A",
        "rho_ctr",
        "A->B_shuf",
        "B->A_shuf",
        "rho_ctr_shuf",
    }
    for layer, values in rows.items():
        if not isinstance(values, dict) or not required.issubset(values):
            errors.append(
                f"{display(path)} layer {layer} lacks complete AA/AB/BB/BA metrics"
            )
    for summary in ("mean_0_31", "mean_1_31"):
        values = data.get(summary)
        if not isinstance(values, dict) or not required.issubset(values):
            errors.append(f"{display(path)} lacks complete {summary} metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-runtime",
        action="store_true",
        help="also require artifacts emitted only by a fresh evaluation run",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLES)
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
    raw_dir = args.raw_dir.resolve()
    table_dir = args.table_dir.resolve()
    errors: list[str] = []

    for stem in RAW_BUNDLES:
        json_path = raw_dir / f"{stem}.json"
        csv_path = raw_dir / f"{stem}.csv"
        for path in (json_path, csv_path):
            if not path.is_file():
                errors.append(f"missing raw artifact: {display(path)}")
        if json_path.is_file():
            try:
                json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"invalid JSON {display(json_path)}: {exc}")
        if csv_path.is_file():
            try:
                _, rows = read_csv(csv_path)
                if not rows:
                    errors.append(f"empty CSV: {display(csv_path)}")
            except Exception as exc:
                errors.append(str(exc))

    for stem in RETIRED_RAW:
        for suffix in (".json", ".csv"):
            path = raw_dir / f"{stem}{suffix}"
            if path.exists():
                errors.append(f"retired raw artifact must be removed: {display(path)}")

    if args.strict_runtime:
        for name in (
            "a_to_b_generation_samples.csv",
            "a_to_b_paired_statistics.csv",
            "a_to_b_kv_cache.csv",
            "b_to_a_generation_samples.csv",
            "b_to_a_paired_statistics.csv",
            "b_to_a_kv_cache.csv",
        ):
            if not (raw_dir / name).is_file():
                errors.append(f"missing fresh-run artifact: {display(raw_dir / name)}")

    validate_a_to_b_semantics(raw_dir, errors)
    validate_cross_path_semantics(raw_dir, errors)
    validate_cross_family_mlp_bundles(raw_dir, errors)
    validate_directional_raw_pairs(raw_dir, errors)
    validate_four_path_layer_metrics(raw_dir, errors)

    training_curve = raw_dir / "training_curve.csv"
    if not training_curve.is_file():
        errors.append(f"missing structured training curve: {display(training_curve)}")
    else:
        try:
            _, rows = read_csv(training_curve)
            if not rows:
                errors.append(f"empty CSV: {display(training_curve)}")
        except Exception as exc:
            errors.append(str(exc))

    manifest_path = table_dir / "MANIFEST.csv"
    if not manifest_path.is_file():
        errors.append(f"missing {display(manifest_path)}")
        manifest_rows = []
    else:
        _, manifest_rows = read_csv(manifest_path)
    manifest_names = {row["table"] for row in manifest_rows}

    for name, expected_rows in EXPECTED_TABLES.items():
        path = table_dir / name
        if not path.is_file():
            errors.append(f"missing paper table: {display(path)}")
            continue
        try:
            _, rows = read_csv(path)
            if len(rows) != expected_rows:
                errors.append(
                    f"{display(path)} has {len(rows)} rows; expected {expected_rows}"
                )
            if name in EXECUTION_PATH_TABLE_COUNTS:
                actual_path_counts: dict[str, int] = {}
                for row in rows:
                    execution_path = row.get("execution_path", "")
                    actual_path_counts[execution_path] = (
                        actual_path_counts.get(execution_path, 0) + 1
                    )
                if actual_path_counts != EXECUTION_PATH_TABLE_COUNTS[name]:
                    errors.append(
                        f"{display(path)} has invalid execution-path coverage: "
                        f"{actual_path_counts}"
                    )
            if name == "paper_cross_family_paired_statistics.csv":
                for row in rows:
                    try:
                        discordant = int(row["mcnemar_b"]) + int(row["mcnemar_c"])
                        p_value = float(row["mcnemar_p"])
                    except (KeyError, TypeError, ValueError):
                        errors.append(f"{display(path)} has invalid McNemar fields")
                        continue
                    if not 0.0 <= p_value <= 1.0:
                        errors.append(
                            f"{display(path)} has out-of-range McNemar p-value"
                        )
                    if discordant and p_value == 0.0:
                        errors.append(
                            f"{display(path)} contains an underflowed McNemar p-value"
                        )
                comparison_keys = [
                    (
                        row.get("comparison_family"),
                        row.get("task"),
                        row.get("config_a"),
                        row.get("config_b"),
                    )
                    for row in rows
                ]
                if len(comparison_keys) != len(set(comparison_keys)):
                    errors.append(
                        f"{display(path)} contains duplicated paired comparisons"
                    )
                cross_path = [
                    row for row in rows if row.get("comparison_family") == "cross_path"
                ]
                expected_pairs = {
                    (
                        task,
                        f"B_to_A@{layer}",
                        f"A_to_B@{layer}",
                    )
                    for task in ("arc_easy", "sciq")
                    for layer in (4, 8, 16, 24)
                }
                actual_pairs = {
                    (row.get("task"), row.get("config_a"), row.get("config_b"))
                    for row in cross_path
                }
                if actual_pairs != expected_pairs:
                    errors.append(
                        f"{display(path)} lacks the eight paired "
                        "B-to-A versus A-to-B comparisons"
                    )
                expected_interventions = {
                    (
                        task,
                        f"{direction}@{layer}",
                        f"{direction}_alpha0@{layer}",
                    )
                    for task in ("arc_easy", "sciq")
                    for direction in ("A_to_B", "B_to_A")
                    for layer in (4, 8, 16, 24)
                }
                actual_interventions = {
                    (row.get("task"), row.get("config_a"), row.get("config_b"))
                    for row in rows
                    if "_alpha0@" in row.get("config_b", "")
                }
                if actual_interventions != expected_interventions:
                    errors.append(
                        f"{display(path)} lacks the 16 matched AB/BA "
                        "same-adapter alpha interventions"
                    )
                expected_parent_pairs = {
                    (task, f"{direction}@{layer}", parent)
                    for task in ("arc_easy", "sciq")
                    for direction in ("A_to_B", "B_to_A")
                    for layer in (4, 8, 16, 24)
                    for parent in ("pythia", "rwkv")
                }
                actual_parent_pairs = {
                    (row.get("task"), row.get("config_a"), row.get("config_b"))
                    for row in rows
                    if row.get("config_b") in {"pythia", "rwkv"}
                }
                if actual_parent_pairs != expected_parent_pairs:
                    errors.append(
                        f"{display(path)} lacks the 32 symmetric AB/BA "
                        "parent comparisons"
                    )
            if name == "paper_cross_family_generation_samples.csv":
                validate_generation_contract(rows, path, errors)
                stale = [
                    row.get("config", "")
                    for row in rows
                    if row.get("config", "").endswith(("-low", "-high"))
                ]
                if stale:
                    errors.append(
                        f"{display(path)} contains retired checkpoint labels: {stale}"
                    )
            if name == "paper_result_coverage.csv":
                expected_coverage_tables = {
                    "paper_four_path_qa_accuracy.csv",
                    "paper_cross_family_qa_accuracy.csv",
                    "paper_representation.csv",
                    "paper_cross_family_linearity.csv",
                    "paper_cross_family_domain_shift.csv",
                    "paper_layer_metrics.csv",
                    "paper_cross_family_mlp_intervention.csv",
                    "paper_cross_family_paired_statistics.csv",
                    "paper_cross_family_serving.csv",
                    "paper_training_curve.csv",
                    "paper_cross_family_generation_samples.csv",
                    "paper_cross_family_memory_accuracy.csv",
                }
                actual_coverage_tables = {row.get("table", "") for row in rows}
                if actual_coverage_tables != expected_coverage_tables:
                    errors.append(
                        f"{display(path)} has incomplete table coverage: "
                        f"{sorted(actual_coverage_tables)}"
                    )
                unmatched = [
                    row.get("table", "")
                    for row in rows
                    if row.get("ab_ba_matched", "").lower() != "true"
                ]
                if unmatched:
                    errors.append(
                        "public result families with unmatched AB/BA coverage: "
                        f"{unmatched}"
                    )
                incomplete_four_path = [
                    row.get("table", "")
                    for row in rows
                    if row.get("four_path_applicable", "").lower() == "true"
                    and row.get("four_path_complete", "").lower() != "true"
                ]
                if incomplete_four_path:
                    errors.append(
                        "applicable tables lack complete AA/AB/BB/BA coverage: "
                        f"{incomplete_four_path}"
                    )
                spurious_four_path = [
                    row.get("table", "")
                    for row in rows
                    if row.get("four_path_applicable", "").lower() == "false"
                    and row.get("four_path_complete", "") != ""
                ]
                if spurious_four_path:
                    errors.append(
                        "non-applicable tables must leave four_path_complete empty: "
                        f"{spurious_four_path}"
                    )
        except Exception as exc:
            errors.append(str(exc))
        if name not in manifest_names:
            errors.append(f"{name} is absent from MANIFEST.csv")

    validate_execution_path_table(table_dir, errors)

    # The final representation table must exactly project the release config.
    release_config = args.release_config.resolve()
    representation = table_dir / "paper_representation.csv"
    if release_config.is_file() and representation.is_file():
        config = json.loads(release_config.read_text(encoding="utf-8"))
        if config.get("generation_contract") != {
            "decoding": "greedy_argmax",
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 40,
            "early_stop": "eos_token_id",
            "seed": 0,
        }:
            errors.append(
                f"{display(release_config)} has the wrong generation contract"
            )
        _, rows = read_csv(representation)
        actual = {row["checkpoint_key"]: float(row["value"]) for row in rows}
        for key, expected in config["evaluation"].items():
            if key not in actual or abs(actual[key] - float(expected)) > 1e-12:
                errors.append(
                    f"paper_representation.csv disagrees with release config for {key}"
                )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(
        f"RESULT_DATA_OK tables={len(EXPECTED_TABLES)} raw_bundles={len(RAW_BUNDLES)}"
    )


if __name__ == "__main__":
    main()
