#!/usr/bin/env python3
"""Build one self-contained, direction-locked NinaXander Hub package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


MODEL_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
GENERATION_CONFIG_FILE = "generation_config.json"
PARENT_FINGERPRINTS_FILE = "parent_fingerprints.json"
CHECKSUM_FILE = "SHA256SUMS"
SUPPORTED_SWITCH_LAYERS = (4, 8, 16, 24)
DIRECTION_ARGUMENTS = {
    "rwkv-to-pythia": "A_to_B",
    "pythia-to-rwkv": "B_to_A",
}
DIRECTION_DISPLAY_NAMES = {
    "A_to_B": "RWKV-to-Pythia",
    "B_to_A": "Pythia-to-RWKV",
}
MODEL_NAME_TEMPLATES = {
    "A_to_A": "NinaXander-RWKV-reconstruction@{layer}",
    "A_to_B": "NinaXander-RWKV-to-Pythia@{layer}",
    "B_to_B": "NinaXander-Pythia-reconstruction@{layer}",
    "B_to_A": "NinaXander-Pythia-to-RWKV@{layer}",
}
REPRESENTATION_EVALUATION_ORDER = (
    "A->A",
    "A->B",
    "B->B",
    "B->A",
    "rho_ctr",
    "rho_raw",
    "f",
)
PARENT_MODEL_FILENAMES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
}
TOKENIZER_FILENAMES = {
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_architecture(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    block_ids = sorted(
        {
            int(name.split(".")[1])
            for name in state
            if name.startswith("eA.") and name.endswith(".fc1.weight")
        }
    )
    expected = list(range(1, len(block_ids) + 1))
    if not block_ids or block_ids != expected:
        raise ValueError(f"Unexpected residual block indices: {block_ids}")
    hidden = int(state[f"eA.{block_ids[0]}.fc1.weight"].shape[0])
    return len(block_ids), hidden


def stack_statistics(
    stats: dict[Any, Any], layers: list[int]
) -> dict[str, torch.Tensor]:
    names = ("a_mean", "a_std", "b_mean", "b_std")
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for layer in layers:
        values = stats[layer] if layer in stats else stats[str(layer)]
        if len(values) != len(names):
            raise ValueError(
                f"Layer {layer} has {len(values)} statistics; expected four"
            )
        for name, value in zip(names, values, strict=True):
            rows[name].append(value.detach().cpu().contiguous())
    return {
        f"statistics.{name}": torch.stack(values).contiguous()
        for name, values in rows.items()
    }


def selection_metrics(
    qa_table: Path,
    memory_table: Path,
    *,
    qa_config_type: str,
    memory_config_prefix: str,
) -> tuple[int, dict[int, dict[str, float]]]:
    by_layer: dict[int, dict[str, float]] = {
        layer: {} for layer in SUPPORTED_SWITCH_LAYERS
    }
    for row in read_csv(qa_table):
        if row["config_type"] != qa_config_type or not row["switch_layer"]:
            continue
        layer = int(row["switch_layer"])
        if layer not in by_layer:
            continue
        by_layer[layer][f"{row['task']}_accuracy_norm"] = float(row["accuracy_norm"])

    for row in read_csv(memory_table):
        if not row["config"].startswith(memory_config_prefix):
            continue
        layer = int(row["switch_layer"])
        if layer not in by_layer:
            continue
        by_layer[layer]["transformer_blocks"] = float(row["transformer_layers"])
        by_layer[layer]["kv_reduction_fraction"] = float(row["kv_reduction_fraction"])

    required = {
        "arc_easy_accuracy_norm",
        "sciq_accuracy_norm",
        "transformer_blocks",
        "kv_reduction_fraction",
    }
    for layer, values in by_layer.items():
        missing = required.difference(values)
        if missing:
            raise ValueError(
                f"Missing selection metrics for switch layer {layer}: {sorted(missing)}"
            )
        values["mean_qa_accuracy_norm"] = (
            values["arc_easy_accuracy_norm"] + values["sciq_accuracy_norm"]
        ) / 2.0

    best_layer = max(
        SUPPORTED_SWITCH_LAYERS,
        key=lambda layer: (by_layer[layer]["mean_qa_accuracy_norm"], -layer),
    )
    return best_layer, by_layer


def model_assets(directory: Path) -> list[Path]:
    assets = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and (path.name in PARENT_MODEL_FILENAMES or path.name.endswith(".safetensors"))
    ]
    required = {"config.json", "model.safetensors.index.json"}
    missing = required.difference(path.name for path in assets)
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing parent-model files: {sorted(missing)}"
        )
    if not any(path.name.endswith(".safetensors") for path in assets):
        raise FileNotFoundError(f"{directory} contains no SafeTensors weight shards")
    return sorted(assets)


def tokenizer_assets(directory: Path) -> list[Path]:
    assets = [directory / name for name in sorted(TOKENIZER_FILENAMES)]
    missing = [path.name for path in assets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} is missing tokenizer files: {missing}")
    return assets


def materialize_assets(
    sources: list[Path],
    destination: Path,
    *,
    overwrite: bool,
    copy_assets: bool,
    clean_destination: bool = True,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {path.name for path in sources}
    stale = (
        [path for path in destination.iterdir() if path.name not in expected]
        if clean_destination
        else []
    )
    if stale and not overwrite:
        raise FileExistsError(
            f"Unexpected files in {destination}: {', '.join(path.name for path in stale)}"
        )
    if overwrite:
        for path in stale:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    for source in sources:
        target = destination / source.name
        if target.exists() or target.is_symlink():
            if not overwrite:
                raise FileExistsError(target)
            target.unlink()
        resolved = source.resolve()
        if copy_assets:
            shutil.copy2(resolved, target)
        else:
            try:
                os.link(resolved, target)
            except OSError:
                shutil.copy2(resolved, target)


def fingerprint_tree(directory: Path) -> dict[str, Any]:
    files = []
    for path in sorted(
        candidate for candidate in directory.iterdir() if candidate.is_file()
    ):
        files.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {"files": files}


def fingerprint_files(paths: list[Path], *, subdir: str) -> dict[str, Any]:
    return {
        "subdir": subdir,
        "files": [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(paths)
        ],
    }


def write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name != CHECKSUM_FILE
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    lines = [
        f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in paths
    ]
    (output_dir / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--execution-path-table", type=Path, required=True)
    parser.add_argument("--cross-family-qa-table", type=Path, required=True)
    parser.add_argument("--cross-family-memory-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--direction",
        choices=tuple(DIRECTION_ARGUMENTS),
        required=True,
        help=(
            "Lock this release to one public execution direction. The best "
            "measured switch layer within that direction becomes the default."
        ),
    )
    parser.add_argument("--source-model", default="RWKV/rwkv-raven-7b")
    parser.add_argument(
        "--target-model", default="allenai/open-instruct-pythia-6.9b-tulu"
    )
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hardlink-assets",
        action="store_true",
        help=(
            "Use space-saving hard links for a local draft. The default makes "
            "independent copies suitable for a release directory."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated files and bundled assets.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    source_dir = args.source_dir.resolve()
    target_dir = args.target_dir.resolve()
    tokenizer_dir = args.tokenizer_dir.resolve()
    execution_path_table = args.execution_path_table.resolve()
    cross_family_qa_table = args.cross_family_qa_table.resolve()
    cross_family_memory_table = args.cross_family_memory_table.resolve()
    output_dir = args.output_dir.resolve()
    for path in (
        checkpoint,
        execution_path_table,
        cross_family_qa_table,
        cross_family_memory_table,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (source_dir, target_dir, tokenizer_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_assets = not args.hardlink_assets

    generated = [
        output_dir / MODEL_FILE,
        output_dir / CONFIG_FILE,
        output_dir / GENERATION_CONFIG_FILE,
        output_dir / PARENT_FINGERPRINTS_FILE,
        output_dir / CHECKSUM_FILE,
    ]
    existing = [path for path in generated if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite generated files: {names}; pass --overwrite"
        )

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    required = {"model", "stats", "cfg", "eval", "step"}
    missing = required.difference(blob)
    if missing:
        raise KeyError(f"Checkpoint is missing required fields: {sorted(missing)}")

    cfg = blob["cfg"]
    layers = [int(layer) for layer in cfg["layers"]]
    if layers != list(range(32)):
        raise ValueError(
            f"Expected the current 32-layer checkpoint, got layers={layers}"
        )
    state = {
        f"adapter.{name}": value.detach().cpu().contiguous()
        for name, value in blob["model"].items()
    }
    blocks, hidden = infer_architecture(blob["model"])
    tensors = {**state, **stack_statistics(blob["stats"], layers)}

    a_to_b_best_layer, a_to_b_metrics = selection_metrics(
        cross_family_qa_table,
        cross_family_memory_table,
        qa_config_type="A_to_B",
        memory_config_prefix="A_to_B@",
    )
    b_to_a_best_layer, b_to_a_metrics = selection_metrics(
        cross_family_qa_table,
        cross_family_memory_table,
        qa_config_type="B_to_A",
        memory_config_prefix="B_to_A@",
    )
    metrics_by_direction = {
        "A_to_B": a_to_b_metrics,
        "B_to_A": b_to_a_metrics,
    }
    best_layers = {
        "A_to_B": a_to_b_best_layer,
        "B_to_A": b_to_a_best_layer,
    }
    best_direction = DIRECTION_ARGUMENTS[args.direction]
    best_layer = best_layers[best_direction]
    metrics = metrics_by_direction[best_direction]
    directions = (best_direction,)
    best_direction_index = 0
    supported = torch.tensor(SUPPORTED_SWITCH_LAYERS, dtype=torch.int64)
    tensors.update(
        {
            "switch.supported_layers": supported,
            "switch.source_blocks": supported.clone(),
            "switch.target_blocks": supported.clone(),
            "switch.target_start_blocks": supported + 1,
            "switch.transformer_blocks": torch.tensor(
                [
                    int(metrics[layer]["transformer_blocks"])
                    for layer in SUPPORTED_SWITCH_LAYERS
                ],
                dtype=torch.int64,
            ),
            "switch.default_layer": torch.tensor([best_layer], dtype=torch.int64),
            "switch.default_direction_index": torch.tensor(
                [best_direction_index],
                dtype=torch.int64,
            ),
            "switch.supported_direction_indices": torch.arange(
                len(directions),
                dtype=torch.int64,
            ),
            "switch.parent_reproduction_threshold": torch.tensor(
                [1e-4], dtype=torch.float32
            ),
            "evaluation.arc_easy_accuracy_norm": torch.tensor(
                [
                    metrics[layer]["arc_easy_accuracy_norm"]
                    for layer in SUPPORTED_SWITCH_LAYERS
                ],
                dtype=torch.float64,
            ),
            "evaluation.sciq_accuracy_norm": torch.tensor(
                [
                    metrics[layer]["sciq_accuracy_norm"]
                    for layer in SUPPORTED_SWITCH_LAYERS
                ],
                dtype=torch.float64,
            ),
            "evaluation.mean_qa_accuracy_norm": torch.tensor(
                [
                    metrics[layer]["mean_qa_accuracy_norm"]
                    for layer in SUPPORTED_SWITCH_LAYERS
                ],
                dtype=torch.float64,
            ),
            "evaluation.kv_reduction_fraction": torch.tensor(
                [
                    metrics[layer]["kv_reduction_fraction"]
                    for layer in SUPPORTED_SWITCH_LAYERS
                ],
                dtype=torch.float64,
            ),
        }
    )
    for direction in directions:
        direction_metrics = metrics_by_direction[direction]
        direction_key = direction.lower()
        prefix = f"evaluation.{direction_key}"
        tensors.update(
            {
                f"switch.{direction_key}.source_post_block": supported.clone(),
                f"switch.{direction_key}.target_pre_block": supported + 1,
                f"switch.{direction_key}.front_block_count": supported + 1,
                f"switch.{direction_key}.back_block_count": 31 - supported,
                f"switch.{direction_key}.transformer_block_count": torch.tensor(
                    [
                        int(direction_metrics[layer]["transformer_blocks"])
                        for layer in SUPPORTED_SWITCH_LAYERS
                    ],
                    dtype=torch.int64,
                ),
            }
        )
        for name in (
            "arc_easy_accuracy_norm",
            "sciq_accuracy_norm",
            "mean_qa_accuracy_norm",
            "kv_reduction_fraction",
        ):
            tensors[f"{prefix}.{name}"] = torch.tensor(
                [direction_metrics[layer][name] for layer in SUPPORTED_SWITCH_LAYERS],
                dtype=torch.float64,
            )

    checkpoint_hash = sha256(checkpoint)
    save_file(
        tensors,
        output_dir / MODEL_FILE,
        metadata={
            "format": "pt",
            "format_version": "4",
            "architecture": "NinaXanderChimeraForCausalLM",
            "contents": (
                "adapter weights, standardization tensors, and a "
                "direction-locked switch plan"
            ),
            "source_checkpoint_sha256": checkpoint_hash,
            "source_checkpoint_step": str(int(blob["step"])),
            "default_switch_layer": str(best_layer),
            "default_direction": best_direction,
            "display_direction": DIRECTION_DISPLAY_NAMES[best_direction],
        },
    )

    materialize_assets(
        model_assets(source_dir),
        output_dir / "source",
        overwrite=args.overwrite,
        copy_assets=copy_assets,
    )
    materialize_assets(
        model_assets(target_dir),
        output_dir / "target",
        overwrite=args.overwrite,
        copy_assets=copy_assets,
    )
    tokenizer_sources = tokenizer_assets(tokenizer_dir)
    legacy_tokenizer_dir = output_dir / "tokenizer"
    if legacy_tokenizer_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Legacy tokenizer directory exists: {legacy_tokenizer_dir}; pass --overwrite"
            )
        shutil.rmtree(legacy_tokenizer_dir)
    materialize_assets(
        tokenizer_sources,
        output_dir,
        overwrite=args.overwrite,
        copy_assets=copy_assets,
        clean_destination=False,
    )
    bundled_tokenizer_paths = [output_dir / path.name for path in tokenizer_sources]
    evaluation_sources = [
        execution_path_table,
        cross_family_qa_table,
        cross_family_memory_table,
    ]
    materialize_assets(
        evaluation_sources,
        output_dir / "evaluation",
        overwrite=args.overwrite,
        copy_assets=True,
    )

    direction_specs = {
        "A_to_B": {
            "model_name_template": MODEL_NAME_TEMPLATES["A_to_B"].replace(
                "{layer}", "{switch_layer}"
            ),
            "front": "RWKV blocks 0..L",
            "source_boundary": "post-RWKV block L residual",
            "translation": "A-to-B with the layer-L standardization tensors",
            "target_boundary": "pre-GPT-NeoX block L+1 residual",
            "back": "GPT-NeoX blocks L+1..31",
        },
        "B_to_A": {
            "model_name_template": MODEL_NAME_TEMPLATES["B_to_A"].replace(
                "{layer}", "{switch_layer}"
            ),
            "front": "GPT-NeoX blocks 0..L",
            "source_boundary": "post-GPT-NeoX block L residual",
            "translation": "B-to-A with the layer-L standardization tensors",
            "target_boundary": "pre-RWKV block L+1 residual",
            "back": "RWKV blocks L+1..31",
        },
    }
    selected_spec = direction_specs[best_direction]
    result_tables = {
        "execution_path_qa": f"evaluation/{execution_path_table.name}",
        "cross_family": {
            "qa": f"evaluation/{cross_family_qa_table.name}",
            "memory": f"evaluation/{cross_family_memory_table.name}",
        },
    }

    config = {
        "format_version": 4,
        "model_type": "ninaxander_chimera",
        "architectures": ["NinaXanderChimeraForCausalLM"],
        "supported_execution_paths": [
            "A_to_A",
            "A_to_B",
            "B_to_B",
            "B_to_A",
        ],
        "execution_paths": {
            "A_to_A": {
                "model_name_template": MODEL_NAME_TEMPLATES["A_to_A"],
                "role": "same_family_control",
                "runtime_causal_lm": False,
            },
            "A_to_B": {
                "model_name_template": MODEL_NAME_TEMPLATES["A_to_B"],
                "role": "cross_family_chimera",
                "runtime_causal_lm": best_direction == "A_to_B",
            },
            "B_to_B": {
                "model_name_template": MODEL_NAME_TEMPLATES["B_to_B"],
                "role": "same_family_control",
                "runtime_causal_lm": False,
            },
            "B_to_A": {
                "model_name_template": MODEL_NAME_TEMPLATES["B_to_A"],
                "role": "cross_family_chimera",
                "runtime_causal_lm": best_direction == "B_to_A",
            },
        },
        "auto_map": {
            "AutoConfig": "configuration_ninaxander.NinaXanderChimeraConfig",
            "AutoModelForCausalLM": (
                "modeling_ninaxander_chimera.NinaXanderChimeraForCausalLM"
            ),
        },
        "library_name": "transformers",
        "dtype": "float16",
        "vocab_size": 50278,
        "hidden_size": int(cfg["dB"]),
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "tie_word_embeddings": False,
        "is_encoder_decoder": False,
        "source_subdir": "source",
        "target_subdir": "target",
        "tokenizer_subdir": ".",
        "switch_tensor_file": MODEL_FILE,
        "release_variant": {
            "type": "direction_best",
            "direction_id": best_direction,
            "direction_slug": args.direction,
            "direction_display_name": DIRECTION_DISPLAY_NAMES[best_direction],
            "direction_locked": True,
            "default_switch_layer": best_layer,
            "boundary_switching_supported": True,
        },
        "adapter": {
            "architecture": "resnet",
            "source_hidden_size": int(cfg["dA"]),
            "target_hidden_size": int(cfg["dB"]),
            "latent_size": int(cfg["z"]),
            "residual_blocks": blocks,
            "block_hidden_size": hidden,
            "latent_norm": {
                "type": "layer_norm",
                "elementwise_affine": False,
            },
            "layer_mapping": [
                {"source_block": layer, "target_block": layer} for layer in layers
            ],
            "weight_dtype": "float32",
            "statistics_dtype": "float32",
        },
        "switch": {
            "default_direction": best_direction,
            "default_model_name": MODEL_NAME_TEMPLATES[best_direction].format(
                layer=best_layer
            ),
            "supported_directions": list(directions),
            "supported_model_name_templates": {
                direction: direction_specs[direction]["model_name_template"]
                for direction in directions
            },
            "default_layer": best_layer,
            "supported_layers": list(SUPPORTED_SWITCH_LAYERS),
            "source_boundary": selected_spec["source_boundary"],
            "translation": selected_spec["translation"],
            "target_boundary": selected_spec["target_boundary"],
            "transformer_blocks_retained": {
                str(layer): int(metrics[layer]["transformer_blocks"])
                for layer in SUPPORTED_SWITCH_LAYERS
            },
            "directions": {
                direction: direction_specs[direction] for direction in directions
            },
            "parent_reproduction_threshold": 1e-4,
            "runtime_activation_is_dynamic": True,
            "direction_locked": True,
        },
        "selection": {
            "checkpoint_rule": "maximum held-out A-to-B R2",
            "checkpoint_step": int(blob["step"]),
            "direction_and_switch_rule": (
                "within the fixed publication direction, maximum mean "
                "normalized accuracy across ARC-Easy and SciQ over the four "
                "measured switch layers; shallower layer wins exact ties"
            ),
            "direction_and_switch_selection_scope": (
                "post-hoc boundary selection on the two reported test sets "
                "across four switch layers in one fixed direction; no "
                "independent deployment-selection split was used"
            ),
            "direction": best_direction,
            "direction_slug": args.direction,
            "direction_display_name": DIRECTION_DISPLAY_NAMES[best_direction],
            "switch_layer": best_layer,
            "evaluated_directions": {
                best_direction: {
                    str(layer): metrics[layer]
                    for layer in SUPPORTED_SWITCH_LAYERS
                }
            },
            "other_direction_evidence_is_bundled_but_not_runtime_selectable": True,
            "result_tables": result_tables,
        },
        "parents": {
            "source": {
                "model_id": args.source_model,
                "local_subdir": "source",
                "role": "RWKV parent A; front in A-to-B, back in B-to-A",
                "bundled": True,
                "revision": None,
                "provenance_status": (
                    "exact local converted fp16 tensors are bundled; the original "
                    "BlinkDL source filename/revision remains unrecorded"
                ),
            },
            "target": {
                "model_id": args.target_model,
                "local_subdir": "target",
                "role": "GPT-NeoX/Tulu parent B; back in A-to-B, front in B-to-A",
                "bundled": True,
                "revision": None,
                "provenance_status": (
                    "exact local fp16 tensors are bundled; the upstream revision "
                    "used for the local resave remains unrecorded"
                ),
            },
        },
        "tokenizer": {
            "model_id": args.tokenizer,
            "local_subdir": ".",
            "bundled": True,
            "revision": "c292233c833e336628618a88a648727eb3dff0a7",
            "requirement": "Both parents receive the identical token-id sequence.",
        },
        "generation_contract": {
            "decoding": "greedy_argmax",
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 40,
            "early_stop": "eos_token_id",
            "seed": args.seed,
        },
        "training": {
            "corpus": "tatsu-lab/alpaca",
            "corpus_split": "train",
            "text_field": "text",
            "tokens_per_layer": int(cfg["N"]),
            "window_tokens": 112,
            "held_out_character_offset": 2_000_000,
            "held_out_windows": 1_000,
            "noise_sigma": float(cfg["sigma"]),
            "alignment_lambda": float(cfg["lam"]),
            "seed": args.seed,
            "step": int(blob["step"]),
            "parent_parameters_frozen": True,
        },
        "representation_evaluation": {
            key: float(blob["eval"][key])
            for key in REPRESENTATION_EVALUATION_ORDER
        },
        "source_checkpoint": {
            "filename": checkpoint.name,
            "size_bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_hash,
            "excluded_training_state": ["optimizer", "scheduler"],
        },
        "provenance_complete": False,
        "provenance_note": (
            "All inference tensors are bundled and fingerprinted. Publication still "
            "requires recovering the original Raven conversion provenance."
        ),
    }
    json_dump(output_dir / CONFIG_FILE, config)
    json_dump(
        output_dir / GENERATION_CONFIG_FILE,
        {
            "_from_model_config": True,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "pad_token_id": 0,
            "do_sample": False,
            "max_new_tokens": 40,
            "transformers_version": "5.2.0",
        },
    )
    json_dump(
        output_dir / PARENT_FINGERPRINTS_FILE,
        {
            "description": (
                "Hashes of every exact local parent-model artifact bundled with "
                "this composite release. No absolute paths are retained."
            ),
            "source": fingerprint_tree(output_dir / "source"),
            "target": fingerprint_tree(output_dir / "target"),
            "tokenizer": fingerprint_files(bundled_tokenizer_paths, subdir="."),
        },
    )
    write_checksums(output_dir)

    print(f"Best checkpoint: step {int(blob['step'])}, SHA-256 {checkpoint_hash}")
    print(f"Best evaluated direction/switch: {best_direction} L={best_layer}")
    print(
        f"Wrote {len(tensors)} tensors to {output_dir / MODEL_FILE} "
        f"({(output_dir / MODEL_FILE).stat().st_size / 2**30:.2f} GiB)"
    )
    print(
        f"Bundled source/target/tokenizer assets under {output_dir} "
        f"({'independent copies' if copy_assets else 'hard links where supported'})"
    )


if __name__ == "__main__":
    main()
