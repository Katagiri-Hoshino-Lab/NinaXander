#!/usr/bin/env python3
"""Export a NinaXander training checkpoint as a safe, inference-only Hub package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


MODEL_FILE = "model.safetensors"
STATS_FILE = "stats.safetensors"
CONFIG_FILE = "config.json"
PARENT_FINGERPRINTS_FILE = "parent_fingerprints.json"
CHECKSUM_FILE = "SHA256SUMS"
EVALUATION_ORDER = ("A->A", "A->B", "B->B", "B->A", "rho_ctr", "rho_raw", "f")
MODEL_NAME_TEMPLATES = {
    "A_to_A": "NinaXander-RWKV-reconstruction@{layer}",
    "A_to_B": "NinaXander-RWKV-to-Pythia@{layer}",
    "B_to_B": "NinaXander-Pythia-reconstruction@{layer}",
    "B_to_A": "NinaXander-Pythia-to-RWKV@{layer}",
}


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def infer_architecture(state: dict[str, torch.Tensor]) -> tuple[str, int, int]:
    block_ids = sorted(
        {
            int(name.split(".")[1])
            for name in state
            if name.startswith("eA.") and name.endswith(".fc1.weight")
        }
    )
    if not block_ids:
        raise ValueError("The checkpoint does not look like a ResNet LatentAdapter")
    expected = list(range(1, len(block_ids) + 1))
    if block_ids != expected:
        raise ValueError(f"Unexpected encoder block indices: {block_ids}; expected {expected}")
    hidden = int(state[f"eA.{block_ids[0]}.fc1.weight"].shape[0])
    return "resnet", len(block_ids), hidden


def flatten_stats(stats: dict[Any, Any], layers: list[int]) -> dict[str, torch.Tensor]:
    names = ("a_mean", "a_std", "b_mean", "b_std")
    flat: dict[str, torch.Tensor] = {}
    for layer in layers:
        values = stats[layer] if layer in stats else stats[str(layer)]
        if len(values) != len(names):
            raise ValueError(f"Layer {layer} has {len(values)} statistics; expected four")
        for name, value in zip(names, values, strict=True):
            flat[f"{name}.{layer:02d}"] = value.detach().cpu().contiguous()
    return flat


def fingerprint_parent(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {"available": False, "reason": "path was absent from the training checkpoint"}
    path = Path(path_value)
    if not path.is_dir():
        return {
            "available": False,
            "reason": "the recorded local directory is no longer available",
            "recorded_basename": path.name,
        }
    files = []
    for candidate in sorted(path.iterdir()):
        if candidate.is_file() and (
            candidate.name.endswith((".safetensors", ".json"))
            or candidate.name in {"LICENSE", "LICENSE.txt"}
        ):
            files.append(
                {
                    "filename": candidate.name,
                    "size_bytes": candidate.stat().st_size,
                    "sha256": sha256(candidate),
                }
            )
    return {
        "available": True,
        "recorded_basename": path.name,
        "files": files,
    }


def write_checksums(output_dir: Path) -> None:
    paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != CHECKSUM_FILE
    )
    lines = [f"{sha256(path)}  {path.name}" for path in paths]
    (output_dir / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model", default="RWKV/rwkv-raven-7b")
    parser.add_argument("--target-model", default="allenai/open-instruct-pythia-6.9b-tulu")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated export files in an existing output directory.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = [
        output_dir / MODEL_FILE,
        output_dir / STATS_FILE,
        output_dir / CONFIG_FILE,
        output_dir / PARENT_FINGERPRINTS_FILE,
        output_dir / CHECKSUM_FILE,
    ]
    existing = [path for path in generated if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite generated files: {names}; pass --overwrite")

    # This is a trusted, locally produced training checkpoint. The exported package
    # never requires pickle loading.
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    required = {"model", "stats", "cfg", "eval", "step"}
    missing = required.difference(blob)
    if missing:
        raise KeyError(f"Checkpoint is missing required fields: {sorted(missing)}")

    state = {
        name: value.detach().cpu().contiguous()
        for name, value in blob["model"].items()
    }
    cfg = blob["cfg"]
    layers = [int(layer) for layer in cfg["layers"]]
    if layers != list(range(32)):
        raise ValueError(f"Expected the current 32-layer checkpoint, got layers={layers}")
    arch, blocks, hidden = infer_architecture(state)
    stats = flatten_stats(blob["stats"], layers)

    checkpoint_hash = sha256(checkpoint)
    common_metadata = {
        "format": "pt",
        "format_version": "3",
        "architecture": "NinaXanderAdapter",
        "source_checkpoint_sha256": checkpoint_hash,
        "source_checkpoint_step": str(int(blob["step"])),
    }
    save_file(
        state,
        output_dir / MODEL_FILE,
        metadata={**common_metadata, "contents": "inference adapter weights"},
    )
    save_file(
        stats,
        output_dir / STATS_FILE,
        metadata={**common_metadata, "contents": "per-layer standardization statistics"},
    )

    config = {
        "format_version": 3,
        "model_type": "ninaxander_shared_latent_adapter",
        "architectures": ["NinaXanderAdapter"],
        "library_name": "pytorch",
        "release_variant": {
            "type": "adapter_only",
            "display_name": "NinaXander RWKV-Raven-7B ↔ Tulu-Pythia-6.9B Adapter",
            "parent_weights_bundled": False,
        },
        "supported_directions": ["A_to_B", "B_to_A"],
        "direction_display_names": {
            "A_to_B": "RWKV-to-Pythia",
            "B_to_A": "Pythia-to-RWKV",
        },
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
            },
            "A_to_B": {
                "model_name_template": MODEL_NAME_TEMPLATES["A_to_B"],
                "role": "cross_family_chimera",
            },
            "B_to_B": {
                "model_name_template": MODEL_NAME_TEMPLATES["B_to_B"],
                "role": "same_family_control",
            },
            "B_to_A": {
                "model_name_template": MODEL_NAME_TEMPLATES["B_to_A"],
                "role": "cross_family_chimera",
            },
        },
        "adapter": {
            "architecture": arch,
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
        "files": {
            "weights": MODEL_FILE,
            "statistics": STATS_FILE,
        },
        "parents": {
            "source": {
                "model_id": args.source_model,
                "revision": None,
                "role": "RWKV parent A; front in A-to-B, back in B-to-A",
                "provenance_status": (
                    "compatible Hub reference; the exact original BlinkDL .pth filename "
                    "and revision used for the local conversion were not preserved"
                ),
            },
            "target": {
                "model_id": args.target_model,
                "revision": None,
                "role": "GPT-NeoX/Tulu parent B; back in A-to-B, front in B-to-A",
                "provenance_status": (
                    "model identity is recorded; the exact downloaded Hub revision of "
                    "the local fp16 resave was not preserved"
                ),
            },
        },
        "tokenizer": {
            "model_id": args.tokenizer,
            "requirement": "Both parents must receive the identical token-id sequence.",
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
        "evaluation": {
            key: float(blob["eval"][key])
            for key in EVALUATION_ORDER
        },
        "source_checkpoint": {
            "filename": checkpoint.name,
            "size_bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_hash,
            "excluded_training_state": ["optimizer", "scheduler"],
        },
        "provenance_complete": False,
        "provenance_note": (
            "The exported tensors are fully fingerprinted, but the exact upstream "
            "revision of each locally resaved parent was not recorded during training."
        ),
    }
    json_dump(output_dir / CONFIG_FILE, config)

    parent_fingerprints = {
        "description": (
            "Hashes of the exact local fp16 parent artifacts recorded in the training "
            "checkpoint. Absolute paths are intentionally omitted."
        ),
        "source": fingerprint_parent(cfg.get("src")),
        "target": fingerprint_parent(cfg.get("tgt")),
    }
    json_dump(output_dir / PARENT_FINGERPRINTS_FILE, parent_fingerprints)
    write_checksums(output_dir)

    model_bytes = (output_dir / MODEL_FILE).stat().st_size
    stats_bytes = (output_dir / STATS_FILE).stat().st_size
    print(f"Exported {len(state)} tensors to {output_dir / MODEL_FILE} ({model_bytes / 2**20:.2f} MiB)")
    print(f"Exported {len(stats)} tensors to {output_dir / STATS_FILE} ({stats_bytes / 2**20:.2f} MiB)")
    print(f"Source checkpoint SHA-256: {checkpoint_hash}")
    print(f"Wrote {CONFIG_FILE}, {PARENT_FINGERPRINTS_FILE}, and {CHECKSUM_FILE}")


if __name__ == "__main__":
    main()
