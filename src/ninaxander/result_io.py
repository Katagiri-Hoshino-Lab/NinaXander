"""Atomic JSON/CSV writers shared by all experiments."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


Scalar = str | int | float | bool | None


def _atomic_replace(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)

    def writer(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    _atomic_replace(destination, writer)
    return destination


def _cell(value: Any) -> Scalar:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    destination = Path(path)
    materialized = [{key: _cell(value) for key, value in row.items()} for row in rows]
    if fieldnames is None:
        ordered: list[str] = []
        for row in materialized:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    if not fieldnames:
        raise ValueError(f"Cannot write a CSV with no columns: {destination}")

    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            csv_writer.writeheader()
            csv_writer.writerows(materialized)

    _atomic_replace(destination, writer)
    return destination


def csv_sidecar(json_path: str | Path) -> Path:
    return Path(json_path).with_suffix(".csv")


def write_bundle(
    json_path: str | Path,
    payload: Any,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    json_destination = write_json(json_path, payload)
    csv_destination = write_csv(csv_sidecar(json_destination), rows, fieldnames)
    return json_destination, csv_destination


def append_csv_row(
    path: str | Path,
    row: Mapping[str, Any],
    fieldnames: Sequence[str],
) -> Path:
    """Append one durable training-event row, creating a header when needed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exists = destination.exists() and destination.stat().st_size > 0
    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow({key: _cell(row.get(key)) for key in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    return destination
