#!/usr/bin/env python3
"""Rebuild integrity manifests for the three public Hugging Face packages."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGES = (
    ROOT / "release" / "huggingface" / "ninaxander-raven7b-tulu69-adapter",
    ROOT / "release" / "huggingface" / "ninaxander-tulu69-to-raven7b-best",
    ROOT / "release" / "huggingface" / "ninaxander-raven7b-to-tulu69-best",
)
MANIFEST = "SHA256SUMS"
TEMP_MANIFEST = ".SHA256SUMS.tmp"


def file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def sha256(
    path: Path,
    cache: dict[tuple[int, int, int, int], str],
    chunk_size: int = 16 * 1024 * 1024,
) -> str:
    identity = file_identity(path)
    if identity not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        cache[identity] = digest.hexdigest()
    return cache[identity]


def package_files(package: Path) -> list[Path]:
    return sorted(
        path
        for path in package.rglob("*")
        if path.is_file()
        and path.name not in {MANIFEST, TEMP_MANIFEST}
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def update_package(
    package: Path,
    cache: dict[tuple[int, int, int, int], str],
) -> None:
    package = package.resolve()
    if not package.is_dir():
        raise NotADirectoryError(package)
    if any(path.is_symlink() for path in package.rglob("*")):
        raise ValueError(f"Symbolic links are not permitted in {package}")
    lines = [
        f"{sha256(path, cache)}  {path.relative_to(package).as_posix()}"
        for path in package_files(package)
    ]
    temporary = package / TEMP_MANIFEST
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(package / MANIFEST)
    print(f"{package.name}: {len(lines)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "packages",
        nargs="*",
        type=Path,
        default=list(DEFAULT_PACKAGES),
        help="Package directories; defaults to the three public releases.",
    )
    args = parser.parse_args()
    cache: dict[tuple[int, int, int, int], str] = {}
    for package in args.packages:
        update_package(package, cache)


if __name__ == "__main__":
    main()
