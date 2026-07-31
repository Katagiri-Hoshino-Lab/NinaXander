#!/usr/bin/env python3
"""Verify that every project Markdown document has en/ja/zh-CN variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


LANGUAGE_SUFFIXES = (".ja.md", ".zh-CN.md")
LANGUAGE_DIRECTORIES = ("ja", "zh-CN")
IGNORED_DIRECTORIES = {".git", ".cache", "__pycache__"}
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]*]\(([^)\s]+)")
REMOTE_PREFIXES = ("http://", "https://", "mailto:", "#")


def english_path(path: Path) -> Path:
    if path.parent.name in LANGUAGE_DIRECTORIES:
        return path.parent.parent / path.name
    name = path.name
    for suffix in LANGUAGE_SUFFIXES:
        if name.endswith(suffix):
            return path.with_name(f"{name.removesuffix(suffix)}.md")
    return path


def variants(base: Path) -> tuple[Path, Path, Path]:
    directory_ja = base.parent / "ja" / base.name
    directory_zh = base.parent / "zh-CN" / base.name
    if directory_ja.exists() or directory_zh.exists():
        return base, directory_ja, directory_zh
    stem = base.name.removesuffix(".md")
    return (
        base,
        base.with_name(f"{stem}.ja.md"),
        base.with_name(f"{stem}.zh-CN.md"),
    )


def markdown_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if not any(part in IGNORED_DIRECTORIES for part in path.relative_to(root).parts)
    )


def structure_signature(text: str) -> tuple[tuple[int, ...], int, int]:
    heading_levels = tuple(
        len(match.group(1))
        for match in re.finditer(r"^(#{1,6})\s", text, flags=re.MULTILINE)
    )
    fence_count = len(re.findall(r"^```", text, flags=re.MULTILINE))
    table_count = len(
        re.findall(r"^\|(?:[-: ]+\|)+$", text, flags=re.MULTILINE)
    )
    return heading_levels, fence_count, table_count


def validate_local_links(path: Path, root: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for raw_target in LINK_PATTERN.findall(text):
        target = raw_target.strip("<>").split("#", maxsplit=1)[0]
        if not target or raw_target.startswith(REMOTE_PREFIXES):
            continue
        if not (path.parent / target).resolve().exists():
            errors.append(
                f"BROKEN_LOCAL_LINK: {path.relative_to(root)} -> {target}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    files = markdown_files(root)
    families = sorted({english_path(path) for path in files})
    errors: list[str] = []

    for base in families:
        english, japanese, chinese = variants(base)
        expected = (english, japanese, chinese)
        for path in expected:
            if not path.is_file():
                errors.append(f"MISSING_TRANSLATION: {path.relative_to(root)}")
                continue
            if not path.read_text(encoding="utf-8").strip():
                errors.append(f"EMPTY_TRANSLATION: {path.relative_to(root)}")

        if not all(path.is_file() for path in expected):
            continue

        signatures = {
            path: structure_signature(path.read_text(encoding="utf-8"))
            for path in expected
        }
        for path in (japanese, chinese):
            if signatures[path] != signatures[english]:
                errors.append(
                    "STRUCTURE_MISMATCH: "
                    f"{path.relative_to(root)} != {english.relative_to(root)}"
                )

        def relative_target(target: Path, source: Path) -> str:
            import os

            return os.path.relpath(target, source.parent).replace(os.sep, "/")

        link_requirements = {
            english: (relative_target(japanese, english), relative_target(chinese, english)),
            japanese: (relative_target(english, japanese), relative_target(chinese, japanese)),
            chinese: (relative_target(english, chinese), relative_target(japanese, chinese)),
        }
        for path, targets in link_requirements.items():
            text = path.read_text(encoding="utf-8")
            for target in targets:
                if f"]({target})" not in text:
                    errors.append(
                        f"MISSING_LANGUAGE_LINK: {path.relative_to(root)} -> {target}"
                    )

    for path in files:
        validate_local_links(path, root, errors)

    if errors:
        print("\n".join(errors))
        return 1

    print(
        "MARKDOWN_LANGUAGES_OK "
        f"families={len(families)} files={len(files)} languages=en,ja,zh-CN"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
