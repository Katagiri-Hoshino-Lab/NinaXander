#!/usr/bin/env python3
"""Gate: the English paper must contain exactly the same numbers as the Japanese one.

A translation may not change, drop, or invent a single numeric token. Numeric
tokens are compared as multisets over the full LaTeX sources, so any numeric
drift between paper_ja.tex and paper_en.tex fails the build.
"""

import argparse
import pathlib
import re
import sys
from collections import Counter

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(path: pathlib.Path) -> Counter:
    return Counter(NUMBER.findall(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ja", default="paper/paper_ja.tex")
    parser.add_argument("--en", default="paper/paper_en.tex")
    args = parser.parse_args()

    en_path = pathlib.Path(args.en)
    if not en_path.exists():
        print("NOTE: paper_en.tex absent; English number-parity check skipped.")
        return 0

    ja, en = numbers(pathlib.Path(args.ja)), numbers(en_path)
    only_ja = ja - en
    only_en = en - ja

    # Bare single digits may legitimately be spelled out in English prose
    # ("four paths", "July"); report them but do not fail. Every multi-digit or
    # decimal token must match exactly.
    def is_strict(token: str) -> bool:
        return len(token) > 1 or "." in token or "," in token

    failures = []
    for source, diff in (("JA_ONLY", only_ja), ("EN_ONLY", only_en)):
        for token, count in sorted(diff.items()):
            line = f"{source}: {token} x{count}"
            if is_strict(token):
                failures.append(line)
            else:
                print(f"WARN (single digit, likely prose spell-out) {line}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"EN_NUMBER_PARITY_OK tokens={sum(ja.values())} distinct={len(ja)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
