#!/usr/bin/env python3
"""Quantify exposure of the reported eval sets inside Tulu's instruction mixture.

Tulu-Pythia is instruction-tuned on a subsample of FLAN V2, whose task inventory
lists ARC and SciQ. This scan reconstructs the evaluation order used by the QA
benchmarks, searches the released mixture for verbatim eval questions, and
recomputes every configuration's accuracy with the matched items removed, using
the per-item correctness dumps. No GPU and no model is needed: the per-item
dumps already record which items each configuration answered correctly.

Writes artifacts/metrics/raw/eval_leakage.json.
"""

import argparse
import json
import os
import pathlib
import re
import sys
import unicodedata

KEEP_COMPONENTS = {"flan_v2", "cot"}
MIN_WORDS = 6  # shorter questions collide by chance


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).lower()).strip()


def eval_questions():
    """Reproduce the item order of the QA benchmarks exactly."""
    from datasets import load_dataset

    arc = [
        r["question"]
        for r in load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        if r["answerKey"] in r["choices"]["label"]
    ]
    sciq = [r["question"] for r in load_dataset("allenai/sciq", split="test")]
    return {"arc_easy": arc, "sciq": sciq}


def scan(questions):
    from datasets import load_dataset

    chunks = []
    for row in load_dataset("allenai/tulu-v1-sft-mixture", split="train", streaming=True):
        if row.get("dataset") in KEEP_COMPONENTS:
            chunks.append(norm(" ".join(m.get("content", "") for m in row["messages"])))
    corpus = "\n".join(chunks)
    hits = {}
    for task, qs in questions.items():
        hits[task] = [
            i
            for i, q in enumerate(qs)
            if len(q.split()) >= MIN_WORDS and corpus.find(norm(q)) != -1
        ]
    return hits, len(chunks), len(corpus)


def recompute(raw_dir, hits):
    """Accuracy with matched items removed, for every dumped configuration."""
    sources = [
        ("A_to_B", "a_to_b_qa_items_arc.json", "arc_easy"),
        ("A_to_B", "a_to_b_qa_items_sciq.json", "sciq"),
        ("B_to_A", "b_to_a_qa_items_arc.json", "arc_easy"),
        ("B_to_A", "b_to_a_qa_items_sciq.json", "sciq"),
    ]
    out = []
    for direction, name, task in sources:
        data = json.loads((raw_dir / name).read_text())
        dropped = set(hits[task])
        keep = [i for i in range(data["N"]) if i not in dropped]
        for cfg in data["cfgs"]:
            v = data["per_item_norm"][cfg]
            full = 100 * sum(v) / data["N"]
            excl = 100 * sum(v[i] for i in keep) / len(keep)
            out.append(
                {
                    "direction": direction,
                    "task": task,
                    "config": cfg,
                    "accuracy_norm_full": full,
                    "accuracy_norm_excluding_matched": excl,
                    "delta": excl - full,
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="artifacts/metrics/raw")
    ap.add_argument("--out", default="artifacts/metrics/raw/eval_leakage.json")
    a = ap.parse_args()
    raw_dir = pathlib.Path(a.raw_dir)

    questions = eval_questions()
    hits, rows_kept, corpus_chars = scan(questions)
    recomputed = recompute(raw_dir, hits)

    # Exposure cannot be blamed for a difficulty gap that the UNEXPOSED parent
    # shows too, so record both parents on the matched subset.
    contrast = []
    for name, task in [("a_to_b_qa_items_arc.json", "arc_easy"), ("a_to_b_qa_items_sciq.json", "sciq")]:
        data = json.loads((raw_dir / name).read_text())
        dropped = sorted(hits[task])
        keep = [i for i in range(data["N"]) if i not in set(dropped)]
        for cfg in ("pythia", "rwkv"):
            v = data["per_item_norm"][cfg]
            contrast.append(
                {
                    "task": task,
                    "config": cfg,
                    "matched_items": len(dropped),
                    "accuracy_on_matched": 100 * sum(v[i] for i in dropped) / len(dropped),
                    "accuracy_on_rest": 100 * sum(v[i] for i in keep) / len(keep),
                }
            )

    worst = max(abs(r["delta"]) for r in recomputed)
    payload = {
        "components_scanned": sorted(KEEP_COMPONENTS),
        "mixture_rows_scanned": rows_kept,
        "corpus_chars": corpus_chars,
        "min_question_words": MIN_WORDS,
        "matched_indices": hits,
        "matched_counts": {k: len(v) for k, v in hits.items()},
        "max_abs_delta_points": worst,
        "recomputed": recomputed,
        "parent_contrast": contrast,
    }
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"matched: {payload['matched_counts']}")
    print(f"max |delta| across {len(recomputed)} configurations: {worst:.3f} points")
    print(f"wrote {a.out}")
    sys.stdout.flush()
    os._exit(0)  # streaming iterator upsets interpreter teardown


if __name__ == "__main__":
    main()
