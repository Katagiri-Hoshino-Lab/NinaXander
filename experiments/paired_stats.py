"""Paired significance tests over the per-item QA dumps. CPU-only, no GPU.

Model comparisons on the SAME question set are PAIRED data, so an independent per-cell binomial SE overstates the
uncertainty of a difference. This reads the per-item path QA dumps and reports, for
each key comparison, McNemar's exact test (discordant pairs b, c) and a paired bootstrap 95% CI on the accuracy gap.

Comparisons (per task):
  A_to_B and B_to_A: every measured switch vs each parent
                     -- avoid per-task best-switch selection
  cross-family path vs path-matched affine control
  parent vs same-family path                       -- decoder-reconstruction loss
  same-family vs cross-family path                 -- substitution term
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ninaxander.paired import (  # noqa: E402
    mcnemar_exact,
    paired_bootstrap_ci as boot_ci,
)
from ninaxander.result_io import write_csv  # noqa: E402


def report(task, D):
    cfgs = D["cfgs"]
    per = D["per_item_norm"]
    acc = D["acc_norm"]
    rows = []

    def has(key):
        return key in per

    direction = D.get("direction", "A_to_B")
    is_b_to_a = direction == "B_to_A"
    adapter_prefix = direction
    linear_prefix = f"{direction}_affine"
    SW = sorted(
        int(k.split("@")[1]) for k in cfgs if k.startswith(f"{adapter_prefix}@")
    )
    print(f"\n===== {task} direction={direction} (N={D['N']}, acc_norm) =====")
    parent = "  ".join(f"{k} {100 * acc[k]:.1f}" for k in ("pythia", "rwkv") if has(k))
    if parent:
        print(f"parents: {parent}")

    def line(name, k1, k2):
        if not (has(k1) and has(k2)):
            return  # skip comparisons whose arms aren't in this dump
        b, c, p = mcnemar_exact(per[k1], per[k2])
        lo, hi = boot_ci(per[k1], per[k2])
        star = "*" if p < 0.05 else " "
        print(
            f"  {name:22s} {k1:8s}-{k2:8s}: dacc {100 * (acc[k1] - acc[k2]):+5.1f}  "
            f"95%CI[{100 * lo:+5.1f},{100 * hi:+5.1f}]  McNemar b={b} c={c} p={p:.3g} {star}"
        )
        rows.append(
            {
                "direction": direction,
                "task": task,
                "comparison": name,
                "config_a": k1,
                "config_b": k2,
                "n_items": D["N"],
                "accuracy_a": acc[k1],
                "accuracy_b": acc[k2],
                "accuracy_difference": acc[k1] - acc[k2],
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "mcnemar_b": b,
                "mcnemar_c": c,
                "mcnemar_p": p,
                "significant_0_05": p < 0.05,
            }
        )

    for L in SW:
        line(
            f"chimera vs Pythia @{L}",
            f"{adapter_prefix}@{L}",
            "pythia",
        )
        line(
            f"chimera vs RWKV @{L}",
            f"{adapter_prefix}@{L}",
            "rwkv",
        )
    for L in SW:
        line(
            f"adapter vs affine @{L}",
            f"{adapter_prefix}@{L}",
            f"{linear_prefix}@{L}",
        )
        if has(f"{direction}_alpha0@{L}"):
            line(
                f"MLP-branch @{L}",
                f"{direction}@{L}",
                f"{direction}_alpha0@{L}",
            )
    if is_b_to_a:
        for L in SW:
            line(f"decoder loss @{L}", "rwkv", f"A_to_A@{L}")
        for L in SW:
            line(f"cross-family @{L}", f"A_to_A@{L}", f"B_to_A@{L}")
    else:
        for L in SW:
            line(f"decoder loss @{L}", "pythia", f"B_to_B@{L}")
        for L in SW:
            line(f"cross-family @{L}", f"B_to_B@{L}", f"A_to_B@{L}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dumps",
        nargs="+",
        required=True,
        help="per-item A-to-B or B-to-A QA dump JSON files",
    )
    ap.add_argument(
        "--out",
        default="artifacts/metrics/raw/a_to_b_paired_statistics.csv",
    )
    a = ap.parse_args()
    rows = []
    for path in a.dumps:
        with open(path) as f:
            D = json.load(f)
        rows.extend(report(D["task"], D))
    output = write_csv(a.out, rows)
    print(f"\nwrote {output}\nPAIRED_STATS_DONE  (* = McNemar p<0.05)")


if __name__ == "__main__":
    main()
