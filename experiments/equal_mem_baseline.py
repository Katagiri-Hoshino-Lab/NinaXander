"""Equal-memory accuracy baseline.

At the same Transformer depth (= same KV budget) as chimera@L, is a plain
shallow Transformer as good? We run EARLY-EXIT Pythia -- embed_in + first K
blocks + final_layer_norm + head -- with K=31-L for RWKV-front/Pythia-back and
K=L+1 for Pythia-front/RWKV-back. This has the same number of Transformer
blocks as the corresponding chimera but no RWKV and no translation. Full test
sets, temperature 0. Read-only, pure Pythia (no RWKV, no adapter).

The per-item correctness is aligned by construction with the path-matched
chimera dumps (same loaders, order, and n=0), so we use McNemar plus paired
bootstrap on the same items. For A-to-B this tests whether the RWKV prefix adds
value over the matched Transformer suffix depth; for B-to-A it tests whether
the RWKV suffix adds value over a Pythia early exit with the same Transformer
KV depth.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from ninaxander.result_io import write_bundle  # noqa: E402
from ninaxander.paired import (  # noqa: E402
    mcnemar_exact,
    paired_bootstrap_ci as boot_ci,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pythia", default="models/tulu-pythia69-fp16")
    ap.add_argument("--tok", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--task", default="arc_easy", choices=["arc_easy", "sciq"])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--switches", default="4,8,16,24")
    ap.add_argument(
        "--direction",
        choices=["A_to_B", "B_to_A"],
        default="A_to_B",
        help="A_to_B matches the Transformer suffix (31-L); B_to_A matches the prefix (L+1)",
    )
    ap.add_argument(
        "--chimera_dump",
        default="",
        help="{a,b}_to_{a,b}_qa_items_{task}.json for paired comparison",
    )
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    if not a.dump:
        task_name = "arc" if a.task == "arc_easy" else a.task
        prefix = "a_to_b_exit_items" if a.direction == "A_to_B" else "b_to_a_exit_items"
        a.dump = f"artifacts/metrics/raw/{prefix}_{task_name}.json"
    dev = a.dev
    torch.manual_seed(0)
    SW = [int(x) for x in a.switches.split(",")]
    Ks = {L: (31 - L if a.direction == "A_to_B" else L + 1) for L in SW}
    # Both comparisons use the same kind of baseline: a truncated B=Pythia
    # parent. The producer path remains separate metadata; it is not part of
    # the baseline model's name.
    exit_prefix = "B_early_exit"
    chimera_prefix = a.direction

    tok = AutoTokenizer.from_pretrained(a.tok)
    B = (
        AutoModelForCausalLM.from_pretrained(
            a.pythia, dtype=torch.float16, low_cpu_mem_usage=True
        )
        .to(dev)
        .eval()
    )
    for p in B.parameters():
        p.requires_grad_(False)
    full_layers = list(B.gpt_neox.layers)  # keep the originals to restore between K's

    from datasets import load_dataset as _ld

    items = []
    if a.task == "arc_easy":
        d = _ld("allenai/ai2_arc", "ARC-Easy", split="test")
        for r in d:
            opts, labs = r["choices"]["text"], r["choices"]["label"]
            if r["answerKey"] not in labs:
                continue
            items.append(
                (
                    f"Question: {r['question']}\nAnswer:",
                    [" " + o for o in opts],
                    labs.index(r["answerKey"]),
                )
            )
    else:
        d = _ld("allenai/sciq", split="test")
        for r in d:
            opts = [
                r["correct_answer"],
                r["distractor1"],
                r["distractor2"],
                r["distractor3"],
            ]
            items.append(
                (f"Question: {r['question']}\nAnswer:", [" " + o for o in opts], 0)
            )
    if a.n > 0:
        items = items[: a.n]
    N = len(items)
    print(
        f"task={a.task} items={N} (full set); early-exit K={sorted(set(Ks.values()))}",
        flush=True,
    )

    @torch.no_grad()
    def opt_logprob(logits, ids, p0):
        lp = torch.log_softmax(logits[0, :-1].float(), -1)
        tgt = ids[0, 1:].to(lp.device)
        sel = lp[torch.arange(len(tgt), device=lp.device), tgt][p0 - 1 :]
        return sel.mean().item()

    cfgs = [f"{exit_prefix}@{Ks[L]}" for L in SW]
    hit = {k: 0 for k in cfgs}
    per = {k: [] for k in cfgs}
    for L in SW:
        K = Ks[L]
        B.gpt_neox.layers = nn.ModuleList(
            full_layers[:K]
        )  # early exit: run first K blocks then final_ln + head
        key = f"{exit_prefix}@{K}"
        h = 0
        for prompt, opts, gold in items:
            p0 = tok(prompt, return_tensors="pt").input_ids.shape[1]
            sc = []
            for o in opts:
                ids = tok(prompt + o, return_tensors="pt").input_ids.to(dev)
                with torch.no_grad():
                    sc.append(
                        opt_logprob(
                            B(ids, use_cache=False).logits,
                            ids,
                            p0,
                        )
                    )
            cn = int(max(range(len(opts)), key=lambda i: sc[i]) == gold)
            hit[key] += cn
            per[key].append(cn)
            h += cn
        formula = f"31-{L}" if a.direction == "A_to_B" else f"{L}+1"
        print(
            f"  {exit_prefix}@{K} (={formula}): acc_norm {100 * hit[key] / N:.1f}",
            flush=True,
        )
    B.gpt_neox.layers = nn.ModuleList(full_layers)  # restore

    print(f"\n== {a.task} N={N} early-exit Pythia acc_norm ==")
    for k in cfgs:
        print(f"  {k:9s} {100 * hit[k] / N:6.1f}")

    # Paired comparison vs the direction-matched chimera at equal KV.
    if a.chimera_dump and os.path.exists(a.chimera_dump):
        with open(a.chimera_dump) as f:
            CD = json.load(f)
        if CD["N"] != N:
            print(
                f"\n[skip paired test: chimera dump N={CD['N']} != {N} (subset run); need matching full-set dump]",
                flush=True,
            )
            CD = None
    else:
        CD = None
    if CD is not None:
        print(
            f"\n== paired: early-exit Pythia vs chimera at EQUAL KV (same {N} items) =="
        )
        comparisons = []
        for L in SW:
            K = Ks[L]
            ex = per[f"{exit_prefix}@{K}"]
            ch = CD["per_item_norm"].get(f"{chimera_prefix}@{L}")
            if ch is None:
                continue
            b, cc, p = mcnemar_exact(ex, ch)
            lo, hi = boot_ci(ex, ch)
            acc_ex = sum(ex) / N
            acc_ch = sum(ch) / N
            star = "*" if p < 0.05 else " "
            print(
                f"  {exit_prefix}@{K} vs {chimera_prefix}@{L}: {100 * acc_ex:5.1f} vs {100 * acc_ch:5.1f}  dacc {100 * (acc_ex - acc_ch):+5.1f} "
                f"95%CI[{100 * lo:+5.1f},{100 * hi:+5.1f}] McNemar b={b} c={cc} p={p:.3g} {star}",
                flush=True,
            )
            comparisons.append(
                {
                    "task": a.task,
                    "switch_layer": L,
                    "transformer_blocks": K,
                    "direction": a.direction,
                    "config_a": f"{exit_prefix}@{K}",
                    "config_b": f"{chimera_prefix}@{L}",
                    "accuracy_a": acc_ex,
                    "accuracy_b": acc_ch,
                    "accuracy_difference": acc_ex - acc_ch,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "mcnemar_b": b,
                    "mcnemar_c": cc,
                    "mcnemar_p": p,
                    "significant_0_05": p < 0.05,
                }
            )
    else:
        comparisons = []

    if a.dump:
        payload = {
            "schema_version": 1,
            "direction": a.direction,
            "task": a.task,
            "N": N,
            "cfgs": cfgs,
            "Ks": {str(L): Ks[L] for L in SW},
            "per_item_norm": per,
            "acc_norm": {k: hit[k] / N for k in cfgs},
            "paired_equal_memory": comparisons,
        }
        rows = [
            {
                "result_type": "accuracy",
                "task": a.task,
                "switch_layer": layer,
                "transformer_blocks": Ks[layer],
                "direction": payload["direction"],
                "config": f"{exit_prefix}@{Ks[layer]}",
                "accuracy_norm": hit[f"{exit_prefix}@{Ks[layer]}"] / N,
                "n_items": N,
            }
            for layer in SW
        ]
        rows.extend({"result_type": "paired_comparison", **row} for row in comparisons)
        json_path, csv_path = write_bundle(a.dump, payload, rows)
        print(f"wrote {json_path} + {csv_path}", flush=True)
    print("EQUAL_MEM_DONE", flush=True)


if __name__ == "__main__":
    main()
