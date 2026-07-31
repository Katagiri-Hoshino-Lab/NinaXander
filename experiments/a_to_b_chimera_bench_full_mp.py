"""Full-set QA for the A-to-B RWKV-prefix -> adapter -> Pythia-suffix chimera.

This is the direction-symmetric counterpart of
``b_to_a_chimera_bench_full_mp.py``. For every switch L it evaluates:

  * FULL evaluation sets by default (ARC-Easy test 2376, SciQ test 1000), not a 250-item subset.
  * ``A_to_B@L``: the complete trained adapter path;
  * ``A_to_B_alpha0@L``: the same path with every learned residual-MLP
    branch scaled to zero, a same-adapter causal intervention;
  * ``A_to_B_affine@L``: a separately fitted affine standardized A-to-B map;
  * A B->B control arm (B_to_B@L): translate B's own layer-L residual through D_B(E_B(std_B(r_B))) and
    inject it back. This runs NO cross-family step, so its gap below pure Pythia isolates the DECODER RECONSTRUCTION
    loss (B->B R^2=0.71) from the cross-family mismatch. The resulting decomposition is
        pythia = oracle >= B_to_B@L (decoder loss only) >= A_to_B@L (decoder loss + cross-family).
    (oracle = injecting the true r_B^L is identical to pure Pythia by the parent-repro gate, rel<1e-4, so it is not
    re-run as a separate arm; it is the pythia row.)
  * A PER-ITEM dump (--dump) of acc_norm correctness for every config, so paired tests (McNemar, paired bootstrap)
    can be computed downstream (paired_stats.py) instead of comparing independent binomial SEs on shared items.

The JSON contains item-aligned correctness for paired tests. A result is never
emitted unless injecting the true post-L Pythia residual reproduces pure
Pythia at every requested switch.
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from ninaxander import adapter as adapter_module
from ninaxander.adapter import LatentAdapter
from ninaxander.result_io import write_bundle
from transformers import AutoModelForCausalLM, AutoTokenizer


_ALPHA = {"value": 1.0}


def _scaled_resblock_forward(module, value):
    return value + _ALPHA["value"] * module.fc2(F.gelu(module.fc1(module.norm(value))))


adapter_module.ResBlock.forward = _scaled_resblock_forward


def rel(a, b):
    return (a - b).float().norm().item() / (b.float().norm().item() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter", default="artifacts/checkpoints/z4096_L32/SNAP_L32_final.pt"
    )
    ap.add_argument("--rwkv", default="models/rwkv-raven-7b-hf-fp16")
    ap.add_argument("--pythia", default="models/tulu-pythia69-fp16")
    ap.add_argument("--tok", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--hid", type=int, default=4096)
    ap.add_argument("--arch", default="resnet")
    ap.add_argument("--task", default="arc_easy", choices=["arc_easy", "sciq"])
    ap.add_argument("--n", type=int, default=0, help="0 = full test set")
    ap.add_argument("--switches", default="4,8,16,24")
    ap.add_argument("--dev_a", default="cuda:0")
    ap.add_argument("--dev_b", default="cuda:1")
    ap.add_argument("--fit_rows", type=int, default=40000)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    if not a.dump:
        task_name = "arc" if a.task == "arc_easy" else a.task
        a.dump = f"artifacts/metrics/raw/a_to_b_qa_items_{task_name}.json"
    dA, dB = a.dev_a, a.dev_b
    torch.manual_seed(0)
    SW = [int(x) for x in a.switches.split(",")]

    c = torch.load(a.adapter, map_location="cpu", weights_only=False)
    cfg = c["cfg"]
    m = (
        LatentAdapter(
            cfg["dA"], cfg["dB"], cfg["z"], blocks=a.blocks, hidden=a.hid, arch=a.arch
        )
        .to(dB)
        .eval()
    )
    m.load_state_dict(c["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    st = {int(j): [t.to(dB) for t in v] for j, v in c["stats"].items()}
    print(
        f"adapter {os.path.basename(a.adapter)} A->B={c['eval']['A->B']:+.4f} devA={dA} devB={dB}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(a.tok)
    A = (
        AutoModelForCausalLM.from_pretrained(
            a.rwkv, dtype=torch.float16, low_cpu_mem_usage=True
        )
        .to(dA)
        .eval()
    )
    B = (
        AutoModelForCausalLM.from_pretrained(
            a.pythia, dtype=torch.float16, low_cpu_mem_usage=True
        )
        .to(dB)
        .eval()
    )
    for p in list(A.parameters()) + list(B.parameters()):
        p.requires_grad_(False)
    Blay = B.gpt_neox.layers

    ids0 = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        offA = (
            1
            if (
                A(ids0.to(dA), output_hidden_states=True).hidden_states[0]
                - A.rwkv.embeddings(ids0.to(dA))
            )
            .abs()
            .max()
            .item()
            < 1e-3
            else 0
        )
        hsB0 = B(ids0.to(dB), output_hidden_states=True).hidden_states
        offB = (
            1
            if (hsB0[0] - B.gpt_neox.embed_in(ids0.to(dB))).abs().max().item() < 1e-3
            else 0
        )
    print(f"off: RWKV={offA} Pythia={offB}", flush=True)

    def std_a(h, L):
        return (h.float() - st[L][0]) / st[L][1]

    def std_b(h, L):
        return (h.float() - st[L][2]) / st[L][3]

    # ---- best linear map per switch (unchanged) ----
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("tatsu-lab/alpaca", split="train")["text"])
    win = 112
    nwin = a.fit_rows // win + 1
    fit_ids = (
        tok(text[2_000_000 : 2_000_000 + nwin * win * 8], return_tensors="pt")
        .input_ids[0][: nwin * win]
        .reshape(nwin, win)
    )
    Blin = {}
    with torch.no_grad():
        XtX = {L: torch.zeros(4097, 4097, dtype=torch.float64) for L in SW}
        XtY = {L: torch.zeros(4097, 4096, dtype=torch.float64) for L in SW}
        for s in range(0, fit_ids.shape[0], 4):
            ib = fit_ids[s : s + 4]
            hsA = A(ib.to(dA), output_hidden_states=True).hidden_states
            hsB = B(ib.to(dB), output_hidden_states=True).hidden_states
            for L in SW:
                va = std_a(hsA[L + offA].to(dB), L).reshape(-1, 4096).double()
                vb = std_b(hsB[L + offB], L).reshape(-1, 4096).double()
                xa = torch.cat(
                    [va, torch.ones(va.shape[0], 1, dtype=torch.float64, device=dB)], 1
                )
                XtX[L] += (xa.T @ xa).cpu()
                XtY[L] += (xa.T @ vb).cpu()
            del hsA, hsB
        for L in SW:
            lam = a.ridge * (torch.diagonal(XtX[L]).sum() / 4097)
            XtX[L] += lam * torch.eye(4097, dtype=torch.float64)
            Blin[L] = torch.linalg.solve(XtX[L], XtY[L]).float().to(dB)
        del XtX, XtY
        torch.cuda.empty_cache()
    print(f"linear maps fit for {SW}", flush=True)

    _inject = {"h": None}

    def prehook(mod, args, kwargs):
        if _inject["h"] is None:
            return None
        h = _inject["h"]
        if len(args):
            return (h,) + args[1:], kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = h
        return args, kwargs

    def pythia_inject(ids_b, h):
        _inject["h"] = h
        try:
            return B(ids_b).logits
        finally:
            _inject["h"] = None

    def translate_adapter(hA, L):
        return (m.dB(m.enc_a(std_a(hA.to(dB), L))) * st[L][3] + st[L][2]).to(B.dtype)

    def translate_linear(hA, L):
        v = std_a(hA.to(dB), L).reshape(-1, 4096)
        v = torch.cat([v, torch.ones(v.shape[0], 1, device=dB)], 1)
        return ((v @ Blin[L]).reshape(1, -1, 4096) * st[L][3] + st[L][2]).to(B.dtype)

    def translate_bb(hBself, L):  # B's own residual through the B autoencoder
        return (m.dB(m.enc_b(std_b(hBself.to(dB), L))) * st[L][3] + st[L][2]).to(
            B.dtype
        )

    @torch.no_grad()
    def opt_logprob(logits, ids_any, p0):
        lp = torch.log_softmax(logits[0, :-1].float(), -1)
        tgt = ids_any[0, 1:].to(lp.device)
        sel = lp[torch.arange(len(tgt), device=lp.device), tgt][p0 - 1 :]
        return sel.sum().item(), sel.mean().item()

    # gate per switch (+ oracle check: injecting true r_B^L reproduces pure Pythia)
    handles = {}
    gate_errors = {}
    with torch.no_grad():
        _ALPHA["value"] = 1.0
        hf0 = B(ids0.to(dB)).logits
        _ = opt_logprob(A(ids0.to(dA)).logits, ids0.to(dB), 2)
        for L in SW:
            idxB = L + offB
            assert 0 < idxB < len(Blay)
            h = Blay[idxB].register_forward_pre_hook(prehook, with_kwargs=True)
            r = rel(pythia_inject(ids0.to(dB), hsB0[idxB].to(dA).to(dB)), hf0)
            assert r < 1e-4, f"GATE/oracle FAILED L={L}: rel={r:.2e}"
            h.remove()
            handles[L] = idxB
            gate_errors[L] = r
    print(
        f"GATE+oracle passed {SW}: injecting true r_B^L reproduces pure Pythia (rel<1e-4) => oracle==pythia\n",
        flush=True,
    )

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
    print(f"task={a.task}  items={len(items)}  (full test set)\n", flush=True)

    cfgs = (
        ["pythia", "rwkv"]
        + [f"A_to_B@{L}" for L in SW]
        + [f"A_to_B_alpha0@{L}" for L in SW]
        + [f"A_to_B_affine@{L}" for L in SW]
        + [f"B_to_B@{L}" for L in SW]
    )
    hit_sum = {k: 0 for k in cfgs}
    hit_norm = {k: 0 for k in cfgs}
    per_item = {k: [] for k in cfgs}  # acc_norm correctness per item (for paired stats)

    for qi, (prompt, opts, gold) in enumerate(items):
        p0 = tok(prompt, return_tensors="pt").input_ids.shape[1]
        sc_s = {k: [] for k in cfgs}
        sc_n = {k: [] for k in cfgs}
        for o in opts:
            ids = tok(prompt + o, return_tensors="pt").input_ids
            ida, idb = ids.to(dA), ids.to(dB)
            with torch.no_grad():
                outA = A(ida, output_hidden_states=True)
                outB = B(
                    idb, output_hidden_states=True
                )  # pythia logits + B's own residuals for bb
                s, n = opt_logprob(outA.logits, ida, p0)
                sc_s["rwkv"].append(s)
                sc_n["rwkv"].append(n)
                s, n = opt_logprob(outB.logits, idb, p0)
                sc_s["pythia"].append(s)
                sc_n["pythia"].append(n)
                for L in SW:
                    hA = outA.hidden_states[L + offA]
                    hBself = outB.hidden_states[L + offB]
                    hnd = Blay[handles[L]].register_forward_pre_hook(
                        prehook, with_kwargs=True
                    )
                    try:
                        _ALPHA["value"] = 1.0
                        lg = pythia_inject(idb, translate_adapter(hA, L))
                        s, n = opt_logprob(lg, idb, p0)
                        sc_s[f"A_to_B@{L}"].append(s)
                        sc_n[f"A_to_B@{L}"].append(n)
                        _ALPHA["value"] = 0.0
                        lg = pythia_inject(idb, translate_adapter(hA, L))
                        s, n = opt_logprob(lg, idb, p0)
                        sc_s[f"A_to_B_alpha0@{L}"].append(s)
                        sc_n[f"A_to_B_alpha0@{L}"].append(n)
                        _ALPHA["value"] = 1.0
                        lg = pythia_inject(idb, translate_linear(hA, L))
                        s, n = opt_logprob(lg, idb, p0)
                        sc_s[f"A_to_B_affine@{L}"].append(s)
                        sc_n[f"A_to_B_affine@{L}"].append(n)
                        lg = pythia_inject(idb, translate_bb(hBself, L))
                        s, n = opt_logprob(lg, idb, p0)
                        sc_s[f"B_to_B@{L}"].append(s)
                        sc_n[f"B_to_B@{L}"].append(n)
                    finally:
                        _ALPHA["value"] = 1.0
                        hnd.remove()
        for k in cfgs:
            cs = int(max(range(len(opts)), key=lambda i: sc_s[k][i]) == gold)
            cn = int(max(range(len(opts)), key=lambda i: sc_n[k][i]) == gold)
            hit_sum[k] += cs
            hit_norm[k] += cn
            per_item[k].append(cn)
        if (qi + 1) % 100 == 0:
            print(
                f"  [{qi + 1}/{len(items)}] "
                + "  ".join(
                    f"{k} {100 * hit_norm[k] / (qi + 1):.0f}"
                    for k in ["pythia", "rwkv"] + [f"A_to_B@{L}" for L in SW]
                ),
                flush=True,
            )

    N = len(items)
    print(f"\n== {a.task}  N={N}  (accuracy %, chance={100 / len(items[0][1]):.0f}) ==")
    print(f"{'config':10s} {'acc(sum)':>9} {'acc(norm)':>10}")
    print("-" * 32)
    for k in cfgs:
        print(f"{k:10s} {100 * hit_sum[k] / N:9.1f} {100 * hit_norm[k] / N:10.1f}")
    if a.dump:
        payload = {
            "schema_version": 2,
            "direction": "A_to_B",
            "source_model": "RWKV-Raven-7B",
            "target_model": "Tulu-Pythia-6.9B",
            "boundary": "RWKV post-block L -> Pythia pre-block L+1",
            "task": a.task,
            "N": N,
            "cfgs": cfgs,
            "switches": SW,
            "gold_chance": 1.0 / len(items[0][1]),
            "parent_reproduction_relative_error": {
                str(layer): error for layer, error in gate_errors.items()
            },
            "checkpoint": os.path.basename(a.adapter),
            "checkpoint_step": c.get("step", ""),
            "checkpoint_a_to_b_r2": c["eval"]["A->B"],
            "linear_fit_rows_requested": a.fit_rows,
            "linear_ridge": a.ridge,
            "same_adapter_intervention": {
                "branch": "ResBlock.fc2(GELU(fc1(LayerNorm(x))))",
                "alphas": [0.0, 1.0],
                "alpha0_config_prefix": "A_to_B_alpha0",
            },
            "per_item_norm": per_item,
            "acc_norm": {k: hit_norm[k] / N for k in cfgs},
            "acc_sum": {k: hit_sum[k] / N for k in cfgs},
        }
        rows = [
            {
                "direction": "A_to_B",
                "source_model": payload["source_model"],
                "target_model": payload["target_model"],
                "task": a.task,
                "config": key,
                "switch_layer": key.rsplit("@", 1)[1] if "@" in key else "",
                "accuracy_sum": hit_sum[key] / N,
                "accuracy_norm": hit_norm[key] / N,
                "n_items": N,
                "chance": payload["gold_chance"],
            }
            for key in cfgs
        ]
        json_path, csv_path = write_bundle(a.dump, payload, rows)
        print(f"wrote per-item JSON {json_path} + summary CSV {csv_path}", flush=True)
    print("BENCHFULL_DONE", flush=True)


if __name__ == "__main__":
    main()
