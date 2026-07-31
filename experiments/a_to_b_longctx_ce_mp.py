"""Long-context language-modelling CE for the 32-layer A-to-B chimera.

Why this exists. The QA benchmarks (ARC/SciQ) score ~50-token items -- SHORT context. But the chimera's only win is
KV memory, and KV memory only matters at LONG context. So the QA trade-off is measured in the wrong regime. This
measures next-token CE as context grows toward where the memory saving is real, for the parents, the ADAPTER chimera,
and the BEST-LINEAR-MAP chimera at each switch.

Honest scope, stated up front:
  * Both parents are short-context models -- RWKV-Raven trained at 1024, Tulu-Pythia at 2048 -- so 2048 is the ceiling;
    beyond it both degrade for their OWN reasons and would confound the chimera. We measure 512/1024/2048 only.
  * CE is language-modelling quality, NOT distant-dependency reading comprehension. It answers "does the chimera keep
    modelling text well as context grows", which is exactly what the KV trade-off needs, but it is not a needle task.
  * Text is WikiText-103 (out of domain): the calibration used Alpaca, and RWKV-Raven is instruction-tuned on
    Alpaca-like data, so Alpaca CE partly measured memorisation. WikiText removes that confound.

Method uses the same trusted, gated model-parallel path as the canonical QA evaluation: RWKV on devA,
Pythia+adapter+stats on devB, and HF runs its own forward with a pre-hook replacing the switched layer's input.
Parents run natively. The
per-switch parent-repro gate (inject B's own residual -> logits unchanged) and the cross-device round-trip are
exercised before any measurement. CE is one-pass mean next-token cross-entropy over non-overlapping windows.
"""

import argparse
import os
import sys
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from ninaxander.adapter import LatentAdapter
from ninaxander.result_io import write_bundle
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    ap.add_argument("--switches", default="4,8,16,24")
    ap.add_argument(
        "--ctx", default="512:48,1024:32,2048:20", help="ctxlen:nwindows,..."
    )
    ap.add_argument("--dev_a", default="cuda:0")
    ap.add_argument("--dev_b", default="cuda:1")
    ap.add_argument(
        "--fit_rows",
        type=int,
        default=40000,
        help="held-out residual rows to fit the linear map",
    )
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument(
        "--text",
        default="wikitext",
        choices=["wikitext", "alpaca"],
        help="wikitext = out-of-domain; alpaca = in-domain control (same text the adapter trained on)",
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if not a.out:
        a.out = f"artifacts/metrics/raw/a_to_b_longctx_{a.text}.json"
    dA, dB = a.dev_a, a.dev_b
    torch.manual_seed(0)
    SW = [int(x) for x in a.switches.split(",")]
    PLAN = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in a.ctx.split(",")]

    c = torch.load(a.adapter, map_location="cpu", weights_only=False)
    cfg = c["cfg"]
    m = (
        LatentAdapter(
            cfg["dA"], cfg["dB"], cfg["z"], blocks=a.blocks, hidden=a.hid, arch="resnet"
        )
        .to(dB)
        .eval()
    )
    m.load_state_dict(c["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    m = m.half()  # fp16 adapter halves its footprint (1.07->0.53 GiB) so devB has room for ctx-2048 activations;
    # the chimera runs the whole B side in fp16 anyway, so this changes nothing downstream.
    st = {
        int(j): [t.to(dB) for t in v] for j, v in c["stats"].items()
    }  # stats stay fp32 (tiny, 4 vecs/layer)
    print(
        f"adapter {os.path.basename(a.adapter)}  A->B={c['eval']['A->B']:+.4f}  devA={dA} devB={dB}  plan={PLAN}",
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

    probe = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        offA = (
            1
            if (
                A(probe.to(dA), output_hidden_states=True).hidden_states[0]
                - A.rwkv.embeddings(probe.to(dA))
            )
            .abs()
            .max()
            .item()
            < 1e-3
            else 0
        )
        offB = (
            1
            if (
                B(probe.to(dB), output_hidden_states=True).hidden_states[0]
                - B.gpt_neox.embed_in(probe.to(dB))
            )
            .abs()
            .max()
            .item()
            < 1e-3
            else 0
        )
    print(f"off: RWKV={offA} Pythia={offB}", flush=True)

    # ---- held-out text. wikitext = OUT-of-domain (the real test); alpaca = IN-domain control (the corpus the
    #      adapter trained on, held-out region) to separate a DOMAIN-shift effect from a CONTEXT-length effect.
    #      Fit rows come from the FRONT, eval windows from further in, so fit and CE never share tokens. ----
    from datasets import load_dataset

    if a.text == "wikitext":
        src = "".join(
            load_dataset("wikitext", "wikitext-103-raw-v1", split="test")["text"]
        )
        all_ids = tok(src, return_tensors="pt").input_ids[0]
        fit_ids = all_ids[: a.fit_rows]
        need_eval = max(n * L for L, n in PLAN)
        eval_ids = all_ids[a.fit_rows : a.fit_rows + need_eval + 10]
    else:  # alpaca in-domain control: use the SAME held-out region adapter7b evaluated on (text[2e6:])
        src = "\n\n".join(load_dataset("tatsu-lab/alpaca", split="train")["text"])
        all_ids = tok(src[2_000_000:2_800_000], return_tensors="pt").input_ids[0]
        fit_ids = all_ids[: a.fit_rows]
        need_eval = max(n * L for L, n in PLAN)
        eval_ids = all_ids[a.fit_rows : a.fit_rows + need_eval + 10]
    assert eval_ids.shape[0] >= need_eval, (
        f"not enough {a.text}: have {eval_ids.shape[0]}, need {need_eval}"
    )
    print(
        f"text={a.text}: {all_ids.shape[0]} tokens; fit={fit_ids.shape[0]}  eval_pool={eval_ids.shape[0]} (disjoint)",
        flush=True,
    )

    def std_a(h, L):
        return (h.float() - st[L][0]) / st[L][1]

    def std_b(h, L):
        return (h.float() - st[L][2]) / st[L][3]

    # ---- fit the best linear map per switch on the FIT slice (fp64 normal equations, CPU accumulators) ----
    Blin = {}
    with torch.no_grad():
        w = 128
        nf = fit_ids.shape[0] // w
        F = fit_ids[: nf * w].reshape(nf, w)
        XtX = {L: torch.zeros(4097, 4097, dtype=torch.float64) for L in SW}
        XtY = {L: torch.zeros(4097, 4096, dtype=torch.float64) for L in SW}
        for s in range(0, nf, 4):
            ib = F[s : s + 4]
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
    print(f"linear maps fit for switches {SW}", flush=True)

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
        z = m.enc_a(std_a(hA.to(dB), L).half())  # adapter is fp16; feed it fp16
        return (m.dB(z).float() * st[L][3] + st[L][2]).to(B.dtype)

    def translate_linear(hA, L):
        v = std_a(hA.to(dB), L).reshape(-1, 4096)  # fp32; Blin is fp32
        v = torch.cat([v, torch.ones(v.shape[0], 1, device=dB)], 1)
        return ((v @ Blin[L]).reshape(1, -1, 4096) * st[L][3] + st[L][2]).to(B.dtype)

    # ---- gate: parent-repro + real cross-device round-trip, per switch, before measuring ----
    parent_gate_errors = {}
    with torch.no_grad():
        hf0 = B(probe.to(dB)).logits
        hsB0 = B(probe.to(dB), output_hidden_states=True).hidden_states
        for L in SW:
            idxB = L + offB
            h = Blay[idxB].register_forward_pre_hook(prehook, with_kwargs=True)
            r = rel(pythia_inject(probe.to(dB), hsB0[idxB].to(dA).to(dB)), hf0)
            h.remove()
            assert r < 1e-4, f"GATE FAILED at L={L}: rel={r:.2e}"
            parent_gate_errors[L] = r
    print(
        f"GATE passed at switches {SW} (parent-repro + cross-device rel<1e-4)\n",
        flush=True,
    )

    ce = torch.nn.functional.cross_entropy

    @torch.no_grad()
    def ce_window(logits, ids_b, chunk=512):
        # CE over the whole window, but fp32-cast only a CHUNK of positions at a time. At ctx 2048 the full
        # logits[0,:-1].float() is a [2047,50278] fp32 tensor (~0.4 GiB) and a second copy inside cross_entropy
        # tips the 16 GiB devB card over (Pythia 13 GiB + adapter + linear maps already fill it). Summing the
        # per-position loss in chunks keeps peak fp32 to [chunk,50278] and is numerically identical.
        lg = logits[0, :-1]
        tgt = ids_b[0, 1:].to(lg.device)
        n = lg.shape[0]
        tot = 0.0
        for s in range(0, n, chunk):
            tot += ce(
                lg[s : s + chunk].float(), tgt[s : s + chunk], reduction="sum"
            ).item()
        return tot / n

    @torch.no_grad()
    def chimera_ce(hA, idsB, L, translate):
        # takes a PRECOMPUTED RWKV residual so the (slow, sequential) RWKV forward runs once per window, not once
        # per switch x method -- the |SW|x re-run the canonical QA implementation already eliminates.
        hB = translate(hA, L)
        hnd = Blay[L + offB].register_forward_pre_hook(prehook, with_kwargs=True)
        try:
            lg = pythia_inject(idsB, hB)
        finally:
            hnd.remove()
        return ce_window(lg, idsB)

    # Capture ONLY the switch-layer RWKV residuals via forward hooks, instead of output_hidden_states=True which
    # keeps all 33 states of [1,T,4096] fp16 (~1 GiB at T=2048) and OOMs the 16 GiB devA card at ctx 2048. RWKV
    # hidden_states[j] = output of block j (offA=0), and HF's periodic /2 is applied AFTER the block returns, so
    # the hook (which sees the pre-/2 value) mirrors it at (j+1)%RE==0 -- same rule the extraction uses.
    RE = A.config.rescale_every
    cap = {}

    def mk(j):
        def hook(mod, inp, out):
            if j + offA in _want:
                h = out[0] if isinstance(out, tuple) else out
                if RE > 0 and (j + 1) % RE == 0:
                    h = h / 2
                cap[j + offA] = h

        return hook

    _want = set(sw + offA for sw in SW)
    Ablk = A.rwkv.blocks
    ah = [Ablk[j].register_forward_hook(mk(j)) for j in range(len(Ablk))]

    # GATE the hook capture against output_hidden_states before trusting it (new capture path; this class of
    # switch-residual bug has silently broken before). The hooked value must equal hidden_states[sw+offA] exactly.
    source_capture_errors = {}
    with torch.no_grad():
        cap.clear()
        hsg = A(probe.to(dA), output_hidden_states=True).hidden_states
        for sw in SW:
            r = rel(cap[sw + offA], hsg[sw + offA])
            assert r < 1e-4, f"HOOK CAPTURE GATE FAILED at switch {sw}: rel={r:.2e}"
            source_capture_errors[sw] = r
    print(
        f"hook-capture gate passed at switches {SW} (matches hidden_states, rel<1e-4)\n",
        flush=True,
    )

    cfgs = (
        ["pure-Pythia", "pure-RWKV"]
        + [f"A_to_B@{L}" for L in SW]
        + [f"A_to_B_affine@{L}" for L in SW]
    )
    OUT = {
        "schema_version": 1,
        "direction": "A_to_B",
        "source_model": "RWKV-Raven-7B",
        "target_model": "Tulu-Pythia-6.9B",
        "boundary": "RWKV post-block L -> Pythia pre-block L+1",
        "adapter": os.path.basename(a.adapter),
        "adapter_dtype": str(next(m.parameters()).dtype),
        "checkpoint_step": c.get("step", ""),
        "checkpoint_a_to_b_r2": c["eval"]["A->B"],
        "text": a.text,
        "switches": SW,
        "fit_rows_requested": a.fit_rows,
        "fit_rows_actual": int(nf * w),
        "linear_ridge": a.ridge,
        "evaluation_windows": {str(context): windows for context, windows in PLAN},
        "parent_reproduction_relative_error": {
            str(layer): error for layer, error in parent_gate_errors.items()
        },
        "source_capture_relative_error": {
            str(layer): error for layer, error in source_capture_errors.items()
        },
        "ctx": {},
    }
    for L, NW in PLAN:
        EV = eval_ids[: NW * L].reshape(NW, L)
        acc = {k: [] for k in cfgs}
        for i in range(NW):
            idsA, idsB = EV[i : i + 1].to(dA), EV[i : i + 1].to(dB)
            with torch.no_grad():
                cap.clear()
                logitsA = A(
                    idsA
                ).logits  # ONE RWKV forward; hooks stash switch residuals
                acc["pure-RWKV"].append(ce_window(logitsA, idsB))
                acc["pure-Pythia"].append(ce_window(B(idsB).logits, idsB))
                for sw in SW:
                    hA = cap[sw + offA]
                    acc[f"A_to_B@{sw}"].append(
                        chimera_ce(hA, idsB, sw, translate_adapter)
                    )
                    acc[f"A_to_B_affine@{sw}"].append(
                        chimera_ce(hA, idsB, sw, translate_linear)
                    )

        def stat(v):
            n = len(v)
            mu = sum(v) / n
            sd = (sum((x - mu) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
            return mu, sd / (n**0.5)

        OUT["ctx"][L] = {
            k: {"ce": stat(acc[k])[0], "se": stat(acc[k])[1]} for k in cfgs
        }
        print(f"== ctx {L} ({NW} windows) ==")
        for k in cfgs:
            mu, se = stat(acc[k])
            print(
                f"  {k:12s} CE {mu:.4f} +-{se:.4f}  ppl {math.exp(mu):7.2f}", flush=True
            )
        print(flush=True)
    window_counts = {context: windows for context, windows in PLAN}
    csv_rows = [
        {
            "direction": "A_to_B",
            "source_model": OUT["source_model"],
            "target_model": OUT["target_model"],
            "domain": a.text,
            "context_tokens": context,
            "windows": window_counts[context],
            "config": config,
            "config_type": config.rsplit("@", 1)[0],
            "switch_layer": (config.rsplit("@", 1)[1] if "@" in config else ""),
            "cross_entropy": values["ce"],
            "standard_error": values["se"],
            "perplexity": math.exp(values["ce"]),
        }
        for context, configurations in OUT["ctx"].items()
        for config, values in configurations.items()
    ]
    write_bundle(a.out, OUT, csv_rows)
    for h in ah:
        h.remove()
    print(f"wrote {a.out} + {os.path.splitext(a.out)[0]}.csv\nLONGCTX_DONE", flush=True)


if __name__ == "__main__":
    main()
