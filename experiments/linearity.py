"""How near-linear are both cross-family paths of the trained adapter?

The paper calls the adapter "near-linear" and justifies it entirely by the initialisation: each ResBlock's second
matrix is zero-initialised, so the map STARTS as exactly linear. That is a fact about step 0, not about step
238,000, and nothing in the paper measures what the map became. From the converged weights alone the picture is
already not "a small perturbation of the identity": the non-linear branch is 2.6-11.6x the size of the identity
skip, and each GELU is used in a regime where a per-unit linear fit explains only 0.41-0.78 of its output.

The same held-out residual rows, fit split, ridge, and layer set are used for
A->B and B->A.  No metric for one path is inferred from the other.

Four quantities are reported per path and layer on rows the fit never saw:

  1. ADAPTER CROSS R2      -- D_B(E_A(.)) for A->B or D_A(E_B(.)) for B->A.
  2. LINEARISED-ADAPTER    -- fit the best affine map L to imitate the ADAPTER's own output, then ask
     (a) R^2(L, adapter)   -- how much of what the adapter does is linear; this is the "near-linear" claim itself
     (b) cross R^2 of L     -- does the affine imitation still deliver the read-out?
  3. BEST AFFINE CROSS R2  -- fit an affine map directly to the path's source/target residuals.

Fitting is by normal equations accumulated in fp64 over chunks (X^T X is only 4097x4097), so the whole training
half can be used without holding it in memory. Ridge is applied at a level far below the signal (1e-3 * trace/d)
purely to keep the solve well-conditioned; it is reported so it cannot hide behind the result.
"""
import argparse, os, sys, json, gc, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from ninaxander.adapter import LatentAdapter, blocks_of
from ninaxander.result_io import write_bundle
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="artifacts/checkpoints/z4096_L32/SNAP_L32_final.pt")
    ap.add_argument("--rwkv", default="models/rwkv-raven-7b-hf-fp16")
    ap.add_argument("--pythia", default="models/tulu-pythia69-fp16")
    ap.add_argument("--tok", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--blocks", type=int, default=1); ap.add_argument("--hid", type=int, default=4096)
    ap.add_argument("--layers", default="4,10,16,22,28")
    ap.add_argument("--win", type=int, default=112); ap.add_argument("--nwin", type=int, default=900)
    ap.add_argument("--ridge", type=float, default=1e-3, help="ridge as a fraction of trace(X^T X)/d")
    ap.add_argument(
        "--out",
        default="artifacts/metrics/raw/cross_family_linearity_L32.json",
    )
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0)
    LAYERS = [int(x) for x in a.layers.split(",")]

    c = torch.load(a.adapter, map_location="cpu", weights_only=False); cfg = c["cfg"]
    m = LatentAdapter(cfg["dA"], cfg["dB"], cfg["z"], blocks=a.blocks, hidden=a.hid, arch="resnet").to(dev).eval()
    m.load_state_dict(c["model"])
    for p in m.parameters(): p.requires_grad_(False)
    st = {int(j): [t.to(dev) for t in v] for j, v in c["stats"].items()}
    print(
        f"adapter {os.path.basename(a.adapter)} step={c.get('step')} "
        f"A->B={c['eval']['A->B']:+.4f} B->A={c['eval']['B->A']:+.4f} "
        f"stats layers {min(st)}..{max(st)}",
        flush=True,
    )
    LAYERS = [j for j in LAYERS if j in st]
    print(f"layers under test: {LAYERS}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.tok)
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("tatsu-lab/alpaca", split="train")["text"])
    ids = tok(text[2_000_000:2_800_000], return_tensors="pt").input_ids[0]
    need = a.nwin * a.win
    assert ids.shape[0] >= need, f"held-out pool {ids.shape[0]} < {need}"
    EV = ids[:need].reshape(a.nwin, a.win)
    print(f"held-out pool: {a.nwin} windows x {a.win} = {need} tokens", flush=True)

    def extract(path, tag):
        """raw output of block j, captured by forward hook -- the same convention adapter7b now uses"""
        M = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, low_cpu_mem_usage=True).to(dev).eval()
        for p in M.parameters(): p.requires_grad_(False)
        with torch.no_grad(): M(EV[:1].to(dev))            # force RWKV's lazy weight rescale before hooking
        blk = blocks_of(M)
        RE = M.config.rescale_every if (hasattr(M, "rwkv") and M.rwkv.layers_are_rescaled) else 0
        cap = {}
        def mk(j):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if RE > 0 and (j + 1) % RE == 0: h = h / 2
                cap[j] = h
            return hook
        hs = [blk[j].register_forward_hook(mk(j)) for j in LAYERS]
        acc = {j: [] for j in LAYERS}
        with torch.no_grad():
            for s in range(0, EV.shape[0], 8):
                M(EV[s:s + 8].to(dev))
                for j in LAYERS: acc[j].append(cap[j].reshape(-1, cap[j].shape[-1]).half().cpu())
        for h in hs: h.remove()
        out = {j: torch.cat(acc[j], 0) for j in LAYERS}
        print(f"  {tag}: {out[LAYERS[0]].shape[0]} rows/layer", flush=True)
        del M, acc, cap; gc.collect(); torch.cuda.empty_cache()
        return out

    RA = extract(a.rwkv, "RWKV"); RB = extract(a.pythia, "Pythia")
    N = RA[LAYERS[0]].shape[0]
    half = N // 2                                     # FIT on the first half, REPORT on the second
    print(f"fit rows {half}, test rows {N - half}\n", flush=True)

    def fit_affine(X, Y, chunk=16384):
        """least squares Y ~ [X 1] B, by normal equations accumulated in fp64"""
        d = X.shape[1]
        XtX = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=dev)
        XtY = torch.zeros(d + 1, Y.shape[1], dtype=torch.float64, device=dev)
        for s in range(0, X.shape[0], chunk):
            xb = X[s:s + chunk].to(dev).double()
            xb = torch.cat([xb, torch.ones(xb.shape[0], 1, dtype=torch.float64, device=dev)], 1)
            yb = Y[s:s + chunk].to(dev).double()
            XtX += xb.T @ xb; XtY += xb.T @ yb
            del xb, yb
        lam = a.ridge * (torch.diagonal(XtX).sum() / (d + 1))
        XtX += lam * torch.eye(d + 1, dtype=torch.float64, device=dev)
        return torch.linalg.solve(XtX, XtY).float(), lam.item()

    def apply_affine(B, X, chunk=16384):
        outs = []
        for s in range(0, X.shape[0], chunk):
            xb = X[s:s + chunk].to(dev)
            xb = torch.cat([xb, torch.ones(xb.shape[0], 1, device=dev)], 1)
            outs.append(xb @ B)
        return torch.cat(outs, 0)

    def r2(pred, tgt):
        """per-dim R^2, centered on the TARGET's own mean -- a constant predictor scores exactly 0"""
        sse = (pred - tgt).pow(2).sum().double()
        sst = (tgt - tgt.mean(0)).pow(2).sum().double()
        return (1 - sse / (sst + 1e-9)).item()

    rows = {"A_to_B": {}, "B_to_A": {}}
    for j in LAYERS:
        mu_a, sd_a, mu_b, sd_b = st[j]
        va = ((RA[j].to(dev).float() - mu_a) / sd_a)      # standardised A residual
        vb = ((RB[j].to(dev).float() - mu_b) / sd_b)      # standardised B residual
        path_specs = (
            ("A_to_B", va, vb, lambda value: m.dB(m.enc_a(value))),
            ("B_to_A", vb, va, lambda value: m.dA(m.enc_b(value))),
        )
        for direction, source, target, translate in path_specs:
            with torch.no_grad():
                adapter_output = torch.cat(
                    [
                        translate(source[s:s + 16384])
                        for s in range(0, N, 16384)
                    ],
                    0,
                )
            source_fit, source_test = source[:half], source[half:]
            target_test = target[half:]
            adapter_fit, adapter_test = adapter_output[:half], adapter_output[half:]

            imitate, ridge_imitate = fit_affine(source_fit, adapter_fit)
            linearised_test = apply_affine(imitate, source_test)
            direct, ridge_direct = fit_affine(source_fit, target[:half])
            direct_test = apply_affine(direct, source_test)

            rows[direction][j] = {
                "adapter_cross_r2": r2(adapter_test, target_test),
                "linearised_fit_r2": r2(linearised_test, adapter_test),
                "linearised_cross_r2": r2(linearised_test, target_test),
                "best_affine_cross_r2": r2(direct_test, target_test),
                "ridge_imitate": ridge_imitate,
                "ridge_direct": ridge_direct,
            }
            result = rows[direction][j]
            print(
                f"  {direction} L{j:2d} adapter {result['adapter_cross_r2']:.4f} | "
                f"affine imitation {result['linearised_fit_r2']:.4f}, "
                f"imitation cross {result['linearised_cross_r2']:.4f} | "
                f"best direct affine {result['best_affine_cross_r2']:.4f}",
                flush=True,
            )
            del adapter_output, imitate, direct, linearised_test, direct_test
            torch.cuda.empty_cache()
        del va, vb
        torch.cuda.empty_cache()

    metric_names = [
        "adapter_cross_r2",
        "linearised_fit_r2",
        "linearised_cross_r2",
        "best_affine_cross_r2",
    ]
    means = {
        direction: {
            metric: sum(rows[direction][j][metric] for j in LAYERS) / len(LAYERS)
            for metric in metric_names
        }
        for direction in rows
    }
    print("\n" + "-" * 92)
    print(f"mean over layers {LAYERS}")
    for direction in ("A_to_B", "B_to_A"):
        result = means[direction]
        print(
            f"  {direction}: adapter={result['adapter_cross_r2']:.4f} "
            f"affine-fit={result['linearised_fit_r2']:.4f} "
            f"affine-cross={result['linearised_cross_r2']:.4f} "
            f"best-direct={result['best_affine_cross_r2']:.4f}"
        )
    print("-" * 92)

    payload = {
        "adapter": os.path.basename(a.adapter),
        "step": c.get("step"),
        "directions": ["A_to_B", "B_to_A"],
        "layers": LAYERS,
        "rows_per_layer": N,
        "fit_rows": half,
        "rows": rows,
        "mean": means,
    }
    csv_rows = [
        {
            "scope": "layer",
            "direction": direction,
            "layer": layer,
            **rows[direction][layer],
            "evaluation_rows": N - half,
            "fit_rows": half,
            "checkpoint_step": c.get("step"),
        }
        for direction in ("A_to_B", "B_to_A")
        for layer in LAYERS
    ]
    csv_rows.extend(
        {
            "scope": "mean",
            "direction": direction,
            "layer": "",
            **means[direction],
            "evaluation_rows": (N - half) * len(LAYERS),
            "fit_rows": half * len(LAYERS),
            "checkpoint_step": c.get("step"),
        }
        for direction in ("A_to_B", "B_to_A")
    )
    json_path, csv_path = write_bundle(a.out, payload, csv_rows)
    print(f"\nwrote {json_path} + {csv_path}\nLINEARITY_DONE", flush=True)


if __name__ == "__main__":
    main()
