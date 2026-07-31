"""All-layer x all-layer correspondence matrix.

The paper pairs block j of A with block j of B and reports a high alignment.
This experiment tests whether that diagonal pairing is actually the best one,
or whether some off-diagonal (j != k) aligns better -- which would undercut the
"same layer number shares a space" reading. We answer with two standard, adapter-free representation-similarity
measures computed for every (j, k) pair, on the same held-out Alpaca residuals, captured by forward hook:

  * linear CKA (Kornblith et al. 2019): centered, scale-invariant, in [0,1].
  * best-linear cross-R^2: ridge-regularised least squares r_A^j -> r_B^k, in-sample R^2 (a linear probe).

For each row j we report argmax_k and whether it lands on the diagonal, plus mean(diagonal) vs mean(off-diagonal).
Read-only: frozen parents, no adapter, no training. Output: a JSON record plus a CSV sidecar.
"""
import argparse, os, sys, json, gc, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from ninaxander.adapter import blocks_of
from ninaxander.result_io import write_bundle
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rwkv", default="models/rwkv-raven-7b-hf-fp16")
    ap.add_argument("--pythia", default="models/tulu-pythia69-fp16")
    ap.add_argument("--tok", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--win", type=int, default=112); ap.add_argument("--nwin", type=int, default=200)
    ap.add_argument("--eval_cut", type=int, default=2_000_000); ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--out", default="artifacts/metrics/raw/all_layer_corr.json")
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(a.tok)
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("tatsu-lab/alpaca", split="train")["text"])
    ids = tok(text[a.eval_cut:a.eval_cut + 900_000], return_tensors="pt").input_ids[0]
    EV = ids[:a.nwin * a.win].reshape(a.nwin, a.win)
    print(f"held-out: {EV.shape[0]} windows x {EV.shape[1]} = {EV.numel()} tokens", flush=True)
    LAYERS = list(range(32))

    def extract(path, tag):
        M = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, low_cpu_mem_usage=True).to(dev).eval()
        for p in M.parameters(): p.requires_grad_(False)
        with torch.no_grad():
            hs0 = M(EV[:1].to(dev), output_hidden_states=True).hidden_states
            off = 1 if (hs0[0] - M.get_input_embeddings()(EV[:1].to(dev))).abs().max().item() < 1e-3 else 0
        L = M.config.num_hidden_layers; blk = blocks_of(M)
        RE = M.config.rescale_every if (hasattr(M, "rwkv") and getattr(M.rwkv, "layers_are_rescaled", False)) else 0
        cap = {}
        def mk(j):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if RE > 0 and (j + 1) % RE == 0: h = h / 2
                cap[j] = h
            return hook
        handles = [blk[j].register_forward_hook(mk(j)) for j in range(L)]
        with torch.no_grad(): hsg = M(EV[:1].to(dev), output_hidden_states=True).hidden_states
        rl = lambda x, y: (x - y).float().norm().item() / (y.float().norm().item() + 1e-9)
        worst = max(rl(cap[j], hsg[j + off]) for j in range(L) if j + off < len(hsg) - 1)
        assert worst < 1e-3, f"{tag}: hook mismatch {worst:.2e}"
        acc = {j: [] for j in LAYERS}
        with torch.no_grad():
            for s in range(0, EV.shape[0], 8):
                M(EV[s:s + 8].to(dev))
                for j in LAYERS: acc[j].append(cap[j].reshape(-1, cap[j].shape[-1]).half().cpu())
        out = {j: torch.cat(acc[j], 0) for j in LAYERS}
        for h in handles: h.remove()
        print(f"  {tag}: off={off} hook ok ({worst:.1e}) rows={out[0].shape[0]}", flush=True)
        del M, acc; gc.collect(); torch.cuda.empty_cache()
        return out

    RA = extract(a.rwkv, "RWKV")
    RB = extract(a.pythia, "Pythia")
    n = RA[0].shape[0]

    # center each layer's features (subtract per-dim mean over samples) once, keep on CPU in fp16 (all 32x2 layers
    # on a 16 GiB card was 46 GiB -> OOM). Move ONE A layer + ONE B layer to the GPU at a time inside the loop.
    def prep(R):
        Z = {}
        for j in LAYERS:
            x = R[j].float(); x = x - x.mean(0)                # centered, on CPU
            Z[j] = x.half()
        return Z
    ZA = prep(RA); del RA; gc.collect()
    ZB = prep(RB); del RB; gc.collect()

    # linear CKA(X,Y) = ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F), features already centered
    nrmB = {}
    for k in LAYERS:
        yk = ZB[k].to(dev).float(); nrmB[k] = (yk.T @ yk).norm().item(); del yk
    torch.cuda.empty_cache()

    cka = np.zeros((32, 32)); linr2 = np.zeros((32, 32))
    for j in LAYERS:
        Xj = ZA[j].to(dev).float(); XtX = Xj.T @ Xj            # one A layer on GPU
        nrmA = XtX.norm().item()
        lam = a.ridge * torch.diagonal(XtX).mean()
        XtXi = torch.linalg.inv(XtX + lam * torch.eye(XtX.shape[0], device=dev))
        for k in LAYERS:
            Yk = ZB[k].to(dev).float()                         # one B layer on GPU
            cka[j, k] = ((Xj.T @ Yk).pow(2).sum() / (nrmA * nrmB[k] + 1e-9)).item()
            W = XtXi @ (Xj.T @ Yk)                             # (4096,4096) ridge LS, centered (no bias needed)
            pred = Xj @ W
            ss_res = (pred - Yk).pow(2).sum(); ss_tot = Yk.pow(2).sum()   # centered -> mean is 0
            linr2[j, k] = (1 - ss_res / (ss_tot + 1e-9)).item()
            del Yk
        del Xj, XtX, XtXi; torch.cuda.empty_cache()
        print(f"  row j={j:2d} done  (CKA diag {cka[j,j]:.3f}, argmax_k {int(cka[j].argmax())})", flush=True)

    def summarize(M, name):
        diag = np.diag(M)
        argmax_k = M.argmax(1)
        on_diag = int((argmax_k == np.arange(32)).sum())
        # within +-1 of the diagonal (adjacent layers are legitimately similar)
        near = int((np.abs(argmax_k - np.arange(32)) <= 1).sum())
        off = M.copy(); np.fill_diagonal(off, np.nan)
        print(f"\n[{name}] diagonal mean {diag.mean():.4f} | off-diagonal mean {np.nanmean(off):.4f} | "
              f"argmax on diagonal {on_diag}/32, within +-1 {near}/32", flush=True)
        return {"diag_mean": float(diag.mean()), "offdiag_mean": float(np.nanmean(off)),
                "argmax_on_diag": on_diag, "argmax_within1": near,
                "argmax_k": argmax_k.tolist(), "diag": diag.tolist()}

    res = {"nwin": a.nwin, "win": a.win, "rows": n,
           "cka": summarize(cka, "linear CKA"), "best_linear_R2": summarize(linr2, "best-linear R^2"),
           "cka_matrix": cka.tolist(), "linr2_matrix": linr2.tolist()}
    csv_rows = [
        {
            "source_layer": source_layer,
            "target_layer": target_layer,
            "linear_cka": float(cka[source_layer, target_layer]),
            "best_linear_r2": float(linr2[source_layer, target_layer]),
            "is_diagonal": source_layer == target_layer,
            "n_windows": a.nwin,
            "window_tokens": a.win,
            "sample_rows": n,
        }
        for source_layer in LAYERS
        for target_layer in LAYERS
    ]
    json_path, csv_path = write_bundle(a.out, res, csv_rows)
    print(f"\nwrote {json_path} + {csv_path}\nALLCORR_DONE", flush=True)


if __name__ == "__main__":
    main()
