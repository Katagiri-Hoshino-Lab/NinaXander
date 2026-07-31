"""Per-layer AA/AB/BB/BA metrics and a shuffled-correspondence control.

The reported 32-layer adapter (SNAP_L32_final.pt, step 415000) and both parent
models remain frozen; this is evaluation only. Every layer emits the complete
four-path matrix in the public order AA, AB, BB, BA.

The alignment rho_ctr=0.90 and the two cross-readouts could in principle be a
distributional artifact -- two clouds that happen to overlap -- rather than a
genuine token-by-token correspondence. To rule that out we re-pair: keep A's
row order, apply one fixed random permutation to the B rows (the same
permutation across all layers), and recompute the cross quantities. If the
correspondence is real, AB, BA, and rho_ctr must collapse toward zero under the
shuffle while AA and BB remain unchanged. The real-minus-shuffled gap is the
effect size.

Residuals are captured for all 32 blocks by FORWARD HOOK (the exact convention the reported adapter trained under:
RWKV's periodic rescale mirrored, GPTNeoX's final_layer_norm'd last entry avoided), so the per-layer numbers here
use the same measurement contract as the final checkpoint. Output: a JSON record plus a CSV sidecar under
artifacts/metrics/raw.
"""

import argparse
import os
import sys
import gc
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from ninaxander.adapter import LatentAdapter, blocks_of
from ninaxander.result_io import write_bundle
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    ap.add_argument("--win", type=int, default=112)
    ap.add_argument("--nwin", type=int, default=400)
    ap.add_argument("--eval_cut", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default="artifacts/metrics/raw/four_path_layer_metrics.json"
    )
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)

    c = torch.load(a.adapter, map_location="cpu", weights_only=False)
    cfg = c["cfg"]
    m = (
        LatentAdapter(
            cfg["dA"], cfg["dB"], cfg["z"], blocks=a.blocks, hidden=a.hid, arch="resnet"
        )
        .to(dev)
        .eval()
    )
    m.load_state_dict(c["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    st_ck = {int(j): [t.to(dev) for t in v] for j, v in c["stats"].items()}
    print(
        f"adapter step={c.get('step')}  A->B(ckpt)={c['eval']['A->B']:+.4f}  "
        f"B->A(ckpt)={c['eval'].get('B->A'):+.4f}  st keys {min(st_ck)}..{max(st_ck)}",
        flush=True,
    )

    # SAME held-out window construction the adapter's own evaluate() used
    tok = AutoTokenizer.from_pretrained(a.tok)
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("tatsu-lab/alpaca", split="train")["text"])
    ids = tok(text[a.eval_cut : a.eval_cut + 900_000], return_tensors="pt").input_ids[0]
    EV = ids[: a.nwin * a.win].reshape(a.nwin, a.win)
    assert EV.shape[0] == a.nwin, (
        f"only {EV.shape[0]} windows tokenized; lower --nwin or raise the char slice"
    )
    print(
        f"held-out: {EV.shape[0]} windows x {EV.shape[1]} tokens = {EV.numel()} tokens",
        flush=True,
    )

    LAYERS = list(range(0, 32))

    def extract(path, tag):
        """residual AFTER block j for all j, captured by FORWARD HOOK -- the reported adapter's convention."""
        M = (
            AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.float16, low_cpu_mem_usage=True
            )
            .to(dev)
            .eval()
        )
        for p in M.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            hs0 = M(
                EV[:1].to(dev), output_hidden_states=True
            ).hidden_states  # warms RWKV rescale too
            emb = M.get_input_embeddings()(EV[:1].to(dev))
            off = 1 if (hs0[0] - emb).abs().max().item() < 1e-3 else 0
        L = M.config.num_hidden_layers
        blk = blocks_of(M)
        assert len(blk) == L
        RE = (
            M.config.rescale_every
            if (hasattr(M, "rwkv") and getattr(M.rwkv, "layers_are_rescaled", False))
            else 0
        )
        cap = {}

        def mk(j):
            def hook(mod, inp, out):
                h = (
                    out[0] if isinstance(out, tuple) else out
                )  # RwkvBlock->tuple, GPTNeoXLayer->tensor
                if RE > 0 and (j + 1) % RE == 0:
                    h = h / 2  # mirror HF's post-block halving
                cap[j] = h

            return hook

        handles = [blk[j].register_forward_hook(mk(j)) for j in range(L)]
        # gate: the hook must match hidden_states on every entry that IS a block output
        with torch.no_grad():
            hsg = M(EV[:1].to(dev), output_hidden_states=True).hidden_states
        def relative_error(x, y):
            return (x - y).float().norm().item() / (
                y.float().norm().item() + 1e-9
            )

        worst = max(
            relative_error(cap[j], hsg[j + off])
            for j in range(L)
            if j + off < len(hsg) - 1
        )
        assert worst < 1e-3, (
            f"{tag}: hook disagrees with hidden_states (worst rel={worst:.2e})"
        )
        fn = (
            getattr(getattr(M, "gpt_neox", None), "final_layer_norm", None)
            or M.rwkv.ln_out
        )
        rt = relative_error(fn(cap[L - 1]), hsg[-1])
        assert rt < 2e-3, (
            f"{tag}: last-block capture does not reproduce hs[-1] under final norm ({rt:.2e})"
        )
        acc = {j: [] for j in LAYERS}
        with torch.no_grad():
            for s in range(0, EV.shape[0], 8):
                M(EV[s : s + 8].to(dev))
                for j in LAYERS:
                    acc[j].append(cap[j].reshape(-1, cap[j].shape[-1]).half().cpu())
        out = {j: torch.cat(acc[j], 0) for j in LAYERS}
        for h in handles:
            h.remove()
        print(
            f"  {tag}: off={off}  hook GATE ok (worst blk rel {worst:.1e}, final-norm {rt:.1e})  "
            f"rows/layer={out[0].shape[0]}",
            flush=True,
        )
        del M, acc
        gc.collect()
        torch.cuda.empty_cache()
        return out

    RA = extract(a.rwkv, "RWKV")
    RB = extract(a.pythia, "Pythia")

    n = RA[0].shape[0]
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(a.seed)
    )  # ONE global reshuffle of B rows

    def r2(p, t):
        return (1 - (p - t).pow(2).sum() / ((t - t.mean(0)).pow(2).sum() + 1e-9)).item()

    def rho(x, y):
        xc, yc = x - x.mean(0), y - y.mean(0)
        return ((xc * yc).sum() / (xc.norm() * yc.norm() + 1e-9)).item()

    rows = {}
    for j in LAYERS:
        va = RA[j].to(dev).float()
        vb = RB[j].to(dev).float()
        if j in st_ck:
            mu_a, sd_a, mu_b, sd_b = st_ck[j]
            src = "ckpt"
        else:
            mu_a, sd_a = va.mean(0), va.std(0) + 1e-5
            mu_b, sd_b = vb.mean(0), vb.std(0) + 1e-5
            src = "NEW"
        e = (va - mu_a) / sd_a
        fv = (vb - mu_b) / sd_b
        pj = perm.to(dev)
        with torch.no_grad():
            za, zb = m.enc_a(e), m.enc_b(fv)
            # REAL correspondence
            ab = r2(m.dB(za), fv)
            ba = r2(m.dA(zb), e)
            aa = r2(m.dA(za), e)
            bb = r2(m.dB(zb), fv)
            rc = rho(za, zb)
            # SHUFFLED correspondence: pair A row i with B row perm[i]. Self-maps are untouched by construction.
            fv_s, zb_s, e_s = (
                fv[pj],
                zb[pj],
                e,
            )  # shuffle B side only; A target e stays with A latent
            ab_s = r2(m.dB(za), fv_s)
            ba_s = r2(m.dA(zb_s), e_s)
            rc_s = rho(za, zb_s)
        rows[j] = {
            "stats": src,
            "A->B": ab,
            "B->A": ba,
            "A->A": aa,
            "B->B": bb,
            "rho_ctr": rc,
            "A->B_shuf": ab_s,
            "B->A_shuf": ba_s,
            "rho_ctr_shuf": rc_s,
        }
        del va, vb, e, fv, za, zb
        torch.cuda.empty_cache()

    print(
        f"\n{'L':>3} {'stats':>5} | {'A->B':>7} {'B->A':>7} {'A->A':>7} {'B->B':>7} {'rho_ct':>7} "
        f"|| shuffled: {'A->B':>7} {'B->A':>7} {'rho_ct':>7}"
    )
    print("-" * 88)
    for j in LAYERS:
        r = rows[j]
        print(
            f"{j:3d} {r['stats']:>5} | {r['A->B']:7.4f} {r['B->A']:7.4f} {r['A->A']:7.4f} {r['B->B']:7.4f} "
            f"{r['rho_ctr']:7.4f} || {r['A->B_shuf']:7.4f} {r['B->A_shuf']:7.4f} {r['rho_ctr_shuf']:7.4f}",
            flush=True,
        )

    def agg(js, k):
        return sum(rows[j][k] for j in js) / len(js)

    A31 = list(range(1, 32))
    A32 = list(range(0, 32))
    print("-" * 88)
    for name, js in (("layers 1..31 (trained)", A31), ("layers 0..31 (all)", A32)):
        print(
            f"MEAN {name:>24}: A->B {agg(js, 'A->B'):.4f}  B->A {agg(js, 'B->A'):.4f}  "
            f"A->A {agg(js, 'A->A'):.4f}  B->B {agg(js, 'B->B'):.4f}  rho_ctr {agg(js, 'rho_ctr'):.4f}",
            flush=True,
        )
    print(
        f"SHUFFLED CONTROL (mean 1..31): A->B {agg(A31, 'A->B_shuf'):+.4f}  B->A {agg(A31, 'B->A_shuf'):+.4f}  "
        f"rho_ctr {agg(A31, 'rho_ctr_shuf'):+.4f}   (real: A->B {agg(A31, 'A->B'):.4f}, rho_ctr {agg(A31, 'rho_ctr'):.4f})",
        flush=True,
    )

    metric_names = [
        "A->B",
        "B->A",
        "A->A",
        "B->B",
        "rho_ctr",
        "A->B_shuf",
        "B->A_shuf",
        "rho_ctr_shuf",
    ]
    payload = {
        "adapter": os.path.basename(a.adapter),
        "step": c.get("step"),
        "nwin": a.nwin,
        "win": a.win,
        "seed": a.seed,
        "rows": rows,
        "mean_1_31": {k: agg(A31, k) for k in metric_names},
        "mean_0_31": {k: agg(A32, k) for k in metric_names},
    }
    csv_rows = [
        {
            "row_type": "layer",
            "layer": layer,
            "statistics": rows[layer]["stats"],
            **{metric: rows[layer][metric] for metric in metric_names},
            "n_windows": a.nwin,
            "window_tokens": a.win,
            "seed": a.seed,
        }
        for layer in LAYERS
    ]
    for label, values in (
        ("mean_1_31", payload["mean_1_31"]),
        ("mean_0_31", payload["mean_0_31"]),
    ):
        csv_rows.append(
            {
                "row_type": label,
                "layer": "",
                "statistics": "mixed",
                **values,
                "n_windows": a.nwin,
                "window_tokens": a.win,
                "seed": a.seed,
            }
        )
    json_path, csv_path = write_bundle(a.out, payload, csv_rows)
    print(f"\nwrote {json_path} + {csv_path}\nBTA_SHUF_DONE", flush=True)


if __name__ == "__main__":
    main()
