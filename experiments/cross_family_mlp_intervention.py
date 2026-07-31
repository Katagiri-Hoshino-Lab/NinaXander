"""Matched AB/BA intervention on the same trained adapter.

Each encoder/decoder is Linear(d->z) -> ResBlock -> Linear(z->d), with ResBlock(x)= x + fc2(GELU(fc1(LN(x)))). The
*only* learned non-linearity is that fc2(GELU(fc1(LN(x)))) branch (fc2 is zero-init, so at t=0 the adapter is exactly
linear). We scale that branch by alpha in {0,.25,.5,.75,1}: alpha=1 is the full trained adapter, alpha=0 is the
adapter's OWN exact linear part (same weights, same layers, same rows -- only the residual MLP branch removed). This
removes the confounds in the earlier adapter-vs-separately-fit-linear comparison (objective,
data, regularisation, layer set, evaluation rows are all identical by construction). The latent LayerNorm remains at
alpha=0 (it is a fixed normalisation, not the learned MLP non-linearity).

This program measures per-layer A->B and B->A R² across alpha on the same
held-out residual rows (forward-hook capture). Both paths are evaluated
directly; neither is inferred from the other. Full-set downstream QA
interventions are emitted symmetrically by the canonical A-to-B and B-to-A QA
programs. ``complete_cross_family_mlp_bundle.py`` joins those item-aligned
outcomes with this representation sweep.
"""

import argparse
import os
import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from ninaxander import adapter as adapter_module
from ninaxander.adapter import LatentAdapter, blocks_of
from ninaxander.result_io import write_bundle
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- monkeypatch ResBlock so its non-linear branch is scaled by a global alpha (alpha=0 => identity block) ----
_ALPHA = {"a": 1.0}


def _scaled_forward(s, x):
    return x + _ALPHA["a"] * s.fc2(F.gelu(s.fc1(s.norm(x))))


adapter_module.ResBlock.forward = _scaled_forward


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
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--r2_layers", default="4,10,16,22,28")
    ap.add_argument("--win", type=int, default=112)
    ap.add_argument("--nwin", type=int, default=400)
    ap.add_argument("--dev_a", default="cuda:0")
    ap.add_argument("--dev_b", default="cuda:1")
    ap.add_argument(
        "--out",
        default="artifacts/metrics/raw/cross_family_mlp_representation.json",
    )
    a = ap.parse_args()
    dA, dB = a.dev_a, a.dev_b
    torch.manual_seed(0)
    ALPHAS = [float(x) for x in a.alphas.split(",")]

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
    st = {int(j): [t.to(dB) for t in v] for j, v in c["stats"].items()}
    print(
        f"adapter {os.path.basename(a.adapter)} A->B(ckpt)={c['eval']['A->B']:+.4f}  alphas={ALPHAS}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(a.tok)

    def load_pair():
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
        return A, B

    A, B = load_pair()

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
        offB = (
            1
            if (
                B(ids0.to(dB), output_hidden_states=True).hidden_states[0]
                - B.gpt_neox.embed_in(ids0.to(dB))
            )
            .abs()
            .max()
            .item()
            < 1e-3
            else 0
        )
    print(f"off: RWKV={offA} Pythia={offB}", flush=True)

    def std_a(h, L):
        return (h.float() - st[L][0]) / st[L][1]

    def std_b(h, L):
        return (h.float() - st[L][2]) / st[L][3]

    # ======================= R^2 sweep over alpha =======================
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("tatsu-lab/alpaca", split="train")["text"])
    ids = tok(text[2_000_000 : 2_000_000 + 900_000], return_tensors="pt").input_ids[0]
    EV = ids[: a.nwin * a.win].reshape(a.nwin, a.win)
    R2L = [int(x) for x in a.r2_layers.split(",")]

    def extract(model, tag):
        blk = blocks_of(model)
        RE = (
            model.config.rescale_every
            if (
                hasattr(model, "rwkv")
                and getattr(model.rwkv, "layers_are_rescaled", False)
            )
            else 0
        )
        cap = {}

        def mk(j):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if RE > 0 and (j + 1) % RE == 0:
                    h = h / 2
                cap[j] = h

            return hook

        hs = [blk[j].register_forward_hook(mk(j)) for j in R2L]
        dev = next(model.parameters()).device
        acc = {j: [] for j in R2L}
        with torch.no_grad():
            for s in range(0, EV.shape[0], 8):
                model(EV[s : s + 8].to(dev))
                for j in R2L:
                    acc[j].append(cap[j].reshape(-1, cap[j].shape[-1]).half().cpu())
        for h in hs:
            h.remove()
        return {j: torch.cat(acc[j], 0) for j in R2L}

    with torch.no_grad():
        A(EV[:1].to(dA))  # warm RWKV rescale
    RA = extract(A, "RWKV")
    RB = extract(B, "Pythia")

    def r2(p, t):
        return (1 - (p - t).pow(2).sum() / ((t - t.mean(0)).pow(2).sum() + 1e-9)).item()

    # The R^2 math needs only the extracted residuals and the adapter.
    del A, B
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "\n== Part A: per-layer AB/BA R^2 vs alpha "
        "(alpha=0 removes the adapter's residual-MLP branches) ==",
        flush=True,
    )
    hdr = "  layer " + " ".join(f"a={al:g}".rjust(8) for al in ALPHAS)
    print(hdr)
    print("  " + "-" * (len(hdr)))
    r2_by_direction_alpha = {
        direction: {al: {} for al in ALPHAS} for direction in ("A_to_B", "B_to_A")
    }
    for j in R2L:
        va = RA[j].to(dB).float()
        ea = std_a(va, j)
        eb = std_b(RB[j].to(dB), j)
        row_ab = []
        row_ba = []
        for al in ALPHAS:
            _ALPHA["a"] = al
            with torch.no_grad():
                pred_ab = torch.cat(
                    [
                        m.dB(m.enc_a(ea[s : s + 16384]))
                        for s in range(0, ea.shape[0], 16384)
                    ],
                    0,
                )
                pred_ba = torch.cat(
                    [
                        m.dA(m.enc_b(eb[s : s + 16384]))
                        for s in range(0, eb.shape[0], 16384)
                    ],
                    0,
                )
            value_ab = r2(pred_ab, eb)
            value_ba = r2(pred_ba, ea)
            r2_by_direction_alpha["A_to_B"][al][j] = value_ab
            r2_by_direction_alpha["B_to_A"][al][j] = value_ba
            row_ab.append(value_ab)
            row_ba.append(value_ba)
        print(
            f"  AB L{j:2d} " + " ".join(f"{v:8.4f}" for v in row_ab),
            flush=True,
        )
        print(
            f"  BA L{j:2d} " + " ".join(f"{v:8.4f}" for v in row_ba),
            flush=True,
        )
        del va, ea, eb, pred_ab, pred_ba
        torch.cuda.empty_cache()
    for direction in ("A_to_B", "B_to_A"):
        print(
            f"  {direction} mean "
            + " ".join(
                f"{np.mean(list(r2_by_direction_alpha[direction][al].values())):8.4f}"
                for al in ALPHAS
            ),
            flush=True,
        )
    del RA, RB
    gc.collect()
    torch.cuda.empty_cache()
    for direction in ("A_to_B", "B_to_A"):
        a0 = np.mean(list(r2_by_direction_alpha[direction][0.0].values()))
        a1 = np.mean(list(r2_by_direction_alpha[direction][1.0].values()))
        print(
            f"  => {direction}: alpha=0 {a0:.4f} vs alpha=1 {a1:.4f}",
            flush=True,
        )
    _ALPHA["a"] = 1.0

    result = {
        "schema_version": 2,
        "adapter": os.path.basename(a.adapter),
        "checkpoint_step": c.get("step", ""),
        "directions": ["A_to_B", "B_to_A"],
        "alphas": ALPHAS,
        "r2_layers": R2L,
        "evaluation_windows": a.nwin,
        "window_tokens": a.win,
        "r2_by_direction_alpha_mean": {
            direction: {
                str(al): float(
                    np.mean(list(r2_by_direction_alpha[direction][al].values()))
                )
                for al in ALPHAS
            }
            for direction in ("A_to_B", "B_to_A")
        },
        "r2_by_direction_alpha_layer": {
            direction: {
                str(al): {str(j): r2_by_direction_alpha[direction][al][j] for j in R2L}
                for al in ALPHAS
            }
            for direction in ("A_to_B", "B_to_A")
        },
    }

    csv_rows = [
        {
            "result_type": "representation_r2",
            "direction": direction,
            "task": "held_out_residuals",
            "alpha": alpha,
            "layer": layer,
            "config": "",
            "value": value,
            "n_items": "",
        }
        for direction, alpha_values in r2_by_direction_alpha.items()
        for alpha, layer_values in alpha_values.items()
        for layer, value in layer_values.items()
    ]
    json_path, csv_path = write_bundle(a.out, result, csv_rows)
    print(
        f"\nwrote {json_path} + {csv_path}\nMLP_REPRESENTATION_INTERVENTION_DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
