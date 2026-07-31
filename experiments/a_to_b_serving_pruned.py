"""Physically pruned AB serving path that frees unused blocks so the memory saving is realised
end-to-end (not merely a per-token KV figure). Read-only; frozen models + frozen adapter.

The A_to_B@L path uses RWKV blocks 0..L and Pythia blocks L+1..31 (plus RWKV embeddings, the adapter, Pythia
final_layer_norm and the LM head). Everything else -- RWKV blocks L+1..31, RWKV ln_out/head, Pythia embed_in,
Pythia blocks 0..L -- is dead weight. We free it correctness-preservingly:

  * Replace the dead blocks with parameter-free PASSTHROUGH modules and `del` the originals, then empty_cache. The
    front Pythia blocks' outputs are overwritten by the injection pre-hook at block L+1, so replacing them with
    identity changes nothing; the RWKV blocks after L never affect hidden_states[L].
  * Feed Pythia `inputs_embeds` (zeros) instead of input_ids so embed_in is never called and can be freed.

This reuses the validated A-to-B execution (HF runs blocks L+1..31 with its own rotary/mask; the pre-hook only
replaces block L+1's input), so the parent-repro gate (inject B's own residual -> logits unchanged, rel<1e-4) still
certifies it. We then measure resident weight bytes, prefill peak memory and prefill latency for: the pruned
A_to_B@L, pure RWKV, pure Pythia, and an equal-Transformer-depth EARLY-EXIT Pythia (first 31-L blocks). The point
is the weight footprint: keeping (L+1) RNN + (31-L) Transformer blocks is ~one model's worth, ~half of loading both.
"""

import argparse
import os
import sys
import time
import gc
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from ninaxander.adapter import LatentAdapter
from ninaxander.result_io import write_bundle
from transformers import AutoModelForCausalLM, AutoTokenizer

MiB = 1024 * 1024
GATE_TOLERANCE = 1e-4


def weight_bytes(mod, dev):
    return sum(
        p.numel() * p.element_size()
        for p in mod.parameters()
        if p.device == torch.device(dev)
    )


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
    ap.add_argument("--ctx", default="512,2048")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dev_a", default="cuda:0")
    ap.add_argument("--dev_b", default="cuda:1")
    ap.add_argument("--out", default="artifacts/metrics/raw/a_to_b_serving_pruned.json")
    a = ap.parse_args()
    dA, dB = a.dev_a, a.dev_b
    torch.manual_seed(0)
    SW = [int(x) for x in a.switches.split(",")]
    CTX = [int(x) for x in a.ctx.split(",")]
    tok = AutoTokenizer.from_pretrained(a.tok)

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

    def sync():
        torch.cuda.synchronize(dA)
        torch.cuda.synchronize(dB)

    def std_a(h, L):
        return (h.float() - st[L][0]) / st[L][1]

    def translate(hA, L):
        return (m.dB(m.enc_a(std_a(hA.to(dB), L))) * st[L][3] + st[L][2]).to(
            torch.float16
        )

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
    print(f"off RWKV={offA} Pythia={offB}", flush=True)

    _inject = {"h": None}

    def prehook(mod, args, kwargs):
        if _inject["h"] is None:
            return None
        if len(args):
            return (_inject["h"],) + args[1:], kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = _inject["h"]
        return args, kwargs

    def chim(A, B, ids, L, inject_idx):
        """RWKV to block L, translate, Pythia from block L+1 via inputs_embeds + injection at layer[inject_idx]."""
        hA = A(ids.to(dA), output_hidden_states=True).hidden_states[L + offA]
        hB = translate(hA, L)
        emb = torch.zeros(
            ids.shape[0],
            ids.shape[1],
            B.config.hidden_size,
            dtype=torch.float16,
            device=dB,
        )
        _inject["h"] = hB
        hnd = B.gpt_neox.layers[inject_idx].register_forward_pre_hook(
            prehook, with_kwargs=True
        )
        try:
            return B(inputs_embeds=emb).logits
        finally:
            hnd.remove()
            _inject["h"] = None

    def prune_chimera(A, B, L):
        """free the dead blocks by TRUNCATING the block lists. RWKV keeps 0..L; Pythia keeps L+1..31. After this the
        Pythia injection index becomes 0 (the first kept block = old block L+1). Truncation (not a passthrough) keeps
        real modules so RWKV's rescale bookkeeping still finds block.attention, and frees the removed weights."""
        A.rwkv.blocks = nn.ModuleList(
            [A.rwkv.blocks[j] for j in range(L + 1)]
        )  # keep 0..L
        A.head = nn.Identity()
        A.rwkv.ln_out = nn.Identity()
        B.gpt_neox.layers = nn.ModuleList(
            [B.gpt_neox.layers[j] for j in range(L + 1, 32)]
        )  # keep L+1..31
        B.gpt_neox.embed_in = nn.Identity()  # unused (inputs_embeds path)
        gc.collect()
        torch.cuda.empty_cache()

    # ---- GATE on the unpruned model: the inputs_embeds injection path == the trusted input_ids chimera ----
    parent_gate_errors = {}
    with torch.no_grad():
        for L in SW:
            hf_ref = B(ids0.to(dB)).logits
            hsB0 = B(ids0.to(dB), output_hidden_states=True).hidden_states
            _inject["h"] = hsB0[L + offB].to(dB)
            hnd = B.gpt_neox.layers[L + offB].register_forward_pre_hook(
                prehook, with_kwargs=True
            )
            emb = torch.zeros(
                1, ids0.shape[1], B.config.hidden_size, dtype=torch.float16, device=dB
            )
            rep = B(inputs_embeds=emb).logits
            hnd.remove()
            _inject["h"] = None
            rel = (rep - hf_ref).float().norm().item() / (
                hf_ref.float().norm().item() + 1e-9
            )
            assert rel < GATE_TOLERANCE, (
                f"GATE(inputs_embeds parent-repro) L={L} rel={rel:.2e}"
            )
            parent_gate_errors[L] = rel
    print(
        f"GATE ok: inputs_embeds injection reproduces pure Pythia (rel<1e-4) for {SW}\n",
        flush=True,
    )
    del A, B
    gc.collect()
    torch.cuda.empty_cache()

    def prof(fwd, ctxs):
        """resident weight bytes are read by caller; here: prefill peak mem + latency per ctx."""
        out = {}
        for P in ctxs:
            ids = tok("A " * (P + 8), return_tensors="pt").input_ids[:, :P]
            try:
                with torch.no_grad():
                    fwd(ids)
                    sync()
                torch.cuda.reset_peak_memory_stats(dA)
                torch.cuda.reset_peak_memory_stats(dB)
                ts = []
                for _ in range(a.reps):
                    sync()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        fwd(ids)
                    sync()
                    ts.append((time.perf_counter() - t0) * 1e3)
                ts.sort()
                out[P] = {
                    "ms": ts[len(ts) // 2],
                    "peak_devA_MiB": torch.cuda.max_memory_allocated(dA) / MiB,
                    "peak_devB_MiB": torch.cuda.max_memory_allocated(dB) / MiB,
                }
            except RuntimeError as ex:
                torch.cuda.empty_cache()
                out[P] = {
                    "error": "OOM"
                    if "out of memory" in str(ex).lower()
                    else str(ex)[:50]
                }
        return out

    results = {
        "schema_version": 1,
        "direction": "A_to_B",
        "source_model": "RWKV-Raven-7B",
        "target_model": "Tulu-Pythia-6.9B",
        "implementation": "physically_pruned",
        "adapter": os.path.basename(a.adapter),
        "checkpoint_step": c.get("step", ""),
        "checkpoint_a_to_b_r2": c["eval"]["A->B"],
        "switches": SW,
        "ctx": CTX,
        "repetitions": a.reps,
        "gate_tolerance": GATE_TOLERANCE,
        "parent_reproduction_relative_error": {
            str(layer): error for layer, error in parent_gate_errors.items()
        },
        "configs": {},
    }

    # ---- baselines: pure Pythia, pure RWKV, early-exit Pythia (first 31-L blocks) ----
    A, B = load_pair()
    wA = weight_bytes(A, dA) / MiB
    wB = weight_bytes(B, dB) / MiB
    print(
        f"unpruned resident weights: RWKV {wA:.0f} MiB (devA) | Pythia+adapter {wB + weight_bytes(m, dB) / MiB:.0f} MiB (devB)\n",
        flush=True,
    )
    results["configs"]["pure-Pythia"] = {
        "weight_MiB_devB": wB,
        "weight_MiB_devA": 0.0,
        "prefill": prof(
            lambda ids, current=B: current(ids.to(dB)).logits,
            CTX,
        ),
    }
    results["configs"]["pure-RWKV"] = {
        "weight_MiB_devA": wA,
        "weight_MiB_devB": 0.0,
        "prefill": prof(
            lambda ids, current=A: current(ids.to(dA)).logits,
            CTX,
        ),
    }
    print("pure-Pythia / pure-RWKV profiled", flush=True)
    del A, B
    gc.collect()
    torch.cuda.empty_cache()  # free the pair before loading early-exit Pythias on devB
    # early-exit Pythia at K = 31-L blocks (equal Transformer depth / KV to chimera@L)
    for L in SW:
        K = 31 - L
        Bx = (
            AutoModelForCausalLM.from_pretrained(
                a.pythia, dtype=torch.float16, low_cpu_mem_usage=True
            )
            .to(dB)
            .eval()
        )
        for p in Bx.parameters():
            p.requires_grad_(False)
        Bx.gpt_neox.layers = nn.ModuleList(
            [Bx.gpt_neox.layers[j] for j in range(K)]
        )  # keep first K
        gc.collect()
        torch.cuda.empty_cache()
        results["configs"][f"B_early_exit@{K}"] = {
            "blocks": K,
            "weight_MiB_devB": weight_bytes(Bx, dB) / MiB,
            "prefill": prof(
                lambda ids, current=Bx: current(ids.to(dB)).logits,
                CTX,
            ),
        }
        print(
            f"  B_early_exit@{K} (=31-{L}) profiled, weights {weight_bytes(Bx, dB) / MiB:.0f} MiB",
            flush=True,
        )
        del Bx
        gc.collect()
        torch.cuda.empty_cache()

    # ---- pruned chimera per switch (reload each; truncation is destructive; gate each against the unpruned path) ----
    for L in SW:
        A, B = load_pair()
        with torch.no_grad():
            ref = chim(
                A, B, ids0, L, L + offB
            )  # trusted path on the UNPRUNED pair (inject at L+offB)
        prune_chimera(A, B, L)  # truncate -> Pythia inject index becomes 0
        with torch.no_grad():
            got = chim(A, B, ids0, L, 0)
        rel = (got - ref).float().norm().item() / (ref.float().norm().item() + 1e-9)
        assert rel < GATE_TOLERANCE, (
            f"PRUNED A_to_B@{L} disagrees with unpruned (rel={rel:.2e})"
        )
        wa = weight_bytes(A, dA) / MiB
        wb = weight_bytes(B, dB) / MiB + weight_bytes(m, dB) / MiB
        pr = prof(
            lambda ids, current_a=A, current_b=B, current_layer=L: chim(
                current_a,
                current_b,
                ids,
                current_layer,
                0,
            ),
            CTX,
        )
        results["configs"][f"pruned-A_to_B@{L}"] = {
            "rwkv_blocks": L + 1,
            "pythia_blocks": 31 - L,
            "parent_gate_rel": parent_gate_errors[L],
            "gate_rel": rel,
            "weight_MiB_devA": wa,
            "weight_MiB_devB": wb,
            "weight_MiB_total": wa + wb,
            "prefill": pr,
        }
        print(
            f"pruned-A_to_B@{L}: gate rel={rel:.1e} | RWKV {L + 1} blk {wa:.0f} MiB (devA) + Pythia {31 - L} blk+adapter "
            f"{wb:.0f} MiB (devB) = {wa + wb:.0f} MiB total",
            flush=True,
        )
        del A, B
        gc.collect()
        torch.cuda.empty_cache()

    csv_rows = []
    for config, values in results["configs"].items():
        base = {
            "direction": "A_to_B",
            "benchmark": "pruned",
            "phase": "prefill",
            "config": config,
            "rwkv_blocks": values.get("rwkv_blocks", 0),
            "pythia_blocks": values.get(
                "pythia_blocks",
                values.get("blocks", 32 if config == "pure-Pythia" else 0),
            ),
            "weight_mib_dev_a": values.get("weight_MiB_devA", 0),
            "weight_mib_dev_b": values.get("weight_MiB_devB", 0),
            "weight_mib_total": values.get(
                "weight_MiB_total",
                values.get("weight_MiB_devA", 0) + values.get("weight_MiB_devB", 0),
            ),
            "parent_gate_relative_error": values.get("parent_gate_rel", ""),
            "source_front_gate_relative_error": "",
            "target_suffix_gate_relative_error": "",
            "gate_relative_error": values.get("gate_rel", ""),
        }
        for context, profile in values["prefill"].items():
            csv_rows.append({**base, "context_tokens": context, **profile})
    json_path, csv_path = write_bundle(a.out, results, csv_rows)
    # summary table
    print(
        f"\n{'config':>22} | {'weights MiB (A+B)':>18} | {'peak MiB @2048 (A/B)':>22}",
        flush=True,
    )
    for k, v in results["configs"].items():
        wt = v.get(
            "weight_MiB_total",
            v.get("weight_MiB_devA", 0) + v.get("weight_MiB_devB", 0),
        )
        pk = v["prefill"].get(2048, v["prefill"].get(CTX[-1], {}))
        pkstr = (
            f"{pk.get('peak_devA_MiB', 0):.0f}/{pk.get('peak_devB_MiB', 0):.0f}"
            if "peak_devB_MiB" in pk
            else pk.get("error", "-")
        )
        print(f"{k:>22} | {wt:18.0f} | {pkstr:>22}", flush=True)
    print(f"\nwrote {json_path} + {csv_path}\nSERVING_PRUNED_DONE", flush=True)


if __name__ == "__main__":
    main()
