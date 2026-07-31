"""Serving-cost microbenchmark: measured peak GPU memory, prefill latency, and decode throughput
for the parents and for the chimera at each switch. Read-only: frozen parents + frozen adapter, no training.

The chimera currently loads both parents in full and
runs them model-parallel (RWKV on dev_a, Tulu-Pythia + adapter on dev_b), so the *weight* footprint does not shrink
with the switch -- what the switch changes is the KV cache (fewer Transformer blocks -> less K/V stored), which
shows up in peak memory at long context. Decode here is the current unoptimised path: each new token re-runs the
whole growing sequence through A, one translation, then B (no cross-switch KV reuse, and HF's RWKV recurrence is
sequential), so the decode throughput is a floor for this implementation, not an architectural limit. All numbers
are labelled as such.

Reported per config: peak allocated bytes on each device (torch.cuda.max_memory_allocated), prefill wall-clock at
ctx P (median of a few forwards after warmup), and decode tokens/sec over a short greedy run at ctx 512.
"""

import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from ninaxander.adapter import LatentAdapter
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
    ap.add_argument("--switches", default="4,8,16,24")
    ap.add_argument(
        "--ctx",
        default="512,2048",
        help="prompt lengths for prefill + peak-memory measurement",
    )
    ap.add_argument("--decode_ctx", type=int, default=512)
    ap.add_argument("--decode_tok", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dev_a", default="cuda:0")
    ap.add_argument("--dev_b", default="cuda:1")
    ap.add_argument("--out", default="artifacts/metrics/raw/a_to_b_serving_bench.json")
    a = ap.parse_args()
    dA, dB = a.dev_a, a.dev_b
    torch.manual_seed(0)
    SW = [int(x) for x in a.switches.split(",")]
    CTX = [int(x) for x in a.ctx.split(",")]
    MiB = 1024 * 1024

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
    w_a = torch.cuda.memory_allocated(dA) / MiB
    w_b = torch.cuda.memory_allocated(dB) / MiB
    print(
        f"weights resident: RWKV {w_a:.0f} MiB on {dA} | Pythia+adapter {w_b:.0f} MiB on {dB}",
        flush=True,
    )

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

    _inject = {"h": None}

    def prehook(mod, args, kwargs):
        if _inject["h"] is None:
            return None
        if len(args):
            return (_inject["h"],) + args[
                1:
            ], kwargs  # hidden_states passed positionally
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = _inject["h"]
        return args, kwargs

    def pythia_inject(ids_b, h):
        _inject["h"] = h
        try:
            return B(ids_b).logits
        finally:
            _inject["h"] = None

    def translate(hA, L):
        return (
            m.dB(m.enc_a((hA.to(dB).float() - st[L][0]) / st[L][1])) * st[L][3]
            + st[L][2]
        ).to(B.dtype)

    def pure_pythia(ids):
        return B(ids.to(dB)).logits

    def pure_rwkv(ids):
        return A(ids.to(dA)).logits

    def chimera(ids, L):
        hA = A(ids.to(dA), output_hidden_states=True).hidden_states[L + offA]
        hB = translate(hA, L)
        hnd = Blay[L + offB].register_forward_pre_hook(prehook, with_kwargs=True)
        try:
            return pythia_inject(ids.to(dB), hB)
        finally:
            hnd.remove()

    CONFIGS = [("pure-Pythia", None), ("pure-RWKV", None)] + [
        (f"A_to_B@{L}", L) for L in SW
    ]

    def run(name, L, ids):
        if name == "pure-Pythia":
            return pure_pythia(ids)
        if name == "pure-RWKV":
            return pure_rwkv(ids)
        return chimera(ids, L)

    def sync():
        torch.cuda.synchronize(dA)
        torch.cuda.synchronize(dB)

    # Gate every published switch: injecting B's own residual must reproduce B.
    parent_gate_errors = {}
    with torch.no_grad():
        hf0 = B(ids0.to(dB)).logits
        hsB = B(ids0.to(dB), output_hidden_states=True).hidden_states
        for L in SW:
            index = L + offB
            hnd = Blay[index].register_forward_pre_hook(
                prehook,
                with_kwargs=True,
            )
            rel = (
                pythia_inject(ids0.to(dB), hsB[index]) - hf0
            ).float().norm().item() / (hf0.float().norm().item() + 1e-9)
            hnd.remove()
            if rel >= 1e-4:
                raise AssertionError(
                    f"parent-reproduction gate failed at L={L}: {rel:.3e}"
                )
            parent_gate_errors[L] = rel
    print(
        "parent-reproduction gate passed "
        + " ".join(
            f"L{layer}={error:.1e}" for layer, error in parent_gate_errors.items()
        ),
        flush=True,
    )

    results = {
        "schema_version": 1,
        "direction": "A_to_B",
        "source_model": "RWKV-Raven-7B",
        "target_model": "Tulu-Pythia-6.9B",
        "implementation": "unpruned_full_parent_reforward",
        "adapter": os.path.basename(a.adapter),
        "checkpoint_step": c.get("step", ""),
        "checkpoint_a_to_b_r2": c["eval"]["A->B"],
        "switches": SW,
        "contexts": CTX,
        "decode_context": a.decode_ctx,
        "decode_tokens": a.decode_tok,
        "repetitions": a.reps,
        "parent_reproduction_relative_error": {
            str(layer): error for layer, error in parent_gate_errors.items()
        },
        "weights_MiB": {
            "rwkv_devA": w_a,
            "pythia_adapter_devB": w_b,
        },
        "prefill": {},
        "decode": {},
    }

    # ---- prefill latency + peak memory, per context length ----
    for P in CTX:
        ids = tok("A " * (P + 8), return_tensors="pt").input_ids[:, :P]
        print(
            f"=== ctx {P} | prefill (median of {a.reps}) + peak GPU memory ===",
            flush=True,
        )
        print(
            f"{'config':>14} | {'prefill ms':>11} | {'peak devA MiB':>13} | {'peak devB MiB':>13}",
            flush=True,
        )
        results["prefill"][P] = {}
        for name, L in CONFIGS:
            try:
                with torch.no_grad():
                    run(name, L, ids)
                    sync()  # warmup
                torch.cuda.reset_peak_memory_stats(dA)
                torch.cuda.reset_peak_memory_stats(dB)
                ts = []
                for _ in range(a.reps):
                    sync()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        run(name, L, ids)
                    sync()
                    ts.append((time.perf_counter() - t0) * 1e3)
                ts.sort()
                ms = ts[len(ts) // 2]
                pa = torch.cuda.max_memory_allocated(dA) / MiB
                pb = torch.cuda.max_memory_allocated(dB) / MiB
                print(f"{name:>14} | {ms:11.1f} | {pa:13.0f} | {pb:13.0f}", flush=True)
                results["prefill"][P][name] = {
                    "ms": ms,
                    "peak_devA_MiB": pa,
                    "peak_devB_MiB": pb,
                }
            except RuntimeError as ex:
                torch.cuda.empty_cache()
                msg = (
                    "OOM"
                    if "out of memory" in str(ex).lower()
                    else f"ERR:{str(ex)[:40]}"
                )
                print(f"{name:>14} | {msg:>11} | {'-':>13} | {'-':>13}", flush=True)
                results["prefill"][P][name] = {"error": msg}

    # ---- decode throughput (unoptimised re-forward path) at ctx decode_ctx ----
    P = a.decode_ctx
    print(
        f"\n=== decode throughput | ctx {P}, generate {a.decode_tok} tokens (greedy re-forward; floor) ===",
        flush=True,
    )
    print(f"{'config':>14} | {'tok/s':>8} | {'ms/token':>9}", flush=True)
    for name, L in CONFIGS:
        ids = tok("A " * (P + 8), return_tensors="pt").input_ids[:, :P]
        try:
            with torch.no_grad():
                run(name, L, ids)
                sync()  # warmup
            sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(a.decode_tok):
                    nxt = run(name, L, ids)[:, -1].argmax(-1, keepdim=True).cpu()
                    ids = torch.cat([ids, nxt], 1)
            sync()
            dt = time.perf_counter() - t0
            tps = a.decode_tok / dt
            print(
                f"{name:>14} | {tps:8.2f} | {dt / a.decode_tok * 1e3:9.1f}", flush=True
            )
            results["decode"][name] = {
                "tok_per_s": tps,
                "ms_per_token": dt / a.decode_tok * 1e3,
            }
        except RuntimeError as ex:
            torch.cuda.empty_cache()
            msg = "OOM" if "out of memory" in str(ex).lower() else f"ERR:{str(ex)[:40]}"
            print(f"{name:>14} | {msg:>8} | {'-':>9}", flush=True)
            results["decode"][name] = {"error": msg}

    csv_rows = []
    for context, configurations in results["prefill"].items():
        for config, values in configurations.items():
            csv_rows.append(
                {
                    "direction": "A_to_B",
                    "benchmark": "unpruned",
                    "phase": "prefill",
                    "context_tokens": context,
                    "config": config,
                    **values,
                }
            )
    for config, values in results["decode"].items():
        csv_rows.append(
            {
                "direction": "A_to_B",
                "benchmark": "unpruned",
                "phase": "decode",
                "context_tokens": a.decode_ctx,
                "config": config,
                **values,
            }
        )
    json_path, csv_path = write_bundle(a.out, results, csv_rows)
    print(f"\nwrote {json_path} + {csv_path}\nSERVING_DONE", flush=True)


if __name__ == "__main__":
    main()
