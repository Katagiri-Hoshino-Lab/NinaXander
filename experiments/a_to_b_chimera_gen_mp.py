"""Model-parallel greedy generation for the current A-to-B chimera.

Same trusted, gated path as the canonical QA evaluation -- RWKV on devA, Pythia+adapter+stats on devB, HF runs its own
forward with a pre-hook replacing the switched layer's input -- but greedy-decodes text instead of scoring options.
The final public checkpoint is run at one switch and both parents are decoded
once for reference. The parent-repro gate (inject B's own residual -> logits
unchanged) exercises the cross-device path before any generation.
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from ninaxander.adapter import LatentAdapter
from ninaxander.result_io import write_csv
from transformers import AutoModelForCausalLM, AutoTokenizer


def rel(a, b):
    return (a - b).float().norm().item() / (b.float().norm().item() + 1e-9)


def load_adapter(path, dB, blocks, hid):
    c = torch.load(path, map_location="cpu", weights_only=False)
    cfg = c["cfg"]
    m = (
        LatentAdapter(
            cfg["dA"], cfg["dB"], cfg["z"], blocks=blocks, hidden=hid, arch="resnet"
        )
        .to(dB)
        .eval()
    )
    m.load_state_dict(c["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    st = {int(j): [t.to(dB) for t in v] for j, v in c["stats"].items()}
    return m, st, c["eval"]["A->B"], c.get("step", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter",
        default="artifacts/checkpoints/z4096_L32/SNAP_L32_final.pt",
    )
    ap.add_argument("--rwkv", default="models/rwkv-raven-7b-hf-fp16")
    ap.add_argument("--pythia", default="models/tulu-pythia69-fp16")
    ap.add_argument("--tok", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--hid", type=int, default=4096)
    ap.add_argument("--switch", type=int, default=4)
    ap.add_argument("--ntok", type=int, default=40)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Reproducibility contract: only 0 is supported (deterministic argmax).",
    )
    ap.add_argument("--dev_a", default="cuda:0")
    ap.add_argument("--dev_b", default="cuda:1")
    ap.add_argument(
        "--out", default="artifacts/metrics/raw/a_to_b_generation_samples.csv"
    )
    a = ap.parse_args()
    if a.temperature != 0.0:
        raise ValueError(
            "This reported generation uses temperature=0 only "
            "(deterministic greedy argmax)."
        )
    dA, dB = a.dev_a, a.dev_b
    torch.manual_seed(0)
    L = a.switch

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
    adapter, stats, a_to_b, checkpoint_step = load_adapter(
        a.adapter,
        dB,
        a.blocks,
        a.hid,
    )
    print(
        f"{os.path.basename(a.adapter)} A->B={a_to_b:+.4f} | "
        f"switch={L} devA={dA} devB={dB}",
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
        hsB0 = B(ids0.to(dB), output_hidden_states=True).hidden_states
        offB = (
            1
            if (hsB0[0] - B.gpt_neox.embed_in(ids0.to(dB))).abs().max().item() < 1e-3
            else 0
        )
    idxB = L + offB
    print(f"off: RWKV={offA} Pythia={offB}  inject at Blay[{idxB}]", flush=True)

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

    # GATE: parent-repro + cross-device round-trip (same as the benchmark's gate)
    with torch.no_grad():
        hf0 = B(ids0.to(dB)).logits
        hnd = Blay[idxB].register_forward_pre_hook(prehook, with_kwargs=True)
        r = rel(pythia_inject(ids0.to(dB), hsB0[idxB].to(dA).to(dB)), hf0)
        hnd.remove()
    assert r < 1e-4, f"GATE FAILED: rel={r:.2e}"
    print(f"GATE passed (parent-repro + cross-device rel={r:.1e})\n", flush=True)

    def translate(hA, m, st):
        return (
            m.dB(m.enc_a((hA.to(dB).float() - st[L][0]) / st[L][1])) * st[L][3]
            + st[L][2]
        ).to(B.dtype)

    @torch.no_grad()
    def greedy(logits_fn, prompt, use_a):
        ids = tok(prompt, return_tensors="pt").input_ids
        p0 = ids.shape[1]
        for _ in range(a.ntok):
            nxt = logits_fn(ids)[:, -1].argmax(-1, keepdim=True).cpu()
            ids = torch.cat([ids, nxt], 1)
            if nxt.item() == tok.eos_token_id:
                break
        return tok.decode(ids[0, p0:], skip_special_tokens=True)

    @torch.no_grad()
    def chimera_logits(ids, m, st):
        hA = A(ids.to(dA), output_hidden_states=True).hidden_states[L + offA]
        hB = translate(hA, m, st)
        hnd = Blay[idxB].register_forward_pre_hook(prehook, with_kwargs=True)
        try:
            return pythia_inject(ids.to(dB), hB)
        finally:
            hnd.remove()

    prompts = [
        "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
        "The three primary colors are",
        "Question: Water is made of hydrogen and\nAnswer:",
        "### Instruction:\nExplain what a black hole is in one sentence.\n\n### Response:\n",
    ]
    generation_metadata = {
        "decoding": "greedy_argmax",
        "temperature": a.temperature,
        "max_new_tokens": a.ntok,
        "seed": 0,
    }
    rows = []
    for prompt_index, pr in enumerate(prompts):
        py = greedy(lambda i: B(i.to(dB)).logits, pr, False)
        rw = greedy(lambda i: A(i.to(dA)).logits, pr, True)
        chimera = greedy(
            lambda i: chimera_logits(i, adapter, stats),
            pr,
            True,
        )
        print(f"PROMPT {pr!r}")
        print(f"  [pure Pythia] {py!r}")
        print(f"  [pure RWKV  ] {rw!r}")
        print(f"  [A_to_B@{L} {a_to_b:.3f}] {chimera!r}\n", flush=True)
        rows.extend(
            [
                {
                    "direction": "B_parent",
                    "source_model": "RWKV-Raven-7B",
                    "target_model": "Tulu-Pythia-6.9B",
                    "prompt_id": prompt_index,
                    "prompt": pr,
                    "config": "pure-Pythia",
                    "switch_layer": "",
                    "adapter_cross_r2": "",
                    "checkpoint_step": "",
                    "parent_gate_relative_error": "",
                    "direct_gate_relative_error": "",
                    **generation_metadata,
                    "output": py,
                },
                {
                    "direction": "A_parent",
                    "source_model": "RWKV-Raven-7B",
                    "target_model": "Tulu-Pythia-6.9B",
                    "prompt_id": prompt_index,
                    "prompt": pr,
                    "config": "pure-RWKV",
                    "switch_layer": "",
                    "adapter_cross_r2": "",
                    "checkpoint_step": "",
                    "parent_gate_relative_error": "",
                    "direct_gate_relative_error": "",
                    **generation_metadata,
                    "output": rw,
                },
                {
                    "direction": "A_to_B",
                    "source_model": "RWKV-Raven-7B",
                    "target_model": "Tulu-Pythia-6.9B",
                    "prompt_id": prompt_index,
                    "prompt": pr,
                    "config": f"A_to_B@{L}",
                    "switch_layer": L,
                    "adapter_cross_r2": a_to_b,
                    "checkpoint_step": checkpoint_step,
                    "parent_gate_relative_error": r,
                    "direct_gate_relative_error": "",
                    **generation_metadata,
                    "output": chimera,
                },
            ]
        )
    csv_path = write_csv(a.out, rows)
    print(f"wrote {csv_path}\nGENMP_DONE", flush=True)


if __name__ == "__main__":
    main()
