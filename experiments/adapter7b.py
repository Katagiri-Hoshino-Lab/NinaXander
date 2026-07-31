"""Build THE 7B shared-latent adapter: RWKV-Raven-7B (pure RNN, instruct) <-> Pythia-6.9b (Transformer).

Scope: create the adapter only. One encoder+decoder per model into ONE latent shared by all layers, trained with the
established recipe (non-affine LayerNorm on z, noise sigma=0.4 before decoding, lambda=1). The two 7B models won't
co-load on a 32GB V100, so residuals for ALL layers are extracted one model at a time, cached to /dev/shm, then the
adapter is trained from the cache. Held-out A->B/B->A/A->A/B->B are logged as a sanity signal; the deliverable is the
saved adapter checkpoint.
"""
import argparse, os, sys, gc, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np
from ninaxander.adapter import LatentAdapter, blocks_of, r2
from ninaxander.result_io import append_csv_row
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- DDP (opt-in: only active under torchrun, so the single-GPU path is untouched) -------------------------------
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
DDP_ON = "RANK" in os.environ and "WORLD_SIZE" in os.environ
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))


def is_main():
    return RANK == 0


def rprint(*args, **kw):
    """Only rank 0 prints, so the log stays readable."""
    if is_main(): print(*args, **kw)


def extract_model(model_id, tr_ids, ev_ids, layers, dev, cdir, tag, bs=8):
    """Load the model ONCE, cache the RAW output of block j for BOTH train and eval windows, then free it.

    Captured by a forward hook on each block, NOT read out of `hidden_states`. hidden_states is not a uniform
    quantity across families and its last entry is not even a block output:
      * GPTNeoX prepends the embeddings (so block j sits at index j+1) and its LAST entry is
        final_layer_norm(block_{L-1}) -- the raw output of the last block is not exposed at all, and the norm is
        not invertible because it discards the per-token mean and scale.
      * RWKV appends inside the block loop (block j sits at index j) and adds ln_out as a SEPARATE final entry, so
        its last block output IS exposed.
    Reading hs[j+off] therefore pairs RWKV's raw final-block output against
    Pythia's normalised one. Hooking the blocks makes all L layers the same kind
    of quantity for both families and removes the off-by-one entirely.

    Gated below: the hook must reproduce hs[j+off] for every layer where hidden_states IS a block output, and for
    the last layer it must differ from hs[-1] while satisfying final_norm(captured) == hs[-1].
    """
    done = os.path.join(cdir, f"{tag}.done")
    # The marker records HOW the cache was built, not just that it was. A cache written by the old
    # A cache from another extraction convention is not interchangeable with a
    # hook-captured 32-layer one.
    sig = f"hook32:{min(layers)}-{max(layers)}:{len(layers)}"
    ld = lambda pfx: {j: np.load(os.path.join(cdir, f"{pfx}_L{j}.npy"), mmap_mode="r") for j in layers}
    if os.path.exists(done):
        parts = open(done).read().split()
        assert len(parts) == 4 and parts[3] == sig, (
            f"stale residual cache in {cdir}: marker={parts!r} but this run needs {sig!r}. "
            f"Delete the cache dir (or use a different --cachedir) -- reusing it would feed the old extraction.")
        off, L, d = [int(x) for x in parts[:3]]
        return ld(tag), ld("e" + tag), off, L, d
    os.makedirs(cdir, exist_ok=True)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, low_cpu_mem_usage=True).to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    hs0 = m(tr_ids[:1].to(dev), output_hidden_states=True).hidden_states
    emb = m.get_input_embeddings()(tr_ids[:1].to(dev))
    off = 1 if (hs0[0] - emb).abs().max().item() < 1e-3 else 0
    L, d = m.config.num_hidden_layers, m.config.hidden_size
    blk = blocks_of(m)
    assert len(blk) == L, f"{len(blk)} block modules but config says {L} layers"

    cap = {}
    # RWKV halves the stream every `rescale_every` blocks -- but HF does it AFTER the block returns, so a forward
    # hook sees the PRE-halved value while hidden_states records the POST-halved one (they differ by exactly 2 at
    # j in {5,11,17,23,29}, which is why the gate reported rel=1.00 on the first run). The value that actually
    # reaches block j+1, and that the chimera's own block loop reproduces, is the halved one. Mirror HF's rule.
    RE = m.config.rescale_every if (hasattr(m, "rwkv") and getattr(m.rwkv, "layers_are_rescaled", False)) else 0
    def mk(j):
        # RwkvBlock returns (hidden, state); GPTNeoXLayer returns a TENSOR. Taking [0] on the latter silently
        # strips the batch dimension -- the exact bug a norm-only probe once failed to catch.
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if RE > 0 and (j + 1) % RE == 0: h = h / 2
            cap[j] = h
        return hook
    handles = [blk[j].register_forward_hook(mk(j)) for j in range(L)]

    # ---- GATE the capture before it is used for anything ----
    with torch.no_grad(): hsg = m(tr_ids[:1].to(dev), output_hidden_states=True).hidden_states
    rl = lambda x, y: (x - y).float().norm().item() / (y.float().norm().item() + 1e-9)
    for j in range(L):
        assert cap[j].shape == hsg[0].shape, f"{tag} L{j}: captured {tuple(cap[j].shape)} vs {tuple(hsg[0].shape)}"
    mism = {j: rl(cap[j], hsg[j + off]) for j in range(L) if j + off < len(hsg) - 1}
    worst = max(mism.values())
    assert worst < 1e-3, f"{tag}: hook disagrees with hidden_states on a block output (worst rel={worst:.2e})"
    last = rl(cap[L - 1], hsg[-1])
    fn = getattr(getattr(m, "gpt_neox", None), "final_layer_norm", None) or m.rwkv.ln_out
    rt = rl(fn(cap[L - 1]), hsg[-1])
    print(f"  [{tag}] hook GATE: agrees with hidden_states on {len(mism)} block outputs (worst {worst:.1e}); "
          f"last block raw-vs-hs[-1]={last:.2e}, final_norm(raw)-vs-hs[-1]={rt:.2e}", flush=True)
    assert rt < 2e-3, f"{tag}: captured last-block output does not reproduce hs[-1] under the final norm ({rt:.2e})"

    def run(ids, pfx):
        acc = {j: [] for j in layers}
        for s in range(0, ids.shape[0], bs):
            with torch.no_grad(): m(ids[s:s + bs].to(dev))
            for j in layers: acc[j].append(cap[j].reshape(-1, cap[j].shape[-1]).half().cpu())
        o = {}
        for j in layers:
            p = os.path.join(cdir, f"{pfx}_L{j}.npy")
            arr = torch.cat(acc[j], 0).numpy(); np.save(p, arr)
            acc[j] = None; del arr; gc.collect()          # drop the in-RAM copy...
            o[j] = np.load(p, mmap_mode="r")               # ...and reopen as mmap, so only ONE model's
        return o                                            # residuals are ever RAM-resident (N=96000 x2 = 48GB > 45GB else)
    RR, EE = run(tr_ids, tag), run(ev_ids, "e" + tag)
    for h in handles: h.remove()
    open(done, "w").write(f"{off} {L} {d} {sig}")
    del m; gc.collect(); torch.cuda.empty_cache()
    return RR, EE, off, L, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--tgt", required=True)
    ap.add_argument("--tok", default="EleutherAI/gpt-neox-20b"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--win", type=int, default=112); ap.add_argument("--N", type=int, default=48000)
    # 400 windows = 44800 held-out tokens. All layers are scored on the same
    # windows, so the independent unit is the window count.
    ap.add_argument("--eval_windows", type=int, default=400); ap.add_argument("--z", type=int, default=2048)
    ap.add_argument("--eval_cut", type=int, default=2_000_000,
                    help="CONSTANT char offset where held-out begins. Must NOT depend on N: it used to be "
                         "N*6+2e6, so runs at different N scored on DISJOINT text and were not comparable. "
                         "Training consumes at most ~465k chars (N=96000), so 2e6 guarantees disjointness for all N.")
    ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--hid", type=int, default=512)
    ap.add_argument("--arch", default="resnet", choices=["resnet", "mlp"])
    ap.add_argument("--gpu_resident", type=int, default=1,
                    help="1=hold residuals on GPU (fast, N<=48000 on 32GB); 0=gather from /tmp mmap (slow, any N)")
    ap.add_argument("--sigma", type=float, default=0.40); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=15000); ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--sched", default="cosine", choices=["cosine", "plateau", "none"])
    ap.add_argument("--patience", type=int, default=3, help="plateau: evals with no held-out A->B gain before LR*0.5")
    ap.add_argument("--cachedir", default="/dev/shm/nx7b_adapt")
    ap.add_argument("--corpus", default="alpaca", choices=["alpaca", "wikitext"],
                    help="alpaca = instruction-formatted text (aligns the INSTRUCT regime); wikitext = plain prose")
    ap.add_argument("--out", default="artifacts/checkpoints/adapter.pt")
    ap.add_argument("--metrics_out", default="artifacts/metrics/raw/training_curve.csv",
                    help="append structured training metrics here; default is <out stem>.metrics.csv")
    ap.add_argument("--resume", default=None, help="warm-start the adapter from this checkpoint (continue training)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed. Was hardcoded to 0, so no seed replicate was runnable and no error bar for "
                         "this config could exist; >=2 seeds/condition are needed to call any A->B delta real.")
    ap.add_argument("--keep_every", type=int, default=0,
                    help="also save a PERMANENT checkpoint every this many steps (0=off). Only _best/_last were "
                         "kept before, so a matched-step comparison could not be reconstructed after the fact.")
    a = ap.parse_args()
    metrics_out = a.metrics_out or os.path.splitext(a.out)[0] + ".metrics.csv"
    if DDP_ON:
        # 4h timeout: ranks 1..3 sit in dist.barrier() while rank 0 extracts residuals, which takes ~25min at
        # N=228k. NCCL's DEFAULT barrier timeout is 30min -- it would abort the job mid-extraction.
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=4))
        torch.cuda.set_device(LOCAL_RANK); dev = f"cuda:{LOCAL_RANK}"
    else:
        dev = a.device
    # Same seed on every rank => identical init (DDP also broadcasts rank0's params). Batch sampling is re-seeded
    # per-rank AFTER model construction, so the ranks draw DIFFERENT rows and the effective batch is WORLD x bs.
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.tok)
    from datasets import load_dataset
    if a.corpus == "alpaca":
        # Instruction-formatted text so BOTH instruct models engage their instruction-following representations
        # (WikiText prose only exercises base-like next-token behaviour). The `text` field is the standard Alpaca
        # "### Instruction / ### Response" rendering. Held-out uses a disjoint character range = disjoint examples.
        text = "\n\n".join(load_dataset("tatsu-lab/alpaca", split="train")["text"])
    else:
        text = "\n".join(t for t in load_dataset("wikitext", "wikitext-103-raw-v1", split="train")["text"] if len(t) > 200)

    def windows(txt, n):
        ids = tok(txt, return_tensors="pt").input_ids[0]
        nw = min(n, ids.shape[0] // a.win); return ids[:nw * a.win].reshape(nw, a.win)
    # HELD-OUT MUST NOT MOVE WITH N. This was `cut = min(len(text), a.N*6 + 2_000_000)`, so N=32000 scored on
    # text[2.192M:2.234M] and N=96000 on text[2.576M:2.618M] -- ZERO overlap (only the first `eval_windows` windows
    # of the slice survive `windows()`, ~42k chars, not the 900k tokenized). Cross-N A->B was therefore a comparison
    # between disjoint text samples, with r2's variance denominator recomputed per run. Now fixed for every N.
    cut = min(len(text), a.eval_cut)
    want_tr = a.N // a.win + 40
    tr_ids = windows(text[:cut], want_tr)                     # train reaches at most ~465k chars (N=96000) << cut
    ev_ids = windows(text[cut:cut + 900_000], a.eval_windows)
    # Disjointness is structural (train tokenizes only text[:cut], eval only text[cut:]). The real risk of a FIXED
    # cut is windows() silently returning fewer windows than asked (nw = min(n, ids//win)), quietly shrinking the
    # train set and faking a data-scaling result. Fail loudly instead.
    assert tr_ids.shape[0] == want_tr, f"train truncated: got {tr_ids.shape[0]}/{want_tr} windows; raise --eval_cut"
    assert ev_ids.shape[0] == a.eval_windows, f"eval truncated: got {ev_ids.shape[0]}/{a.eval_windows} windows"
    # ALL 32 blocks. Layer 0 was previously excluded for no stated reason (the `off` correction already handles
    # the embedding, so j=0 is well defined for both families), and layer 31 was the one paired unfairly. With the
    # hook capture both ends are clean, so the adapter finally sees every block of both models.
    layers = list(range(0, 32))   # both are 32L; pair block j<->block j
    rprint(f"== 7B ADAPTER | {a.src} -> {a.tgt} | {len(layers)} layers | N/layer={a.N} z={a.z} sigma={a.sigma} "
           f"| world={WORLD} eff_batch={a.bs*WORLD} | shared-tok ids sum={int(tr_ids.sum())} ==", flush=True)
    t0 = time.time()
    # ONLY RANK 0 EXTRACTS. Four ranks each loading a 14GB fp16 7B model onto a 16GB V100 would OOM instantly, and
    # they would race writing the same .npy files. Rank 0 builds the cache; the others wait, then mmap it read-only
    # (one physical copy in /dev/shm shared by all ranks).
    if DDP_ON and not is_main():
        dist.barrier()
    RA, EA, offA, LA, dA = extract_model(a.src, tr_ids, ev_ids, layers, dev, a.cachedir, "A")
    RB, EB, offB, LB, dB = extract_model(a.tgt, tr_ids, ev_ids, layers, dev, a.cachedir, "B")
    if DDP_ON and is_main():
        dist.barrier()          # release the others only once BOTH caches carry their .done markers
    assert LA == LB, f"block count differs {LA} vs {LB}"
    rprint(f"  A(src) off{offA} d{dA} | B(tgt) off{offB} d{dB} | extract {time.time()-t0:.0f}s", flush=True)

    # per-layer standardisation stats
    st = {}
    for j in layers:
        va = torch.from_numpy(np.asarray(RA[j])).float(); vb = torch.from_numpy(np.asarray(RB[j])).float()
        st[j] = (va.mean(0).to(dev), (va.std(0) + 1e-5).to(dev), vb.mean(0).to(dev), (vb.std(0) + 1e-5).to(dev))
    P = min(min(RA[j].shape[0] for j in layers), min(RB[j].shape[0] for j in layers))
    if a.gpu_resident:
        # Hold train residuals RESIDENT ON GPU (fp16). Fast (batch gather is a GPU op), but ~18GB at N=32000 caps
        # N<=~48000 on a 32GB V100.
        RAg = {j: torch.from_numpy(np.asarray(RA[j])[:P]).to(dev) for j in layers}
        RBg = {j: torch.from_numpy(np.asarray(RB[j])[:P]).to(dev) for j in layers}
        del RA, RB; gc.collect()
        rprint(f"  residuals resident on GPU: {sum(v.numel()*2 for v in RAg.values())/2**30 + sum(v.numel()*2 for v in RBg.values())/2**30:.1f}GB fp16", flush=True)
        def batch(j, n):
            idx = torch.randint(0, P, (n,), device=dev)
            va = RAg[j][idx].float(); vb = RBg[j][idx].float()
            return (va - st[j][0]) / st[j][1], (vb - st[j][2]) / st[j][3]
    else:
        # Disk-backed: residuals stay as mmaps; each batch gathers just its rows (on /dev/shm this is RAM-speed).
        rprint(f"  residuals DISK-BACKED (mmap {a.cachedir}): {P*4096*2*len(layers)*2/2**30:.1f}GB, gathered per-batch",
               flush=True)
        def batch(j, n):
            idx = torch.randint(0, P, (n,)).numpy()
            va = torch.from_numpy(np.ascontiguousarray(RA[j][idx])).to(dev).float()
            vb = torch.from_numpy(np.ascontiguousarray(RB[j][idx])).to(dev).float()
            return (va - st[j][0]) / st[j][1], (vb - st[j][2]) / st[j][3]

    @torch.no_grad()
    def evaluate(m, chunk=8192):
        # Residual-space cross/self maps PLUS the alignment IN the shared latent z itself (the semantic space).
        # rho_ctr = CENTERED correlation of z_A,z_B (common mode removed) = the honest alignment; raw corr is a trap
        # (satisfiable by a token-independent common mode). f = token-varying fraction of z's ENERGY; under the
        # non-affine LN ||z||^2 is pinned per token, so f = 1 - ||mean z||^2/d -- pure angular spread, exactly
        # invariant to gain. Only ONE direction is valid: f->0 proves z collapsed to a constant and carries nothing.
        # f large does NOT mean more information (at step 0 of the z=4096 run f=0.94, its maximum, with B->B=-0.02);
        # dispersion comes from informative and uninformative variation alike. Informativeness is established by the
        # CENTERED R^2 read-outs below, where a token-independent predictor scores exactly 0 -- not by f.
        #
        # CHUNKED, via ONE-PASS SUFFICIENT STATISTICS. Every metric here needs statistics over the WHOLE eval set
        # (r2's SST needs the target mean; rho_ctr/f need the per-dim token mean), so the eval set cannot simply be
        # split and averaged. But all of them are recoverable from running sums, using
        #     sum((x-xbar)*(y-ybar)) = S_xy - (V_x . V_y)/T      with S_xy = sum(x*y), V_x = sum_t x, T = #tokens.
        # Materialising the whole eval set per layer would need eval_tok x 4096 x 4B = 1.8GB per tensor at
        # eval_windows=1000 -- too much for a 16GB V100 alongside the model. Results are identical to the
        # unchunked version up to float round-off.
        acc = {k: [] for k in ["A->B", "A->A", "B->A", "B->B", "rho_raw", "rho_ctr", "f"]}
        for j in layers:
            EAj, EBj = np.asarray(EA[j]), np.asarray(EB[j])
            n = min(EAj.shape[0], EBj.shape[0])
            T = 0
            S_ab = S_aa = S_bb = 0.0                      # latent sums
            sum_e2 = sum_f2 = 0.0                         # target square-sums (for SST)
            sse = {"A->B": 0.0, "A->A": 0.0, "B->A": 0.0, "B->B": 0.0}
            V_za = V_zb = V_e = V_f = None                # per-dim token sums
            for s in range(0, n, chunk):
                e = (torch.from_numpy(np.ascontiguousarray(EAj[s:s + chunk])).to(dev).float() - st[j][0]) / st[j][1]
                fv = (torch.from_numpy(np.ascontiguousarray(EBj[s:s + chunk])).to(dev).float() - st[j][2]) / st[j][3]
                za, zb = m.enc_a(e), m.enc_b(fv)
                sse["A->B"] += (m.dB(za) - fv).pow(2).sum().item(); sse["A->A"] += (m.dA(za) - e).pow(2).sum().item()
                sse["B->A"] += (m.dA(zb) - e).pow(2).sum().item(); sse["B->B"] += (m.dB(zb) - fv).pow(2).sum().item()
                S_ab += (za * zb).sum().item(); S_aa += za.pow(2).sum().item(); S_bb += zb.pow(2).sum().item()
                sum_e2 += e.pow(2).sum().item(); sum_f2 += fv.pow(2).sum().item()
                z0, z1, e0, f0 = za.sum(0), zb.sum(0), e.sum(0), fv.sum(0)
                V_za = z0 if V_za is None else V_za + z0; V_zb = z1 if V_zb is None else V_zb + z1
                V_e = e0 if V_e is None else V_e + e0;    V_f = f0 if V_f is None else V_f + f0
                T += e.shape[0]
            # SST = sum(t^2) - ||sum_t t||^2 / T  (= sum over the centred target)
            sst_e = sum_e2 - (V_e.pow(2).sum() / T).item()
            sst_f = sum_f2 - (V_f.pow(2).sum() / T).item()
            for k, sst in (("A->B", sst_f), ("A->A", sst_e), ("B->A", sst_e), ("B->B", sst_f)):
                acc[k].append(1 - sse[k] / (sst + 1e-9))
            acc["rho_raw"].append(S_ab / ((S_aa ** 0.5) * (S_bb ** 0.5) + 1e-9))
            # centred latent sums, common mode removed
            c_ab = S_ab - (V_za * V_zb).sum().item() / T
            c_aa = S_aa - (V_za.pow(2).sum() / T).item()
            c_bb = S_bb - (V_zb.pow(2).sum() / T).item()
            acc["rho_ctr"].append(c_ab / ((c_aa ** 0.5) * (c_bb ** 0.5) + 1e-9))
            acc["f"].append((c_aa / (S_aa + 1e-9) + c_bb / (S_bb + 1e-9)) / 2)
        return {k: sum(v) / len(v) for k, v in acc.items()}

    @torch.no_grad()
    def train_ab(m, n=2000):
        # A->B on TRAINING residuals. Gap vs held-out A->B is the overfitting diagnostic (train>>eval => too big).
        # Uses batch() so it works in BOTH gpu-resident and disk-backed modes (RAg/RBg exist only when resident).
        v = []
        for j in layers:
            e, f = batch(j, n)
            v.append(r2(m.dB(m.enc_a(e)), f))
        return sum(v) / len(v)

    m = LatentAdapter(dA, dB, a.z, blocks=a.blocks, hidden=a.hid, arch=a.arch).to(dev)
    nparam = sum(p.numel() for p in m.parameters())
    rprint(f"  adapter params: {nparam/1e6:.1f}M  (pooled train pairs {P*len(layers)/1e6:.2f}M -> "
           f"{nparam/(P*len(layers)):.0f} params/sample)", flush=True)
    net = m                                   # `net` = the raw module: eval/save/state_dict always go through it
    if DDP_ON:
        m = DDP(m, device_ids=[LOCAL_RANK])   # `m` = the DDP wrapper: only the training step goes through it
        net = m.module
        # Re-seed AFTER construction so each rank draws DIFFERENT rows in batch(); the model init above is still
        # identical across ranks (and DDP broadcasts rank0's params anyway). Without this every rank would sample the
        # SAME batch and the effective batch would stay bs, not bs*WORLD -- a silent 4x loss of the whole point.
        torch.manual_seed(a.seed + 100003 * RANK)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    import math
    if a.sched == "cosine":
        def _lr(step):
            if step < a.warmup: return (step + 1) / a.warmup
            prog = (step - a.warmup) / max(1, a.steps - a.warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    elif a.sched == "plateau":
        # Dynamic: drop LR by 0.5 when held-out A->B stops improving for `patience` evals. Adapts to real
        # convergence instead of a fixed decay; no need to know total steps. Stepped at EVAL time with the metric.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=a.patience,
                                                           threshold=1e-3, min_lr=1e-5)
    else:
        sched = None
    # Rank-INVARIANT generator for the per-step layer choice. batch()'s RNG is re-seeded per rank (so the ranks draw
    # different ROWS), which would also make them draw different LAYERS -- and DDP would then average gradients taken
    # on unrelated layers' residuals. Drawing j from this separate, identically-seeded generator keeps the layer in
    # lockstep across ranks while the rows stay independent.
    jgen = torch.Generator(); jgen.manual_seed(a.seed + 777)
    start_step, best = 0, (-1e9, -1)
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])                    # net, not m: DDP would prefix keys with "module."
        start_step = int(ck.get("step", 0))                 # continue the GLOBAL step count
        prev = ck.get("eval", {}).get("A->B")
        if prev is not None: best = (float(prev), start_step)  # don't overwrite _best.pt with a worse transient value
        # ...and never let a resume LOWER the bar below what _best.pt already holds. The watchdog resumes from the
        # newest _step*.pt, which is normally NOT the best one, so seeding `best` from it alone would let a run that
        # never recovers its old peak overwrite _best.pt with a worse model.
        bp = a.out.replace(".pt", "_best.pt")
        if os.path.exists(bp):
            try:
                bck = torch.load(bp, map_location="cpu", weights_only=False)
                bv, bs = float(bck["eval"]["A->B"]), int(bck.get("step", 0))
                if bv > best[0]:
                    best = (bv, bs)
                    rprint(f"  _best.pt on disk is better ({bv:+.4f} @ {bs}); keeping it as the bar", flush=True)
            except Exception as e:
                rprint(f"  could not read {bp} ({e}); proceeding with the resumed value as the bar", flush=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            if sched is not None and "sched" in ck: sched.load_state_dict(ck["sched"])
            optnote = "opt+sched state RESTORED (seamless continuation)"
        else:
            optnote = "opt state ABSENT -> AdamW COLD-START (transient expected; original momentum was not saved)"
        rprint(f"  RESUMED from {a.resume} @ global step {start_step}, A->B {prev} | {optnote}", flush=True)
    if is_main(): os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rprint(f"  training: +{a.steps} steps (global {start_step}..{start_step+a.steps}) x {a.bs}/rank x {WORLD} ranks "
           f"= {a.bs*WORLD} effective | pooled {P*len(layers)} pairs", flush=True)
    for i in range(a.steps):
        gstep = start_step + i                              # global step across resumes
        # Every rank must train the SAME layer j, or DDP would average gradients computed on different layers'
        # residuals. j is drawn from the rank-invariant generator below, NOT the per-rank batch RNG.
        j = layers[int(torch.randint(0, len(layers), (1,), generator=jgen).item())]
        va, vb = batch(j, a.bs)                             # rows DIFFER per rank => effective batch = bs * WORLD
        za, zb, ra, rb = m(va, vb, a.sigma)
        loss = F.mse_loss(ra, va) + F.mse_loss(rb, vb) + a.lam * F.mse_loss(za, zb)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if a.sched == "cosine": sched.step()
        if i % a.log_every == 0 or i == a.steps - 1:
            # Every rank evaluates. evaluate() is deterministic given weights (DDP keeps them identical) and reads the
            # same eval cache, so all ranks get the same metric and the plateau scheduler stays in lockstep WITHOUT
            # any collective. train_ab() uses batch(), so it differs slightly per rank -- rank 0's is the one logged.
            m.eval(); ev = evaluate(net); tr = train_ab(net); m.train()
            if a.sched == "plateau": sched.step(ev["A->B"])
            rprint(f"     step {gstep:6d}  lr={opt.param_groups[0]['lr']:.2e} loss={loss.item():.4f} | "
                  f"A->B(tr)={tr:+.4f} A->B(ev)={ev['A->B']:+.4f} gap={tr-ev['A->B']:+.4f} | "
                  f"B->A={ev['B->A']:+.4f} A->A={ev['A->A']:+.4f} B->B={ev['B->B']:+.4f} | "
                  f"z-align: rho_ctr={ev['rho_ctr']:+.4f} f={ev['f']:.4f} (rho_raw={ev['rho_raw']:+.4f})  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            # `net`, not `m`: DDP's state_dict prefixes every key with "module.", which would break evaluation
            # consumers and --resume. Only rank 0 writes -- 4 ranks racing on one path corrupts the file.
            if is_main():
                append_csv_row(
                    metrics_out,
                    {
                        "step": gstep,
                        "learning_rate": opt.param_groups[0]["lr"],
                        "loss": loss.item(),
                        "train_a_to_b_r2": tr,
                        "eval_a_to_b_r2": ev["A->B"],
                        "eval_b_to_a_r2": ev["B->A"],
                        "eval_a_to_a_r2": ev["A->A"],
                        "eval_b_to_b_r2": ev["B->B"],
                        "latent_rho_centered": ev["rho_ctr"],
                        "latent_rho_raw": ev["rho_raw"],
                        "latent_f": ev["f"],
                        "elapsed_seconds": time.time() - t0,
                        "seed": a.seed,
                        "tokens_per_layer": a.N,
                        "latent_size": a.z,
                    },
                    [
                        "step", "learning_rate", "loss", "train_a_to_b_r2", "eval_a_to_b_r2",
                        "eval_b_to_a_r2", "eval_a_to_a_r2", "eval_b_to_b_r2",
                        "latent_rho_centered", "latent_rho_raw", "latent_f", "elapsed_seconds",
                        "seed", "tokens_per_layer", "latent_size",
                    ],
                )
                blob = {"model": net.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict() if sched is not None else None, "step": gstep, "eval": ev,
                        "stats": {j: [t.cpu() for t in st[j]] for j in layers},
                        "cfg": {"dA": dA, "dB": dB, "z": a.z, "layers": layers, "sigma": a.sigma, "lam": a.lam,
                                "src": a.src, "tgt": a.tgt, "N": a.N}}
                torch.save(blob, a.out)
                if ev["A->B"] > best[0]:
                    best = (ev["A->B"], gstep); torch.save(blob, a.out.replace(".pt", "_best.pt"))
                # _best.pt is SELECTED on held-out A->B across ~20 evals, so it is biased upward and can land on a
                # terminal outlier (the N=32000 baseline's 0.4286 sat ~5sd above its own plateau at its last eval).
                # Keeping periodic checkpoints allows an unbiased matched-step comparison after the fact.
                if a.keep_every and gstep % a.keep_every == 0:
                    torch.save(blob, a.out.replace(".pt", f"_step{gstep}.pt"))
            elif ev["A->B"] > best[0]:
                best = (ev["A->B"], gstep)                  # keep `best` in sync on every rank (scheduler parity)
    rprint(f"  BEST held-out A->B = {best[0]:+.4f} at global step {best[1]} -> {a.out.replace('.pt','_best.pt')}",
           flush=True)
    if DDP_ON: dist.destroy_process_group()
    rprint("ADAPTER7B_DONE")


if __name__ == "__main__":
    main()
