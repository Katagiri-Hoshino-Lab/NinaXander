# 02 · Current findings

English | [日本語](ja/02-findings.md) | [简体中文](zh-CN/02-findings.md)

All values in this document belong to the current public experiment: the
32-layer RWKV-4-Raven-7B / Tulu-Pythia-6.9B pair, the step-415,000
forward-hook adapter, and the structured artifacts rebuilt by
`./reproduce.sh tables`. No result outside that contract is included.

## Result 1 · The reported adapter: 32 layers, forward-hook, converged to a plateau

The reported adapter uses an uncompressed latent (`z = d = 4096`) and pairs all 32 blocks fairly.

`z = d = 4096` (no compression), one zero-initialised ResBlock in each encoder and decoder (hidden 4096), σ=0.4, N/layer=228,000, batch 4096
over 4 GPUs, lr 1e-3 with plateau decay to 1e-5. Residuals are captured for **all 32 blocks by forward hook** (not
`hidden_states`). Training ran to **416,000 steps**
across two walltime cycles (jobs 1832 → 1837) before the allocation ended; all reported numbers use the
step-415,000 checkpoint, the held-out A→B maximum. 268.5M adapter parameters = 1.9% of the 14.2B frozen weights. Held-out
at the deepest point (step 415,000):

| | value |
|---|---|
| A→A (self) | 0.6950 |
| A→B (cross, held-out) | **0.5588** |
| B→B (self, reconstruction) | **0.7112** |
| B→A (cross, B-prefix) | 0.5900 |
| ρ_ctr (centered alignment) | **0.9006** |
| f (token-varying energy fraction) | 0.3501 |
| ρ_raw | 0.9654 |

The paired orderings are the finding: **B→B 0.711 > A→B 0.559** and
**A→A 0.695 > B→A 0.590**, under a centered latent alignment of **ρ_ctr 0.901**
(a correlation, not on the same scale as these R² values). At every one of the 32 layers, AB is
below the BB self-map using the same B decoder, and BA is below the AA self-map
using the same A decoder. **Even under high latent alignment, same-family reconstruction is already
imperfect, and both cross read-outs fall below it; whether that error comes from the encoder, the latent
normalization, the decoder or the training-time noise is not isolated.**
**What binds either self-map is unresolved.** The latent is as wide as the residual and the non-affine LN
drops only two degrees of freedom per token (mean and norm), so width or rank is unlikely to be a strong
constraint. The
decoder is trained to rebuild from z+ξ, ξ~N(0,σ²), so its best coefficient shrinks to 1/(1+σ²); but 1/(1+σ²)≈0.862
is the R² of that coefficient evaluated on *noisy* input, whereas we evaluate on **clean z (ξ=0)**, for which the
scalar-linear ceiling is 1−(σ²/(1+σ²))² ≈ **0.98** (numerically confirmed). So both self-map scores are far below the
noise ceiling and **noise is not established as the binding constraint** — a matched σ sweep
{0,0.1,0.2,0.4,0.8} is required. B→A (0.590) slightly exceeds
A→B (0.559): the RNN-side residual is a hair easier to rebuild from the Transformer latent than the A→B read-out.

**Negative control (`four_path_layer_metrics.py`, forward-hook, 400 windows).** To rule out that ρ_ctr 0.90 and either cross-readout are a
distributional overlap rather than a token-by-token correspondence, keep A's row order and apply one fixed random
permutation to the B rows, then recompute the cross quantities. They **collapse**: ρ_ctr 0.9007 → +0.0000 and
A→B 0.5601 → −0.5909, while B→A 0.5911 → −0.6083 (self-maps A→A/B→B are unchanged by construction).
The re-eval also reproduces the reported means, so both cross-readouts depend on the genuine token correspondence.

Training signal: 2,075 windows × 112 tokens (≈232k independent token positions) of Alpaca, pooled over 32 layers =
7.44M residual pairs, cycled ~228 times.

Each metric is computed **per layer over all its tokens** (centered / SST statistics removed within the layer) and
**averaged over the 32 layers**, not pooled across layers. Honest caveats: single seed; over steps 408k–415k all
four read-outs move by less than 0.001 (AB 0.5584→0.5588, BA 0.5899→0.5900,
AA 0.6941→0.6950, BB 0.7106→0.7112), so this is an **asymptotic plateau, not
strict convergence** (reported at the deepest point).

## Result 2 · What both execution paths buy: measured KV cache

Measured, not assumed (`kvcache_measure.py`, job 1809): Tulu-Pythia caches **16.0 KiB per token per layer**,
512 KiB/token over 32 layers, matching 2·L·d·2 bytes exactly. The RWKV recurrent state that replaces it is
**64 KiB per layer and constant in sequence length** (computed from the reference implementation, which holds
2 of its 5 state vectors in fp16 and 3 in fp32).

AB retains `31−L` Transformer suffix blocks; BA retains `L+1` Transformer prefix blocks. The measured
memory/accuracy table therefore has both paths from the outset:

| config | Transformer layers | KV/token | KV @ 4k ctx | saved | ARC / SciQ (acc_norm) |
|---|---|---|---|---|---|
| pure Tulu-Pythia | 32 | 512 KiB | 2.00 GiB | — | 65.3 / 81.0 |
| pure RWKV-Raven | 0 | 0 | 0 | 100% | 61.8 / 67.0 |
| AB@4 | 27 | 432 KiB | 1.69 GiB | 15.6% | 59.6 / 70.9 |
| **BA@4** | **5** | **80 KiB** | **0.31 GiB** | **84.4%** | **62.2 / 69.2** |
| AB@8 | 23 | 368 KiB | 1.44 GiB | 28.1% | 59.0 / 67.7 |
| BA@8 | 9 | 144 KiB | 0.56 GiB | 71.9% | 57.3 / 64.6 |
| AB@16 | 15 | 240 KiB | 0.94 GiB | 53.1% | 48.6 / 54.5 |
| BA@16 | 17 | 272 KiB | 1.06 GiB | 46.9% | 57.5 / 68.0 |
| AB@24 | 7 | 112 KiB | 0.44 GiB | 78.1% | 47.0 / 51.1 |
| BA@24 | 25 | 400 KiB | 1.56 GiB | 21.9% | 59.6 / 69.9 |

QA uses the **full test sets** (ARC-Easy 2376 / SciQ 1000, acc_norm, temperature 0) and paired
McNemar/bootstrap comparisons (Result 5). The combined SciQ non-dominated sequence is
Pythia → AB@4 → BA@24 → BA@4 → RWKV; this is a post-hoc one-task frontier, not a universal ranking.
BA@4 retains nearly the two-task mean of AB@4 (65.70 vs 65.25) with five rather than 27 Transformer blocks.
Pure RWKV still has no KV at all and is not significantly worse than BA@4, so the new point is a trade-off,
not a dominating architecture. The earlier "no point achieves this accuracy at this memory" framing was wrong.
An equal-KV early-exit Pythia comparison was subsequently run (Result 8): it ties the shallow switches and loses to the
deep switches, but is lighter in resident weights. The curve is still not compared with KV quantisation,
sliding-window attention, or a purpose-trained efficient Transformer. Also not free: in our setup HF's
RWKV runs its recurrence sequentially, so the current reference runtime is
**about 60× slower in pruned prefill** at ctx 2048 (22.6 s vs 0.378 s) and
**64× slower in unpruned cacheless decode** (0.148 vs 9.49 token/s). These are
implementation properties, not architectural limits, but they are real for
the released reference path.

**Serving microbench (both `a_to_b_*` and `b_to_a_*`, measured).** The unpruned reference implementations load
both parents in full and override the boundary inside a complete parent forward, so neither realizes the
resident-weight saving. The physically pruned AB and BA paths remove unused blocks and reproduce their unpruned
references exactly; their resident weights are 13.93–14.55 GiB and 13.49–14.11 GiB, respectively. Speed remains
dominated by HF RWKV's sequential execution and the cacheless reference decoder. These are implementation
properties, not architectural limits, but they bound what the memory axis buys today.

Result 8 supplies that physically pruned path and gates it exactly against this
unpruned reference. It realizes the resident-weight reduction, while the
unpruned timings above remain the reference-runtime measurements rather than
an optimized throughput claim.

## Result 3 · Neither cross-family path is "near-linear"

Measured on the reported 32-layer adapter over five probe layers {4, 10, 16, 22, 28}, on real held-out residuals
(`artifacts/metrics/raw/cross_family_linearity_L32.json`,
`experiments/linearity.py`; 50,400 fit rows /
100,800 disjoint eval rows per layer, fp64 normal
equations, ridge 1e-3·tr(XᵀX)/(d+1) ≈ 50):

| path | adapter R² | affine-imitation R² | irreducibly non-linear | imitation cross-R² | best direct affine R² |
|---|---:|---:|---:|---:|---:|
| AB | 0.558 | 0.866 | **13.4%** | 0.490 | 0.487 |
| BA | 0.576 | 0.880 | **12.0%** | 0.516 | 0.535 |

Two consequences. (1) "E and D each begin as a linear map" is an *initialisation* property (each ResBlock's
second matrix is zero-init; with the non-affine LN between them, the full cross map is not exactly linear even then); after 415k steps the map is clearly non-linear, so "near-linear" is **retracted**.
(2) Best *direct* affine maps still reach AB 0.487 and BA 0.535 — low-capacity
maps that cannot fit arbitrary correspondences — so the shared-space claim is
carried by these direct-affine baselines, not by the heavy adapter. The gap to
the adapter is +0.071 for AB and +0.041 for BA. Downstream, separately fitted
affine maps are much weaker than the adapter in AB (8–23 points), whereas BA
differences are smaller and sometimes change sign. Those between-map
comparisons are descriptive; Result 7 uses the same-adapter alpha intervention for
the causal test in both paths.

## Result 4 · The translation is domain-specific — the degradation is domain shift, not context length

The adapter is trained on Alpaca (instruction text). To separate "long context breaks it" from "out-of-domain text
breaks it", the matched `a_to_b_longctx_ce_mp.py` and `b_to_a_longctx_ce_mp.py` runs measure next-token CE at
ctx 512 / 1024 / 2048 on both **in-domain Alpaca** and **out-of-domain WikiText-103**. A best affine map is
measured alongside in each path, but it is **refit separately within each domain** on that domain's first 40,000
tokens and evaluated on a disjoint suffix. The WikiText affine arms are therefore domain-adapted supervised
oracles, not Alpaca-fitted maps transferred out of domain.

- **Domain, not length, is the axis in both paths.** At ctx 512, AB@4 moves from Alpaca/WikiText perplexity
  6.40/93.87 and BA@4 from 3.85/45.00; both parents degrade far less. BA mitigates the shift, but its WikiText
  perplexity remains well above the parents.
- **Longer context does not collapse either chimera.** From ctx 512 → 2048, WikiText perplexity improves
  93.87 → 83.40 for AB@4 and 45.00 → 34.78 for BA@4. The better parent is still 8.27 at ctx 2048, so the failure
  is not a context-length limit and BA does not solve the domain problem.
- **The within-domain affine control is descriptive, not a clean non-linearity ablation.** WikiText lin@8 CE
  5.65 → 6.28 from ctx 512 → 2048 vs adapter@8 4.74 → 4.72, but the two maps have different objectives and
  training data. This does not identify the adapter's non-linearity as the
  cause; the matched same-adapter α intervention in Result 7 removes this confound
  for both AB and BA, but it too shows only what removing the branches from this
  jointly trained adapter does, not that non-linearity is generally necessary.

This bounds the memory buy (Result 2) in both directions: the KV savings are real, but only where the input resembles
the training domain.

## Result 5 · Symmetric full-set QA with paired tests

The AB and BA evaluators run the **full** ARC-Easy (2376) and SciQ (1000) test sets in identical item order.
They include both same-family arms: BB uses B's residual through `D_B(E_B(·))`, and AA uses A's residual through
`D_A(E_A(·))`. `paired_stats.py` compares every switch with both parents and compares AB with BA on the same
items using McNemar's exact test and a paired bootstrap. Full-set acc_norm:

| task / layer | AA | AB | BB | BA |
|---|---:|---:|---:|---:|
| ARC-Easy @4 | 62.8 | 59.6 | 62.2 | 62.2 |
| ARC-Easy @8 | 59.8 | 59.0 | 62.7 | 57.3 |
| ARC-Easy @16 | 54.8 | 48.6 | 57.4 | 57.5 |
| ARC-Easy @24 | 52.2 | 47.0 | 59.4 | 59.6 |
| SciQ @4 | 68.5 | 70.9 | 74.8 | 69.2 |
| SciQ @8 | 66.7 | 67.7 | 74.8 | 64.6 |
| SciQ @16 | 59.8 | 54.5 | 71.1 | 68.0 |
| SciQ @24 | 57.9 | 51.1 | 70.1 | 69.9 |

- **Every AB and BA switch is significantly below the stronger Pythia parent on both tasks.**
- **The weak-parent relation is path/task dependent.** AB@4 beats RWKV on SciQ (+3.9, p=0.018) but loses on ARC
  (−2.2, p=0.042). No BA switch significantly beats RWKV; BA@4 is tied on ARC (+0.4, p=0.66) and SciQ
  (+2.2, p=0.14).
- **The same-family arm is not a universal additive "encode–decode round-trip cost."** For the B suffix, BB sits below Pythia and
  AB sits another 3–19 points below BB. For the A suffix, deep BA instead exceeds AA: at L=16/24 by
  +2.7/+7.4 points on ARC and +8.2/+12.0 on SciQ. AA/BB isolate the loss of the encode–decode round trip, which
  reaches downstream accuracy significantly for BB at every switch layer on both tasks and for AA on both tasks
  only at L=16/24; but the cross-family difference
  also changes the prefix parent and therefore cannot be interpreted as translation error alone.
- **Depth is path-dependent.** AB falls monotonically with L, whereas BA recovers at L=16/24 and beats the
  same-layer AB by 8.9–18.8 points. Local cross-readout R² alone does not predict the end-to-end ranking.

The canonical structured result is `paper_four_path_qa_accuracy.csv`; all parent, affine, equal-memory and
intervention rows for both directions are in `paper_cross_family_qa_accuracy.csv`.

## Result 6 · Same-numbered layers are not intrinsically special — the alignment is learned

`all_layer_corr.py` computes an adapter-free 32×32 correspondence between A's layer j and B's layer k (linear CKA and
in-sample best-linear R²). By **linear CKA** the raw cross-family representations are only weakly similar (diagonal
mean 0.049 vs off-diagonal 0.035), and A's layer j is the most-similar to B's layer j in only **2/32** rows (argmax
usually the last B layer). The best-linear R² was high everywhere (~0.85) but that is in-sample inflation and
uninformative. So ρ_ctr=0.90 is an alignment the adapter **learns on the chosen same-layer pairs**, not an intrinsic
layer-to-layer correspondence — consistent with narrowing the claim from "shared *semantic* space" to "a token-level
representation correspondence recoverable under matched conditions".

## Result 7 · Removing the adapter's nonlinear branches collapses both AB and BA read-outs

The Result 3/linear-map comparison used a *separately-fit* linear map (different objective/data/layer-set), so it
has a possible confound. `cross_family_mlp_intervention.py` and the two canonical QA evaluators remove it:
they scale the **same trained adapter's** only learned non-linearity — the ResBlock branches
fc2(GELU(fc1(·))) — by α (α=1 full, α=0 = the *same weights/layers/rows* with the branches disabled).
AB passes through the ResBlocks of E_A and D_B and BA through those of E_B and D_A, so the two paths do not
share one branch; each of E and D is one zero-initialised ResBlock between two Linears (a cross path passes
through two), and α scales all of them together.
The representation sweep and both full-set, per-item-paired QA paths are stored together:

| path / α | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| mean AB R² | **−0.41** | −0.75 | 0.35 | 0.52 | **0.56** |
| mean BA R² | **−0.30** | −0.66 | 0.36 | 0.53 | **0.58** |

Disabling the branches (α=0) collapses AB to **−0.41** and BA to **−0.30**.
Full-set normalized QA shows the same effect directly in both execution paths
(entries are α=1 → α=0 accuracy percentages; chance is 25):

| L | ARC-Easy AB | ARC-Easy BA | SciQ AB | SciQ BA |
|---:|---:|---:|---:|---:|
| 4 | 59.6 → 31.9 | 62.2 → 31.8 | 70.9 → 31.3 | 69.2 → 32.6 |
| 8 | 59.0 → 29.9 | 57.3 → 35.5 | 67.7 → 31.4 | 64.6 → 37.0 |
| 16 | 48.6 → 28.5 | 57.5 → 43.7 | 54.5 → 31.7 | 68.0 → 47.3 |
| 24 | 47.0 → 34.8 | 59.6 → 48.4 | 51.1 → 40.0 | 69.9 → 49.1 |

The branches contribute **+11.1 to +39.6 points in AB** and **+11.2 to
+36.6 points in BA**; all 16 item-paired McNemar tests have p<2e-9. Two
points follow. (1) Both α sweeps are **non-monotone** (0.25 is worse than 0),
so the read-out does not degrade gracefully as the branches are scaled down. (2) Each
adapter's own bare linear skeleton (AB −0.41, BA −0.30) is worse than its
separately fitted direct affine map (AB 0.487, BA 0.535): the surrounding
Linears were trained to work with the ResBlocks, and removing them collapses both
paths. This is a property of removing the branches from this jointly trained
adapter; it does not show that non-linearity is generally necessary, and the
separately fitted affine control differs little from the adapter in BA accuracy
(−3.4 to +2.0 points; Result 3), so the adapter does not consistently outperform it.

## Result 8 · Equal-KV Transformer baseline and actually-pruned serving

The paired `a_to_b_serving_pruned.py` / `b_to_a_serving_pruned.py` paths
physically free unused blocks and reproduce their unpruned chimeras at rel=0.
`equal_mem_baseline.py` supplies a Pythia early exit with the exact Transformer
depth/KV of each AB or BA switch.

- **Pruned weights ≈ half in both paths.** AB occupies 13.93–14.55 GiB and BA
  13.49–14.11 GiB, versus about 27 GiB for both complete parents. The matched
  early-exit Pythia baselines remain lighter because a chimera also carries
  RWKV blocks and the adapter.
- **The non-Transformer blocks add capability at low Transformer depth.** For AB, @16 vs exit@15
  ARC +11.2 / SciQ +12.8; AB@24 vs exit@7 +18.5 / +18.7 (all McNemar p<1e-10). At shallow switches they tie
  (AB@4 vs exit@27 ±1 pt, p>0.5). For BA against exits with 5/9/17/25 blocks,
  the gains are ARC +35.4/+27.6/+16.9/+2.1 and SciQ
  +38.7/+31.6/+17.9/+0.5; the first three are significant and the 25-block
  comparison is tied. Thus either an RWKV prefix or suffix can add real
  capability when the retained Transformer is short.
- **The honest frontier includes both paths and both parents.** Pure RWKV has
  zero KV and matched Pythia exits are lighter in resident weights. KV
  quantisation and sliding-window attention were not compared.
