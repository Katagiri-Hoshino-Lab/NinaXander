# 03 · Roadmap

English | [日本語](ja/03-roadmap.md) | [简体中文](zh-CN/03-roadmap.md)

## Done: the 7B instruct pair is built and its adapter is characterised

**RWKV-4-Raven-7B (pure RNN, instruct) ↔ allenai/open-instruct-pythia-6.9b-tulu (Transformer, instruct).** Both 32L,
d4096, GPT-NeoX-20B tokenizer, verified bit-identical tokenization and convention auto-detect (A: RNN, post-append
`off=0`; B: Transformer, pre-append `off=1`). Dolly-v2-7b, the originally planned partner, was **deleted from HF**;
Tulu is the substitute (alternatives: `usvsnsp/pythia-6.9b-sft`, `dvruette/oasst-pythia-6.9b`).

Raven ships as a BlinkDL `.pth` → converted with a one-off helper that is not bundled (patched:
`shard_checkpoint` →
`save_pretrained` for transformers 5.x), then re-saved **fp16** (fp32 was 28GB and OOM'd). 7B×2 won't co-load on a
32GB V100, so residuals are extracted **one model at a time** to a per-node cache. Its location and capacity are
site settings (`NINAXANDER_TRAIN_CACHE`); the reported 32GB probe node had only 23GB of shared memory.
Results: see [02-findings.md](02-findings.md). A shared space of token-aligned representations exists under favorable conditions (not a
general semantic space) — the reported adapter is
z=4096 (no compression), 268.5M params, σ=0.4, **415k steps** (jobs 1832 → 1837), residuals captured for **all 32
blocks by forward hook**, reaching ρ_ctr 0.901, AA/AB/BB/BA = 0.695/0.559/0.711/0.590. Both AB≤BB and BA≤AA
hold at all 32 layers: even under high latent alignment, same-family reconstruction is already imperfect, and both
cross-readouts **fall below it at every layer**. Whether that error comes from the encoder, the latent normalization,
the decoder, or the training-time noise is not isolated, and what limits the self-maps is unresolved (whether the
noise σ sets a ceiling is unknown; a σ sweep would test it).

## Done since: the fair retrain, both chimera paths, controls, and deployable paths

The reported checkpoint captures all 32 blocks by forward hook. All adapter
numbers are from the step-415,000 checkpoint.

Both 7B execution paths, AB and BA, were built and measured. They **converse** — both
answer the capital-of-France prompt — while remaining factually or
format-degraded on other fixed prompts. On the full paired test sets, AB@4
scores ARC 59.6 / SciQ 70.9 and BA@4 scores 62.2 / 69.2, versus RWKV
61.8 / 67.0 and Pythia 65.3 / 81.0. AB@4 beats RWKV only on SciQ
(p=0.02); BA@4 is statistically tied with RWKV on both tasks; both paths are
below Pythia. The complete AB/BA table spans 15.6–84.4% measured KV
reduction; on SciQ the combined non-dominated sequence is Pythia → AB@4 →
BA@24 → BA@4 → RWKV (post-hoc, one-task frontier).

Two further characterisations landed: the adapter is **not near-linear** —
13.4% of AB and 12.0% of BA output is irreducibly non-linear, and setting this
jointly trained adapter's residual-block branches to α=0 (those of E_A and D_B
on AB, of E_B and D_A on BA) collapses both read-outs and downstream accuracy.
That does not show that nonlinearity is generally necessary: in BA accuracy the separately
fitted affine control differs little from the adapter (−3.4 to +2.0 points), so the adapter does not consistently outperform it. Translation is also
**domain-specific in both paths**: at context 512, moving from Alpaca to
WikiText changes AB@4 perplexity 6.40→93.87 and BA@4 3.85→45.00, while the
parents degrade much less. This is domain shift, not a context-length limit.

Each path is measured directly rather than inferred from representation R²:
full-set QA, paired parent and same-layer path tests, equal-KV early exits,
Alpaca/WikiText context 512/1024/2048, fixed-prompt generation, measured KV,
and unpruned/pruned serving. The applicable boundary, direct-suffix, and
pruning gates are exact zero. BA@4 keeps five Transformer blocks (84.375% KV
reduction) and scores ARC 62.2 / SciQ 69.2. It is the best measured two-task
mean across the eight path/switch combinations, but only by 0.45 points over
AB@4 and with no independent selection split; it is a post-hoc release
default, not a validated general winner. A fused RWKV option batch failed the
numerical gate, so reported BA QA remains batch 1.

## Harness invariants (learned the hard way — do not regress)

- The **held-out window must not depend on any training hyper-parameter**. It used to be `cut = N*6 + 2e6`, so runs at
  different N scored on **disjoint** text. Now a constant `--eval_cut`, with asserts that fail loudly on truncation.
- `windows()` keeps only the **first** n windows, so `--eval_windows` *is* the eval size (80 → only 8,960 tokens).
  Now 400. All 32 layers are scored on the *same* windows, so the independent unit is the **window**, not the layer.
- **Never compare runs at `_best`** (selection-biased, ~20 evals, can be a terminal outlier). Compare at matched
  steps; `--keep_every` now retains periodic checkpoints so that is reconstructable after the fact.
- **`--seed` exists now.** Claims about run-to-run variation require multiple
  seeds per condition; the present paper reports one seed.

## Open questions

- **What limits self-reconstruction?** At 7B, B→B remains at 0.711 even though width/rank is unlikely to be a
  strong constraint (z = d, and the non-affine LN drops only the per-token mean and norm). What limits it remains
  **open**: the error is not isolated among the encoder, the latent normalization, the decoder, and the
  training-time noise, and whether the noise σ sets a ceiling is unknown. A matched σ sweep {0, 0.1, 0.2, 0.4, 0.8} on the current
  32-layer setup tests the noise candidate directly.
- **Cost of the cross-reconstruction target** — requires matched multi-seed
  current-scale runs; no estimate is reported.
- **Can the translation be made near-lossless?** Cross-map R² remains a plausible lever, but the current controls
  do not assign the whole chimera cost to it: the encode–decode round-trip loss and front-parent capability are
  also present.
  Levers not yet tried at scale include richer/deeper decoders, a
  vocabulary-anchored closed-form initialisation, and per-layer rather than
  shared decoders.

## Reporting guardrails

- **Raw corr / "emergent" framing** — report centered ρ and f; call both
  cross-readouts "no cross-reconstruction target required".
- **Instruct cross-arch pairs with different tokenizers** — the method needs a shared tokenizer for position-matched
  pairs. Falcon-Mamba-Instruct (pure, 64L) has no same-tokenizer 32L Transformer partner; RecurrentGemma↔Gemma-2
  match block count and tokenizer but RecurrentGemma is a hybrid (has attention).

## Deferred ablations

z-dim sweep, λ_cross sweep, σ sweep, data-scale — none run on the 32-layer setup; per-layer vs shared decoder for
the chimera; utility-versus-alignment analysis; and a same-architecture (two models of one family) alignment control.
No current runnable script is claimed for these deferred experiments.

Done at 7B (no longer deferred): the **shuffled-correspondence negative control** (`four_path_layer_metrics.py`) — permuting
the B rows collapses ρ_ctr 0.90 → 0.00, A→B 0.56 → −0.59, and B→A 0.59 → −0.61, confirming that both
cross-readouts depend on token-level correspondence.
