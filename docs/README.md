# NinaXander

English | [日本語](ja/README.md) | [简体中文](zh-CN/README.md)

> **NinaXander names the family of composed chimera language models** this project produces — each assembled
> from frozen blocks of different architecture families via one learned shared-latent adapter. An individual model
> is specified by its cross-family path and switch layer, e.g. `NinaXander-BA@4` (Pythia→RWKV, L=4). "NinaXander"
> always refers to these composed models, not the research area.

Building **chimera** language models by composing frozen pretrained blocks from *different architecture families*
(RNN/SSM ↔ Transformer) through a learned **shared-latent adapter**.

The adapter is a means, not the end: the target is a chimera that *functions*. Residual-alignment R² is a proxy.

## Documents

| File | What |
|---|---|
| [00-overview.md](00-overview.md) | What NinaXander is, the thesis, the target model pair |
| [01-method.md](01-method.md) | The shared-latent adapter, its loss, and the chimera construction |
| [02-findings.md](02-findings.md) | Established results (each traced to a run), honestly stated |
| [03-roadmap.md](03-roadmap.md) | What is done, open questions, and known dead ends |
| [../paper/](../paper/) | The write-up (`paper_ja.tex`) |

## One-paragraph status

The pair is **RWKV-4-Raven-7B (pure RNN) ↔ open-instruct-pythia-6.9b-tulu (Transformer)**, both 32 blocks,
d4096, GPT-NeoX tokenizer, aligned on Alpaca. A single shared-latent adapter — 268.5M parameters, latent width
z=4096 (no compression), σ=0.4, **415k steps** (jobs 1832 → 1837), residuals captured for **all 32 blocks** by
forward hook — reaches held-out **centered** ρ_ctr=0.901 with AA/AB/BB/BA
read-outs 0.695/0.559/0.711/0.590. Both
cross read-outs **fall below same-family reconstruction at all 32 layers** (A→B below B→B, B→A below A→A): even
under high latent alignment, same-family reconstruction is already imperfect, and whether that error originates in
the encoder, the latent normalization, the decoder, or the training-time noise is not isolated. Removing compression
does not make either same-family reconstruction near-lossless; since the non-affine LN drops only two degrees of
freedom per token, width/rank is unlikely to be a strong constraint, and what limits the self-maps is unresolved
(whether the noise σ sets a ceiling is unknown; a σ sweep would test it). The adapter
is **not** near-linear in either direction: 13.4% of AB and 12.0% of BA output is irreducibly non-linear.
Disabling the residual-block branches of this jointly trained adapter (α=0; AB passes through those of E_A and D_B,
BA through those of E_B and D_A) collapses both read-outs and downstream accuracy, but that does not show that
nonlinearity is generally necessary: in BA accuracy the separately fitted affine control differs little from the adapter
(−3.4 to +2.0 points), so the adapter does not consistently outperform it. The 7B chimeras exist and **converse** — both answer "What is the capital of France?" — but both
show factual or format degradation on the remaining fixed prompts. On the **full test sets**
(SciQ N=1000, ARC-Easy N=2376), AB@4 scores 70.9/59.6 and beats RWKV only on SciQ. BA@4
scores 69.2/62.2 and is statistically tied with RWKV on both tasks; every configuration in both
cross-family paths remains below the stronger Pythia parent. BA@16/@24 gains 8.9–18.8 points over the same-layer
AB path, establishing that the depth effect is path-dependent. The measured release default is
`NinaXander-BA@4`: it keeps five Transformer blocks and removes **84.375%** of KV, while its two-task mean (65.70) is only 0.45
points above AB@4. That is a post-hoc choice among 2 cross-family paths × 4 switches on the two reported test sets,
not independent deployment validation. The same-family reconstruction controls (AA/BB) isolate the loss of the
encode–decode round trip; their gaps to the cross paths in either direction include both front-parent capability
and translation and are not pure translation error. The interface remains **domain-specific**: BA@4
improves WikiText context-2048 perplexity from AB@4 83.4 to 34.8, but the better parent is still 8.27.

One central question remains open: whether the translation can be made near-lossless. The current controls do not
assign the whole chimera cost to cross-map R²: the encode–decode round-trip loss and the capability of the
selected front parent also contribute.

## Reproduce

Programs live in `../experiments/` with role-based launchers in `../experiments/slurm/`.
`../reproduce.sh check|tables|map|paper|gpu-eval` validates, aggregates, and launches them. Experiment outputs are
written to `../artifacts/metrics/raw/`, then normalized into paper-facing CSV files under
`../artifacts/metrics/tables/`. The reported runs used one V100-32GB node for early probes and a
4×V100-16GB node for training/evaluation. Model-parallel evaluators (`*_mp.py`) split one pair across two
16 GiB cards. Python, partition, CPU allocation, GPU request, and log paths are configured through the gitignored
`.env`; public launchers contain no site-local paths or cluster names.
