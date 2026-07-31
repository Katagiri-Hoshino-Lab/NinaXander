# NinaXander

English | [日本語](ja/README.md) | [简体中文](zh-CN/README.md)

> **NinaXander is the name of the final chimera model** this project produces — a single model
> assembled from frozen blocks of different architecture families via a learned shared-latent adapter. The work
> below builds toward it; "NinaXander" always refers to that composed model, not the research area.

Building a **chimera** language model by composing frozen pretrained blocks from *different architecture families*
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
read-outs are **bounded by reconstruction, not alignment**: A→B is below B→B and B→A is below A→A at all 32 layers.
Removing compression does not make either decoder near-lossless; what limits the self-maps is unresolved
(the 1/(1+σ²) ≈ 0.862 "noise
ceiling" of an earlier draft was the *noisy*-eval value — under clean-z eval the ceiling is ≈0.98, so noise is not
the binding constraint; a σ sweep is needed). The adapter is **not** near-linear in either direction:
13.4% of AB and 12.0% of BA output is irreducibly non-linear, and disabling the same nonlinear branch collapses
both read-outs. The 7B chimeras exist and **converse** — both answer "What is the capital of France?" — but both
show factual or format degradation on the remaining fixed prompts. On the **full test sets**
(SciQ N=1000, ARC-Easy N=2376), AB@4 scores 70.9/59.6 and beats RWKV only on SciQ. BA@4
scores 69.2/62.2 and is statistically tied with RWKV on both tasks; every configuration in both
cross-family paths remains below the stronger Pythia parent. BA@16/@24 gains 8.9–18.8 points over the same-layer
AB path, establishing that the depth effect is path-dependent. The measured release default is
`NinaXander-BA@4`: it keeps five Transformer blocks and removes **84.375%** of KV, while its two-task mean (65.70) is only 0.45
points above AB@4. That is a post-hoc choice among 2 cross-family paths × 4 switches on the two reported test sets,
not independent deployment validation. Autoencoder-control gaps in either direction include both front-parent
capability and translation and are not pure translation error. The interface remains **domain-specific**: BA@4
improves WikiText context-2048 perplexity from AB@4 83.4 to 34.8, but the better parent is still 8.27.

One central question remains open: whether the translation can be made near-lossless. The current controls do not
assign the whole chimera cost to cross-map R²: decoder reconstruction and the capability of the selected front
parent also contribute.

## Reproduce

Programs live in `../experiments/` with role-based launchers in `../experiments/slurm/`.
`../reproduce.sh check|tables|map|paper|gpu-eval` validates, aggregates, and launches them. Experiment outputs are
written to `../artifacts/metrics/raw/`, then normalized into paper-facing CSV files under
`../artifacts/metrics/tables/`. The reported runs used one V100-32GB node for early probes and a
4×V100-16GB node for training/evaluation. Model-parallel evaluators (`*_mp.py`) split one pair across two
16 GiB cards. Python, partition, CPU allocation, GPU request, and log paths are configured through the gitignored
`.env`; public launchers contain no site-local paths or cluster names.
