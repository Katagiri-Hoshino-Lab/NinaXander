# NinaXander — report

English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

The write-up of the shared-latent chimera work.

- **[paper_ja.tex](paper_ja.tex)** — the paper (日本語, LaTeX / LuaTeX). The sole, complete write-up: abstract,
  figures, tables, a reproducibility appendix and a bibliography. Every reported
  result has a normalized CSV representation in `../artifacts/metrics/tables/`; raw JSON/CSV remains under
  `../artifacts/metrics/raw/`. (The earlier English `paper.tex` was retired.)

Build with `../reproduce.sh paper`, or directly:

```bash
lualatex paper_ja.tex &&
bibtex paper_ja &&
lualatex paper_ja.tex &&
lualatex paper_ja.tex
```

Every figure is self-contained TikZ (no external image files). Every number is from the current **32-layer**
adapter (`SNAP_L32_final.pt`, step 415,000): held-out ρ_ctr 0.901 and
AA/AB/BB/BA read-outs 0.695/0.559/0.711/0.590; the
matched linearity probe (AB/BA non-linear fraction 13.4%/12.0%, best direct
affine 0.487/0.535); full-set QA for AA/AB/BB/BA; measured KV
(16.0 KiB/token/layer); equal-KV early exits; and pruned/unpruned serving and long-context domain shift. The best
measured cross-family configuration is `NinaXander-BA@4`: ARC-Easy 62.2, SciQ 69.2, five Transformer blocks,
and 84.375% KV reduction. Its public-default status is explicitly reported as a post-hoc choice over eight
cross-family path/switch configurations without an independent selection split. See
[../docs/02-findings.md](../docs/02-findings.md) for per-finding provenance and guardrails.
The qualitative generation table is regenerated at the matched `L=4` point
for both cross-family paths with verbatim English prompts, temperature 0,
greedy argmax, and at most 40 new tokens.

The paper reflects the corrected framing established during the experiments: centered alignment (not raw corr),
"no cross-reconstruction target required" (not "emergent"/"unsupervised"), the chimera as **degraded-but-functional**
(not quality-preserving), the adapter as low-*capacity* and provably **not** "near-linear", and no single-number
"% of supervised ceiling" (within seed spread).
