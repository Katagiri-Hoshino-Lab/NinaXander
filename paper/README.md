# NinaXander — report

English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

The write-up of the shared-latent chimera work.

- **[paper_ja.tex](paper_ja.tex)** — the paper (日本語, pLaTeX, IPSJ SIG Technical Report format), the primary write-up: abstract,
  figures, tables, reproduction information and metric definitions (now part of the Method section) and a bibliography. Every reported
  result has a normalized CSV representation in `../artifacts/metrics/tables/`; raw JSON/CSV remains under
  `../artifacts/metrics/raw/`.
- **[paper_en.tex](paper_en.tex)** — the English mirror of `paper_ja.tex`. `../tools/check_en_number_parity.py`
  requires every multi-digit and decimal number to match between the two sources (single-digit differences only warn).

Build with `../reproduce.sh paper`, or directly. The Japanese paper uses the IPSJ class (pLaTeX); the English mirror
uses the IEEE conference format (`pdflatex paper_en && bibtex paper_en && pdflatex paper_en && pdflatex paper_en`):

```bash
platex -kanji=utf8 -interaction=nonstopmode -halt-on-error paper_ja.tex &&
pbibtex -kanji=utf8 paper_ja &&
platex -kanji=utf8 -interaction=nonstopmode -halt-on-error paper_ja.tex &&
platex -kanji=utf8 -interaction=nonstopmode -halt-on-error paper_ja.tex &&
dvipdfmx paper_ja.dvi
```

The Japanese paper uses the IPSJ SIG Technical Report class (情報処理学会研究報告; `ipsj.cls` v4.1 with `ipsjtech.sty`
and `ipsjunsrt.bst`), vendored in this directory from IPSJ's `ipsj_v4-1.zip` (UTF-8 edition, 2025-02-05). The class
runs only under pLaTeX, so the Japanese build needs TeX Live's pLaTeX, pBibTeX and dvipdfmx, plus `japanese-otf` and
`ptex-fontmaps` configured with `kanji-config-updmap-sys haranoaji` (dvipdfmx then embeds the Harano Aji fonts).
The English paper uses TeX Live's standard `IEEEtran` class and `IEEEtran.bst` and builds with pdfLaTeX, as on arXiv.

Every figure except the layer-correspondence figure is self-contained TikZ (no external image files); `fig_layer_cka.pdf` and
`fig_layer_cka_en.pdf` are generated from the tables by `../tools/make_layer_cka_figure.py` (`../reproduce.sh paper`
regenerates them and also builds `paper_en.pdf`). Every number is from the current **32-layer**
adapter (`SNAP_L32_final.pt`, step 415,000): held-out ρ_ctr 0.901 and
AA/AB/BB/BA read-outs 0.695/0.559/0.711/0.590; the
matched linearity probe (AB/BA non-linear fraction 13.4%/12.0%, best direct
affine 0.487/0.535); full-set QA for AA/AB/BB/BA; measured KV
(16.0 KiB/token/layer); and long-context domain shift. The equal-KV early-exit baseline and the pruned/unpruned
serving measurements remain in the tables but are not printed in the paper. The best
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
