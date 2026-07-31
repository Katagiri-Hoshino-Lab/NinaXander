# NinaXander — 論文

[English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

shared-latent chimera研究の論文です。

- **[paper_ja.tex](paper_ja.tex)** — 論文（日本語、LaTeX / LuaTeX）。abstract、図、table、
  再現性appendix、参考文献を含む唯一の完全版です。報告結果はすべて
  `../artifacts/metrics/tables/`に正規化CSV表現を持ち、raw JSON/CSVは
  `../artifacts/metrics/raw/`に保持します（以前の英語版`paper.tex`は廃止しました）。

`../reproduce.sh paper`、または次のcommandでbuildします。

```bash
lualatex paper_ja.tex &&
bibtex paper_ja &&
lualatex paper_ja.tex &&
lualatex paper_ja.tex
```

すべての図は外部画像を使わないself-contained TikZです。全数値は現在の**32層**adapter
（`SNAP_L32_final.pt`、step 415,000）に基づきます。held-out ρ_ctr 0.901、
AA/AB/BB/BA read-out 0.695/0.559/0.711/0.590、対応linear probe
（AB/BA nonlinear fraction 13.4%/12.0%、最良direct affine 0.487/0.535）、
AA/AB/BB/BA全量QA、実測KV（16.0 KiB/token/layer）、等KV early exit、
pruned/unpruned serving、長文context domain shiftを含みます。

実測で最良のcross-family構成は`NinaXander-BA@4`です。ARC-Easy 62.2、SciQ 69.2、
Transformer block 5個、KV削減84.375%です。公開既定値であることは、独立selection splitを使わず、
8つのcross-family path/switch構成からpost-hocに選んだものと明記しています。
findingごとの来歴とguardrailは[../docs/ja/02-findings.md](../docs/ja/02-findings.md)を参照してください。
定性生成tableは、両cross-family pathの対応する`L=4`点について、英語promptを逐語的に使用し、
temperature 0、greedy argmax、最大40 new tokenで再生成します。

論文は実験で確立した修正後の枠組みを反映しています。raw corrではなくcentered alignment、
「emergent」「unsupervised」ではなく「cross-reconstruction target不要」、chimeraを
quality-preservingではなく**degraded-but-functional**、adapterを「near-linear」ではなく
low-*capacity*かつ明確に**非near-linear**と記述し、単一の「supervised ceiling比%」
（seed間変動内）も示しません。
