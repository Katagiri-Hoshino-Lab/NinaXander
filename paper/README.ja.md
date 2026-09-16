# NinaXander — 論文

[English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

shared-latent chimera研究の論文です。

- **[paper_ja.tex](paper_ja.tex)** — 論文（日本語、pLaTeX、情報処理学会研究報告形式）。abstract、図、table、
  再現情報と指標の定義（現在は手法の節に統合）、参考文献を含む主たる版です。報告結果はすべて
  `../artifacts/metrics/tables/`に正規化CSV表現を持ち、raw JSON/CSVは
  `../artifacts/metrics/raw/`に保持します。
- **[paper_en.tex](paper_en.tex)** — `paper_ja.tex`の英語版です。`../tools/check_en_number_parity.py`が
  両者の複数桁・小数の数値が一致することを検査します（1桁の数字の差は警告のみ）。

`../reproduce.sh paper`、または次のcommandでbuildします（英語版は`paper_en`で同じ手順を繰り返します）。

```bash
platex -kanji=utf8 -interaction=nonstopmode -halt-on-error paper_ja.tex &&
pbibtex -kanji=utf8 paper_ja &&
platex -kanji=utf8 -interaction=nonstopmode -halt-on-error paper_ja.tex &&
platex -kanji=utf8 -interaction=nonstopmode -halt-on-error paper_ja.tex &&
dvipdfmx paper_ja.dvi
```

両論文は情報処理学会研究報告（IPSJ SIG Technical Report）のclass（`ipsj.cls` v4.1と`ipsjtech.sty`、
日本語版は`ipsjunsrt.bst`、英語版は`ipsjunsrt-e.bst`）を使用します。これらはIPSJ配布の
`ipsj_v4-1.zip`（UTF-8版、2025-02-05）からこのdirectoryに同梱しています。classはpLaTeXでのみ動作するため、
buildにはTeX LiveのpLaTeX、pBibTeX、dvipdfmxに加え、`japanese-otf`と、
`kanji-config-updmap-sys haranoaji`で設定した`ptex-fontmaps`が必要です（dvipdfmxが原ノ味フォントを埋め込みます）。

層対応の図を除くすべての図は外部画像を使わないself-contained TikZです。`fig_layer_cka.pdf`と
`fig_layer_cka_en.pdf`は`../tools/make_layer_cka_figure.py`がtableから生成します（`../reproduce.sh paper`は
これらの再生成と`paper_en.pdf`のbuildも行います）。全数値は現在の**32層**adapter
（`SNAP_L32_final.pt`、step 415,000）に基づきます。held-out ρ_ctr 0.901、
AA/AB/BB/BA read-out 0.695/0.559/0.711/0.590、対応linear probe
（AB/BA nonlinear fraction 13.4%/12.0%、最良direct affine 0.487/0.535）、
AA/AB/BB/BA全量QA、実測KV（16.0 KiB/token/layer）、長文context domain shiftを含みます。
等KV early exit baselineとpruned/unpruned servingの測定はtableに残していますが、論文には掲載していません。

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
