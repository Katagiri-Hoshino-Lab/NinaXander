# NinaXander — 论文

[English](README.md) | [日本語](README.ja.md) | 简体中文

共享潜空间奇美拉工作的论文说明。

- **[paper_ja.tex](paper_ja.tex)** — 论文（日文，LaTeX / LuaTeX）。这是唯一完整版本，
  包含摘要、图、表、复现附录和参考文献。每个报告结果在`../artifacts/metrics/tables/`
  中都有规范化CSV表示；raw JSON/CSV保留在`../artifacts/metrics/raw/`下
  （此前的英文`paper.tex`已停用）。

使用`../reproduce.sh paper`或直接执行以下命令构建：

```bash
lualatex paper_ja.tex &&
bibtex paper_ja &&
lualatex paper_ja.tex &&
lualatex paper_ja.tex
```

每幅图都是无需外部图片文件的self-contained TikZ。所有数值均来自当前**32层**适配器
（`SNAP_L32_final.pt`，step 415,000）：held-out ρ_ctr 0.901，
AA/AB/BB/BA read-out 0.695/0.559/0.711/0.590；匹配的线性probe
（AB/BA非线性比例13.4%/12.0%，最佳direct affine 0.487/0.535）；
AA/AB/BB/BA完整QA；实测KV（16.0 KiB/token/layer）；等KV提前退出；
剪枝/未剪枝serving及长上下文domain shift。

实测最佳跨族配置为`NinaXander-BA@4`：ARC-Easy 62.2、SciQ 69.2、
5个Transformer块、KV削减84.375%。论文明确说明其发布默认地位来自
8种跨族路径/切换配置上的post-hoc选择，未使用独立selection split。
各项发现的来源和guardrail见[../docs/zh-CN/02-findings.md](../docs/zh-CN/02-findings.md)。
定性生成表在两条跨族路径共同的`L=4`点重新生成，逐字使用英文prompt，
temperature为0、greedy argmax、最多40个新token。

论文采用实验期间确立的修正表述：报告centered alignment而非raw corr；
称“无需cross-reconstruction target”，而非“emergent”或“unsupervised”；
将奇美拉描述为**有退化但可运行**，而非保持质量；将适配器描述为低*容量*且明确
**并非近线性**；也不报告单一的“supervised ceiling百分比”（其差异处于seed波动内）。
