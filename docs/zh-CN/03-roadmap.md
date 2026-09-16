# 03 · 路线图

[English](../03-roadmap.md) | [日本語](../ja/03-roadmap.md) | 简体中文

## 已完成：构建7B instruct模型对并完成适配器特性分析

**RWKV-4-Raven-7B（纯RNN、instruct）↔ allenai/open-instruct-pythia-6.9b-tulu
（Transformer、instruct）。** 两者均为32L、d4096、GPT-NeoX-20B tokenizer；
已验证bit-identical tokenization和convention auto-detect
（A：RNN，post-append `off=0`；B：Transformer，pre-append `off=1`）。
原计划的Dolly-v2-7b已**从HF删除**，所以改用Tulu
（备选：`usvsnsp/pythia-6.9b-sft`、`dvruette/oasst-pythia-6.9b`）。

Raven上游以BlinkDL `.pth`发布，经一个未收录的一次性helper转换
（为transformers 5.x把`shard_checkpoint`改为`save_pretrained`），
随后重新保存为**fp16**（fp32为28GB并发生OOM）。两个7B模型无法同时装入32GB V100，
因此每次只从**一个模型**提取残差，写入逐节点cache。位置与容量属于站点设置
（`NINAXANDER_TRAIN_CACHE`）；报告的32GB probe节点仅有23GB shared memory。

结果见[02-findings.md](02-findings.md)。在有利条件下确实存在token对齐表征的共享空间（而非一般的语义空间）：
报告适配器为z=4096（无压缩）、268.5M参数、σ=0.4、**415k步**
（作业1832 → 1837），通过forward hook捕获**全部32个块**的残差，达到
ρ_ctr 0.901、AA/AB/BB/BA = 0.695/0.559/0.711/0.590。
全部32层均满足AB≤BB与BA≤AA：即使在较高的潜空间对齐下，同族重构阶段就已存在误差，
两种cross-readout**在每一层都低于同族重构**。该误差来自encoder、潜空间归一化、decoder还是训练时noise，
尚未区分；限制self-map的因素仍未解决。
noise σ是否设定了ceiling尚不清楚，可通过σ sweep检验。

## 后续已完成：公平重训、两条奇美拉路径、对照与可部署路径

报告checkpoint通过forward hook捕获全部32个块。所有适配器数值均来自step-415,000 checkpoint。

两条7B执行路径AB和BA都已构建并测量。它们都能回答法国首都prompt，因而**可以对话**，
但在其余固定prompt上有事实或格式退化。在完整配对测试集上，AB@4为
ARC 59.6 / SciQ 70.9，BA@4为62.2 / 69.2；RWKV为61.8 / 67.0，
Pythia为65.3 / 81.0。AB@4只在SciQ上超过RWKV（p=0.02）；
BA@4在两个任务上与RWKV统计持平；两条路径均低于Pythia。
完整AB/BA表覆盖15.6–84.4%的实测KV削减；SciQ合并non-dominated序列为
Pythia → AB@4 → BA@24 → BA@4 → RWKV（post-hoc单任务frontier）。

另完成两项特性分析：适配器**不是近线性的**——AB与BA输出分别有13.4%和12.0%为不可约非线性，
把这个联合训练的适配器的residual-block branch设为α=0（AB为E_A与D_B中的块，BA为E_B与D_A中的块）
会使两种read-out和下游准确率崩溃。但这并不说明非线性普遍必要：在BA准确率上，单独拟合的affine对照
与适配器的差距很小（−3.4至+2.0分），适配器并未一致地更优。translation在**两条路径上都具有domain-specific性**：
context 512时从Alpaca转到WikiText，AB@4 perplexity由6.40升至93.87，
BA@4由3.85升至45.00，而父模型退化小得多。这是domain shift，而不是上下文长度限制。

每条路径都直接测量，不从表征R²推断：包含完整QA、父模型与同层路径的配对检验、
等KV提前退出、Alpaca/WikiText context 512/1024/2048、固定prompt生成、实测KV，
以及未剪枝/剪枝serving。适用的boundary、direct-suffix和pruning gate均为exact zero。
BA@4保留5个Transformer块（KV削减84.375%），得到ARC 62.2 / SciQ 69.2。
它在8种路径/切换组合中双任务均值实测最高，但只比AB@4高0.45个百分点，
且没有独立selection split；它是post-hoc发布默认值，不是经验证的普遍胜者。
fused RWKV option batch未通过numerical gate，所以报告BA QA仍采用batch 1。

## harness不变量（从错误中总结，禁止回归）

- **held-out window不得依赖任何训练hyper-parameter。** 过去使用
  `cut = N*6 + 2e6`，导致不同N的run在**不同文本**上评分。现在使用常量`--eval_cut`，
  并在truncation时通过assert明确失败。
- `windows()`只保留**最前面的**n个window，因此`--eval_windows`就是eval size
  （80意味着仅8,960个token），现为400。全部32层在*相同*window上评分，
  独立单位是**window**而不是layer。
- **绝不能在`_best`比较run**（存在selection bias，约20次评估，终点可能是outlier）。
  应比较匹配step；`--keep_every`现在保留周期checkpoint，可事后重建。
- **现在已提供`--seed`。** 关于run间波动的主张需要每个条件使用多个seed；
  当前论文只报告一个seed。

## 开放问题

- **什么限制self-reconstruction？** 在7B上，宽度或秩不太可能是强约束（z = d，且non-affine LN每个token
  只丢弃均值与范数），但B→B仍停留在0.711。真实原因仍然**开放**：误差来自encoder、潜空间归一化、
  decoder还是训练时noise尚未区分，noise σ是否设定了ceiling也尚不清楚。当前32层设置上匹配的σ sweep {0, 0.1, 0.2, 0.4, 0.8}可直接检验noise这一候选。
- **cross-reconstruction target的代价。** 需要当前规模的匹配多seed运行，尚无报告估计。
- **能否使translation接近无损？** cross-map R²仍是可能的改进杠杆，
  但当前对照无法把整个奇美拉代价归于它；encode–decode往返造成的损失与front-parent能力同样存在。
  尚未按当前规模尝试的方向包括更丰富/更深的decoder、vocabulary-anchored closed-form初始化，
  以及逐层而非共享decoder。

## 报告guardrail

- **raw corr /“emergent”表述。** 报告centered ρ与f，并把两种cross-readout称为
  “不需要cross-reconstruction target”。
- **tokenizer不同的instruct跨架构模型对。** 方法需要共享tokenizer以形成position-matched pair。
  Falcon-Mamba-Instruct（纯、64L）没有同tokenizer的32L Transformer partner；
  RecurrentGemma↔Gemma-2的块数和tokenizer匹配，但RecurrentGemma是含attention的hybrid。

## 延后ablation

z-dim sweep、λ_cross sweep、σ sweep和data-scale均未在32层设置上运行；
还包括奇美拉逐层decoder对共享decoder、utility对alignment分析，以及同架构
（同一模型族两个模型）的alignment对照。目前不声称这些延后实验已有可运行脚本。

7B上已完成、不再延后的项目是**shuffled-correspondence negative control**
（`four_path_layer_metrics.py`）：置换B行会使
ρ_ctr 0.90 → 0.00、A→B 0.56 → −0.59、B→A 0.59 → −0.61，
确认两种cross-readout都依赖token级对应关系。
