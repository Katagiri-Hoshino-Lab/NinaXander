# NinaXander

[English](../README.md) | [日本語](../ja/README.md) | 简体中文

> **NinaXander是本项目最终生成的奇美拉模型名称。** 它是一个单一模型，
> 由不同架构族的冻结模块通过学习得到的共享潜空间适配器组装而成。下述工作均以它为目标；
> “NinaXander”始终指这个组合模型，而不是这一研究领域。

本项目通过学习得到的**共享潜空间适配器**，组合来自不同架构族
（RNN/SSM ↔ Transformer）的冻结预训练模块，构建一个**奇美拉**语言模型。

适配器是手段，而不是终点；目标是得到一个真正*能运行*的奇美拉。残差对齐R²只是代理指标。

## 文档

| 文件 | 内容 |
|---|---|
| [00-overview.md](00-overview.md) | NinaXander是什么、核心论点和目标模型对 |
| [01-method.md](01-method.md) | 共享潜空间适配器、损失函数及奇美拉构造 |
| [02-findings.md](02-findings.md) | 可追溯到具体运行、如实表述的已确立结果 |
| [03-roadmap.md](03-roadmap.md) | 已完成事项、开放问题和已知失败路线 |
| [../paper/](../../paper/) | 论文（`paper_ja.tex`） |

## 一段话概述现状

模型对为**RWKV-4-Raven-7B（纯RNN）↔ open-instruct-pythia-6.9b-tulu（Transformer）**：
两者均为32个块、d4096、GPT-NeoX tokenizer，并在Alpaca上对齐。一个包含268.5M参数、
潜空间宽度z=4096（无压缩）、σ=0.4的共享潜空间适配器训练了**415k步**
（作业1832 → 1837），通过forward hook捕获**全部32个块**的残差；held-out
**centered** ρ_ctr为0.901，AA/AB/BB/BA read-out为0.695/0.559/0.711/0.590。
两条read-out都受**重构而非对齐**限制：在全部32层上A→B低于B→B，B→A低于A→A。
取消压缩也未使任一decoder接近无损，限制self-map的因素仍未解决。旧稿中的
1/(1+σ²) ≈ 0.862“noise ceiling”其实是*noisy*-eval值；clean-z eval的ceiling约为0.98，
因此noise不是binding constraint，仍需进行σ sweep。

适配器在两个方向上都**不是近线性的**：AB和BA输出分别有13.4%和12.0%为不可约非线性成分，
禁用同一个nonlinear branch会使两条read-out崩溃。7B奇美拉已经构建并且**能够对话**——
两条路径都能回答“What is the capital of France?”——但在其余固定prompt上出现事实或格式退化。
在**完整测试集**（SciQ N=1000、ARC-Easy N=2376）上，AB@4得到70.9/59.6，
只在SciQ上超过RWKV；BA@4得到69.2/62.2，在两个任务上与RWKV统计持平。
两条跨族路径的全部配置都低于更强的Pythia父模型。BA@16/@24比同层AB高8.9至18.8个百分点，
说明深度效应依赖路径。

实测发布默认值为`NinaXander-BA@4`：它仅保留5个Transformer块，削减**84.375%**的KV，
但双任务均值65.70只比AB@4高0.45个百分点。这是在相同报告测试集上从
2条跨族路径 × 4个切换点中post-hoc选出的结果，并非独立deployment-selection split的验证。
两个方向的autoencoder对照差异都同时包含前半父模型能力与translation，不能解释为纯translation error。
接口依然具有明显的**domain-specific**特性：BA@4将WikiText context-2048 perplexity从
AB@4的83.4降至34.8，但更好的父模型仍为8.27。

一个核心问题仍然开放：translation能否做到接近无损。当前对照不能把奇美拉的全部损失归因于
cross-map R²；decoder重构以及所选前半父模型的能力也会产生影响。

## 复现

程序位于`../experiments/`，按角色组织的launcher位于`../experiments/slurm/`。
`../reproduce.sh check|tables|map|paper|gpu-eval`负责验证、汇总和提交任务。实验输出写入
`../artifacts/metrics/raw/`，随后规范化为`../artifacts/metrics/tables/`下的论文CSV。
报告运行使用一台V100-32GB节点完成早期probe，并使用一台4×V100-16GB节点进行训练和评估。
model-parallel evaluator（`*_mp.py`）将一个模型对拆到两张16 GiB GPU上。Python、partition、
CPU分配、GPU request和日志路径均通过Git忽略的`.env`配置；公开launcher不包含站点本地路径或集群名。
