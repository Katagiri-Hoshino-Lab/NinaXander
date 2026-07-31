# NinaXander：组合不同模型族的冻结语言模型

[English](README.md) | [日本語](README.ja.md) | 简体中文

本仓库包含**“NinaXander：基于共享潜在表示的跨模型族冻结语言模型组合——成立条件与局限”**的代码、结构化结果数据、
运行日志和日文论文（[`paper/paper_ja.pdf`](paper/paper_ja.pdf)）。

**一句话概述。** 两个来自不同架构族的*冻结*7B语言模型——RWKV-4-Raven-7B（纯RNN）和
open-instruct-pythia-6.9b-tulu（Transformer）——在匹配条件下的对应层共享token级表征空间；
一个小型学习适配器能够充分恢复这种对应，从而构建可运行的**AB与BA奇美拉**，
每个模型只需一次残差转换。路径很重要：在深层边界上，Pythia前半→RWKV后半明显更强。
在@4处，AB与BA双任务均值分别为65.25%和65.70%，BA只保留5个Transformer块。
奇美拉能够对话，但事实能力退化，且适配器仍然具有很强的domain-specific性。

> **NinaXander**这一名称来自《钢之炼金术师》中的奇美拉——融合成功了，它也能运行（会说话），
> 却不及融合前的存在。本项目的奇美拉同样如此：可以运行，但达不到更强的父模型。
> 这个名称正是取自这一对应关系。

> 范围：一个模型对、一份语料（Alpaca）、一个seed。参见论文§Limitations。
> 公开结果contract只包含当前32层RWKV-Raven/Tulu-Pythia实验。

## 核心数值

全部来自当前适配器：**z=4096（无压缩）、268.5M参数、σ=0.4、32层、415,000步**
（作业1832 → 1837），通过forward hook捕获每个块的残差。
held-out最深点（step 415,000）：

| | |
|---|---|
| centered latent alignment ρ_ctr | **0.901** |
| 四路径read-out AA / AB / BB / BA | 0.695 / **0.559** / **0.711** / 0.590 |
| 最佳direct affine cross-map AB / BA | 0.487 / 0.535（不可约非线性输出：13.4% / 12.0%） |
| SciQ acc_norm（N=1000）：Pythia / AB@4 / BA@4 / RWKV | 81.0 / 70.9 / **69.2** / 67.0 |
| ARC-Easy acc_norm（N=2376）：Pythia / AB@4 / BA@4 / RWKV | 65.3 / 59.6 / **62.2** / 61.8 |
| @4删除的KV：AB（RWKV→Pythia）/ BA（Pythia→RWKV） | 15.625% / **84.375%** |

公开结果采用对称的**四路径记法**。第一个字母表示prefix模型族，第二个表示
suffix/read-out模型族，`@L`表示切换层：

| prefix ↓ / suffix → | A = RWKV | B = Pythia |
|---|---:|---:|
| **A = RWKV** | `NinaXander-AA@L`（同族对照） | `NinaXander-AB@L`（奇美拉） |
| **B = Pythia** | `NinaXander-BA@L`（奇美拉） | `NinaXander-BB@L`（同族对照） |

`L=4`的完整normalized accuracy：

| 任务 | AA | AB | BB | BA |
|---|---:|---:|---:|---:|
| ARC-Easy | 62.8 | 59.6 | 62.2 | 62.2 |
| SciQ | 68.5 | 70.9 | 74.8 | 69.2 |

支撑结论的四项发现：

1. **两种cross-readout都低于匹配的self-reconstruction。**
   ρ_ctr 0.901 > B→B 0.711 > A→B 0.559，以及
   ρ_ctr 0.901 > A→A 0.695 > B→A 0.590；这些不等式在32层全部成立。
   潜空间之间的一致程度远高于任一decoder重构残差的能力，所以稀缺资源是*重构*而不是对齐。
   取消压缩（z = d = 4096）也没有让任一self-map接近1，限制因素仍未解决。
   旧稿宣称1/(1+σ²) ≈ 0.862为“noise ceiling”是错误的：那是*noisy*-eval值；
   本研究在clean z上评估，ceiling约为0.98，所以B→B 0.711远低于它，
   noise**不是**binding constraint，需要σ sweep确定真正原因。
2. **两条跨族路径都不是“近线性的”。** 最佳affine imitation解释AB适配器输出的86.6%、
   BA输出的88.0%，剩余13.4% / 12.0%为不可约非线性。direct affine map达到
   AB 0.487、BA 0.535；移除同一个已训练residual-MLP branch会使相应read-out跌至−0.41与−0.30。
3. **两条执行路径都能生成文本，但事实能力都有退化。** 四个prompt逐字以英文提供，
   使用确定性greedy argmax（temperature 0），最多40个新token，只在EOS时提前停止。
   论文样本使用共同frontier点`L=4`：两条路径都回答“Paris”并正确开始primary-colors答案；
   两者都未完成水的问题；AB能给出正确的一句black-hole定义，BA则在颜色问题上漂移到对话，
   并给出循环的black-hole定义。在完整配对测试集上，每个AB/BA切换点都高于随机猜测，
   但低于更强的Pythia父模型。AB@4只在SciQ上超过RWKV（+3.9，p=0.019），
   在ARC上更差（−2.2，p=0.042）；没有BA切换点显著超过RWKV。
   对称BB/AA对照也说明差距并非纯translation error：AB低于BB，而深层BA可能高于AA，
   因为prefix父模型同时改变。接口在两条路径上都具有**domain-specific性**：
   context 2048的WikiText perplexity为AB@4 83.4、BA@4 34.8，而更好的父模型为8.27。
4. **AB与BA的深度profile不同。** BA@16/@24比同层AB配置高8.9–18.8个百分点；
   BA@4削减84.375%的Transformer KV，同时得到62.2/69.2。它仍在两个任务上低于Pythia，
   且未显著超过RWKV。双任务均值65.70只比AB@4高0.45个百分点；把它作为发布默认值是在
   两个报告测试集上对2条跨族路径 × 4个切换点进行的**post-hoc**排序，
   没有独立deployment-selection split。BA@4减轻WikiText损害，但context 2048的
   perplexity仍为34.8，而更好的父模型为8.27。

匹配的同一适配器干预也按对称方式报告。每项为
完整适配器 → 禁用residual-MLP后的normalized accuracy（%）：

| L | ARC-Easy AB | ARC-Easy BA | SciQ AB | SciQ BA |
|---:|---:|---:|---:|---:|
| 4 | 59.6 → 31.9 | 62.2 → 31.8 | 70.9 → 31.3 | 69.2 → 32.6 |
| 8 | 59.0 → 29.9 | 57.3 → 35.5 | 67.7 → 31.4 | 64.6 → 37.0 |
| 16 | 48.6 → 28.5 | 57.5 → 43.7 | 54.5 → 31.7 | 68.0 → 47.3 |
| 24 | 47.0 → 34.8 | 59.6 → 48.4 | 51.1 → 40.0 | 69.9 → 49.1 |

全部16项逐题配对比较均有McNemar p<2e-9。因此学习得到的nonlinear branch在AB和BA中
都起关键作用，并非从单条路径推断出的效果。

## 快速开始

```bash
pip install -r requirements.txt      # torch、transformers==5.2.0、datasets
cp .env.example .env                 # 随后编辑Python、模型和SLURM设置
./reproduce.sh slurm-config          # 显示解析后的非secret配置

./reproduce.sh check     # 验证可移植源文件和现有结构化数据（数秒、无需GPU）
./reproduce.sh tables    # 从raw JSON/CSV产物重建论文CSV
./reproduce.sh map       # 列出每个论文CSV、行数和说明
./reproduce.sh paper     # 重建paper/paper_ja.pdf
./reproduce.sh hf-check  # 对三个Hub发布包执行静态/checkpoint验证
./reproduce.sh hf-gpu-check  # 提交两个锁定方向的GPU smoke test
./reproduce.sh gpu-eval  # 提交匹配的AB/BA非训练评估
```

`check`、`tables`、`map`、`paper`和`hf-check`只需CPU。
`hf-gpu-check`和评估程序需要两个冻结7B父模型、最终适配器及GPU。`gpu-eval`从不提交训练。

SLURM启动器不包含本地checkout路径、partition、输出路径、account、QoS、GRES或CPU数量。
这些值从Git忽略的`.env`读取，由`experiments/slurm/submit.sh`注入。
原因是`#SBATCH`行会在shell source `.env`之前解析，直接写在指令中的变量无法可靠展开。
配置变量与直接提交示例见
[`experiments/slurm/README.zh-CN.md`](experiments/slurm/README.zh-CN.md)。

## 结果数据contract

每项评估直接写出机器可读数据。嵌套/逐题数据保留为JSON，论文中的每项数值关系都规范化为CSV：

- `artifacts/metrics/raw/`：每个实验一份JSON/CSV bundle，外加配对检验与生成CSV。
- 生成行记录`decoding=greedy_argmax`、`temperature=0`、
  `max_new_tokens=40`和`seed=0`；prompt本身逐字保存。
- `artifacts/metrics/raw/training_curve.csv`：公开417点训练轨迹，无需本地调度器日志。
- `artifacts/metrics/tables/`：用于审计论文报告值的稳定、已连接CSV表。
- `artifacts/metrics/tables/MANIFEST.csv`：表名、schema、行数和说明。
- `artifacts/metrics/tables/paper_result_coverage.csv`：逐表AA/AB/BB/BA计数；
  每个公开跨族结果系列必须同时覆盖匹配AB与BA。截断Pythia baseline单独记为
  `B_early_exit`，绝不作为BB适配器self-map。四路径完整标志只在方法上适用
  AA/BB self-map对照的结果系列中填写。
- `tools/build_result_tables.py`：确定性的raw-to-table转换。
- `tools/validate_results.py`：验证schema、行数、JSON、manifest、release metadata、
  匹配18-arm AB/BA QA、两条路径的boundary gate（含BA direct-suffix/concurrency gate）、
  pruning/generation gate、精确等内存深度、serving/KV一致性、配对raw-CSV schema一致性，
  以及跨路径parent-item一致性。

AB和BA的端到端测量都不从表征R²推断。每条路径直接执行，并以匹配的
`a_to_b_*` / `b_to_a_*` raw名称输出。标准四路径QA矩阵为
`paper_four_path_qa_accuracy.csv`。其余配对AB/BA评估合并到
`paper_cross_family_*`，保留独立`producer_path`来源字段和真实`execution_path`列，
不重复父模型行；规范化公开表不会拆成AB-only与BA-only副本。

每次评估后运行`./reproduce.sh tables && ./reproduce.sh check`。
runtime日志只用于来源与诊断，不再作为论文数值的数据源。

## Hugging Face发布包

轻量adapter-only Hub候选包staged在本地
`release/huggingface/ninaxander-raven7b-tulu69-adapter`（被Git忽略；由`tools/export_hf_adapter.py`生成）。
它以SafeTensors包含FP32适配器及逐层统计量，并提供standalone loader、
带gate的奇美拉生成示例、父产物fingerprint、model card和适用的research-only license。
3.1 GiB训练checkpoint中的optimizer/scheduler state及本地绝对路径均被排除。

完整模型（同样由`tools/export_hf_chimera.py`在本地生成，被Git忽略）按明确执行路线拆分：

- `release/huggingface/ninaxander-tulu69-to-raven7b-best`
  为Pythia→RWKV，实测最佳默认L=4：Pythia块`0..4`、一次转换、RWKV块`5..31`。
- `release/huggingface/ninaxander-raven7b-to-tulu69-best`
  为RWKV→Pythia，实测最佳默认L=4：RWKV块`0..4`、一次转换、Pythia块`5..31`。

每个完整包约28 GiB，包含两个准确fp16父模型、tokenizer、step-415,000适配器、
所有逐层标准化tensor及显式switch-plan tensor。路线被锁定；
L=8/16/24只用于同一路线内受控比较。两个L=4默认值都是在报告测试上post-hoc选择的，
没有独立selection split。

三个发布包均通过与`SNAP_L32_final.pt`的精确tensor比较；完整bundle还支持
`AutoModelForCausalLM`加载、parent-reproduction injection gate和双GPU生成。
每个包均提供英文、日文和简体中文model card。完整静态发布审计运行`./reproduce.sh hf-check`。

private Hub snapshot（需要访问权限）：
[collection](https://huggingface.co/collections/kotama7/ninaxander-private-inference-releases-6a6b02ec40c18d6b41ed040a)、
[adapter](https://huggingface.co/kotama7/ninaxander-raven7b-tulu69-adapter)、
[Pythia→RWKV best](https://huggingface.co/kotama7/ninaxander-tulu69-to-raven7b-best)和
[RWKV→Pythia best](https://huggingface.co/kotama7/ninaxander-raven7b-to-tulu69-best)。
`tools/export_hf_adapter.py`与`tools/export_hf_chimera.py`分别是轻量和完整包的确定性exporter。
只修改文档后，`tools/update_hf_release_checksums.py`会更新三个完整性manifest，
且共享hard-link父模型shard只hash一次。

由于本地HF转换时没有保留原始BlinkDL Raven `.pth`的准确文件名/revision，
三个包仍是release candidate；最终发布前需恢复该provenance。

## 目录结构

```
src/ninaxander/       可复用适配器、RWKV-suffix runtime和result-I/O包
.env.example          本地Python/模型/SLURM设置的公开模板
experiments/          当前32层训练/评估入口
  slurm/              按角色组织的启动器：train、evaluate、aggregate
tools/                表格builder、validator和Hub exporter
artifacts/
  checkpoints/        已训练适配器（本地，Git忽略）
  metrics/raw/        实验直接输出
  metrics/tables/     规范化论文CSV
  logs/               本地、Git忽略的SLURM/评估诊断
  runs/               结构化提交记录（本地，Git忽略）
paper/                论文源文件与构建的PDF
docs/                 项目级方法、发现和路线图
release/huggingface/  三个仅推理Hugging Face发布包（exporter输出，Git忽略）
models/               本地冻结父模型权重（Git忽略）
```

## 复现某个具体数值

`./reproduce.sh map`列出规范化CSV表。例如，
`paper_four_path_qa_accuracy.csv`保存AA/AB/BB/BA矩阵；
`paper_cross_family_qa_accuracy.csv`保存AB与BA的parent/chimera/control合并行；
`paper_cross_family_paired_statistics.csv`保存两条路径及其同题比较的McNemar检验和
paired-bootstrap区间；`paper_cross_family_linearity.csv`保存匹配AB/BA非线性分析；
`paper_cross_family_mlp_intervention.csv`保存同一适配器alpha干预，
其任务级raw bundle也包含匹配AB/BA汇总与逐题结果；
`paper_cross_family_domain_shift.csv`保存域内/域外交叉熵研究的合并结果。

`raw/cross_family_runtime_smoke.{json,csv}`记录四个切换点经GPU验证的AB/BA boundary gate。
标准QA程序为`experiments/a_to_b_chimera_bench_full_mp.py`和
`experiments/b_to_a_chimera_bench_full_mp.py`；适配器训练程序为`experiments/adapter7b.py`。

## 硬件 / 环境

- **需要GPU。** 报告集群用一台V100-32GB节点完成早期probe，
  用一台4×V100-16GB节点完成训练/评估。模型对需要28.9 GiB resident memory
  （RWKV 14 GiB + Pythia 13 GiB，fp16），所以16 GiB GPU评估把每对模型拆到两张卡上
  （`*_mp.py`：RWKV在`cuda:0`，Pythia在`cuda:1`；适配器与路径的suffix模型同卡）。
  BA评估器只把可达E_B/D_A部分放到GPU；BA QA还保留E_A用于AA decoder-loss对照。
  公开脚本不嵌入集群partition/GRES名称。
- **walltime。** 适配器训练超过12小时限制并恢复一次；`continue_adapter.sbatch`
  恢复optimizer、scheduler和global step（运行跨两个周期达416,000步；所有报告数值均使用held-out A→B最优的step 415,000检查点）。
- 数据（Alpaca、ARC-Easy、SciQ）通过`datasets`按需下载，缓存于`~/.cache/huggingface`。
- `models/`下的独立父模型副本仅供本地使用且被忽略。
  `release/huggingface/*-best/`下两个方向最佳候选有意打包准确fp16父模型，
  并上传至与源码项目分离的仓库。RWKV-Raven上游为BlinkDL `.pth`，经转换得到HF格式；
  两个父模型共享GPT-NeoX-20B tokenizer（硬性要求，见`docs/zh-CN/00-overview.md`）。
