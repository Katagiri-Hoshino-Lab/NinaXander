# NinaXander — 论文的参考仓库

[English](README.md) | [日本語](README.ja.md) | 简体中文

本仓库是论文《NinaXander：通过共享潜在表示组合不同模型族的冻结语言模型——成立条件与局限》
（[`paper/paper_en.pdf`](paper/paper_en.pdf)、[`paper/paper_ja.pdf`](paper/paper_ja.pdf)）的配套成果。
其目的是让论文读者能够定位任一报告数值背后的数据、从原始评测输出重建面向论文的表格，并重新运行把两者绑定起来的检查。
它不是通用库。

本研究冻结两个来自不同族系的 7B 指令跟随模型——具有线性注意力与递归推理形式的 RWKV-4-Raven-7B，以及属于 Transformer 的
open-instruct-pythia-6.9b-tulu——并在对应层的残差配对上训练单一的共享潜在适配器。组合方式为：运行一方父模型的前 `L+1`
个块，对残差只翻译一次，再运行另一方父模型的其余块；两个方向使用同一个适配器。

## 从论断追溯到数据

论文中印出的每个数值都源自规范化的 CSV 表，`tools/check_paper_claims.py` 会重新计算，一旦不一致就让构建失败。
从论断到数据的对应如下。

| 论文中的论断 | `artifacts/metrics/tables/` 下的表 |
|---|---|
| 潜在对齐、四路径读出 | `paper_representation.csv` |
| 逐层与全层对应 | `paper_layer_metrics.csv`、`paper_layer_correspondence.csv` |
| 四路径多项选择准确率 | `paper_four_path_qa_accuracy.csv` |
| 各边界上的嵌合体与仿射对照 | `paper_cross_family_qa_accuracy.csv` |
| 显著性检验与自助法区间 | `paper_cross_family_paired_statistics.csv` |
| 适配器非线性、α 干预 | `paper_cross_family_linearity.csv`、`paper_cross_family_mlp_intervention.csv` |
| KV 缓存与准确率的权衡 | `paper_cross_family_memory_accuracy.csv` |
| 域内与域外困惑度 | `paper_cross_family_domain_shift.csv` |
| 生成样例 | `paper_cross_family_generation_samples.csv` |
| 训练轨迹 | `paper_training_curve.csv` |
| 评测集暴露分析 | `artifacts/metrics/raw/eval_leakage.json` |

`./reproduce.sh map` 会列出所有表及其模式、行数与说明。
`artifacts/metrics/raw/` 保存未经变换的逐实验输出，其中包括配对检验所用的逐条正误记录。

## 运行检查

```bash
pip install -r requirements.txt
cp .env.example .env                 # Python、模型与 SLURM 的本地设置

./reproduce.sh check     # 结构、数据模式，以及论文与数据的一致性（CPU，数秒）
./reproduce.sh tables    # 从原始输出重建面向论文的 CSV 表
./reproduce.sh map       # 列出每个表及其模式与来源
./reproduce.sh paper     # 重新构建 PDF
```

`check` 会运行四道闸门：`validate_results.py`（模式、行数、跨路径一致性）、`check_paper_claims.py`（把每个头条
数值与表核对）、`check_en_number_parity.py`（英文版与日文版必须包含相同数值）、`validate_markdown_languages.py`。
四者均只需 CPU，不需要模型权重。

`tools/check_eval_leakage.py` 用于量化所报告的评测集有多少出现在 Transformer 一侧父模型的指令微调混合数据中。
它从逐条记录中剔除命中项后重新计算准确率，因此同样不需要 GPU。

## 复现实验

评测与训练的入口确实需要把两个冻结的 7B 父模型放在 GPU 上。

```bash
./reproduce.sh gpu-eval        # 提交评测（绝不提交训练）
./reproduce.sh hf-check        # 对三个发布包做静态验证
./reproduce.sh hf-gpu-check    # 对方向锁定的包做双 GPU 冒烟测试
```

两个父模型以 fp16 常驻需 28.9 GiB，因此 `*_mp.py` 评测器会把每一对拆到两张卡上。SLURM 启动脚本中不含任何检出路径、
分区、账户、QoS 或 GRES 取值；这些经由被 gitignore 的 `.env` 通过 `experiments/slurm/submit.sh` 注入，因为
`#SBATCH` 指令行会在 shell 读取 `.env` 之前被解析。参见
[`experiments/slurm/README.md`](experiments/slurm/README.md)。

适配器训练为 `experiments/adapter7b.py`。它超过 12 小时上限并恢复一次，会还原 optimizer、scheduler 与 global step。
所报告的检查点是 416,000 步运行中、留出集上 RWKV→Pythia 读出取得最大值的 step 415,000。

## Hugging Face 发布包

三个推理用包由 `tools/export_hf_adapter.py` 与 `tools/export_hf_chimera.py` 生成到
`release/huggingface/`（已 gitignore），并发布到公开的 Hub 仓库。

| 包 | 内容 |
|---|---|
| [`ninaxander-raven7b-tulu69-adapter`](https://huggingface.co/Katagiri-Hoshino-Lab/ninaxander-raven7b-tulu69-adapter) | 仅适配器与逐层统计量，附独立加载器 |
| [`ninaxander-tulu69-to-raven7b-best`](https://huggingface.co/Katagiri-Hoshino-Lab/ninaxander-tulu69-to-raven7b-best) | Pythia→RWKV 的 `L=4`，内含 fp16 双亲 |
| [`ninaxander-raven7b-to-tulu69-best`](https://huggingface.co/Katagiri-Hoshino-Lab/ninaxander-raven7b-to-tulu69-best) | RWKV→Pythia 的 `L=4`，内含 fp16 双亲 |

这些仓库已公开，并归入一个
[合集](https://huggingface.co/collections/Katagiri-Hoshino-Lab/ninaxander-inference-releases-6a9010088ba6e224ea2ea8dd)。
`release/huggingface/REMOTE_RELEASES.csv` 记录每个仓库经过验证的修订号、文件数与完整性状态。方向锁定包中的默认
`L=4` 是在所报告测试集上的事后选择，没有独立的选择划分；其他边界仍可在同一包自身的方向内选用。

许可遵循限制更强的父模型（AI2 AI Model License，仅限非商业研究），每个包都配有英文、日文与简体中文的模型卡。

## 论断的适用范围

论文的论断有意收窄，本仓库不会将其扩大。对应关系是在共享分词器、宽度、层数与基础预训练语料的一对模型上测得的；
训练只用了一个随机种子与一份指令格式语料；切换层是在所报告的同一批测试集上选定的。组合模型从未超过更强的父模型，
翻译在域外显著劣化。哪些结论成立、哪些不成立，论文的局限一节已写明；在复用此处任何数值之前请先阅读该节。

## 目录结构

```
paper/                论文源文件与构建好的 PDF
src/ninaxander/       适配器、RWKV 后段运行时、结果读写
experiments/          训练与评测入口
  slurm/              按角色划分的启动脚本
tools/                建表、校验、导出、泄漏检查
artifacts/
  metrics/raw/        实验直接输出、逐条记录
  metrics/tables/     规范化的面向论文 CSV
  checkpoints/        训练好的适配器（gitignore）
  logs/, runs/        诊断与提交记录（gitignore）
docs/                 方法、发现、路线图
release/huggingface/  导出器输出（gitignore）
models/               冻结的父模型权重（gitignore）
```

本仓库中的每份 Markdown 文档都有英文、日文与简体中文三种版本，`validate_markdown_languages.py` 会强制三者在结构上
保持一致。
