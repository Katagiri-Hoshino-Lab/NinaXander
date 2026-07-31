# 研究产物

[English](README.md) | [日本語](README.ja.md) | 简体中文

生成的研究产物和本地研究产物按用途分开存放：

- `checkpoints/`：适配器检查点。仅限本地；整个目录被Git忽略。
- `metrics/raw/`：实验程序的直接输出。JSON保留嵌套结构和逐题记录；配套CSV提供矩形测量数据。
- `metrics/tables/`：由`tools/build_result_tables.py`确定性生成、面向论文的CSV文件。
- `logs/slurm/`：本地调度器标准输出和历史诊断信息。
- `logs/evaluation/`：当前各评估程序的本地诊断日志。
- `runs/`：由`reproduce.sh gpu-eval`生成的结构化作业提交记录。
- `figures/`：按需生成的图形资源。

日志被Git忽略，不是权威数值数据集。公开实验记录应使用`metrics/raw/`，论文所报告关系应使用
`metrics/tables/`。特别是，`metrics/raw/training_curve.csv`替代了此前对站点本地训练日志的依赖。

`metrics/tables/paper_result_coverage.csv`是机器可读的对称性审计。每一类公开的跨模型族结果都必须
同时包含匹配的AB和BA行；标准QA矩阵还要在每个任务和切换层包含AA与BB。Pythia提前退出被归类为
`B_early_exit`，不会误计为BB适配器对照。若某测量系列在方法上不适用AA/BB对照，
`four_path_complete`将留空，而不会把匹配的AB/BA实验错误标成不完整。
