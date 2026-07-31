# 当前32层实验

[English](README.md) | [日本語](README.ja.md) | 简体中文

本目录仅包含当前可运行的NinaXander流水线。

## Python入口

- `adapter7b.py`：适配器训练及结构化训练曲线输出。
- `a_to_b_chimera_bench_full_mp.py`：标准AB完整ARC-Easy/SciQ评估，包含两个父模型以及匹配的
  AB、AB α=0、affine和BB arm。
- `b_to_a_chimera_bench_full_mp.py`：对应的BA评估器，包含两个父模型以及匹配的BA、BA α=0、
  affine和AA arm。答案选项保持RWKV batch size为1；四个arm采用分别通过gate的CUDA stream，
  而不是数值不同的fused batch。
- `linearity.py`：匹配的AB/BA affine imitation与direct-affine probe。
- `cross_family_mlp_intervention.py`：匹配的AB/BA residual-MLP表征干预。
  两个标准QA程序也直接输出相应downstream α=0和α=1 arm。
- `four_path_layer_metrics.py`、`all_layer_corr.py`：AA/AB/BB/BA表征与层级对照。
- `a_to_b_longctx_ce_mp.py`、`b_to_a_longctx_ce_mp.py`：AB和BA域内/域外长上下文交叉熵。
- `a_to_b_chimera_gen_mp.py`、`b_to_a_chimera_gen_mp.py`：结构化定性生成样本。
- `paired_stats.py`：两条跨族路径的同题配对统计。
- `a_to_b_serving_bench_mp.py`、`b_to_a_serving_bench_mp.py`：匹配的AB/BA未剪枝延迟、内存和解码测量。
- `a_to_b_serving_pruned.py`、`b_to_a_serving_pruned.py`：匹配的AB/BA物理剪枝serving路径。
- `kvcache_measure.py`：可按方向参数化的AB/BA实测KV核算。
- `equal_mem_baseline.py`：按方向参数化、逐题对齐的Pythia提前退出对照，其Transformer深度与
  每条AB/BA路径保留的深度完全相同。

共享适配器定义和原子结果写入器位于`../src/ninaxander/`；实验脚本不会重复定义它们。

## SLURM入口

`slurm/`按角色组织：

- `train_adapter.sbatch`、`continue_adapter.sbatch`：仅用于训练。
- `eval_representation.sbatch`、`eval_qa.sbatch`、`eval_robustness.sbatch`、
  `eval_generation.sbatch`、`eval_cross_family_interventions.sbatch`、
  `eval_efficiency.sbatch`：按测量角色组织的匹配AB/BA非训练评估。
- `check_hf_release.sbatch`：对适配器以及分别锁定方向的Pythia→RWKV、RWKV→Pythia
  Hugging Face发布包执行精确checkpoint检查和双GPU执行检查，包括匹配的跨族runtime-smoke bundle。
- `build_tables.sbatch`：评估依赖完成后重建并验证规范化论文表。

站点专属设置不写入`*.sbatch`。将`../.env.example`复制为`../.env`并编辑，
再运行`../reproduce.sh slurm-config`检查解析后的值。单阶段使用`slurm/submit.sh`，
完整依赖链评估流水线使用`../reproduce.sh gpu-eval`。详见
[`slurm/README.zh-CN.md`](slurm/README.zh-CN.md)。

raw输出写入`../artifacts/metrics/raw/`，确定性的论文CSV写入
`../artifacts/metrics/tables/`。方向raw前缀保留生成记录的评估器来源，
面向论文的表始终连接匹配的AB/BA对。`../reproduce.sh check`会拒绝没有对应文件的
`a_to_b_*.py`或`b_to_a_*.py`入口，结果验证器也对raw JSON/CSV产物应用相同的双向配对规则。
