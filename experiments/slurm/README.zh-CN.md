# 可移植SLURM启动器

[English](README.md) | [日本語](README.ja.md) | 简体中文

`*.sbatch`文件只包含阶段逻辑、walltime和作业名。它们特意不包含checkout路径，也不包含站点专属的
`--partition`、`--output`、`--cpus-per-task`、`--gres`、account、QoS或constraint。

SLURM会在作业shell启动前解析`#SBATCH`指令，因此
`#SBATCH --partition=$MY_PARTITION`之类的指令无法可靠展开从`.env`加载的值。
支持的入口因此是`submit.sh`：它读取项目根目录的`.env`，把这些值作为`sbatch`命令行参数传入，
从而覆盖或补充可移植指令。

## 配置

```bash
cp .env.example .env
# 编辑.env
./reproduce.sh slurm-config
```

主要变量如下：

| 变量 | 用途 |
|---|---|
| `NINAXANDER_PYTHON`, `NINAXANDER_TORCHRUN` | Python环境 |
| `NINAXANDER_SLURM_GPU_PARTITION` | 训练和GPU评估partition |
| `NINAXANDER_SLURM_CPU_PARTITION` | 最终表格汇总partition |
| `NINAXANDER_SLURM_GPU_CPUS`, `NINAXANDER_SLURM_CPU_CPUS` | 每个作业申请的CPU数 |
| `NINAXANDER_SLURM_GRES` | 可选申请，例如`gpu:4`；GPU为隐式分配时留空 |
| `NINAXANDER_SLURM_ACCOUNT`, `NINAXANDER_SLURM_QOS`, `NINAXANDER_SLURM_CONSTRAINT` | 可选站点策略 |
| `NINAXANDER_GPU_0` … `NINAXANDER_GPU_3` | 作为两组model-parallel pair使用的四个物理GPU编号 |
| `NINAXANDER_RWKV_MODEL`, `NINAXANDER_PYTHIA_MODEL`, `NINAXANDER_TOKENIZER` | 父模型产物 |
| `NINAXANDER_ARTIFACTS_DIR` | checkpoint、metric、日志和run记录的可选位置 |
| `NINAXANDER_TRAIN_CACHE` | 训练时使用的大型临时残差cache |

默认值和可选设置见`.env.example`。`.env`被Git忽略，可以包含`HF_TOKEN`；切勿提交该文件。

## 提交

```bash
# 单个阶段
experiments/slurm/submit.sh eval_generation

# 匹配的AB/BA线性分析与同一适配器干预
experiments/slurm/submit.sh eval_cross_family_interventions

# 在两张GPU上验证适配器及两个锁定方向的Hugging Face发布包
experiments/slurm/submit.sh check_hf_release
# 带确认gate和结构化run记录的等价入口
./reproduce.sh hf-gpu-check

# 添加普通sbatch选项
experiments/slurm/submit.sh eval_qa --time=18:00:00

# 将sbatch选项与传给作业脚本的位置参数分隔
experiments/slurm/submit.sh continue_adapter --dependency=afterany:1234 -- /path/to/checkpoint.pt

# 运行完整评估DAG，随后重建表格并严格验证
./reproduce.sh gpu-eval

# 运行匹配的18-arm AB/BA QA和干预，随后验证表格
./reproduce.sh cross-family-eval
```

各角色阶段（`eval_qa`、`eval_robustness`、`eval_generation`和`eval_efficiency`）
都会产生匹配的AB与BA测量。父模型和self-path对照在规范化表中明确标为AA与BB，
不存在单独补齐某一方向的阶段。

复现driver将提交的作业ID和依赖写入
`artifacts/runs/*evaluation-submission-*.csv`；`./reproduce.sh status`同时列出实时队列与持久记录。

使用`--test-only`可让SLURM验证提交，而不创建作业：

```bash
experiments/slurm/submit.sh eval_generation --test-only
experiments/slurm/submit.sh build_tables --test-only
```

从项目根目录直接执行`sbatch experiments/slurm/eval_qa.sbatch`在语法上仍然有效，
但partition、CPU和stdout将由调度器默认值决定。`submit.sh`才是可复现路径；
它还会固定作业工作目录，使每个复制出的SLURM脚本都能加载`experiments/slurm/_common.sh`。

`check_hf_release`应在三个release目录全部重新export后运行。它将adapter-only包与可信checkpoint精确比较，
在两个命名方向上运行其standalone示例，然后对锁定方向的Pythia→RWKV、RWKV→Pythia完整包执行
静态完整性检查及单token GPU smoke test。
