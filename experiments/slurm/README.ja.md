# 可搬なSLURM launcher

[English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

`*.sbatch`ファイルには、stage logic、walltime、job nameだけを記述します。checkout pathやサイト固有の
`--partition`、`--output`、`--cpus-per-task`、`--gres`、account、QoS、constraintは含めません。

SLURMはjob shellの起動前に`#SBATCH` directiveを解析するため、
`#SBATCH --partition=$MY_PARTITION`のようなdirectiveでは`.env`から読み込んだ値を確実に展開できません。
そのため対応entry pointは`submit.sh`です。project rootの`.env`を読み、これらの値を
`sbatch` command-line引数として渡すことで、可搬なdirectiveを上書きまたは補完します。

## 設定

```bash
cp .env.example .env
# .envを編集
./reproduce.sh slurm-config
```

主な変数は次のとおりです。

| 変数 | 用途 |
|---|---|
| `NINAXANDER_PYTHON`, `NINAXANDER_TORCHRUN` | Python環境 |
| `NINAXANDER_SLURM_GPU_PARTITION` | 学習とGPU評価用partition |
| `NINAXANDER_SLURM_CPU_PARTITION` | 最終table集計用partition |
| `NINAXANDER_SLURM_GPU_CPUS`, `NINAXANDER_SLURM_CPU_CPUS` | jobごとの要求CPU数 |
| `NINAXANDER_SLURM_GRES` | `gpu:4`などの任意request。GPUが暗黙割当される場合は空欄 |
| `NINAXANDER_SLURM_ACCOUNT`, `NINAXANDER_SLURM_QOS`, `NINAXANDER_SLURM_CONSTRAINT` | 任意のサイトpolicy |
| `NINAXANDER_GPU_0` … `NINAXANDER_GPU_3` | 2つのmodel-parallel pairとして使う4個の物理GPU index |
| `NINAXANDER_RWKV_MODEL`, `NINAXANDER_PYTHIA_MODEL`, `NINAXANDER_TOKENIZER` | 親成果物 |
| `NINAXANDER_ARTIFACTS_DIR` | checkpoint、metric、log、run記録の任意保存先 |
| `NINAXANDER_TRAIN_CACHE` | 学習中に使う大容量一時residual cache |

既定値と任意設定は`.env.example`を参照してください。`.env`はGit管理外で、`HF_TOKEN`を含められますが、
絶対にcommitしないでください。

## 投入

```bash
# 単一stage
experiments/slurm/submit.sh eval_generation

# 対応するAB/BA linearityおよび同一adapter介入
experiments/slurm/submit.sh eval_cross_family_interventions

# adapterと方向固定Hugging Face package 2件を2 GPUで検証
experiments/slurm/submit.sh check_hf_release
# 確認gateと構造化run recordを備えた等価entry point
./reproduce.sh hf-gpu-check

# 通常のsbatch optionを追加
experiments/slurm/submit.sh eval_qa --time=18:00:00

# sbatch optionとjob scriptへ渡す位置引数を分離
experiments/slurm/submit.sh continue_adapter --dependency=afterany:1234 -- /path/to/checkpoint.pt

# 評価DAG全体を実行し、その後table再構築と厳格検証
./reproduce.sh gpu-eval

# 対応する18-arm AB/BA QAと介入を実行し、その後table検証
./reproduce.sh cross-family-eval
```

各role stage（`eval_qa`、`eval_robustness`、`eval_generation`、`eval_efficiency`）は、
対応するAB/BA測定を生成します。親モデルとself-path対照は正規化tableでAA・BBと明示し、
一方向だけを後から補うstageはありません。

再現driverは、投入job IDとdependencyを
`artifacts/runs/*evaluation-submission-*.csv`へ記録します。`./reproduce.sh status`は、
live queueと永続記録の両方を表示します。

`--test-only`を使うと、jobを作成せずSLURMへ投入内容の検証だけを依頼できます。

```bash
experiments/slurm/submit.sh eval_generation --test-only
experiments/slurm/submit.sh build_tables --test-only
```

project rootから`sbatch experiments/slurm/eval_qa.sbatch`を直接実行しても構文上は有効ですが、
partition、CPU、stdoutはschedulerの既定値に従います。再現可能な経路は`submit.sh`です。
これはjob working directoryも固定するため、copyされたすべてのSLURM scriptが
`experiments/slurm/_common.sh`を読み込めます。

`check_hf_release`は3つのrelease directoryをすべて再exportした後に実行します。
adapter-only packageを信頼済みcheckpointと照合し、そのstandalone exampleを両方向で実行した後、
方向固定Pythia→RWKV・RWKV→Pythia完全packageにstatic integrity checkと1-token GPU smoke testを行います。
