# 現行32層実験

[English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

このdirectoryには、現在実行可能なNinaXander pipelineだけを置いています。

## Python entry point

- `adapter7b.py`：adapter学習と構造化されたtraining curveの出力。
- `a_to_b_chimera_bench_full_mp.py`：両親モデルと、対応するAB、AB α=0、affine、BB armを含む、
  標準AB全量ARC-Easy/SciQ評価。
- `b_to_a_chimera_bench_full_mp.py`：両親モデルと、対応するBA、BA α=0、affine、AA armを含む
  対応BA evaluator。選択肢はRWKVのbatch size 1を維持します。4 armは、数値的に異なる
  fused batchではなく、個別にgateしたCUDA streamを使います。
- `linearity.py`：対応するAB/BA affine imitationとdirect-affine probe。
- `cross_family_mlp_intervention.py`：対応するAB/BA residual-MLP表現介入。
  2つの標準QA programは、対応するdownstream α=0/α=1 armも直接出力します。
- `four_path_layer_metrics.py`、`all_layer_corr.py`：AA/AB/BB/BA表現と層対照。
- `a_to_b_longctx_ce_mp.py`、`b_to_a_longctx_ce_mp.py`：AB/BAのdomain内外の長文context cross-entropy。
- `a_to_b_chimera_gen_mp.py`、`b_to_a_chimera_gen_mp.py`：構造化された定性生成sample。
- `paired_stats.py`：両cross-family pathの同一項目paired statistics。
- `a_to_b_serving_bench_mp.py`、`b_to_a_serving_bench_mp.py`：対応するAB/BA unpruned latency、
  memory、decode測定。
- `a_to_b_serving_pruned.py`、`b_to_a_serving_pruned.py`：対応するAB/BA物理pruned serving path。
- `kvcache_measure.py`：AB/BA両方向をparameter指定できる実測KV計算。
- `equal_mem_baseline.py`：各AB/BA pathで保持するTransformer depthに合わせた、方向指定可能で
  項目整列済みのPythia early-exit対照。

共有adapter定義とatomic result writerは`../src/ninaxander/`にあり、実験script側では重複定義しません。

## SLURM entry point

`slurm/`は役割別に整理しています。

- `train_adapter.sbatch`、`continue_adapter.sbatch`：学習専用。
- `eval_representation.sbatch`、`eval_qa.sbatch`、`eval_robustness.sbatch`、
  `eval_generation.sbatch`、`eval_cross_family_interventions.sbatch`、
  `eval_efficiency.sbatch`：測定の役割ごとに整理した、対応するAB/BA非学習評価。
- `check_hf_release.sbatch`：adapterと、方向固定のPythia→RWKV・RWKV→Pythia Hugging Face packageを
  正確なcheckpointおよび2 GPU実行で検査します。対応するcross-family runtime-smoke bundleも含みます。
- `build_tables.sbatch`：評価dependency完了後、正規化済み論文tableを再構築・検証。

サイト固有設定は`*.sbatch`に含めません。`../.env.example`を`../.env`へcopyして編集し、
`../reproduce.sh slurm-config`で解決済み値を確認してください。単一stageには`slurm/submit.sh`、
dependency付き評価pipeline全体には`../reproduce.sh gpu-eval`を使います。詳細は
[`slurm/README.ja.md`](slurm/README.ja.md)にあります。

raw出力は`../artifacts/metrics/raw/`へ、決定論的な論文用CSVは
`../artifacts/metrics/tables/`へ書きます。方向別raw prefixは、どのevaluatorが記録を生成したかを
保持しますが、論文用tableは常に対応するAB/BA pairを結合します。
`../reproduce.sh check`は対応相手のない`a_to_b_*.py`または`b_to_a_*.py` entry pointを拒否し、
結果validatorもraw JSON/CSV成果物に同じ双方向pairing ruleを適用します。
