# Current 32-layer experiments

English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

This directory contains only the runnable current NinaXander pipeline.

## Python entry points

- `adapter7b.py`: adapter training and structured training-curve output.
- `a_to_b_chimera_bench_full_mp.py`: canonical AB full-set ARC-Easy/SciQ
  evaluation with both parents and matched AB, AB α=0, affine, and BB arms.
- `b_to_a_chimera_bench_full_mp.py`: the corresponding BA evaluator with
  both parents and matched BA, BA α=0, affine, and AA arms. Answer options
  remain at RWKV batch size one; its four arms use separately gated CUDA
  streams rather than a numerically different fused batch.
- `linearity.py`: matched AB/BA affine-imitation and direct-affine probes.
- `cross_family_mlp_intervention.py`: matched AB/BA residual-MLP representation intervention.
  Both canonical QA programs emit the corresponding downstream α=0 and α=1
  arms directly.
- `four_path_layer_metrics.py`, `all_layer_corr.py`: AA/AB/BB/BA representation and layer controls.
- `a_to_b_longctx_ce_mp.py`, `b_to_a_longctx_ce_mp.py`: in-domain and out-of-domain long-context cross-entropy for AB and BA.
- `a_to_b_chimera_gen_mp.py`, `b_to_a_chimera_gen_mp.py`: structured qualitative generation samples.
- `paired_stats.py`: same-item paired statistics for both cross-family paths.
- `a_to_b_serving_bench_mp.py`, `b_to_a_serving_bench_mp.py`: matched AB/BA
  unpruned latency, memory, and decode measurements.
- `a_to_b_serving_pruned.py`, `b_to_a_serving_pruned.py`: matched AB/BA
  physically pruned serving paths.
- `kvcache_measure.py`: direction-parameterized measured KV accounting for
  both AB and BA.
- `equal_mem_baseline.py`: direction-parameterized, item-aligned Pythia
  early-exit controls at the Transformer depth retained by each AB/BA path.

Shared adapter definitions and atomic result writers live in `../src/ninaxander/`; experiment scripts do not
duplicate those definitions.

## SLURM entry points

`slurm/` is organized by role:

- `train_adapter.sbatch`, `continue_adapter.sbatch`: training only.
- `eval_representation.sbatch`, `eval_qa.sbatch`, `eval_robustness.sbatch`,
  `eval_generation.sbatch`, `eval_cross_family_interventions.sbatch`,
  `eval_efficiency.sbatch`: matched AB/BA non-training evaluations, organized by measurement role.
- `check_hf_release.sbatch`: exact-checkpoint and two-GPU execution checks for
  the adapter plus the separate direction-locked Pythia→RWKV and RWKV→Pythia
  Hugging Face packages, including the matched cross-family runtime-smoke bundle.
- `build_tables.sbatch`: rebuild and validate the normalized paper tables after evaluation dependencies finish.

Site-specific settings are kept out of `*.sbatch`. Copy `../.env.example` to `../.env`, edit it, and inspect the
resolved values with `../reproduce.sh slurm-config`. Use `slurm/submit.sh` for a single stage or
`../reproduce.sh gpu-eval` for the dependency-linked evaluation pipeline. Full details are in
[`slurm/README.md`](slurm/README.md).

Raw outputs go to
`../artifacts/metrics/raw/`; deterministic paper-facing CSV files go to `../artifacts/metrics/tables/`.
Directional raw prefixes preserve which evaluator produced the record; the
paper-facing tables always join the matched AB/BA pair.
`../reproduce.sh check` rejects an `a_to_b_*.py` or `b_to_a_*.py` entry point
without its peer, and the result validator applies the same bidirectional
pairing rule to raw JSON/CSV artifacts.
