#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -J nx-qa

source "experiments/slurm/_common.sh"

# Phase 1: B-suffix/read-out evaluations (AB chimera and BB control).
CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u experiments/a_to_b_chimera_bench_full_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --task arc_easy --n 0 \
  --dev_a cuda:0 --dev_b cuda:1 --switches 4,8,16,24 \
  --dump "$RAW_METRICS/a_to_b_qa_items_arc.json" \
  > "$EVAL_LOGS/a_to_b_qa_arc_easy.log" 2>&1 &
pid_ab_arc=$!

CUDA_VISIBLE_DEVICES="$GPU_2,$GPU_3" "$PYTHON" -u experiments/a_to_b_chimera_bench_full_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --task sciq --n 0 \
  --dev_a cuda:0 --dev_b cuda:1 --switches 4,8,16,24 \
  --dump "$RAW_METRICS/a_to_b_qa_items_sciq.json" \
  > "$EVAL_LOGS/a_to_b_qa_sciq.log" 2>&1 &
pid_ab_sciq=$!

wait_all "$pid_ab_arc" "$pid_ab_sciq"
"$PYTHON" -u experiments/paired_stats.py \
  --dumps "$RAW_METRICS/a_to_b_qa_items_arc.json" "$RAW_METRICS/a_to_b_qa_items_sciq.json" \
  --out "$RAW_METRICS/a_to_b_paired_statistics.csv" \
  > "$EVAL_LOGS/a_to_b_paired_statistics.log" 2>&1

# Phase 2: validate both cross-family runtime boundaries, including the extra
# RWKV suffix diagnostics required by BA, before running the A-suffix/read-out
# evaluations (AA control and BA chimera).
CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u tools/check_cross_family_chimera.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/cross_family_runtime_smoke.json" \
  > "$EVAL_LOGS/cross_family_runtime_smoke.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u experiments/b_to_a_chimera_bench_full_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --task arc_easy --n 0 \
  --dev_a cuda:0 --dev_b cuda:1 --switches 4,8,16,24 \
  --dump "$RAW_METRICS/b_to_a_qa_items_arc.json" \
  > "$EVAL_LOGS/b_to_a_qa_arc.log" 2>&1 &
pid_ba_arc=$!

CUDA_VISIBLE_DEVICES="$GPU_2,$GPU_3" "$PYTHON" -u experiments/b_to_a_chimera_bench_full_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --task sciq --n 0 \
  --dev_a cuda:0 --dev_b cuda:1 --switches 4,8,16,24 \
  --dump "$RAW_METRICS/b_to_a_qa_items_sciq.json" \
  > "$EVAL_LOGS/b_to_a_qa_sciq.log" 2>&1 &
pid_ba_sciq=$!

wait_all "$pid_ba_arc" "$pid_ba_sciq"
"$PYTHON" -u experiments/paired_stats.py \
  --dumps "$RAW_METRICS/b_to_a_qa_items_arc.json" "$RAW_METRICS/b_to_a_qa_items_sciq.json" \
  --out "$RAW_METRICS/b_to_a_paired_statistics.csv" \
  > "$EVAL_LOGS/b_to_a_paired_statistics.log" 2>&1

tail -n 24 "$EVAL_LOGS/a_to_b_qa_arc_easy.log"
tail -n 24 "$EVAL_LOGS/a_to_b_qa_sciq.log"
tail -n 24 "$EVAL_LOGS/b_to_a_qa_arc.log"
tail -n 24 "$EVAL_LOGS/b_to_a_qa_sciq.log"
tail -n 20 "$EVAL_LOGS/a_to_b_paired_statistics.log"
tail -n 20 "$EVAL_LOGS/b_to_a_paired_statistics.log"
echo "EVAL_FOUR_PATH_QA_AA_AB_BB_BA_DONE"
