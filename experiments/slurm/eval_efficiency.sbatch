#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -J nx-efficient

source "experiments/slurm/_common.sh"

# Phase 1: the two-GPU unpruned serving benchmark and the short one-GPU KV probe.
CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u experiments/a_to_b_serving_bench_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/a_to_b_serving_bench.json" \
  > "$EVAL_LOGS/a_to_b_serving_bench.log" 2>&1 &
pid_serving=$!

CUDA_VISIBLE_DEVICES="$GPU_2" "$PYTHON" -u experiments/kvcache_measure.py \
  --model "$PYTHIA_MODEL" --tokenizer "$TOKENIZER" \
  --direction A_to_B --device cuda --out "$RAW_METRICS/a_to_b_kv_cache.csv" \
  > "$EVAL_LOGS/a_to_b_kv_cache.log" 2>&1 &
pid_kv=$!

wait_all "$pid_serving" "$pid_kv"

# Phase 2: actual block-pruned serving plus equal-KV early-exit baselines.
CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u experiments/a_to_b_serving_pruned.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/a_to_b_serving_pruned.json" \
  > "$EVAL_LOGS/a_to_b_serving_pruned.log" 2>&1 &
pid_pruned=$!

CUDA_VISIBLE_DEVICES="$GPU_2" "$PYTHON" -u experiments/equal_mem_baseline.py \
  --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --direction A_to_B --task arc_easy --n 0 --switches 4,8,16,24 \
  --chimera_dump "$RAW_METRICS/a_to_b_qa_items_arc.json" \
  --dump "$RAW_METRICS/a_to_b_exit_items_arc.json" \
  > "$EVAL_LOGS/a_to_b_equal_memory_arc.log" 2>&1 &
pid_exit_arc=$!

CUDA_VISIBLE_DEVICES="$GPU_3" "$PYTHON" -u experiments/equal_mem_baseline.py \
  --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --direction A_to_B --task sciq --n 0 --switches 4,8,16,24 \
  --chimera_dump "$RAW_METRICS/a_to_b_qa_items_sciq.json" \
  --dump "$RAW_METRICS/a_to_b_exit_items_sciq.json" \
  > "$EVAL_LOGS/a_to_b_equal_memory_sciq.log" 2>&1 &
pid_exit_sciq=$!

wait_all "$pid_pruned" "$pid_exit_arc" "$pid_exit_sciq"

# Phase 3: matched BA unpruned serving and KV probe.
CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u experiments/b_to_a_serving_bench_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/b_to_a_serving_bench.json" \
  > "$EVAL_LOGS/b_to_a_serving_bench.log" 2>&1 &
pid_ba_serving=$!

CUDA_VISIBLE_DEVICES="$GPU_2" "$PYTHON" -u experiments/kvcache_measure.py \
  --model "$PYTHIA_MODEL" --tokenizer "$TOKENIZER" \
  --direction B_to_A --device cuda --out "$RAW_METRICS/b_to_a_kv_cache.csv" \
  > "$EVAL_LOGS/b_to_a_kv_cache.log" 2>&1 &
pid_ba_kv=$!

wait_all "$pid_ba_serving" "$pid_ba_kv"

# Phase 4: matched BA pruned serving and equal-KV controls.
CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u experiments/b_to_a_serving_pruned.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/b_to_a_serving_pruned.json" \
  > "$EVAL_LOGS/b_to_a_serving_pruned.log" 2>&1 &
pid_ba_pruned=$!

CUDA_VISIBLE_DEVICES="$GPU_2" "$PYTHON" -u experiments/equal_mem_baseline.py \
  --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --direction B_to_A --task arc_easy --n 0 --switches 4,8,16,24 \
  --chimera_dump "$RAW_METRICS/b_to_a_qa_items_arc.json" \
  --dump "$RAW_METRICS/b_to_a_exit_items_arc.json" \
  > "$EVAL_LOGS/b_to_a_equal_memory_arc.log" 2>&1 &
pid_ba_exit_arc=$!

CUDA_VISIBLE_DEVICES="$GPU_3" "$PYTHON" -u experiments/equal_mem_baseline.py \
  --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --direction B_to_A --task sciq --n 0 --switches 4,8,16,24 \
  --chimera_dump "$RAW_METRICS/b_to_a_qa_items_sciq.json" \
  --dump "$RAW_METRICS/b_to_a_exit_items_sciq.json" \
  > "$EVAL_LOGS/b_to_a_equal_memory_sciq.log" 2>&1 &
pid_ba_exit_sciq=$!

wait_all "$pid_ba_pruned" "$pid_ba_exit_arc" "$pid_ba_exit_sciq"
tail -n 30 "$EVAL_LOGS/a_to_b_serving_bench.log"
tail -n 30 "$EVAL_LOGS/a_to_b_serving_pruned.log"
cat "$EVAL_LOGS/a_to_b_kv_cache.log"
tail -n 30 "$EVAL_LOGS/a_to_b_equal_memory_arc.log"
tail -n 30 "$EVAL_LOGS/a_to_b_equal_memory_sciq.log"
tail -n 30 "$EVAL_LOGS/b_to_a_serving_bench.log"
tail -n 30 "$EVAL_LOGS/b_to_a_serving_pruned.log"
cat "$EVAL_LOGS/b_to_a_kv_cache.log"
tail -n 30 "$EVAL_LOGS/b_to_a_equal_memory_arc.log"
tail -n 30 "$EVAL_LOGS/b_to_a_equal_memory_sciq.log"
echo "EVAL_EFFICIENCY_AB_BA_DONE"
