#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -J nx-repr

source "experiments/slurm/_common.sh"

CUDA_VISIBLE_DEVICES="$GPU_1" "$PYTHON" -u experiments/four_path_layer_metrics.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --nwin 400 \
  --out "$RAW_METRICS/four_path_layer_metrics.json" \
  > "$EVAL_LOGS/four_path_layer_metrics.log" 2>&1 &
pid_layers=$!

CUDA_VISIBLE_DEVICES="$GPU_2" "$PYTHON" -u experiments/all_layer_corr.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --nwin 200 \
  --out "$RAW_METRICS/all_layer_corr.json" \
  > "$EVAL_LOGS/all_layer_corr.log" 2>&1 &
pid_corr=$!

wait_all "$pid_layers" "$pid_corr"
tail -n 20 "$EVAL_LOGS/four_path_layer_metrics.log"
tail -n 20 "$EVAL_LOGS/all_layer_corr.log"
echo "EVAL_REPRESENTATION_DONE"
