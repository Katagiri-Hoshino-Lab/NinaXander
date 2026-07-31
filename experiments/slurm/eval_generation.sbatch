#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -J nx-gen

source "experiments/slurm/_common.sh"

CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u experiments/a_to_b_chimera_gen_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" \
  --switch 4 --ntok 40 --temperature 0 --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/a_to_b_generation_samples.csv" \
  > "$EVAL_LOGS/a_to_b_generation_samples.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u experiments/b_to_a_chimera_gen_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" \
  --switch 4 --ntok 40 --temperature 0 --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/b_to_a_generation_samples.csv" \
  > "$EVAL_LOGS/b_to_a_generation_samples.log" 2>&1

cat "$EVAL_LOGS/a_to_b_generation_samples.log"
cat "$EVAL_LOGS/b_to_a_generation_samples.log"
echo "EVAL_GENERATION_AB_BA_DONE"
