#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -J nx-robust

source "experiments/slurm/_common.sh"

# Phase 1: AB.
CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u experiments/a_to_b_longctx_ce_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --text wikitext \
  --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/a_to_b_longctx_wikitext.json" \
  > "$EVAL_LOGS/a_to_b_longctx_wikitext.log" 2>&1 &
pid_wiki=$!

CUDA_VISIBLE_DEVICES="$GPU_2,$GPU_3" "$PYTHON" -u experiments/a_to_b_longctx_ce_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --text alpaca \
  --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/a_to_b_longctx_alpaca.json" \
  > "$EVAL_LOGS/a_to_b_longctx_alpaca.log" 2>&1 &
pid_alpaca=$!

wait_all "$pid_wiki" "$pid_alpaca"

# Phase 2: BA under the same domains and context windows.
CUDA_VISIBLE_DEVICES="$GPU_0,$GPU_1" "$PYTHON" -u experiments/b_to_a_longctx_ce_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --text wikitext \
  --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/b_to_a_longctx_wikitext.json" \
  > "$EVAL_LOGS/b_to_a_longctx_wikitext.log" 2>&1 &
pid_ba_wiki=$!

CUDA_VISIBLE_DEVICES="$GPU_2,$GPU_3" "$PYTHON" -u experiments/b_to_a_longctx_ce_mp.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --text alpaca \
  --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/b_to_a_longctx_alpaca.json" \
  > "$EVAL_LOGS/b_to_a_longctx_alpaca.log" 2>&1 &
pid_ba_alpaca=$!

wait_all "$pid_ba_wiki" "$pid_ba_alpaca"
tail -n 36 "$EVAL_LOGS/a_to_b_longctx_wikitext.log"
tail -n 36 "$EVAL_LOGS/a_to_b_longctx_alpaca.log"
tail -n 36 "$EVAL_LOGS/b_to_a_longctx_wikitext.log"
tail -n 36 "$EVAL_LOGS/b_to_a_longctx_alpaca.log"
echo "EVAL_ROBUSTNESS_AB_BA_DONE"
