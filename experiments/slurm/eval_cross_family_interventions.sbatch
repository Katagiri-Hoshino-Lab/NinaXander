#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -J nx-cross

source "experiments/slurm/_common.sh"

# The upstream eval_qa job has already emitted item-aligned alpha=1/alpha=0
# outcomes for both AB and BA. Run the two representation analyses in parallel,
# then join those QA outcomes into direction-balanced intervention bundles.
CUDA_VISIBLE_DEVICES="$GPU_0" "$PYTHON" -u experiments/linearity.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --blocks 1 --hid 4096 \
  --layers 4,10,16,22,28 --nwin 900 \
  --out "$RAW_METRICS/cross_family_linearity_L32.json" \
  > "$EVAL_LOGS/cross_family_linearity.log" 2>&1 &
pid_linearity=$!

CUDA_VISIBLE_DEVICES="$GPU_2,$GPU_3" "$PYTHON" -u experiments/cross_family_mlp_intervention.py \
  --rwkv "$RWKV_MODEL" --pythia "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --adapter "$FINAL_ADAPTER" --blocks 1 --hid 4096 \
  --r2_layers 4,10,16,22,28 --nwin 400 \
  --dev_a cuda:0 --dev_b cuda:1 \
  --out "$RAW_METRICS/cross_family_mlp_representation.json" \
  > "$EVAL_LOGS/cross_family_mlp_representation.log" 2>&1 &
pid_representation=$!

wait_all "$pid_linearity" "$pid_representation"
"$PYTHON" -u tools/complete_cross_family_mlp_bundle.py \
  --raw-dir "$RAW_METRICS"

"$PYTHON" -u experiments/paired_stats.py \
  --dumps "$RAW_METRICS/a_to_b_qa_items_arc.json" "$RAW_METRICS/a_to_b_qa_items_sciq.json" \
  --out "$RAW_METRICS/a_to_b_paired_statistics.csv" \
  > "$EVAL_LOGS/a_to_b_paired_statistics.log" 2>&1 &
pid_ab_stats=$!

"$PYTHON" -u experiments/paired_stats.py \
  --dumps "$RAW_METRICS/b_to_a_qa_items_arc.json" "$RAW_METRICS/b_to_a_qa_items_sciq.json" \
  --out "$RAW_METRICS/b_to_a_paired_statistics.csv" \
  > "$EVAL_LOGS/b_to_a_paired_statistics.log" 2>&1 &
pid_ba_stats=$!

wait_all "$pid_ab_stats" "$pid_ba_stats"

tail -n 24 "$EVAL_LOGS/cross_family_linearity.log"
tail -n 30 "$EVAL_LOGS/cross_family_mlp_representation.log"
tail -n 12 "$EVAL_LOGS/a_to_b_paired_statistics.log"
tail -n 12 "$EVAL_LOGS/b_to_a_paired_statistics.log"
echo "EVAL_CROSS_FAMILY_INTERVENTIONS_AB_BA_DONE"
