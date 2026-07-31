#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -J nx-train

source "experiments/slurm/_common.sh"

# Canonical training contract: all 32 raw block outputs, captured by forward
# hooks, with one shared latent and the fixed public hyperparameters below.
[[ -n "$TORCHRUN" ]] || { echo "torchrun not found; set NINAXANDER_TORCHRUN in .env" >&2; exit 1; }
mkdir -p "$CHECKPOINTS_DIR/z4096_L32" "$TRAIN_CACHE"
df -h "$TRAIN_CACHE" | tail -1
"$TORCHRUN" --standalone --nproc-per-node="$TRAIN_PROCESSES" experiments/adapter7b.py \
  --src "$RWKV_MODEL" \
  --tgt "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --N 228000 --z 4096 --blocks 1 --hid 4096 --sigma 0.40 \
  --steps 100000 --bs 1024 --lr 1e-3 --sched plateau --patience 3 \
  --eval_cut 2000000 --eval_windows 1000 --keep_every 10000 --seed 0 \
  --gpu_resident 0 --cachedir "$TRAIN_CACHE" --corpus alpaca \
  --out "$CHECKPOINTS_DIR/z4096_L32/z4096_L32_b1_h4096_N228k.pt" \
  --metrics_out "$RAW_METRICS/training_curve.csv"
echo "L32_DONE"
