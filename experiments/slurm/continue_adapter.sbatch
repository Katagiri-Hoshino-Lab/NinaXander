#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH -J nx-continue

source "experiments/slurm/_common.sh"

# Continuation of the 32-layer run. Hyper-parameters are identical to
# train_adapter.sbatch except --steps and --resume. The resume source is $1 so the watchdog can hand over the newest
# checkpoint it has VERIFIED loadable; the default is _best.pt, never the periodic --out file, because SIGTERM at
# walltime truncates that one mid-write (this cost a recovery cycle on the previous run).
# The /dev/shm cache is NOT removed on exit: re-extraction costs ~35 min and the tmpfs is dedicated to this node.
RESUME=${1:-$CHECKPOINTS_DIR/z4096_L32/z4096_L32_b1_h4096_N228k_best.pt}
echo "RESUMING FROM: $RESUME"
[[ -n "$TORCHRUN" ]] || { echo "torchrun not found; set NINAXANDER_TORCHRUN in .env" >&2; exit 1; }
mkdir -p "$TRAIN_CACHE"
df -h "$TRAIN_CACHE" | tail -1; du -sh "$TRAIN_CACHE" 2>/dev/null && echo "  cache OK -> extract 0s"
"$TORCHRUN" --standalone --nproc-per-node="$TRAIN_PROCESSES" experiments/adapter7b.py \
  --src "$RWKV_MODEL" \
  --tgt "$PYTHIA_MODEL" --tok "$TOKENIZER" \
  --N 228000 --z 4096 --blocks 1 --hid 4096 --sigma 0.40 \
  --steps 150000 --bs 1024 --lr 1e-3 --sched plateau --patience 3 \
  --eval_cut 2000000 --eval_windows 1000 --keep_every 10000 --seed 0 \
  --gpu_resident 0 --cachedir "$TRAIN_CACHE" --corpus alpaca \
  --resume "$RESUME" \
  --out "$CHECKPOINTS_DIR/z4096_L32/z4096_L32_b1_h4096_N228k.pt" \
  --metrics_out "$RAW_METRICS/training_curve.csv"
echo "L32MORE_DONE"
