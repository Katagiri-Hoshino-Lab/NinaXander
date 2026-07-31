#!/bin/bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_config.sh"

cd "$PROJECT_ROOT"
mkdir -p "$RAW_METRICS" "$TABLES" "$SLURM_LOGS" "$EVAL_LOGS" "$RUNS_DIR"

export HF_HUB_OFFLINE="$NINAXANDER_HF_OFFLINE"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$NINAXANDER_OMP_THREADS"

wait_all() {
  local status=0
  local pid
  for pid in "$@"; do
    wait "$pid" || status=1
  done
  return "$status"
}
