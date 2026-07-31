#!/bin/bash
#
# Runtime configuration shared by local commands, SLURM jobs, and submit.sh.
# Repository paths are derived from this file, so the checkout can live anywhere.
# Site-specific values belong in the gitignored project-root .env file.

NINAXANDER_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$NINAXANDER_SLURM_DIR/../.." && pwd)"
NINAXANDER_ENV_FILE="${NINAXANDER_ENV_FILE:-$PROJECT_ROOT/.env}"
[[ "$NINAXANDER_ENV_FILE" == /* ]] || NINAXANDER_ENV_FILE="$PROJECT_ROOT/$NINAXANDER_ENV_FILE"

if [[ -f "$NINAXANDER_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$NINAXANDER_ENV_FILE"
  set +a
fi

resolve_executable() {
  local candidate="$1"
  if [[ "$candidate" == */* ]]; then
    [[ -x "$candidate" ]] && { printf '%s\n' "$candidate"; return 0; }
  else
    command -v "$candidate" 2>/dev/null && return 0
  fi
  return 1
}

project_path() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$PROJECT_ROOT" "${value#./}"
  fi
}

model_reference() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    printf '%s\n' "$value"
  elif [[ "$value" == ./* || "$value" == ../* || "$value" == models/* || -e "$PROJECT_ROOT/$value" ]]; then
    project_path "$value"
  else
    # A non-local value such as organization/model is left as a Hugging Face Hub ID.
    printf '%s\n' "$value"
  fi
}

python_candidate="${NINAXANDER_PYTHON:-python3}"
if ! PYTHON="$(resolve_executable "$python_candidate")"; then
  echo "Python executable not found: $python_candidate" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -n "${NINAXANDER_TORCHRUN:-}" ]]; then
  TORCHRUN="$NINAXANDER_TORCHRUN"
elif [[ -x "$(dirname "$PYTHON")/torchrun" ]]; then
  TORCHRUN="$(dirname "$PYTHON")/torchrun"
else
  TORCHRUN="$(command -v torchrun 2>/dev/null || true)"
fi

ARTIFACTS_DIR="$(project_path "${NINAXANDER_ARTIFACTS_DIR:-artifacts}")"
CHECKPOINTS_DIR="$ARTIFACTS_DIR/checkpoints"
RAW_METRICS="$ARTIFACTS_DIR/metrics/raw"
TABLES="$ARTIFACTS_DIR/metrics/tables"
SLURM_LOGS="$(project_path "${NINAXANDER_SLURM_LOG_DIR:-${ARTIFACTS_DIR#$PROJECT_ROOT/}/logs/slurm}")"
EVAL_LOGS="$(project_path "${NINAXANDER_EVAL_LOG_DIR:-${ARTIFACTS_DIR#$PROJECT_ROOT/}/logs/evaluation}")"
RUNS_DIR="$ARTIFACTS_DIR/runs"

FINAL_ADAPTER="$(project_path "${NINAXANDER_FINAL_ADAPTER:-${CHECKPOINTS_DIR#$PROJECT_ROOT/}/z4096_L32/SNAP_L32_final.pt}")"
RWKV_MODEL="$(model_reference "${NINAXANDER_RWKV_MODEL:-models/rwkv-raven-7b-hf-fp16}")"
PYTHIA_MODEL="$(model_reference "${NINAXANDER_PYTHIA_MODEL:-models/tulu-pythia69-fp16}")"
TOKENIZER="${NINAXANDER_TOKENIZER:-EleutherAI/gpt-neox-20b}"
TRAIN_CACHE="$(project_path "${NINAXANDER_TRAIN_CACHE:-${SLURM_TMPDIR:-/dev/shm}/ninaxander_L32}")"

GPU_0="${NINAXANDER_GPU_0:-0}"
GPU_1="${NINAXANDER_GPU_1:-1}"
GPU_2="${NINAXANDER_GPU_2:-2}"
GPU_3="${NINAXANDER_GPU_3:-3}"
TRAIN_PROCESSES="${NINAXANDER_TRAIN_PROCESSES:-4}"

SLURM_GPU_PARTITION="${NINAXANDER_SLURM_GPU_PARTITION:-}"
SLURM_CPU_PARTITION="${NINAXANDER_SLURM_CPU_PARTITION:-}"
SLURM_GPU_CPUS="${NINAXANDER_SLURM_GPU_CPUS:-8}"
SLURM_CPU_CPUS="${NINAXANDER_SLURM_CPU_CPUS:-2}"
SLURM_GRES="${NINAXANDER_SLURM_GRES:-}"
SLURM_ACCOUNT="${NINAXANDER_SLURM_ACCOUNT:-}"
SLURM_QOS="${NINAXANDER_SLURM_QOS:-}"
SLURM_CONSTRAINT="${NINAXANDER_SLURM_CONSTRAINT:-}"

NINAXANDER_HF_OFFLINE="${NINAXANDER_HF_OFFLINE:-0}"
NINAXANDER_OMP_THREADS="${NINAXANDER_OMP_THREADS:-2}"
