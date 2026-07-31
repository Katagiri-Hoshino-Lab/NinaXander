#!/bin/bash
# Submit one NinaXander stage with site settings injected from .env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_config.sh"

usage() {
  cat <<'EOF'
Usage:
  experiments/slurm/submit.sh config
  experiments/slurm/submit.sh STAGE [SBATCH_OPTIONS...] [-- SCRIPT_ARGUMENTS...]

Stages:
  train_adapter, continue_adapter
  eval_representation, eval_qa, eval_robustness
  eval_generation, eval_cross_family_interventions, eval_efficiency
  check_hf_release
  build_tables

Partition, output path, CPU count, GRES, account, QoS, and constraint are read
from the project-root .env. Extra sbatch options may be supplied before `--`.
EOF
}

if [[ "${1:-}" == "config" ]]; then
  printf '%-32s %s\n' \
    "project_root" "$PROJECT_ROOT" \
    "environment_file" "$NINAXANDER_ENV_FILE" \
    "python" "$PYTHON" \
    "torchrun" "${TORCHRUN:-<not found>}" \
    "gpu_partition" "${SLURM_GPU_PARTITION:-<scheduler default>}" \
    "cpu_partition" "${SLURM_CPU_PARTITION:-<scheduler default>}" \
    "gpu_cpus" "$SLURM_GPU_CPUS" \
    "cpu_cpus" "$SLURM_CPU_CPUS" \
    "gres" "${SLURM_GRES:-<not requested>}" \
    "slurm_log_dir" "$SLURM_LOGS" \
    "rwkv_model" "$RWKV_MODEL" \
    "pythia_model" "$PYTHIA_MODEL" \
    "final_adapter" "$FINAL_ADAPTER"
  exit 0
fi

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
stage="$1"
shift

case "$stage" in
  train_adapter)       script="train_adapter.sbatch";       class="gpu"; log="train-adapter" ;;
  continue_adapter)    script="continue_adapter.sbatch";    class="gpu"; log="continue-adapter" ;;
  eval_representation) script="eval_representation.sbatch"; class="gpu"; log="eval-representation" ;;
  eval_qa)             script="eval_qa.sbatch";             class="gpu"; log="eval-qa" ;;
  eval_robustness)     script="eval_robustness.sbatch";     class="gpu"; log="eval-robustness" ;;
  eval_generation)     script="eval_generation.sbatch";     class="gpu"; log="eval-generation" ;;
  eval_cross_family_interventions)
                        script="eval_cross_family_interventions.sbatch"
                        class="gpu"; log="eval-cross-family-interventions" ;;
  eval_efficiency)     script="eval_efficiency.sbatch";     class="gpu"; log="eval-efficiency" ;;
  check_hf_release)    script="check_hf_release.sbatch";    class="gpu"; log="check-hf-release" ;;
  build_tables)        script="build_tables.sbatch";        class="cpu"; log="build-tables" ;;
  *) echo "Unknown stage: $stage" >&2; usage >&2; exit 2 ;;
esac

sbatch_options=()
script_arguments=()
parsing_script_arguments=0
for argument in "$@"; do
  if [[ "$argument" == "--" && "$parsing_script_arguments" -eq 0 ]]; then
    parsing_script_arguments=1
  elif [[ "$parsing_script_arguments" -eq 0 ]]; then
    sbatch_options+=("$argument")
  else
    script_arguments+=("$argument")
  fi
done

mkdir -p "$SLURM_LOGS"
options=(
  --parsable
  --chdir="$PROJECT_ROOT"
  --output="$SLURM_LOGS/$log-%j.log"
)

if [[ "$class" == "gpu" ]]; then
  [[ -n "$SLURM_GPU_PARTITION" ]] && options+=(--partition="$SLURM_GPU_PARTITION")
  [[ -n "$SLURM_GPU_CPUS" ]] && options+=(--cpus-per-task="$SLURM_GPU_CPUS")
  [[ -n "$SLURM_GRES" ]] && options+=(--gres="$SLURM_GRES")
else
  [[ -n "$SLURM_CPU_PARTITION" ]] && options+=(--partition="$SLURM_CPU_PARTITION")
  [[ -n "$SLURM_CPU_CPUS" ]] && options+=(--cpus-per-task="$SLURM_CPU_CPUS")
fi
[[ -n "$SLURM_ACCOUNT" ]] && options+=(--account="$SLURM_ACCOUNT")
[[ -n "$SLURM_QOS" ]] && options+=(--qos="$SLURM_QOS")
[[ -n "$SLURM_CONSTRAINT" ]] && options+=(--constraint="$SLURM_CONSTRAINT")

sbatch "${options[@]}" "${sbatch_options[@]}" "$SCRIPT_DIR/$script" "${script_arguments[@]}"
