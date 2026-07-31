#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH -J nx-tables

source "experiments/slurm/_common.sh"
"$PYTHON" -u tools/build_result_tables.py \
  --raw-dir "$RAW_METRICS" --table-dir "$TABLES" --log-dir "$SLURM_LOGS"
./reproduce.sh check
"$PYTHON" -u tools/validate_results.py \
  --raw-dir "$RAW_METRICS" --table-dir "$TABLES" --strict-runtime
echo "BUILD_TABLES_DONE"
