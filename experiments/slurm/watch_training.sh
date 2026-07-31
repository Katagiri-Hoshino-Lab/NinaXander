#!/bin/bash
# Watchdog for the 32-layer adapter run. USER-AUTHORISED to resubmit for THIS run only.
#
# Every failure mode below has actually happened on this project, and each fix here is a scar:
#   * WALLTIME. Long runs may require a continuation; this is expected and auto-recovered.
#   * TRUNCATED CHECKPOINT. SIGTERM at walltime lands mid-write on the periodic --out file. A previous watchdog
#     resumed from it and poisoned the run. We therefore never trust a file we have not torch.load-ed ourselves,
#     and we prefer the highest-step _step*.pt that actually opens, falling back to _best.pt.
#   * WRONG DONE MARKER. A previous watchdog grepped for a marker the script never printed and kept relaunching a
#     finished run. The markers here are the two the sbatch and the script really emit.
#   * STALL. NCCL can hang without dying; the job then holds four GPUs doing nothing until walltime. If no new eval
#     line appears for `limit` minutes we scancel to reclaim the node, then resume.
#   * NOT AUTO-FIXABLE. Disk-full and CUDA OOM are reported and the loop STOPS. Silently retrying either wastes the
#     allocation and, for disk-full, can make things worse.
#
# Usage:  nohup bash experiments/slurm/watch_training.sh <jobid> >> <state> 2>&1 &
[[ $# -eq 1 ]] || { echo "usage: $0 JOB_ID" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_config.sh"
SUBMIT="$SCRIPT_DIR/submit.sh"
LOGDIR="$SLURM_LOGS"
CKDIR="$CHECKPOINTS_DIR/z4096_L32"
STATE="$LOGDIR/watch_training.state"
PY="$PYTHON"
JOB=$1
mkdir -p "$LOGDIR"
tries=0
prev_evals=0
echo "$(date +%F' '%H:%M:%S) watchdog: tracking job $JOB" | tee -a "$STATE"

logfile() {
  for p in "$LOGDIR/train-adapter-$1.log" "$LOGDIR/continue-adapter-$1.log"; do [ -f "$p" ] && { echo "$p"; return; }; done
  echo "$LOGDIR/continue-adapter-$1.log"
}

newest_safe_ckpt() {   # highest-step _step*.pt that torch.load can actually open; else _best.pt
  for f in $(ls -1 "$CKDIR"/*_step*.pt 2>/dev/null | awk -F'step' '{print $2" "$0}' | sort -rn | awk '{print $2}'); do
    if "$PY" -c "import torch;torch.load('$f',map_location='cpu',weights_only=False)" 2>/dev/null; then echo "$f"; return; fi
  done
  B="$CKDIR/z4096_L32_b1_h4096_N228k_best.pt"
  if "$PY" -c "import torch;torch.load('$B',map_location='cpu',weights_only=False)" 2>/dev/null; then echo "$B"; return; fi
  echo ""                                                    # nothing safe -> caller must stop
}

while true; do
  sleep 60
  st=$(squeue -j "$JOB" -h -o "%t" 2>/dev/null)
  L=$(logfile "$JOB")
  if [ -n "$st" ]; then
    last=$(grep -ac "     step " "$L" 2>/dev/null); last=${last:-0}
    now=$(date +%s)
    if [ "$last" -gt "$prev_evals" ]; then prev_evals=$last; last_prog=$now; fi
    : "${last_prog:=$now}"
    stalled=$(( (now - last_prog) / 60 ))
    limit=75; [ "$last" -gt 0 ] && limit=25        # 75 min covers extract+standardise before the first eval
    echo "$(date +%H:%M:%S) job $JOB $st evals=$last stalled=${stalled}m/${limit}m $(grep -a '     step ' "$L" 2>/dev/null | tail -1 | grep -o 'A->B(ev)=[+-][0-9.]*')" >> "$STATE"
    if [ "$stalled" -ge "$limit" ]; then
      echo "$(date +%H:%M:%S) job $JOB STALLED ${stalled}m -> scancel to reclaim the node" | tee -a "$STATE"
      scancel "$JOB"; sleep 30; prev_evals=0; unset last_prog
    else
      continue
    fi
  fi

  # the job is gone. finished cleanly?
  if grep -qa "BEST held-out" "$L" 2>/dev/null && grep -qaE "L32_DONE|L32MORE_DONE" "$L" 2>/dev/null; then
    echo "$(date +%H:%M:%S) job $JOB COMPLETED cleanly" | tee -a "$STATE"; break
  fi

  tries=$((tries+1))
  if [ "$tries" -gt 6 ]; then echo "$(date +%H:%M:%S) 6 recoveries used -- STOPPING, needs a human" | tee -a "$STATE"; break; fi

  # Diagnose most-specific first and STOP at the first match. The previous order let a harmless startup line --
  # torch prints "ProcessGroupNCCL.cpp:5138 Guessing device ID" on every single run -- overwrite the correct
  # walltime diagnosis, so two ordinary walltime cycles were reported as NCCL failures. Match the NCCL *failure*
  # strings, never the module name alone.
  CAUSE="unknown"
  if   grep -qa "AssertionError" "$L" 2>/dev/null;                     then CAUSE="ASSERTION (a gate fired -- NOT auto-fixed)"
  elif grep -qai "No space left" "$L" 2>/dev/null;                     then CAUSE="disk-full (NOT auto-fixed)"
  elif grep -qai "CUDA out of memory" "$L" 2>/dev/null;                then CAUSE="CUDA OOM (NOT auto-fixed)"
  elif grep -qai "DUE TO TIME LIMIT" "$L" 2>/dev/null;                 then CAUSE="walltime"
  elif grep -qaiE "NCCL.*(timeout|error|abort|unhandled)" "$L" 2>/dev/null; then CAUSE="NCCL failure"
  fi
  echo "$(date +%H:%M:%S) job $JOB DIED cause=[$CAUSE] attempt=$tries" | tee -a "$STATE"
  case "$CAUSE" in *"NOT auto-fixed"*) echo "  not auto-recoverable -- STOPPING" | tee -a "$STATE"; break;; esac

  RES=$(newest_safe_ckpt)
  if [ -z "$RES" ]; then
    echo "  NO loadable checkpoint exists yet (died before the first save?) -- STOPPING" | tee -a "$STATE"; break
  fi
  STEP=$("$PY" -c "import torch;print(torch.load('$RES',map_location='cpu',weights_only=False)['step'])" 2>/dev/null)
  echo "  resuming from $(basename "$RES") (step $STEP)" | tee -a "$STATE"
  JOB=$("$SUBMIT" continue_adapter -- "$RES")
  echo "  relaunched as job $JOB" | tee -a "$STATE"
  prev_evals=0; unset last_prog
done
