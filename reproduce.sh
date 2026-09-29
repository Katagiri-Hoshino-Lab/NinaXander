#!/bin/bash
# Reproduction driver for the current 32-layer NinaXander study.
#
# Usage:
#   ./reproduce.sh check                  validate structure and existing result data
#   ./reproduce.sh tables                 rebuild normalized paper-facing CSV tables
#   ./reproduce.sh map                    list every structured table and its provenance
#   ./reproduce.sh paper                  rebuild paper/paper_ja.pdf and paper/paper_en.pdf
#                                         (IPSJ ipsj.cls: platex + pbibtex + dvipdfmx)
#   ./reproduce.sh hf-check               validate all three Hugging Face release packages
#   ./reproduce.sh hf-gpu-check [--yes]   submit two-GPU checks for both direction-best packages
#   ./reproduce.sh gpu-eval [--yes]       submit all non-training evaluations
#   ./reproduce.sh cross-family-eval [--yes] submit symmetric AB/BA interventions
#   ./reproduce.sh gpu [--yes]            submit training, continuation, then evaluations
#   ./reproduce.sh slurm-config            show resolved non-secret cluster settings
#   ./reproduce.sh status                 show queued jobs and recorded submissions
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SLURM_DIR="$ROOT/experiments/slurm"
source "$SLURM_DIR/_config.sh"
SUBMIT="$SLURM_DIR/submit.sh"
MODE="${1:-check}"
ASSUME_YES="${2:-}"

RUNS="$RUNS_DIR"
CHECKPOINT="$FINAL_ADAPTER"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/ninaxander-pycache"

confirm() {
  [[ "$ASSUME_YES" == "--yes" ]] && return 0
  printf "proceed? [y/N] "
  read -r answer
  [[ "$answer" == "y" ]]
}

require_file() {
  [[ -f "$ROOT/$1" ]] || {
    echo "MISSING: $1"
    return 1
  }
}

# grep exit 0 = violation found, 1 = clean, >=2 = the gate could not scan its
# inputs. All three must be distinguished: treating >=2 as clean would silently
# disable a gate (the same failure mode as a missing search binary).
run_gate() {
  local message="$1"
  shift
  local status=0
  grep "$@" || status=$?
  if [[ "$status" -eq 0 ]]; then
    echo "ERROR: $message"
    failure=1
  elif [[ "$status" -ge 2 ]]; then
    echo "ERROR: gate could not scan its inputs (grep exit $status): $message"
    failure=1
  fi
}

submit_eval_pipeline() {
  local upstream="${1:-}"
  local dependency=()
  [[ -n "$upstream" ]] && dependency=(--dependency="afterok:$upstream")

  local representation qa robustness generation interventions efficiency tables
  representation=$("$SUBMIT" eval_representation "${dependency[@]}")
  qa=$("$SUBMIT" eval_qa "${dependency[@]}")
  robustness=$("$SUBMIT" eval_robustness "${dependency[@]}")
  generation=$("$SUBMIT" eval_generation "${dependency[@]}")
  interventions=$("$SUBMIT" eval_cross_family_interventions --dependency="afterok:$qa")
  efficiency=$("$SUBMIT" eval_efficiency --dependency="afterok:$qa")
  tables=$("$SUBMIT" build_tables \
    --dependency="afterok:$representation:$qa:$robustness:$generation:$interventions:$efficiency")

  mkdir -p "$RUNS"
  local record="$RUNS/evaluation-submission-${tables}.csv"
  "$PYTHON" - "$record" "$representation" "$qa" "$robustness" "$generation" \
    "$interventions" "$efficiency" "$tables" <<'PY'
import csv
import datetime
import sys

path, *job_ids = sys.argv[1:]
names = [
    "representation", "qa", "robustness", "generation",
    "cross_family_interventions", "efficiency", "tables",
]
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
with open(path, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["stage", "job_id", "submitted_at_utc"])
    writer.writeheader()
    writer.writerows(
        {"stage": name, "job_id": job_id, "submitted_at_utc": stamp}
        for name, job_id in zip(names, job_ids)
    )
PY

  printf "%-18s %s\n" \
    "representation" "$representation" \
    "qa" "$qa" \
    "robustness" "$robustness" \
    "generation" "$generation" \
    "cross interventions" "$interventions" \
    "efficiency" "$efficiency (after QA)" \
    "tables/check" "$tables (after all evaluations)"
  echo "submission record: ${record#$ROOT/}"
}

case "$MODE" in
  check)
    cd "$ROOT"
    failure=0
    required=(
      pyproject.toml
      .env.example
      src/ninaxander/adapter.py
      src/ninaxander/paired.py
      src/ninaxander/result_io.py
      src/ninaxander/b_to_a_chimera.py
      tools/build_result_tables.py
      tools/complete_cross_family_mlp_bundle.py
      tools/check_cross_family_chimera.py
      tools/validate_markdown_languages.py
      tools/validate_results.py
      tools/check_paper_claims.py
      tools/make_layer_cka_figure.py
      tools/check_en_number_parity.py
      paper/paper_en.tex
      experiments/adapter7b.py
      experiments/four_path_layer_metrics.py
      experiments/all_layer_corr.py
      experiments/a_to_b_chimera_bench_full_mp.py
      experiments/a_to_b_chimera_gen_mp.py
      experiments/a_to_b_longctx_ce_mp.py
      experiments/a_to_b_serving_bench_mp.py
      experiments/a_to_b_serving_pruned.py
      experiments/equal_mem_baseline.py
      experiments/kvcache_measure.py
      experiments/paired_stats.py
      experiments/b_to_a_chimera_bench_full_mp.py
      experiments/b_to_a_chimera_gen_mp.py
      experiments/b_to_a_longctx_ce_mp.py
      experiments/b_to_a_serving_bench_mp.py
      experiments/b_to_a_serving_pruned.py
      experiments/linearity.py
      experiments/cross_family_mlp_intervention.py
      experiments/slurm/eval_representation.sbatch
      experiments/slurm/eval_qa.sbatch
      experiments/slurm/eval_robustness.sbatch
      experiments/slurm/eval_generation.sbatch
      experiments/slurm/eval_cross_family_interventions.sbatch
      experiments/slurm/eval_efficiency.sbatch
      experiments/slurm/check_hf_release.sbatch
      experiments/slurm/build_tables.sbatch
      experiments/slurm/_config.sh
      experiments/slurm/_common.sh
      experiments/slurm/submit.sh
      paper/paper_ja.tex
      paper/ipsj.cls
      paper/ipsjtech.sty
      paper/ipsjunsrt.bst
    )
    have_release=0
    [[ -d release/huggingface ]] && have_release=1
    release_required=(
      release/huggingface/README.md
      release/huggingface/README.ja.md
      release/huggingface/README.zh-CN.md
      release/huggingface/REMOTE_RELEASES.csv
      release/huggingface/ninaxander-raven7b-tulu69-adapter/config.json
      release/huggingface/ninaxander-raven7b-tulu69-adapter/README.ja.md
      release/huggingface/ninaxander-raven7b-tulu69-adapter/README.zh-CN.md
      release/huggingface/ninaxander-raven7b-tulu69-adapter/NOTICE.ja.md
      release/huggingface/ninaxander-raven7b-tulu69-adapter/NOTICE.zh-CN.md
      release/huggingface/ninaxander-tulu69-to-raven7b-best/config.json
      release/huggingface/ninaxander-tulu69-to-raven7b-best/model.safetensors
      release/huggingface/ninaxander-tulu69-to-raven7b-best/source/model.safetensors.index.json
      release/huggingface/ninaxander-tulu69-to-raven7b-best/target/model.safetensors.index.json
      release/huggingface/ninaxander-tulu69-to-raven7b-best/README.ja.md
      release/huggingface/ninaxander-tulu69-to-raven7b-best/README.zh-CN.md
      release/huggingface/ninaxander-tulu69-to-raven7b-best/NOTICE.ja.md
      release/huggingface/ninaxander-tulu69-to-raven7b-best/NOTICE.zh-CN.md
      release/huggingface/ninaxander-raven7b-to-tulu69-best/config.json
      release/huggingface/ninaxander-raven7b-to-tulu69-best/model.safetensors
      release/huggingface/ninaxander-raven7b-to-tulu69-best/source/model.safetensors.index.json
      release/huggingface/ninaxander-raven7b-to-tulu69-best/target/model.safetensors.index.json
      release/huggingface/ninaxander-raven7b-to-tulu69-best/README.ja.md
      release/huggingface/ninaxander-raven7b-to-tulu69-best/README.zh-CN.md
      release/huggingface/ninaxander-raven7b-to-tulu69-best/NOTICE.ja.md
      release/huggingface/ninaxander-raven7b-to-tulu69-best/NOTICE.zh-CN.md
    )
    if [[ "$have_release" -eq 1 ]]; then
      required+=("${release_required[@]}")
    else
      echo "NOTE: release/huggingface absent (gitignored); release-package checks skipped."
      echo "      Stage the packages with tools/export_hf_adapter.py and tools/export_hf_chimera.py."
    fi
    for path in "${required[@]}"; do require_file "$path" || failure=1; done
    for retired in heteroblock results huggingface docs/report; do
      [[ ! -e "$ROOT/$retired" ]] || {
        echo "RETIRED PATH STILL PRESENT: $retired"
        failure=1
      }
    done
    # bash -n only parses its FIRST operand (the rest become positional
    # parameters), so each script must be checked individually.
    for script in reproduce.sh experiments/slurm/*.sh experiments/slurm/*.sbatch; do
      bash -n "$script" || failure=1
    done
    compile_targets=(src/ninaxander/*.py experiments/*.py tools/*.py)
    if [[ "$have_release" -eq 1 ]]; then
      compile_targets+=(
        release/huggingface/ninaxander-raven7b-tulu69-adapter/*.py
        release/huggingface/ninaxander-tulu69-to-raven7b-best/*.py
        release/huggingface/ninaxander-raven7b-to-tulu69-best/*.py
      )
    fi
    "$PYTHON" -m py_compile "${compile_targets[@]}" || failure=1
    "$PYTHON" tools/validate_results.py \
      --raw-dir "$RAW_METRICS" --table-dir "$TABLES" || failure=1
    "$PYTHON" tools/check_paper_claims.py \
      --tex paper/paper_ja.tex --table-dir "$TABLES" \
      --release-dir release/huggingface || failure=1
    "$PYTHON" tools/check_en_number_parity.py || failure=1
    "$PYTHON" tools/validate_markdown_languages.py --root "$ROOT" || failure=1
    [[ -f "$CHECKPOINT" ]] || echo "NOTE: final checkpoint absent; gpu-eval cannot run."
    # Plain GNU grep, not ripgrep: rg is not guaranteed in a fresh checkout, and a
    # missing binary inside `if rg ...` would skip these gates while check still
    # prints PROJECT_CHECK_OK. The --exclude list stands in for rg's gitignore
    # awareness (LaTeX byproducts, bytecode).
    run_gate "active source/documentation still contains retired paths." \
      -rnIE --exclude='*.pdf' --exclude-dir=__pycache__ \
      --exclude='*.aux' --exclude='*.bbl' --exclude='*.blg' --exclude='*.log' \
      --exclude='*.out' --exclude='*.toc' --exclude='*.nav' --exclude='*.snm' \
      '(^|[[:space:]`(])(heteroblock/|docs/report/|results/|huggingface/)' \
      README.md README.ja.md README.zh-CN.md docs experiments paper tools src
    run_gate "site-specific partition/output/CPU directives must be injected by submit.sh." \
      -nE '^#SBATCH[[:space:]]+(-p|--partition|-o|--output|-c|--cpus-per-task)' \
      experiments/slurm/*.sbatch
    local_path_pattern="/""home/"
    local_cluster_pattern="sx""40|sx""login"
    run_gate "public SLURM sources still contain a local absolute path or cluster name." \
      -rnIE --exclude-dir=__pycache__ \
      "$local_path_pattern|$local_cluster_pattern" experiments/slurm reproduce.sh .env.example
    directional_term_pattern="rever""se-order|for""ward-order|逆""順|正""順|正""方向|逆""方向"
    run_gate "public documentation must use explicit model-family route names." \
      -rniE --include='*.md' --include='*.tex' "$directional_term_pattern" \
      README.md README.ja.md README.zh-CN.md docs paper experiments
    if [[ "$have_release" -eq 1 ]]; then
      expected_hf_packages=$'ninaxander-raven7b-to-tulu69-best\nninaxander-raven7b-tulu69-adapter\nninaxander-tulu69-to-raven7b-best'
      actual_hf_packages=$(
        for pkg in release/huggingface/*/; do
          [[ -d "$pkg" ]] || continue
          basename "$pkg"
        done | sort
      )
      if [[ "$actual_hf_packages" != "$expected_hf_packages" ]]; then
        echo "ERROR: release/huggingface must contain exactly the three public packages."
        printf 'actual:\n%s\n' "$actual_hf_packages"
        failure=1
      fi
    fi
    for path in experiments/a_to_b_*.py; do
      counterpart="experiments/b_to_a_${path#experiments/a_to_b_}"
      [[ -f "$counterpart" ]] || {
        echo "ERROR: AB-only experiment entry point lacks BA pair: $path"
        failure=1
      }
    done
    for path in experiments/b_to_a_*.py; do
      counterpart="experiments/a_to_b_${path#experiments/b_to_a_}"
      [[ -f "$counterpart" ]] || {
        echo "ERROR: BA-only experiment entry point lacks AB pair: $path"
        failure=1
      }
    done
    while IFS= read -r referenced_script; do
      [[ -f "$referenced_script" ]] || {
        echo "ERROR: SLURM launcher references a missing script: $referenced_script"
        failure=1
      }
    done < <(
      grep -hoE 'experiments/[A-Za-z0-9_]+\.py' \
        experiments/slurm/*.sbatch | sort -u
    )
    [[ "$failure" -eq 0 ]] || exit 1
    echo "PROJECT_CHECK_OK"
    ;;

  tables)
    cd "$ROOT"
    "$PYTHON" tools/build_result_tables.py \
      --raw-dir "$RAW_METRICS" --table-dir "$TABLES" --log-dir "$SLURM_LOGS"
    "$PYTHON" tools/validate_results.py \
      --raw-dir "$RAW_METRICS" --table-dir "$TABLES"
    "$PYTHON" tools/check_paper_claims.py \
      --tex paper/paper_ja.tex --table-dir "$TABLES" \
      --release-dir release/huggingface
    ;;

  map)
    cd "$ROOT"
    "$PYTHON" - "$TABLES" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
with (root / "MANIFEST.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
print("paper-facing table                              rows  description")
print("-" * 108)
for row in rows:
    print(f"{row['table']:<48} {int(row['rows']):>5}  {row['description']}")
with (root / "paper_result_coverage.csv").open(
    encoding="utf-8",
    newline="",
) as handle:
    coverage = list(csv.DictReader(handle))
print("\nexecution-path coverage")
print("table                                      AA     AB     BB     BA  B-exit   AB=BA")
print("-" * 86)
for row in coverage:
    print(
        f"{row['table']:<42} "
        f"{int(row['A_to_A']):>5} "
        f"{int(row['A_to_B']):>6} "
        f"{int(row['B_to_B']):>6} "
        f"{int(row['B_to_A']):>6} "
        f"{int(row['B_early_exit']):>7}   "
        f"{row['ab_ba_matched']}"
    )
print(f"\nAll tables: {root}")
PY
    ;;

  paper)
    cd "$ROOT"
    # Both papers use the vendored IPSJ SIG Technical Report class (paper/ipsj.cls v4.1),
    # which runs only under pLaTeX: platex -> pbibtex -> platex x2 -> dvipdfmx.
    for tool in platex pbibtex dvipdfmx; do
      command -v "$tool" >/dev/null 2>&1 || {
        echo "MISSING: $tool (TeX Live pLaTeX toolchain; see paper/README.md)"
        exit 1
      }
    done
    "$PYTHON" tools/make_layer_cka_figure.py \
      --table-dir "$TABLES" --out paper/fig_layer_cka.pdf
    "$PYTHON" tools/make_layer_cka_figure.py \
      --table-dir "$TABLES" --out paper/fig_layer_cka_en.pdf --lang en
    cd "$ROOT/paper"
    build_log="${TMPDIR:-/tmp}/ninaxander-paper-build.log"
    : >"$build_log"
    # Every step appends to the log; on failure, say which step failed and show the error.
    run_tex() { "$@" >>"$build_log" 2>&1 || { echo "PAPER BUILD FAILED: $* (full log: $build_log)"; grep -nE '^!|Error' "$build_log" | tail -20; tail -20 "$build_log"; exit 1; }; }
    run_tex platex -kanji=utf8 -interaction=nonstopmode -halt-on-error paper_ja.tex
    run_tex pbibtex -kanji=utf8 paper_ja
    run_tex platex -kanji=utf8 -interaction=nonstopmode -halt-on-error paper_ja.tex
    run_tex platex -kanji=utf8 -interaction=nonstopmode -halt-on-error paper_ja.tex
    run_tex dvipdfmx paper_ja.dvi
    pages=$(pdfinfo paper_ja.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
    echo "paper/paper_ja.pdf OK (${pages:-unknown} pages)"
    if [[ -f paper_en.tex ]]; then
      # The English paper uses the IEEE conference format and pdfLaTeX (as on arXiv).
      run_tex pdflatex -interaction=nonstopmode -halt-on-error paper_en.tex
      run_tex bibtex paper_en
      run_tex pdflatex -interaction=nonstopmode -halt-on-error paper_en.tex
      run_tex pdflatex -interaction=nonstopmode -halt-on-error paper_en.tex
      pages_en=$(pdfinfo paper_en.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
      echo "paper/paper_en.pdf OK (${pages_en:-unknown} pages)"
    fi
    ;;

  hf-check)
    cd "$ROOT"
    [[ -d release/huggingface ]] || {
      echo "release/huggingface absent (gitignored); stage the packages with"
      echo "tools/export_hf_adapter.py and tools/export_hf_chimera.py first."
      exit 1
    }
    "$PYTHON" release/huggingface/ninaxander-raven7b-tulu69-adapter/validate_package.py \
      --package release/huggingface/ninaxander-raven7b-tulu69-adapter \
      --source-checkpoint "$CHECKPOINT"
    "$PYTHON" release/huggingface/ninaxander-tulu69-to-raven7b-best/validate_package.py \
      --package release/huggingface/ninaxander-tulu69-to-raven7b-best \
      --source-checkpoint "$CHECKPOINT"
    "$PYTHON" release/huggingface/ninaxander-raven7b-to-tulu69-best/validate_package.py \
      --package release/huggingface/ninaxander-raven7b-to-tulu69-best \
      --source-checkpoint "$CHECKPOINT"
    ;;

  hf-gpu-check)
    command -v sbatch >/dev/null || { echo "sbatch not found"; exit 1; }
    [[ -f "$CHECKPOINT" ]] || { echo "missing final checkpoint: $CHECKPOINT"; exit 1; }
    echo "Submit static, exact-checkpoint, and GPU checks for all three release packages."
    confirm || { echo "aborted"; exit 0; }
    hf_job=$("$SUBMIT" check_hf_release)
    mkdir -p "$RUNS"
    record="$RUNS/hf-release-evaluation-submission-${hf_job}.csv"
    "$PYTHON" - "$record" "$hf_job" <<'PY'
import csv
import datetime
import sys

path, job_id = sys.argv[1:]
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
with open(path, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["stage", "job_id", "submitted_at_utc", "status"],
    )
    writer.writeheader()
    writer.writerow(
        {
            "stage": "huggingface_direction_locked_gpu_validation",
            "job_id": job_id,
            "submitted_at_utc": stamp,
            "status": "submitted",
        }
    )
PY
    echo "Hugging Face validation: $hf_job"
    echo "submission record: ${record#$ROOT/}"
    ;;

  gpu-eval)
    command -v sbatch >/dev/null || { echo "sbatch not found"; exit 1; }
    [[ -f "$CHECKPOINT" ]] || { echo "missing final checkpoint: $CHECKPOINT"; exit 1; }
    [[ -d "$RWKV_MODEL" || -d "$ROOT/$RWKV_MODEL" ]] || echo "NOTE: RWKV reference is not a local directory: $RWKV_MODEL"
    [[ -d "$PYTHIA_MODEL" || -d "$ROOT/$PYTHIA_MODEL" ]] || echo "NOTE: Pythia reference is not a local directory: $PYTHIA_MODEL"
    echo "Submit current paper evaluations only; no training job will run."
    confirm || { echo "aborted"; exit 0; }
    submit_eval_pipeline
    ;;

  cross-family-eval)
    command -v sbatch >/dev/null || { echo "sbatch not found"; exit 1; }
    [[ -f "$CHECKPOINT" ]] || { echo "missing final checkpoint: $CHECKPOINT"; exit 1; }
    echo "Submit matched 18-arm AB/BA QA, linearity, and same-adapter interventions."
    confirm || { echo "aborted"; exit 0; }
    qa=$("$SUBMIT" eval_qa)
    interventions=$("$SUBMIT" eval_cross_family_interventions \
      --dependency="afterok:$qa")
    tables=$("$SUBMIT" build_tables --dependency="afterok:$interventions")
    mkdir -p "$RUNS"
    record="$RUNS/cross-family-evaluation-submission-${tables}.csv"
    "$PYTHON" - "$record" "$qa" "$interventions" "$tables" <<'PY'
import csv
import datetime
import sys

path, qa_job, intervention_job, tables_job = sys.argv[1:]
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
with open(path, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["stage", "job_id", "dependency", "submitted_at_utc", "status"],
    )
    writer.writeheader()
    writer.writerows(
        [
            {
                "stage": "matched_qa",
                "job_id": qa_job,
                "dependency": "",
                "submitted_at_utc": stamp,
                "status": "submitted",
            },
            {
                "stage": "cross_family_interventions",
                "job_id": intervention_job,
                "dependency": f"afterok:{qa_job}",
                "submitted_at_utc": stamp,
                "status": "pending_dependency",
            },
            {
                "stage": "tables_and_validation",
                "job_id": tables_job,
                "dependency": f"afterok:{intervention_job}",
                "submitted_at_utc": stamp,
                "status": "pending_dependency",
            },
        ]
    )
PY
    printf "%-22s %s\n" \
      "matched QA" "$qa" \
      "cross interventions" "$interventions (after QA)" \
      "tables/check" "$tables (after interventions)"
    echo "submission record: ${record#$ROOT/}"
    ;;

  gpu)
    command -v sbatch >/dev/null || { echo "sbatch not found"; exit 1; }
    echo "Submit adapter training, continuation, and all evaluations."
    confirm || { echo "aborted"; exit 0; }
    train=$("$SUBMIT" train_adapter)
    continuation=$("$SUBMIT" continue_adapter --dependency="afterany:$train")
    echo "training: $train; continuation: $continuation"
    submit_eval_pipeline "$continuation"
    ;;

  slurm-config)
    "$SUBMIT" config
    ;;

  status)
    command -v squeue >/dev/null && \
      squeue -u "${USER:-$(id -un)}" -o "%.18i %.12P %.24j %.8T %.10M %.9l %R"
    echo
    while IFS= read -r record; do
      printf '%s\n' "${record#$ROOT/}"
    done < <(
      find "$RUNS" -maxdepth 1 -name '*evaluation-submission-*.csv' \
        -type f -print 2>/dev/null | sort
    )
    ;;

  *)
    sed -n '2,12p' "$0"
    exit 2
    ;;
esac
