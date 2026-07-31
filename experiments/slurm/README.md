# Portable SLURM launchers

English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

The `*.sbatch` files contain only stage logic, walltime, and a job name. They intentionally do not contain a
checkout path or site-specific `--partition`, `--output`, `--cpus-per-task`, `--gres`, account, QoS, or constraint.

SLURM parses `#SBATCH` directives before the job shell starts, so a directive such as
`#SBATCH --partition=$MY_PARTITION` does not reliably expand a value loaded from `.env`. The supported entry point
is therefore `submit.sh`: it reads the project-root `.env` and supplies these values as `sbatch` command-line
arguments, which override or complement the portable directives.

## Configure

```bash
cp .env.example .env
# edit .env
./reproduce.sh slurm-config
```

The main variables are:

| Variable | Purpose |
|---|---|
| `NINAXANDER_PYTHON`, `NINAXANDER_TORCHRUN` | Python environment |
| `NINAXANDER_SLURM_GPU_PARTITION` | partition for training and GPU evaluation |
| `NINAXANDER_SLURM_CPU_PARTITION` | partition for final table aggregation |
| `NINAXANDER_SLURM_GPU_CPUS`, `NINAXANDER_SLURM_CPU_CPUS` | CPUs requested per job |
| `NINAXANDER_SLURM_GRES` | optional request such as `gpu:4`; leave empty when GPUs are implicit |
| `NINAXANDER_SLURM_ACCOUNT`, `NINAXANDER_SLURM_QOS`, `NINAXANDER_SLURM_CONSTRAINT` | optional site policy |
| `NINAXANDER_GPU_0` … `NINAXANDER_GPU_3` | four physical GPU indices used as two model-parallel pairs |
| `NINAXANDER_RWKV_MODEL`, `NINAXANDER_PYTHIA_MODEL`, `NINAXANDER_TOKENIZER` | parent artifacts |
| `NINAXANDER_ARTIFACTS_DIR` | optional location for checkpoints, metrics, logs, and run records |
| `NINAXANDER_TRAIN_CACHE` | large temporary residual cache used during training |

See the documented defaults and optional settings in `.env.example`. `.env` is gitignored and may contain
`HF_TOKEN`; never commit it.

## Submit

```bash
# One stage
experiments/slurm/submit.sh eval_generation

# Matched AB/BA linearity and same-adapter interventions
experiments/slurm/submit.sh eval_cross_family_interventions

# Validate the adapter and both direction-locked Hugging Face packages on two GPUs
experiments/slurm/submit.sh check_hf_release
# Equivalent confirmation-gated entry point with a structured run record
./reproduce.sh hf-gpu-check

# Add an ordinary sbatch option
experiments/slurm/submit.sh eval_qa --time=18:00:00

# Separate sbatch options from positional arguments passed to the job script
experiments/slurm/submit.sh continue_adapter --dependency=afterany:1234 -- /path/to/checkpoint.pt

# Entire evaluation DAG, followed by table rebuilding and strict validation
./reproduce.sh gpu-eval

# Matched 18-arm AB/BA QA and interventions, followed by table validation
./reproduce.sh cross-family-eval
```

The role stages (`eval_qa`, `eval_robustness`, `eval_generation`, and
`eval_efficiency`) each produce matched AB and BA measurements. Parent and
self-path controls are labeled explicitly as AA and BB in the normalized
tables; there is no separate one-direction catch-up stage.

The reproduction drivers write their submitted job IDs and dependencies to
`artifacts/runs/*evaluation-submission-*.csv`; `./reproduce.sh status` lists
both the live queue and these durable records.

Use `--test-only` to ask SLURM to validate a submission without creating the job:

```bash
experiments/slurm/submit.sh eval_generation --test-only
experiments/slurm/submit.sh build_tables --test-only
```

Direct `sbatch experiments/slurm/eval_qa.sbatch` remains syntactically valid when submitted from the project root,
but then scheduler defaults determine partition, CPUs, and stdout. Using `submit.sh` is the reproducible path; it
also fixes the job working directory so every copied SLURM script can load `experiments/slurm/_common.sh`.

`check_hf_release` is intended to run after all three release directories have
been re-exported. It checks the adapter-only package against the trusted
checkpoint, runs its standalone example in both named routes, and then performs
static integrity checks plus one-token GPU smoke tests for the direction-locked
Pythia→RWKV and RWKV→Pythia complete packages.
