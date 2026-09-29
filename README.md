# NinaXander — reference repository for the paper

English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

This repository is the artifact accompanying *"NinaXander: Composing Frozen Language Models Across Model
Families via a Shared Latent Representation — Conditions and Limits"*
([`paper/paper_en.pdf`](paper/paper_en.pdf), [`paper/paper_ja.pdf`](paper/paper_ja.pdf)). It exists so that a
reader of the paper can locate the data behind any reported number, re-derive the paper-facing tables from the
raw evaluation output, and re-run the checks that bind the two together. It is not a general-purpose library.

The study freezes two 7B instruction-following models from different families — RWKV-4-Raven-7B, a
linear-attention architecture with a recurrent inference mode, and open-instruct-pythia-6.9b-tulu, a Transformer —
and trains one shared-latent adapter on paired residual activations at corresponding layers. A chimera is then
composed by running one parent's first `L+1` blocks, translating the residual once, and running the other
parent's remaining blocks, in both directions with the same adapter.

## Finding the number behind a claim

Every number printed in the paper is derived from a normalized CSV table, and `tools/check_paper_claims.py`
re-derives it and fails the build if the two disagree. To go from a claim to its data:

| Claim in the paper | Table under `artifacts/metrics/tables/` |
|---|---|
| Latent alignment, four-path read-outs | `paper_representation.csv` |
| Per-layer and all-layer correspondence | `paper_layer_metrics.csv`, `paper_layer_correspondence.csv` |
| Four-path multiple-choice accuracy | `paper_four_path_qa_accuracy.csv` |
| Chimera vs. affine control, per boundary | `paper_cross_family_qa_accuracy.csv` |
| Significance tests and bootstrap intervals | `paper_cross_family_paired_statistics.csv` |
| Adapter non-linearity, α intervention | `paper_cross_family_linearity.csv`, `paper_cross_family_mlp_intervention.csv` |
| KV cache and accuracy trade-off | `paper_cross_family_memory_accuracy.csv` |
| In-domain vs. out-of-domain perplexity | `paper_cross_family_domain_shift.csv` |
| Generation samples | `paper_cross_family_generation_samples.csv` |
| Training trajectory | `paper_training_curve.csv` |
| Evaluation-set exposure analysis | `artifacts/metrics/raw/eval_leakage.json` |

`./reproduce.sh map` prints every table with its schema, row count, and description.
`artifacts/metrics/raw/` holds the untransformed per-experiment output, including per-item correctness dumps
used by the paired tests.

## Running the checks

```bash
pip install -r requirements.txt
cp .env.example .env                 # local Python, model, and SLURM settings

./reproduce.sh check     # structure, data schemas, and paper-to-data agreement (CPU, seconds)
./reproduce.sh tables    # rebuild the paper-facing CSV tables from raw output
./reproduce.sh map       # list every table with schema and provenance
./reproduce.sh paper     # rebuild the PDFs
```

`check` runs four gates: `validate_results.py` (schemas, row counts, cross-path consistency),
`check_paper_claims.py` (every headline number against the tables), `check_en_number_parity.py` (the English and
Japanese papers must contain the same numbers), and `validate_markdown_languages.py`. All four are CPU-only and
need no model weights.

`tools/check_eval_leakage.py` quantifies how much of the reported evaluation sets appears in the instruction
mixture the Transformer parent was tuned on. It recomputes accuracy with the matched items removed from the
per-item dumps, so it also needs no GPU.

## Reproducing the experiments

The evaluation and training entry points do need the two frozen 7B parents on GPUs.

```bash
./reproduce.sh gpu-eval        # submit the evaluations (never submits training)
./reproduce.sh hf-check        # static validation of the three release packages
./reproduce.sh hf-gpu-check    # two-GPU smoke tests for the direction-locked packages
```

The parents need 28.9 GiB resident in fp16, so the `*_mp.py` evaluators split each pair across two cards.
SLURM launchers contain no checkout path, partition, account, QoS, or GRES value; those come from the gitignored
`.env` through `experiments/slurm/submit.sh`, because `#SBATCH` directives are parsed before a shell could source
it. See [`experiments/slurm/README.md`](experiments/slurm/README.md).

Adapter training is `experiments/adapter7b.py`; it exceeds a 12-hour limit and resumes once, restoring optimizer,
scheduler, and global step. The reported checkpoint is step 415,000, the held-out maximum of the RWKV→Pythia
read-out, from a 416,000-step run.

## Hugging Face packages

Three inference packages are staged locally by `tools/export_hf_adapter.py` and `tools/export_hf_chimera.py`
under `release/huggingface/` (gitignored), and published as public Hub repositories:

| Package | Contents |
|---|---|
| [`ninaxander-raven7b-tulu69-adapter`](https://huggingface.co/Katagiri-Hoshino-Lab/ninaxander-raven7b-tulu69-adapter) | Adapter and per-layer statistics only, with a standalone loader |
| [`ninaxander-tulu69-to-raven7b-best`](https://huggingface.co/Katagiri-Hoshino-Lab/ninaxander-tulu69-to-raven7b-best) | Pythia→RWKV at `L=4`, with both fp16 parents bundled |
| [`ninaxander-raven7b-to-tulu69-best`](https://huggingface.co/Katagiri-Hoshino-Lab/ninaxander-raven7b-to-tulu69-best) | RWKV→Pythia at `L=4`, with both fp16 parents bundled |

The repositories are public and are grouped in a
[collection](https://huggingface.co/collections/Katagiri-Hoshino-Lab/ninaxander-inference-releases-6a9010088ba6e224ea2ea8dd).
`release/huggingface/REMOTE_RELEASES.csv` records the verified revision, file count, and integrity status of each.
The `L=4` default in both direction-locked packages is a post-hoc choice on the reported test sets, with no
independent selection split; the other boundaries remain selectable within a package's own direction.

Licensing follows the more restrictive parent (AI2 AI Model License, non-commercial research), and each package
carries English, Japanese, and Simplified Chinese model cards.

## Scope

The paper's claims are deliberately narrow, and the repository does not widen them. The correspondence is
measured on one model pair that shares a tokenizer, a width, a depth, and a base pretraining corpus; training
uses a single random seed and one instruction-formatted corpus; the switch layer was selected on the same test
sets that are reported. The composed model never surpasses the stronger parent, and translation degrades sharply
out of domain. The paper's Limitations section states what is and is not established; read it before reusing any
number here.

## Layout

```
paper/                paper sources and built PDFs
src/ninaxander/       adapter, RWKV-suffix runtime, result I/O
experiments/          training and evaluation entry points
  slurm/              role-based launchers
tools/                table builder, validators, exporters, leakage check
artifacts/
  metrics/raw/        direct experiment output, per-item dumps
  metrics/tables/     normalized paper-facing CSV
  checkpoints/        trained adapters (gitignored)
  logs/, runs/        diagnostics and submission records (gitignored)
docs/                 method, findings, roadmap
release/huggingface/  exporter output (gitignored)
models/               frozen parent weights (gitignored)
```

Every Markdown document in this repository exists in English, Japanese, and Simplified Chinese, and
`validate_markdown_languages.py` enforces that the three stay structurally identical.
