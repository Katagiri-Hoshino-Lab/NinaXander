# NinaXander: Composing Frozen Cross-Family Language Models

English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

Code, structured result data, run logs, and the Japanese write-up for
**"NinaXander: Composing Frozen Language Models Across Model Families via a Shared Latent Representation --- Conditions and Limits"**
([`paper/paper_ja.pdf`](paper/paper_ja.pdf)).

**One line.** Two *frozen* 7B language models from different architecture families — RWKV-4-Raven-7B (pure RNN)
and open-instruct-pythia-6.9b-tulu (Transformer) — share a token-level representation space at matching layers
(under matched conditions), and a small learned adapter recovers it well enough to build working **AB and BA
chimeras**, each joined by one residual translation. The path matters: Pythia-front → RWKV-back is much stronger at
deep boundaries. At @4, AB and BA score 65.25% and 65.70% on the two-task mean, while BA retains only five
Transformer blocks. The chimera converses but is factually degraded, and the
adapter remains strongly domain-specific.

> **NinaXander** is the name of the composed model, taken from the *Fullmetal Alchemist* chimera — the fusion
> succeeds and the creature functions (it speaks), yet it does not match the beings it was made from. Likewise,
> our chimera works but does not reach its stronger parent; the name mirrors that correspondence.

> Scope: one model pair, one corpus (Alpaca), single seed. See the paper's §Limitations. Only the current
> 32-layer RWKV-Raven/Tulu-Pythia experiment belongs to the public result contract.

## Headline numbers

All from the current adapter: **z=4096 (no compression), 268.5M params, σ=0.4, 32 layers, 415,000 steps**
(jobs 1832 → 1837), residuals captured for every block by forward hook. Held-out, deepest point (step 415,000):

| | |
|---|---|
| centered latent alignment ρ_ctr | **0.901** |
| four-path read-out AA / AB / BB / BA | 0.695 / **0.559** / **0.711** / 0.590 |
| best direct affine cross-map AB / BA | 0.487 / 0.535 (irreducibly non-linear output: 13.4% / 12.0%) |
| SciQ acc_norm (N=1000): Pythia / AB@4 / BA@4 / RWKV | 81.0 / 70.9 / **69.2** / 67.0 |
| ARC-Easy acc_norm (N=2376): Pythia / AB@4 / BA@4 / RWKV | 65.3 / 59.6 / **62.2** / 61.8 |
| KV deleted at @4: AB (RWKV→Pythia) / BA (Pythia→RWKV) | 15.625% / **84.375%** |

Public results use a symmetric **four-path notation**. The first letter is the
prefix family, the second is the suffix/read-out family, and `@L` is the switch
layer:

| prefix ↓ / suffix → | A = RWKV | B = Pythia |
|---|---:|---:|
| **A = RWKV** | `NinaXander-AA@L` (same-family control) | `NinaXander-AB@L` (chimera) |
| **B = Pythia** | `NinaXander-BA@L` (chimera) | `NinaXander-BB@L` (same-family control) |

At `L=4`, the full-set normalized accuracies are:

| task | AA | AB | BB | BA |
|---|---:|---:|---:|---:|
| ARC-Easy | 62.8 | 59.6 | 62.2 | 62.2 |
| SciQ | 68.5 | 70.9 | 74.8 | 69.2 |

Four load-bearing findings:

1. **Both cross-readouts are below their matching self-reconstruction:** ρ_ctr 0.901 > B→B 0.711 > A→B 0.559,
   and ρ_ctr 0.901 > A→A 0.695 > B→A 0.590. These inequalities hold at every one of the 32 layers. The latents
   agree far better than either decoder can rebuild a residual, so the scarce resource is *reconstruction*, not
   alignment. Removing compression (z = d = 4096) does not lift either self-map near one, and what limits them is
   unresolved. (The 1/(1+σ²) ≈ 0.862 "noise ceiling" claimed in
   an earlier draft was wrong: that is the *noisy*-eval value; since we evaluate on clean z the ceiling is ≈0.98, so
   B→B 0.711 is far below it and noise is **not** the binding constraint — a σ sweep is needed to say what is.)
2. **Neither cross-family path is "near-linear."** Best affine imitations
   explain 86.6% of AB and 88.0% of BA adapter output; the remaining 13.4% /
   12.0% is irreducibly non-linear. Direct affine maps reach AB 0.487 and BA
   0.535, while removing the same trained residual-MLP branch collapses the
   corresponding read-outs to −0.41 and −0.30.
3. **Both execution paths generate, but both are factually degraded.** The four prompts are passed verbatim in
   English and decoded with deterministic greedy argmax (temperature 0), at most 40 new tokens, stopping early on
   EOS. The paper-facing samples use the shared `L=4` frontier point: both paths answer “Paris” and start the
   primary-colors answer correctly; both miss the water completion, AB gives a correct one-sentence black-hole
   definition, and BA drifts into dialogue on colors and gives a circular black-hole definition. On the full
   paired test sets every AB/BA switch beats guessing and loses to the stronger Pythia parent. AB@4 beats RWKV
   only on SciQ (+3.9, p=0.019) and loses on ARC (−2.2, p=0.042); no BA switch significantly beats RWKV.
   The symmetric BB/AA controls also show why their gaps are not pure translation error: AB is below BB, while
   deep BA can exceed AA because the prefix parent changes at the same time. The interface is **domain-specific**
   on both paths: at context 2048, WikiText perplexity is 83.4 for AB@4 and 34.8 for BA@4 versus 8.27 for the
   better parent.
4. **AB and BA have different depth profiles.** BA@16/@24 exceeds the same-layer AB configuration by
   8.9–18.8 points, and BA@4 scores 62.2/69.2 while deleting 84.375% of Transformer KV. It still falls below
   Pythia on both tasks and does not significantly beat RWKV. Its 65.70 two-task mean is only 0.45 points above
   AB@4; choosing it as the public default is a **post-hoc** ranking over 2 cross-family paths × 4 switches on the two
   reported test sets, with no independent deployment-selection split. BA@4 reduces the WikiText damage but
   still has 34.8 perplexity at context 2048 versus the better parent's 8.27.

The matched same-adapter intervention is also reported symmetrically. Entries
are full-adapter → residual-MLP-disabled normalized accuracies (%):

| L | ARC-Easy AB | ARC-Easy BA | SciQ AB | SciQ BA |
|---:|---:|---:|---:|---:|
| 4 | 59.6 → 31.9 | 62.2 → 31.8 | 70.9 → 31.3 | 69.2 → 32.6 |
| 8 | 59.0 → 29.9 | 57.3 → 35.5 | 67.7 → 31.4 | 64.6 → 37.0 |
| 16 | 48.6 → 28.5 | 57.5 → 43.7 | 54.5 → 31.7 | 68.0 → 47.3 |
| 24 | 47.0 → 34.8 | 59.6 → 48.4 | 51.1 → 40.0 | 69.9 → 49.1 |

All 16 item-paired comparisons have McNemar p<2e-9. Thus the learned
nonlinear branch is load-bearing in both AB and BA, not an effect inferred
from one path.

## Quick start

```bash
pip install -r requirements.txt      # torch, transformers==5.2.0, datasets
cp .env.example .env                 # then edit Python, model, and SLURM settings
./reproduce.sh slurm-config          # show the resolved non-secret configuration

./reproduce.sh check     # validate portable sources and existing structured data (seconds, no GPU)
./reproduce.sh tables    # rebuild paper-facing CSV files from raw JSON/CSV artifacts
./reproduce.sh map       # list every paper-facing CSV, row count, and description
./reproduce.sh paper     # rebuild paper/paper_ja.pdf
./reproduce.sh hf-check  # static/checkpoint validation for all three Hub packages
./reproduce.sh hf-gpu-check  # submit two direction-locked GPU smoke tests
./reproduce.sh gpu-eval  # submit matched AB/BA non-training evaluations
```

`check`, `tables`, `map`, `paper`, and `hf-check` are CPU-only.
`hf-gpu-check` and the evaluation programs need the two frozen 7B parents and
the final adapter on GPUs. `gpu-eval` never submits training.

SLURM launchers contain no local checkout path, partition, output path, account, QoS, GRES, or CPU-count setting.
Those values are read from the gitignored `.env` and injected by
`experiments/slurm/submit.sh`. This is necessary because `#SBATCH` lines are parsed before a shell can source
`.env`, so variables placed directly in those directives would not expand reliably. See
[`experiments/slurm/README.md`](experiments/slurm/README.md) for the configuration variables and direct submission
examples.

## Result-data contract

Every evaluation writes machine-readable data directly. Nested/per-item data is retained as JSON and every
paper-facing numeric relation is normalized as CSV:

- `artifacts/metrics/raw/`: one JSON/CSV bundle per experiment, plus paired-test and generation CSV files.
- Generation rows record `decoding=greedy_argmax`, `temperature=0`,
  `max_new_tokens=40`, and `seed=0`; the prompts themselves are stored verbatim.
- `artifacts/metrics/raw/training_curve.csv`: public 417-point training trajectory; local scheduler logs are unnecessary.
- `artifacts/metrics/tables/`: stable, joined CSV tables used to audit the values reported in the paper.
- `artifacts/metrics/tables/MANIFEST.csv`: table names, schemas, row counts, and descriptions.
- `artifacts/metrics/tables/paper_result_coverage.csv`: per-table
  AA/AB/BB/BA counts; every public cross-family result family must have
  matched AB and BA coverage. Truncated Pythia baselines are counted
  separately as `B_early_exit`, never as the BB adapter self-map. The
  four-path completeness flag is populated only for result families where
  AA/BB self-map controls are methodologically applicable.
- `tools/build_result_tables.py`: deterministic raw-to-table transformation.
- `tools/validate_results.py`: schema, row-count, JSON, manifest,
  release-metadata, matched 18-arm AB/BA QA, both paths' boundary gates
  (plus BA direct-suffix/concurrency gates), pruning/generation gates, exact
  equal-memory depths, serving/KV consistency, identical paired raw-CSV
  schemas, and cross-path parent-item consistency checks.

Neither AB nor BA end-to-end measurements are inferred from representation
R². Each path is executed directly and emitted under matched `a_to_b_*` /
`b_to_a_*` raw names. The canonical four-path QA
matrix is `paper_four_path_qa_accuracy.csv`. All remaining paired AB/BA
evaluations are combined under `paper_cross_family_*`, with a distinct
`producer_path` provenance field, an actual `execution_path` column, and no duplicated parent rows; normalized public
tables are never split into an AB-only and BA-only copy.

Run `./reproduce.sh tables && ./reproduce.sh check` after any evaluation. Runtime logs are provenance and
diagnostics; they are no longer the data source for paper numbers.

## Hugging Face release packages

The lightweight adapter-only Hub release candidate is staged locally under
`release/huggingface/ninaxander-raven7b-tulu69-adapter` (gitignored; generated
by `tools/export_hf_adapter.py`).
It contains the FP32 adapter and per-layer statistics as SafeTensors, a standalone
loader, a gated chimera-generation example, parent-artifact fingerprints, a
model card, and the applicable research-only license. Optimizer/scheduler state
and local absolute paths from the 3.1 GiB training checkpoint are excluded.

The complete models (staged the same way by `tools/export_hf_chimera.py`) are
split by explicit execution route:

- `release/huggingface/ninaxander-tulu69-to-raven7b-best`
  is Pythia→RWKV with measured-best default L=4: Pythia blocks `0..4`,
  one translation, then RWKV blocks `5..31`.
- `release/huggingface/ninaxander-raven7b-to-tulu69-best`
  is RWKV→Pythia with measured-best default L=4: RWKV blocks `0..4`,
  one translation, then Pythia blocks `5..31`.

Each is an approximately 28 GiB offline bundle containing both exact fp16
parents, the tokenizer, the step-415,000 adapter, all per-layer
standardization tensors, and explicit switch-plan tensors. Its route is locked;
L=8/16/24 remain selectable only for within-route comparisons. Both L=4
defaults are post-hoc choices on the reported tests without an independent
selection split.

All three packages pass exact tensor comparison against
`SNAP_L32_final.pt`; the complete bundles additionally support
`AutoModelForCausalLM` loading, the parent-reproduction injection gate, and
two-GPU generation. Every package has English, Japanese, and Simplified
Chinese model cards. Run `./reproduce.sh hf-check` for the full static release
audit.

Private Hub snapshots (access permission required):
[collection](https://huggingface.co/collections/kotama7/ninaxander-private-inference-releases-6a6b02ec40c18d6b41ed040a),
[adapter](https://huggingface.co/kotama7/ninaxander-raven7b-tulu69-adapter),
[Pythia→RWKV best](https://huggingface.co/kotama7/ninaxander-tulu69-to-raven7b-best),
and [RWKV→Pythia best](https://huggingface.co/kotama7/ninaxander-raven7b-to-tulu69-best).
`tools/export_hf_adapter.py` and `tools/export_hf_chimera.py` are the
corresponding deterministic exporters for the lightweight and complete
packages. After documentation-only edits,
`tools/update_hf_release_checksums.py` refreshes all three integrity manifests
while hashing shared hard-linked parent shards only once.

All three remain release candidates because the exact original BlinkDL Raven `.pth`
filename/revision was not retained when the local HF conversion was made;
recover that provenance before final publication.

## Layout

```
src/ninaxander/       reusable adapter, RWKV-suffix runtime, and result-I/O package
.env.example          public template for local Python/model/SLURM settings
experiments/          current 32-layer training/evaluation entry points
  slurm/              role-based launchers: train, evaluate, aggregate
tools/                table builder, validator, and Hub exporter
artifacts/
  checkpoints/        trained adapters (local, gitignored)
  metrics/raw/        direct experiment outputs
  metrics/tables/     normalized paper-facing CSV files
  logs/               local, gitignored SLURM/evaluation diagnostics
  runs/               structured submission records (local, gitignored)
paper/                paper source and built PDF
docs/                 project-level method, findings, and roadmap
release/huggingface/  three inference-only Hugging Face packages (exporter output; gitignored)
models/               local frozen parent weights (gitignored)
```

## Reproducing a specific number

`./reproduce.sh map` lists the normalized CSV tables. For example,
`paper_four_path_qa_accuracy.csv` holds the AA/AB/BB/BA matrix,
`paper_cross_family_qa_accuracy.csv` holds the combined
parent/chimera/control rows for AB and BA,
`paper_cross_family_paired_statistics.csv` holds McNemar tests and
paired-bootstrap intervals for both paths and their same-item comparison,
`paper_cross_family_linearity.csv` holds the matched AB/BA non-linearity
analysis, `paper_cross_family_mlp_intervention.csv` holds the same-adapter
alpha intervention for both paths; its task-level raw bundles likewise contain
matched AB/BA summaries and per-item outcomes, and
`paper_cross_family_domain_shift.csv` holds the combined
in-domain/out-of-domain CE study.
`raw/cross_family_runtime_smoke.{json,csv}` records GPU-validated AB/BA
boundary gates at all four switches. The canonical QA programs are
`experiments/a_to_b_chimera_bench_full_mp.py` and
`experiments/b_to_a_chimera_bench_full_mp.py`; adapter training is
`experiments/adapter7b.py`.

## Hardware / environment

- **GPU required.** The reported cluster used one V100-32GB node for early probes and a 4×V100-16GB node for
  training/evaluation. The pair needs 28.9 GiB resident (RWKV 14 GiB + Pythia 13 GiB, fp16), so evaluation on
  16 GiB GPUs splits each pair across two cards (`*_mp.py` = RWKV on `cuda:0`, Pythia on `cuda:1`; the adapter
  lives with the path's suffix model). The BA evaluator keeps only the reachable E_B/D_A half on GPU;
  BA QA additionally keeps E_A for its AA decoder-loss control.
  Cluster partition/GRES names are deliberately not embedded in the public scripts.
- **Walltime.** Adapter training exceeds the 12 h limit and resumes once; `continue_adapter.sbatch` handles that and
  restores optimiser + scheduler + global step (the run reached 416,000 steps across the two cycles; all reported numbers use the step-415,000 checkpoint, the held-out A→B maximum).
- Data is downloaded on the fly (Alpaca, ARC-Easy, SciQ via `datasets`) and cached under `~/.cache/huggingface`.
- Standalone parent copies under `models/` are local and ignored. The two
  direction-best candidates under `release/huggingface/*-best/` intentionally
  bundle the exact fp16 parents and are uploaded as separate repositories from
  the source project. RWKV-Raven ships
  upstream as a BlinkDL `.pth` and was converted to HF format; both parents
  share the GPT-NeoX-20B tokenizer (a hard requirement — see
  `docs/00-overview.md`).
