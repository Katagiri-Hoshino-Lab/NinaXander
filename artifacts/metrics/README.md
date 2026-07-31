# Metric data

English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

The metric pipeline has two layers.

`raw/` is written by experiment programs. An experiment with nested output writes both `name.json` and a
rectangular `name.csv`; naturally tabular experiments write CSV directly. Per-item correctness remains in JSON so
paired tests can be reproduced.

`raw/training_curve.csv` is the public training trajectory. It uses the same append-only schema as current training;
the initial 417 historical points were migrated once from the authoritative scheduler logs so those local logs are
not required in a public checkout.

`tables/` is rebuilt deterministically:

```bash
./reproduce.sh tables
```

`tables/MANIFEST.csv` describes every paper-facing table and records its row count and schema.
`tools/validate_results.py` verifies JSON readability, CSV headers/non-emptiness,
manifest coverage, expected row counts, and consistency between the final
representation table and the Hugging Face release metadata. AB and BA
validation enforce their explicit path metadata, full ARC-Easy/SciQ item
counts, and the same 18 QA configurations: two parents, four self paths, four
cross paths, four same-adapter alpha-zero interventions, and four affine
controls. Both paths enforce aligned per-item arrays and path-specific
boundary gates below `1e-4`; BA additionally records direct-suffix and
concurrent-arm gates because its RWKV suffix uses a custom batch-1 runtime.
The two pure-parent item-outcome arrays must also exactly match across the AB
and BA QA runs, which guards dataset order and scoring symmetry. Equal-memory
controls use 27/23/15/7 Pythia blocks for AB and 5/9/17/25 for BA, with full
per-item arrays in both paths. Long-context window counts, serving
configuration sets, generation gates, and the measured KV layer
counts/reduction fractions are checked as well. Serving cells accept
only finite positive measurements or an explicit structured runtime error.
Checkpoint step, linear-fit rows/ridge, context-window counts, concurrency
cutoffs, serving repetitions, KV probe length, and final-checkpoint generation
identity are part of the semantic contract rather than informal log metadata.
Long-context affine controls are refit on a disjoint 40,000-token prefix of
each evaluated domain. In particular, the WikiText affine arm is a
domain-adapted oracle, not an Alpaca-fitted map tested out of domain.
Four historical AB long-context/serving JSON files predated structured run
metadata. Their measured cells are unchanged and carry an explicit
`legacy_metadata_enrichment` record for fields recovered from the canonical
invocation; runtime gate values that were not persisted are listed rather
than reconstructed.
`cross_family_runtime_smoke` records parent-reproduction gates, finite logits,
and output-token checks at all four switches for both AB and BA. BA additionally
records direct-vs-hook, independent CUDA-stream, and fused batch-3 diagnostics
because its RWKV suffix is batch-sensitive. The fused acceptance flag must
agree with the measured `1e-4` threshold, but rejection is valid: fused options
are never used for reported QA, whose canonical RWKV path remains independent
batch-1 calls.
Paired QA, MLP intervention, equal-memory, and same-layer AB/BA comparisons
are joined in `tables/paper_cross_family_paired_statistics.csv`. AB and BA QA
both compare every one of the four switches with both parents; neither selects
a task-specific best switch before testing. All producers and the table builder
use `src/ninaxander/paired.py`, whose exact binomial tail is evaluated in log
space; validation rejects an underflowed `mcnemar_p=0`.

`raw/cross_family_mlp_representation.json` contains 25 matched intervention
cells for each of AB and BA. The task-level
`raw/cross_family_mlp_intervention_{arc,sciq}.json` bundles are also symmetric:
each joins that sweep with eight α=0/1 QA configurations and per-item outcomes
for each path. `tools/complete_cross_family_mlp_bundle.py` performs this
lossless join after the canonical AB and BA QA evaluators finish.
The superseded one-direction `linearity_L32.*` and `mlp_ablation_*.*`
bundles are rejected by validation and are not part of the result contract.
Historical `mlp_items_*` and `paired_statistics_mlp.csv` intermediates are
folded into the canonical QA/task bundles once and then rejected.

The canonical `paper_four_path_qa_accuracy.csv` table contains exactly
2 tasks × 4 switch layers × 4 paths (AA, AB, BB, BA). Raw artifacts retain the
`a_to_b_` and `b_to_a_` producer prefixes only as provenance. Public tables
separate `producer_path` from the actual `execution_path`, deduplicate identical
parent rows, and interleave matched AB/BA conditions. In particular, a
Pythia-only early exit is classified as `B_early_exit`, not as an AB/BA or
BB adapter path merely because a cross-path evaluator requested it.
Every paired directional CSV must also have the identical column schema; the
normalizer rewrites legacy generation, KV, long-context, and serving sidecars
through shared AB/BA code paths before validation.
`paper_result_coverage.csv` audits every
public result family and validation rejects any unmatched AB-only family. The
paired table includes same-item BA-versus-AB comparisons at every canonical
switch rather than treating the paths as independent samples.

`raw/four_path_layer_metrics.{json,csv}` likewise records AA, AB, BB, and BA
for every one of the 32 layers in a single bundle. Its name is deliberately
path-neutral; the former BA catch-up name is retired.
