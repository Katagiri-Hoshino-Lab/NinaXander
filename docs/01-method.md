# 01 · Method

English | [日本語](ja/01-method.md) | [简体中文](zh-CN/01-method.md)

## Residual pairs (the data)

Feed the **same text** to two frozen models A and B that share a tokenizer. At token position *t* and block index
*j*, take the output of block *j* from each — an **index-matched, position-matched pair**.

Residuals are captured by a **forward hook on every block**, not read out of `hidden_states`. `hidden_states` is not
a uniform quantity across families, and its last entry is not even a block output:

- RNN/SSM models (RWKV, Mamba) append *inside* the block loop, so `hidden_states[j]` is the output of block *j*, and
  `ln_out` is appended as a separate final entry.
- Transformers (GPTNeoX) prepend the embeddings, so block *j* sits at `hidden_states[j+1]`, **and the last entry is
  `final_layer_norm(block_{L-1})`** — the raw output of the last block is never exposed, and the norm is not
  invertible because it discards the per-token mean and scale.

The reported adapter therefore uses forward-hook capture: all layers are the
same kind of quantity for both families, layer 0 is included, and every one of
the 32 blocks is paired against a raw block output. Two hazards the hook must
handle and that are **gated**, not assumed: RWKV halves the
stream every `config.rescale_every` blocks *after* the block returns (so the hook must mirror it), and `RwkvBlock`
returns a tuple while `GPTNeoXLayer` returns a bare tensor.

The current extractor builds both caches from the same prompt/token/layer indices and gates hook-captured values
against Hugging Face's forward outputs.

## The shared-latent adapter

One encoder and one decoder **per model**, both encoders mapping into **one latent** `z ∈ ℝ^d_z` shared by **all
layers** (no layer conditioning — the same E_A serves layer 1 and layer 23):

```
z_A = LN(E_A(r_A)),   z_B = LN(E_B(r_B))          LN = non-affine LayerNorm
r̂_A = D_A(z_A + ξ),  r̂_B = D_B(z_B + ξ)          ξ ~ N(0, σ² I)
L = ‖r̂_A − r_A‖² + ‖r̂_B − r_B‖² + λ · ‖z_A − z_B‖²        λ = 1
```

Each term is a per-element mean (`F.mse_loss`), not a squared norm. Residuals are per-dimension standardised before
entering the adapter.

**Neither cross-map is fit to a cross-family target.** For
`A→B := D_B(E_A(r_A))`, `r_A` does not appear in B's reconstruction loss; the
same statement holds symmetrically for `B→A := D_A(E_B(r_B))`. They are not
trained directly. This is **not** "unsupervised", though: the alignment term
`λ‖z_A − z_B‖²` consumes exactly the position-aligned cross-family pairs a
supervised bridge would, and supplies the pairing. The honest claim is narrower
— **no cross-reconstruction target is required in either path**.

### Two design choices that are load-bearing

- **Non-affine LayerNorm on z.** Without it the align term is scale-degenerate: the optimiser shrinks `z` and the
  decoders re-amplify, so align → 0 while nothing is aligned. LN fixes `‖z‖²`.
- **Noise ξ before decoding (σ > 0).** LN fixes the *norm* of z but not the split between a token-independent
  common mode and the token-varying signal. The reported 32-layer model fixes
  σ=0.4. No current matched σ sweep exists, so this work does not claim that
  0.4 is optimal or isolate noise as the cause of the final reconstruction
  ceiling.

An optional `lam_cross > 0` adds the supervised cross terms
`‖D_B(z_A) − r_B‖² + ‖D_A(z_B) − r_A‖²`. It is not used by the reported
checkpoint; the current result uses `lam_cross = 0`.

## The four execution paths

We use **A = RWKV** and **B = Tulu-Pythia** throughout. A path name records
prefix then suffix/read-out: `AA`, `AB`, `BB`, or `BA`. The public model name
is `NinaXander-XY@L`, while structured data uses `X_to_Y`.

Both cross-family paths switch families **once** at layer *L* and are executed
directly:

| path | prefix blocks | one translation | suffix blocks + head | retained Pythia blocks |
|---|---|---|---|---:|
| `NinaXander-AB@L` | RWKV `0..L` | A-space → B-space | Pythia `L+1..31` | `31−L` |
| `NinaXander-BA@L` | Pythia `0..L` | B-space → A-space | RWKV `L+1..31` | `L+1` |

```
h_A = A.hidden_states[L]                                  # output of RWKV block L  (A ran L+1 blocks: 0..L)
h_B = unstd_B( D_B( E_A( std_A(h_A) ) ) )                # A-space → B-space, ONE translation
inject h_B as the input to B.block_{L+1}                  # B runs 31-L blocks + head

h_B = B.hidden_states[L+1]                                # output of Pythia block L
h_A = unstd_A( D_A( E_B( std_B(h_B) ) ) )                # B-space → A-space, ONE translation
inject h_A as the input to A.block_{L+1}                  # RWKV runs 31-L blocks + head
```

The split is **L+1 / 31−L**, not L / 32−L: both prefixes have already
executed *L+1* blocks. Thus AB@16 is 17 RWKV + 15 Pythia blocks, while BA@16
is 17 Pythia + 15 RWKV blocks. AB's remaining Pythia blocks are run by
**HF's own forward** with a pre-hook that replaces the injected layer's input,
rather than by re-implementing GPTNeoX's partial rotary and causal mask by
hand. BA is likewise evaluated directly, not inferred from its representation
R². Three
independent gates protect this less conventional path: re-injecting RWKV's true
post-L residual through an HF pre-hook must recover pure-RWKV logits; a
prefix-free direct RWKV suffix must recover the same logits; and the physically
pruned Pythia-front/RWKV-back implementation must match the unpruned hook path.
Every one of these gates must remain below `1e-4`.
The direct suffix is used for full QA, generation, and long-context CE so the
discarded RWKV prefix is not recomputed for every arm. The unpruned serving
benchmark deliberately measures the full re-forward implementation, while the
pruned benchmark measures the deployable block-truncated path.
QA answer options always use the canonical RWKV batch size of one. Concatenating
the adapter, affine, and A→A-control arms changed the logits by a relative
`1.8e-3` and was rejected by the equivalence gate. Independent batch-one arms
may run on separate CUDA streams only after a sequential/concurrent gate passes
at every switch; this preserves the batch-one kernel path while avoiding a
numerically different batched result.
The bidirectional checkpoint itself is unchanged. When executing BA alone,
the unreachable E_A/D_B modules can remain on CPU; QA retains E_A because its
A→A control uses it, while D_B is unreachable in every BA arm.
Every quantitative BA comparison fixes the same public step-415,000
checkpoint selected by held-out A→B; it does not reselect a checkpoint after
looking at B→A or BA QA.

The remaining two cells, `NinaXander-AA@L` and `NinaXander-BB@L`, are
same-family reconstruction controls. They use the same encoder/decoder
interface at the same boundary but do not change model family. All four cells
are reported together in `paper_four_path_qa_accuracy.csv`.

The current construction uses one family boundary and therefore one
translation. Multiple-boundary compositions are outside the reported
experiment.

## Metrics

- **Held-out per-dim R²** for cross-maps `A→B`, `B→A`, and self-maps `A→A`, `B→B`. Held-out = disjoint corpus half;
  the current extractor fixes the split.
- **Centered alignment** ρ and the **token-varying fraction** f = tr(Cov_t(z)) / E‖z‖².
  `align = 2·f·(1 − ρ)`. Report ρ and f — **never raw corr(z_A, z_B)**, which equals `1 − align/2` by construction.
- **Chimera CE / perplexity** and deterministic greedy generations for the
  functional test. The four prompts are supplied verbatim in English;
  decoding uses temperature 0 (argmax), at most 40 new tokens, and stops early
  only when EOS is emitted.

## What NOT to report (corrections banked this session)

- Raw `corr(z_A, z_B)` as evidence of a shared space — it is the training loss restated, and its optimum is
  reachable at zero information.
- "emergent" / "unsupervised" for either `A→B` or `B→A` — say "no
  cross-reconstruction target required in either path".
- A single "% of supervised ceiling" — the reported checkpoint has no
  matched multi-seed supervised control.
- `A→A`/`B→B` as a *ceiling* on `B→A`/`A→B`, or either gap as a
  "price of sharing" — the targets occupy different spaces; the real
  reference ceiling is a directly supervised per-layer bridge.

## Why the adapter must be lightweight (load-bearing)

NinaXander tests the hypothesis that matching layers of heterogeneous models share a semantic space after a *simple*
transform. This makes adapter capacity a methodological constraint, not just an overfitting concern: a heavy adapter
(a deep/wide MLP or a large ResNet) can approximate an arbitrary map between the two spaces, so high AB/BA read-outs under a
heavy adapter is **not** evidence for a shared space. Only a **light, low-capacity** map aligning the two spaces
supports the hypothesis — and the cleanest capacity control is a *linear* map, which cannot fit an arbitrary
correspondence yet still reaches AB 0.487 and BA 0.535. Those matched
linear-reachability measurements carry the shared-space claim.

The reported latent is uncompressed (`z = d = 4096`). The lightweight
argument is therefore about *capacity*, not width: the adapter is one
zero-initialised residual block between two linear maps, so it begins as an
exactly linear map and grows only the non-linearity it needs. Its 268.5M
parameters are 1.9% of the 14.2B frozen weights. Report both
cross-readouts *together with* the parameter count and architecture; either direction is evidence only when the
map is correspondingly constrained.

How near-linear it remains **after** training is itself measurable, and the
answer retracts the "near-linear" framing: on identical real residual
rows, best affine imitations explain **86.6%** of AB and **88.0%** of BA
adapter output. Thus 13.4% / 12.0% is irreducibly non-linear. Removing the
same trained residual-MLP branch collapses both cross read-outs (AB
0.56→−0.41, BA 0.58→−0.30). The adapter must therefore be reported as
low-*capacity*, never as "near-linear."
