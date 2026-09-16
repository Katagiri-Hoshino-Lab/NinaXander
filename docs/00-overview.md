# 00 · Overview

English | [日本語](ja/00-overview.md) | [简体中文](zh-CN/00-overview.md)

## The goal

**On the name.** NinaXander is named, intentionally, after the *Fullmetal Alchemist* chimera --- Nina Tucker and
her dog Alexander, fused into one talking creature. The reference is chosen for the correspondence, not just
"chimera = fusion": the fusion succeeds and the chimera *functions* (it speaks), yet it does not match the beings
it was made from. That is exactly this project's finding --- the composed models work, but none reaches the
stronger parent.


**NinaXander** names the family of chimera language models this project produces: composed models whose blocks come
from different pretrained architecture families, remain **frozen**, and are connected by one learned shared-latent
adapter. An individual NinaXander model is specified by its cross-family path and switch layer --- for example
`NinaXander-BA@4`, Pythia→RWKV at L=4 (see the naming below). The current
study evaluates one family boundary at four fixed locations; it does **not** report a search over arbitrary
per-layer choices. Only the current 32-layer experiment belongs to the
reproduction contract.

The families keep their internal representations in incompatible spaces (different widths, different bases). The
adapter's job is to translate a residual from one family's space into another's so a foreign block can run in place.

## Why this is hard

The two frozen models use incompatible residual spaces. A useful interface
must preserve enough information for the other family to continue inference,
while remaining small enough that it is not simply learning a new model
between the parents.

This project revisits this with a **shared-latent adapter** (one latent space both families encode into and decode
from, trained on all layers at once) and asks the composition question *functionally*, not just as alignment R².

## Four-path naming

The public notation is a 2×2 execution-path matrix: **A = RWKV** and
**B = Tulu-Pythia**; the first letter names the prefix family and the second
names the suffix/read-out family.

| prefix ↓ / suffix → | A (RWKV) | B (Pythia) |
|---|---|---|
| A (RWKV) | `NinaXander-AA@L` (same-family control) | `NinaXander-AB@L` (cross-family chimera) |
| B (Pythia) | `NinaXander-BA@L` (cross-family chimera) | `NinaXander-BB@L` (same-family control) |

`@L` means that the prefix runs through block L and the read-out/suffix starts
at block L+1. CSV metadata uses `A_to_A`, `A_to_B`, `B_to_B`, and `B_to_A`.
The diagonal cells `NinaXander-AA@L` and `NinaXander-BB@L` reuse the naming scheme only as same-family controls;
they are not NinaXander models in the sense defined above, which always cross a family boundary.

## The thesis (current, honest)

1. A shared latent that both families encode into **does** exist and is learnable across all layers at once — but
   its quality must be measured as **centered** alignment, not raw correlation (a non-affine LayerNorm makes raw
   corr an algebraic restatement of the loss; the optimiser satisfies it for free via a token-independent common
   mode that carries no information). See [02-findings.md](02-findings.md).
2. With the shared latent, **single-boundary chimeras function** — degraded but not collapsed. This is new
   relative to the prior evo catastrophe.
3. The residual interface is **lossy**, but the current 7B controls do not identify translation as the sole cost.
   The observed gap contains a same-family reconstruction (encode–decode round-trip) term and a
   cross-family-substitution term; the latter mixes the capability difference between the two possible prefixes
   with translation error. At 7B the paired read-outs are **A→B 0.559 below B→B 0.711** and
   **B→A 0.590 below A→A 0.695**, at every measured layer — so even under high latent alignment (ρ_ctr 0.901),
   same-family reconstruction is already imperfect, and the cross read-outs fall below it. Whether that error
   originates in the encoder, the latent normalization, the decoder, or the training-time noise is not isolated;
   together with the domain shift in item 5, the obstacles are imperfect alignment, reconstruction error, and
   domain shift. What limits the self-maps is unresolved: the latent is as wide as the residual and the non-affine
   LN drops only two degrees of freedom per token (mean and norm), so width/rank is unlikely to be a strong
   constraint; whether the noise σ sets a ceiling on B→B 0.711 is unknown, and a σ sweep would test it.
   The adapter is **not near-linear**
   in either cross-family path: best affine imitations explain 86.6% of AB and
   88.0% of BA output, while direct affine maps reach 0.487 / 0.535. Setting the
   residual-block branches of this jointly trained adapter to α=0 collapses both read-outs to negative R² and
   collapses downstream accuracy on both paths; that does not show that nonlinearity is generally necessary,
   and in BA accuracy the separately fitted affine control differs little from the adapter (−3.4 to +2.0 points).
4. Neither cross-family path beats the **stronger** Pythia parent on the full test sets. AB@4
   scores SciQ 70.9 / ARC 59.6; it beats RWKV only on SciQ and loses on ARC. BA@4
   scores 69.2 / 62.2 and is statistically tied with RWKV on both tasks. Deep BA
   switches are much stronger than their same-layer AB counterparts (+8.9 to +18.8 points at L=16/24), so the
   depth effect is path-dependent. The same-family reconstruction controls (AA/BB) isolate the loss of the
   encode–decode round trip, which shows up significantly in downstream accuracy for BB at every switch layer on
   both tasks and for AA on both tasks at L=16 and L=24; their gaps to the cross paths, however, mix front-parent
   capability and cross-family translation and are not pure estimates of translation error.
5. The best measured memory point is `NinaXander-BA@4`: five Transformer blocks, **84.375% KV removed**, ARC 62.2 and SciQ
   69.2. Its two-task mean 65.70 is only 0.45 points above AB@4's 65.25, and that default was selected
   post-hoc over 2 cross-family paths × 4 switches on the same reported tests, without an independent deployment-selection
   split. The physically pruned BA model occupies 14.11 GiB and exactly reproduces the unpruned path. Pure RWKV
   still has zero KV, and matched Pythia early exits carry fewer weights; the composition is therefore a trade-off,
   not a universally Pareto-optimal architecture. Domain shift also remains: BA@4 improves WikiText context-2048
   perplexity from AB@4's 83.4 to 34.8, but the better parent is still 8.27.

## Target model pair (built; adapter converged; chimera measured)

To ask "does a chimera keep *conversational* ability" — impossible on the base 130M models used earlier, which
cannot converse at all — this project uses a pure-non-Transformer **instruct** pair on the shared GPT-NeoX tokenizer:

| | RWKV-4-Raven-7B | open-instruct-pythia-6.9b-tulu |
|---|---|---|
| architecture | pure RNN (zero attention) | Transformer |
| blocks | 32 | 32 |
| width | 4096 | 4096 |
| tokenizer | GPT-NeoX-20B | GPT-NeoX-20B |
| tuning | instruct (Alpaca/ShareGPT) | instruct (AllenAI Tulu SFT) |

The originally-planned Dolly-v2-7b was deleted from HF; `allenai/open-instruct-pythia-6.9b-tulu` (Pythia-6.9b + Tulu
SFT) is the substituted instruct Transformer. Both are aligned on Alpaca instruction text.

Same block count and tokenizer are **required** by the shared-latent method (residuals are paired at matching token
positions, so both models must tokenize identically, and block *j* ↔ block *j* needs equal depth). This constraint
is why an instruct pair was hard to assemble: most instruct models across architectures use incompatible tokenizers.
RWKV-Raven ships as a BlinkDL `.pth` and was converted to HF format with a one-off helper that is not bundled; the
conversion was patched for transformers 5.x. See [03-roadmap.md](03-roadmap.md).
