# 03 · ロードマップ

[English](../03-roadmap.md) | 日本語 | [简体中文](../zh-CN/03-roadmap.md)

## 完了：7B instruct pairを構築し、adapterを特性評価

**RWKV-4-Raven-7B（pure RNN、instruct）↔ allenai/open-instruct-pythia-6.9b-tulu
（Transformer、instruct）。** 両方とも32L、d4096、GPT-NeoX-20B tokenizerで、
bit-identical tokenizationとconvention auto-detectを確認済みです
（A：RNN、post-append `off=0`、B：Transformer、pre-append `off=1`）。
当初予定したpartnerのDolly-v2-7bは**HFから削除**されたため、Tuluを代用しました
（候補：`usvsnsp/pythia-6.9b-sft`、`dvruette/oasst-pythia-6.9b`）。

RavenはBlinkDL `.pth`として配布されるため、同梱していない一回限りのhelperで変換しました
（transformers 5.x向けに`shard_checkpoint` → `save_pretrained`へpatch）。
その後**fp16**で再保存しました。fp32は28GBでOOMしたためです。
7B×2は32GB V100へ同時loadできないので、残差は**1モデルずつ**node単位cacheへ抽出します。
cacheの場所と容量はサイト設定（`NINAXANDER_TRAIN_CACHE`）であり、報告した32GB probe nodeの
shared memoryは23GBしかありませんでした。

結果は[02-findings.md](02-findings.md)を参照してください。shared spaceは実在します。
報告adapterはz=4096（圧縮なし）、268.5M parameter、σ=0.4、**415k step**
（job 1832 → 1837）で、forward hookにより**全32 block**の残差を取得し、
ρ_ctr 0.901、AA/AB/BB/BA = 0.695/0.559/0.711/0.590へ到達しました。
全32層でAB≤BB、BA≤AAであるため、両cross-readoutはalignmentではなく
**reconstruction-bound**です。self-mapを制約する要因は未解決です。以前のdraftにあった
noise ceiling 1/(1+σ²) ≈ 0.862は*noisy*-eval値で、clean-z evalでは約0.98です。
noiseはbinding constraintではなく、σ sweepが必要です。

## その後に完了：公平な再学習、両chimera path、対照、deploy可能path

報告checkpointはforward hookで全32 blockを取得します。adapter値はすべて
step-415,000 checkpointによります。

7Bの両実行path、ABとBAを構築し測定しました。両方とも首都France promptへ答え、
**会話できます**が、他の固定promptでは事実またはformatが劣化します。全量対応test setでは、
AB@4がARC 59.6 / SciQ 70.9、BA@4が62.2 / 69.2、対してRWKVは61.8 / 67.0、
Pythiaは65.3 / 81.0です。AB@4がRWKVを上回るのはSciQだけ（p=0.02）で、
BA@4は両taskでRWKVと統計的に同等、両pathともPythiaを下回ります。
完全なAB/BA tableは実測KV削減15.6〜84.4%を網羅します。SciQの統合non-dominated sequenceは
Pythia → AB@4 → BA@24 → BA@4 → RWKVです（post-hocな単一task frontier）。

さらに2つを特性評価しました。adapterは**near-linearではなく**、
AB出力の13.4%、BA出力の12.0%は不可約なnon-linear成分で、同じnonlinear branchを除くと
両read-outが崩壊します。translationは**両pathでdomain-specific**です。context 512で
AlpacaからWikiTextへ移ると、AB@4 perplexityは6.40→93.87、BA@4は3.85→45.00となり、
親モデルの劣化ははるかに小さいです。これはcontext-length limitではなくdomain shiftです。

各pathはrepresentation R²から推測せず直接測定します。全量QA、親および同層pathの対応test、
等KV early exit、Alpaca/WikiText context 512/1024/2048、固定prompt generation、
実測KV、unpruned/pruned servingを含みます。該当するboundary、direct-suffix、pruning gateは
exact zeroです。BA@4はTransformer blockを5個保持（KV 84.375%削減）し、
ARC 62.2 / SciQ 69.2です。8つのpath/switch構成では実測2 task平均が最良ですが、
AB@4との差は0.45 pointだけで独立selection splitはありません。これはpost-hocなrelease defaultで、
検証済みの普遍的winnerではありません。fused RWKV option batchはnumerical gateに不合格だったため、
報告BA QAはbatch 1のままです。

## harnessの不変条件（失敗から得たもの。regressさせない）

- **held-out windowをtraining hyper-parameterへ依存させてはいけません。**
  以前は`cut = N*6 + 2e6`だったため、Nの異なるrunを**別text**上で採点していました。
  現在は一定の`--eval_cut`を使い、truncation時はassertで明示的に失敗します。
- `windows()`は**最初の**n windowだけを残すため、`--eval_windows`がeval sizeそのものです
  （80なら8,960 tokenだけ）。現在は400です。全32層を*同じ*window上で採点するため、
  独立unitはlayerではなく**window**です。
- **`_best`でrunを比較してはいけません**（selection-biasedで約20回評価され、
  終端outlierの可能性があります）。対応stepで比較してください。
  `--keep_every`は周期checkpointを保持するため、後から再構築できます。
- **現在は`--seed`があります。** run間変動の主張には各条件で複数seedが必要です。
  現行論文は1 seedを報告します。

## 未解決問題

- **self-reconstructionを制約するものは何か。** 7Bのclean-z evalでnoise由来ceilingは約0.98です
  （0.862は*noisy*-eval値）。B→B 0.711は大幅に下回るため、
  **noiseはbinding constraintではありません**。原因は**未解決**であり、
  現行32層setup上の対応σ sweep {0, 0.1, 0.2, 0.4, 0.8}が決定的な実験です。
- **cross-reconstruction targetのcost。** 現行scaleで対応するmulti-seed runが必要で、
  推定値は報告していません。
- **translationをnear-losslessにできるか。** cross-map R²は改善leverの候補ですが、
  現行対照だけではchimera cost全体をそこへ帰属できません。decoder reconstructionと
  front-parent capabilityも含まれます。まだscaleを揃えて試していないleverには、
  より豊かで深いdecoder、vocabulary-anchored closed-form initialization、
  shared decoderではなくlayer別decoderがあります。

## 報告上のguardrail

- **raw corr /「emergent」という枠組み。** centered ρとfを報告し、両cross-readoutを
  「cross-reconstruction target不要」と呼びます。
- **tokenizerが異なるinstruct cross-architecture pair。** methodはposition-matched pairのため
  shared tokenizerを必要とします。Falcon-Mamba-Instruct（pure、64L）には同じtokenizerの
  32L Transformer partnerがなく、RecurrentGemma↔Gemma-2はblock数とtokenizerが一致しますが、
  RecurrentGemmaはattentionを含むhybridです。

## 延期したablation

z-dim sweep、λ_cross sweep、σ sweep、data-scaleはいずれも32層setupでは未実行です。
chimera用layer別対shared decoder、utility対alignment分析、同一architecture
（1つのfamilyに属する2 model）のalignment対照も延期しています。
これらの延期実験について、現在実行可能なscriptがあるとは主張しません。

7Bで完了済み（もはや延期ではない）のは**shuffled-correspondence negative control**
（`four_path_layer_metrics.py`）です。B rowをpermutationすると
ρ_ctr 0.90 → 0.00、A→B 0.56 → −0.59、B→A 0.59 → −0.61へ崩壊し、
両cross-readoutがtoken-level correspondenceに依存することを確認しました。
