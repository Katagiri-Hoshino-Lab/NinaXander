# 02 · 現在の知見

[English](../02-findings.md) | 日本語 | [简体中文](../zh-CN/02-findings.md)

この文書の全数値は、現行公開実験、すなわち32層RWKV-4-Raven-7B /
Tulu-Pythia-6.9B pair、step-415,000のforward-hook adapter、
`./reproduce.sh tables`で再構築した構造化成果物に属します。このcontract外の結果は含みません。

## 結果1 · 報告adapter：32層、forward hook、plateauまで収束

報告adapterは非圧縮latent（`z = d = 4096`）を使い、32 blockすべてを公平にpairにします。

`z = d = 4096`（圧縮なし）、encoderとdecoderそれぞれにzero-initialized ResBlock 1個（hidden 4096）、σ=0.4、
N/layer=228,000、4 GPUでbatch 4096、lr 1e-3をplateau decayで1e-5まで低下させました。
残差は`hidden_states`ではなく**全32 blockのforward hook**で取得します。
2回のwalltime cycle（job 1832 → 1837）にわたり**416,000 step**走行した時点で計算資源上限により終了し、報告値はすべてheld-out A→B最良の**step 415,000** checkpointを用います。
adapter 268.5M parameterは14.2B frozen weightの1.9%です。最深点（step 415,000）のheld-out結果：

| | 値 |
|---|---|
| A→A（self） | 0.6950 |
| A→B（cross、held-out） | **0.5588** |
| B→B（self、reconstruction） | **0.7112** |
| B→A（cross、B-prefix） | 0.5900 |
| ρ_ctr（centered alignment） | **0.9006** |
| f（token-varying energy fraction） | 0.3501 |
| ρ_raw | 0.9654 |

知見は対応する順序関係、すなわち**B→B 0.711 > A→B 0.559**と
**A→A 0.695 > B→A 0.590**で、そのときのcentered latent alignmentは**ρ_ctr 0.901**です
（ρ_ctrは相関係数であり、これらのR²と同じ尺度では比較できません）。32層すべてでABは同じB decoderを使う
BB self-mapを下回り、BAは同じA decoderを使うAA self-mapを下回ります。
**高いlatent alignmentのもとでもsame-family reconstructionの段階ですでに誤差が残り、両cross read-outはそれをさらに下回ります。
この誤差がencoder、latent normalization、decoder、training時noiseのいずれに由来するかは切り分けていません。**

**どちらのself-mapを制約している要因も未解決です。** latentの幅はresidualと等しく、非affine LNが落とすのも
tokenごとの平均とnormの2自由度にすぎないため、幅やrankが強い制約だとは考えにくいです。decoderはz+ξ、ξ~N(0,σ²)から
再構築するよう学習するため、最良係数は1/(1+σ²)へ縮みます。ただし1/(1+σ²)≈0.862は
その係数を*noisy* input上で評価したR²であり、本研究は**clean z（ξ=0）**で評価します。
scalar-linear ceilingは1−(σ²/(1+σ²))² ≈ **0.98**で、数値的にも確認しました。
したがって両self-map scoreはnoise ceilingを大幅に下回り、
**noiseがbinding constraintだとは確立されていません**。対応するσ sweep
{0,0.1,0.2,0.4,0.8}が必要です。B→A（0.590）はA→B（0.559）をわずかに上回ります。
Transformer latentからRNN側残差を再構築する方が、その逆よりわずかに容易です。

**negative control（`four_path_layer_metrics.py`、forward hook、400 window）。**
ρ_ctr 0.90と両cross-readoutがtokenごとの対応ではなく分布の重なりにすぎない可能性を排除するため、
Aのrow順序を維持し、B rowへ1つの固定random permutationを適用してcross量を再計算しました。
結果は**崩壊**し、ρ_ctr 0.9007 → +0.0000、A→B 0.5601 → −0.5909、
B→A 0.5911 → −0.6083となりました（self-map A→A/B→Bは構造上不変）。
再評価は報告meanも再現しており、両cross-readoutが実際のtoken対応へ依存することを示します。

training signalはAlpacaの2,075 window × 112 token（独立token位置約232k）を32層へpoolした
7.44M residual pairで、約228 cycle使いました。

各metricはlayer内でcentered / SST統計を除き、**各層の全tokenについて計算した後32層平均**を取ります。
層をpoolしません。率直な注意点としてseedは1つです。またstep 408k〜415kで4 read-outの変化は
すべて0.001未満（AB 0.5584→0.5588、BA 0.5899→0.5900、
AA 0.6941→0.6950、BB 0.7106→0.7112）なので、これは**厳密な収束ではなく漸近plateau**であり、
最深点を報告しています。

## 結果2 · 両実行経路で得られるもの：実測KV cache

仮定ではなく実測（`kvcache_measure.py`、job 1809）すると、Tulu-Pythiaは
**1 token・1 layerあたり16.0 KiB**、32層合計512 KiB/tokenをcacheし、
2·L·d·2 byteと正確に一致します。代替するRWKV recurrent stateは
**1 layerあたり64 KiBで、sequence lengthに依存しません**
（5つのstate vectorのうち2つをfp16、3つをfp32で保持するreference実装からの算出値）。

ABは`31−L`個のTransformer suffix block、BAは`L+1`個のTransformer prefix blockを保持します。
したがって実測memory/accuracy tableは最初から両経路を含みます。

| 構成 | Transformer層数 | KV/token | KV @ 4k ctx | 削減 | ARC / SciQ（acc_norm） |
|---|---|---|---|---|---|
| pure Tulu-Pythia | 32 | 512 KiB | 2.00 GiB | — | 65.3 / 81.0 |
| pure RWKV-Raven | 0 | 0 | 0 | 100% | 61.8 / 67.0 |
| AB@4 | 27 | 432 KiB | 1.69 GiB | 15.6% | 59.6 / 70.9 |
| **BA@4** | **5** | **80 KiB** | **0.31 GiB** | **84.4%** | **62.2 / 69.2** |
| AB@8 | 23 | 368 KiB | 1.44 GiB | 28.1% | 59.0 / 67.7 |
| BA@8 | 9 | 144 KiB | 0.56 GiB | 71.9% | 57.3 / 64.6 |
| AB@16 | 15 | 240 KiB | 0.94 GiB | 53.1% | 48.6 / 54.5 |
| BA@16 | 17 | 272 KiB | 1.06 GiB | 46.9% | 57.5 / 68.0 |
| AB@24 | 7 | 112 KiB | 0.44 GiB | 78.1% | 47.0 / 51.1 |
| BA@24 | 25 | 400 KiB | 1.56 GiB | 21.9% | 59.6 / 69.9 |

QAは**全test set**（ARC-Easy 2376 / SciQ 1000、acc_norm、temperature 0）と、
対応McNemar/bootstrap比較（結果5）を使います。SciQの統合non-dominated sequenceは
Pythia → AB@4 → BA@24 → BA@4 → RWKVですが、これはpost-hocな単一task frontierで、
普遍的rankingではありません。BA@4はAB@4に近い2 task平均（65.70対65.25）を保ちながら、
Transformer blockは27個ではなく5個です。

pure RWKVはKVを全く持たず、BA@4より有意に悪くないため、新しい点はtrade-offであって
dominating architectureではありません。以前の「このmemoryでこのaccuracyに達する点はない」
という枠組みは誤りでした。その後、等KV early-exit Pythia比較を実行しました（結果8）。
浅いswitchとは同等、深いswitchには負けますが、resident weightは軽量です。このcurveは
KV quantization、sliding-window attention、効率化目的で学習したTransformerとは比較していません。

また無償ではありません。現在の構成でHF RWKVはrecurrenceを逐次実行するため、reference runtimeは
ctx 2048のpruned prefillで**約60倍低速**（22.6 s対0.378 s）、unpruned cacheless decodeで
**64倍低速**（0.148対9.49 token/s）です。これはarchitectureの限界ではなく実装特性ですが、
公開reference pathでは現実の制約です。

**serving microbench（`a_to_b_*`と`b_to_a_*`の両方、実測）。**
unpruned reference実装は両親を完全にloadし、親全体のforward内で境界をoverrideするため、
resident-weight削減を実現しません。物理pruned AB/BA pathは未使用blockを削除し、
unpruned referenceを正確に再現します。resident weightはABが13.93〜14.55 GiB、
BAが13.49〜14.11 GiBです。speedはHF RWKVの逐次実行とcacheless reference decoderに
依然支配されます。これらはarchitecture限界ではありませんが、現時点でmemory軸がもたらす価値を制約します。

結果8が物理pruned pathを提供し、このunpruned referenceに対して正確にgateします。
resident-weight削減を実現する一方、上記unpruned timingは最適化済みthroughputの主張ではなく、
reference-runtime測定として残します。

## 結果3 · どちらのcross-family pathも「near-linear」ではない

報告32層adapterの5 probe layer {4, 10, 16, 22, 28}について、実held-out residual上で測定しました
（`artifacts/metrics/raw/cross_family_linearity_L32.json`、
`experiments/linearity.py`。layerごとに50,400 fit row /
100,800 disjoint eval row、fp64 normal equation、
ridge 1e-3·tr(XᵀX)/(d+1) ≈ 50）。

| 経路 | adapter R² | affine-imitation R² | 不可約non-linear | imitation cross-R² | 最良direct affine R² |
|---|---:|---:|---:|---:|---:|
| AB | 0.558 | 0.866 | **13.4%** | 0.490 | 0.487 |
| BA | 0.576 | 0.880 | **12.0%** | 0.516 | 0.535 |

結果は2つあります。（1）「EとDがそれぞれlinear mapとして学習を始める」というのは
initialization特性（各ResBlock第2 matrixのzero-init）にすぎず、間に非affine LNが入るため
cross map全体はその時点でも厳密にはlinearではありません。415k step後のmapは明確に
non-linearであり、「near-linear」という表現は**撤回**します。（2）最良*direct* affine mapでも
AB 0.487、BA 0.535へ到達します。任意correspondenceをfitできないlow-capacity mapであり、
shared-space claimを支えるのはheavy adapterではなく、これらのdirect-affine baselineです。
adapterとの差はAB +0.071、BA +0.041です。

downstreamでは、別途fitしたaffine mapはABでadapterより8〜23 point弱い一方、
BAでは差が小さく、符号が変わる場合もあります。map間比較は記述的です。
結果7は、両pathの因果検定に同一adapterのalpha介入を使います。

## 結果4 · translationはdomain-specific：劣化はdomain shiftでありcontext lengthではない

adapterはAlpaca（instruction text）で学習します。「long contextで壊れる」と
「out-of-domain textで壊れる」を分離するため、対応する`a_to_b_longctx_ce_mp.py`と
`b_to_a_longctx_ce_mp.py` runは、**in-domain Alpaca**と**out-of-domain WikiText-103**の
ctx 512 / 1024 / 2048でnext-token CEを測ります。各pathで最良affine mapも測定しますが、
各domainの先頭40,000 tokenで**domainごとにfitし直し**、分離したsuffixで評価します。
したがってWikiText affine armはdomain-adapted supervised oracleであり、
Alpaca-fit mapをdomain外へ移したものではありません。

- **両pathで軸となるのはlengthではなくdomainです。** ctx 512でAB@4のAlpaca/WikiText
  perplexityは6.40/93.87、BA@4は3.85/45.00です。両親の劣化ははるかに小さいです。
  BAはshiftを緩和しますが、WikiText perplexityは親モデルよりかなり高いままです。
- **contextを長くしてもどちらのchimeraも崩壊しません。** ctx 512 → 2048で、
  WikiText perplexityはAB@4が93.87 → 83.40、BA@4が45.00 → 34.78へ改善します。
  より良い親はctx 2048で8.27なので、失敗はcontext-length limitではなく、
  BAもdomain問題を解決しません。
- **domain内affine対照は記述的で、cleanなnon-linearity ablationではありません。**
  WikiText lin@8 CEはctx 512 → 2048で5.65 → 6.28、adapter@8は4.74 → 4.72ですが、
  2つのmapはobjectiveもtraining dataも異なります。これだけではadapterのnon-linearityを
  原因と特定できません。結果7の対応する同一adapter α介入は両AB/BAでこのconfoundを除きますが、
  それが示すのも同時に学習したこのadapterからbranchを外した場合の性質であり、
  non-linearityが一般に必要であることではありません。

これは両方向でmemory効果（結果2）の範囲を限定します。KV削減は実在しますが、
inputがtraining domainに似る場合に限ります。

## 結果5 · 対応検定を伴う対称な全量QA

AB/BA evaluatorは**全量**ARC-Easy（2376）とSciQ（1000）test setを同一項目順で実行します。
両方のsame-family armを含みます。BBはBの残差を`D_B(E_B(·))`へ通し、
AAはAの残差を`D_A(E_A(·))`へ通します。`paired_stats.py`は各switchを両親と比較し、
同一項目上でABとBAを比較します。McNemar exact testとpaired bootstrapを使います。
全量acc_norm：

| task / layer | AA | AB | BB | BA |
|---|---:|---:|---:|---:|
| ARC-Easy @4 | 62.8 | 59.6 | 62.2 | 62.2 |
| ARC-Easy @8 | 59.8 | 59.0 | 62.7 | 57.3 |
| ARC-Easy @16 | 54.8 | 48.6 | 57.4 | 57.5 |
| ARC-Easy @24 | 52.2 | 47.0 | 59.4 | 59.6 |
| SciQ @4 | 68.5 | 70.9 | 74.8 | 69.2 |
| SciQ @8 | 66.7 | 67.7 | 74.8 | 64.6 |
| SciQ @16 | 59.8 | 54.5 | 71.1 | 68.0 |
| SciQ @24 | 57.9 | 51.1 | 70.1 | 69.9 |

- **全AB/BA switchは、両taskでより強いPythia親を有意に下回ります。**
- **弱い親との関係はpath/task依存です。** AB@4はSciQでRWKVを上回ります
  （+3.9、p=0.018）が、ARCでは下回ります（−2.2、p=0.042）。
  どのBA switchもRWKVを有意には上回らず、BA@4はARC（+0.4、p=0.66）、
  SciQ（+2.2、p=0.14）で同等です。
- **same-family armは普遍的な加算「encode–decode往復cost」ではありません。**
  B suffixではBBがPythiaを下回り、ABはさらにBBより3〜19 point低くなります。
  A suffixでは逆に深いBAがAAを上回り、L=16/24でARC +2.7/+7.4、
  SciQ +8.2/+12.0 pointです。AA/BBはencode–decode往復による損失を分離し、この損失はBBでは全switch層の両taskで、
  AAでは両taskともL=16/24でのみ有意に下流accuracyへ現れます。ただし
  cross-family差はprefix親も変えるため、translation errorだけとは解釈できません。
- **depthはpath依存です。** ABはLとともに単調低下しますが、BAはL=16/24で回復し、
  同層ABを8.9〜18.8 point上回ります。local cross-readout R²だけではend-to-end rankingを予測できません。

標準構造化結果は`paper_four_path_qa_accuracy.csv`です。両方向の全parent、affine、
equal-memory、intervention rowは`paper_cross_family_qa_accuracy.csv`にあります。

## 結果6 · 同じ番号の層が本質的に特別なのではない：alignmentは学習される

`all_layer_corr.py`は、Aのlayer jとBのlayer kのadapter-free 32×32 correspondence
（linear CKAとin-sample best-linear R²）を計算します。**linear CKA**ではraw cross-family表現の
類似性は弱く（diagonal mean 0.049、off-diagonal 0.035）、Aのlayer jに最も似るB layerが
同じjなのは**2/32** rowだけです（argmaxは通常Bの最終層）。best-linear R²は全体に高い
（約0.85）ものの、in-sample inflationで情報価値がありません。

したがってρ_ctr=0.90は、本質的なlayer-to-layer correspondenceではなく、
選択したsame-layer pair上でadapterが**学習した**alignmentです。これは主張を
「共有*semantic* space」から「対応条件下で回収可能なtoken-level representation correspondence」へ
狭めることと整合します。

## 結果7 · adapterのnonlinear branchを外すとAB/BA両read-outが崩壊する

結果3のlinear-map比較は、objective/data/layer-setの異なる*別fit* linear mapを使ったため、
confoundの可能性があります。`cross_family_mlp_intervention.py`と2つの標準QA evaluatorは
それを取り除きます。同じ学習済みadapter唯一のnon-linearity、すなわちResBlock branch
fc2(GELU(fc1(·)))をαでscaleします。α=1はfull、α=0は*同じweight/layer/row*でbranchだけを
無効にしたものです。ABはE_AとD_B、BAはE_BとD_AのResBlockを通るため、両pathが1つのbranchを
共有するわけではありません。EとDはそれぞれ2つのLinearに挟まれたzero-initialized ResBlock 1個で
（cross pathはその2個を通ります）、αはそれらすべてをまとめてscaleします。
representation sweepと両full-set・項目対応QA pathをまとめて保存します。

| path / α | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| mean AB R² | **−0.41** | −0.75 | 0.35 | 0.52 | **0.56** |
| mean BA R² | **−0.30** | −0.66 | 0.36 | 0.53 | **0.58** |

branchを無効化（α=0）すると、ABは**−0.41**、BAは**−0.30**へ崩壊します。
全量normalized QAでも、両実行pathで同じ効果を直接確認できます
（各entryはα=1 → α=0 accuracy %、chanceは25）。

| L | ARC-Easy AB | ARC-Easy BA | SciQ AB | SciQ BA |
|---:|---:|---:|---:|---:|
| 4 | 59.6 → 31.9 | 62.2 → 31.8 | 70.9 → 31.3 | 69.2 → 32.6 |
| 8 | 59.0 → 29.9 | 57.3 → 35.5 | 67.7 → 31.4 | 64.6 → 37.0 |
| 16 | 48.6 → 28.5 | 57.5 → 43.7 | 54.5 → 31.7 | 68.0 → 47.3 |
| 24 | 47.0 → 34.8 | 59.6 → 48.4 | 51.1 → 40.0 | 69.9 → 49.1 |

branchの寄与は**ABで+11.1〜+39.6 point**、**BAで+11.2〜+36.6 point**で、
16個すべての項目対応McNemar testがp<2e-9です。2点が分かります。
（1）両α sweepは**非単調**で、0.25は0より悪くなります。branchを縮小してもread-outは
滑らかには劣化しません。（2）各adapter自身のbare linear skeleton（AB −0.41、BA −0.30）は、
別fit direct affine map（AB 0.487、BA 0.535）より悪くなります。周囲のLinearは
ResBlockと協調するよう学習されており、それを除けば両pathが崩壊します。
ただしこれは同時に学習したこのadapterからbranchを外した場合の性質であり、
non-linearityが一般に必要であることは示しません。少なくともBAの正答率では、別fitのaffine対照と
adapterの差は小さく（−3.4〜+2.0点、結果3）、adapterが一貫して上回るわけではありません。

## 結果8 · 等KV Transformer baselineと実際にpruneしたserving

対応する`a_to_b_serving_pruned.py` / `b_to_a_serving_pruned.py` pathは未使用blockを物理的に解放し、
rel=0でunpruned chimeraを再現します。`equal_mem_baseline.py`は、各AB/BA switchと
完全に同じTransformer depth/KVを持つPythia early exitを提供します。

- **両pathでpruned weightは約半分です。** ABは13.93〜14.55 GiB、BAは13.49〜14.11 GiBで、
  両親完全版の約27 GiBに対して小さくなります。対応early-exit Pythia baselineは、
  chimeraがRWKV blockとadapterも保持するため、さらに軽量です。
- **Transformer depthが浅いとき、non-Transformer側は実際の能力を加えます。**
  ABでは@16対exit@15がARC +11.2 / SciQ +12.8、AB@24対exit@7が+18.5 / +18.7
  （すべてMcNemar p<1e-10）です。浅いswitchは同等です
  （AB@4対exit@27は±1 point、p>0.5）。BAを5/9/17/25 blockのexitと比較したgainは、
  ARC +35.4/+27.6/+16.9/+2.1、SciQ +38.7/+31.6/+17.9/+0.5です。
  最初の3比較は有意で、25-block比較は同等です。したがって保持Transformerが短い場合、
  RWKV prefixでもsuffixでも実質的な能力を追加できます。
- **率直なfrontierには両pathと両親が含まれます。** pure RWKVはKVゼロで、
  対応Pythia exitはresident weightが軽量です。KV quantizationとsliding-window attentionは
  比較していません。
