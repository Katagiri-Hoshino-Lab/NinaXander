# 00 · 概要

[English](../00-overview.md) | 日本語 | [简体中文](../zh-CN/00-overview.md)

## 目的

**名称について。** NinaXanderは意図的に、『鋼の錬金術師』のchimera、すなわち
Nina Tuckerと愛犬Alexanderが融合され、言葉を話す一体の生物にされた存在に由来します。
単に「chimera = fusion」という意味ではなく、その対応関係を選びました。融合は成立し、そのchimeraは
（話すという意味で）*機能*しますが、元の存在には及びません。これは本研究の結果そのものです。
合成モデルは動作するものの、どれも強い方の親には届きません。

**NinaXander**は本プロジェクトが生成するchimera言語モデル群の名称です。各モデルは異なるpretrained
architecture family由来のblockを持ち、それらを**凍結**したまま、1つの学習済みshared-latent adapterで
接続した合成モデルです。個々のNinaXanderモデルはcross-family pathとswitch層で指定します
（例：`NinaXander-BA@4`、L=4のPythia→RWKV。下記の命名を参照）。現在の研究は4つの固定位置における1つのfamily境界を評価しており、
任意の層選択を探索したとは報告しません。再現contractに含まれるのは現行32層実験だけです。

各familyの内部表現は互換性のない空間（異なる幅、異なる基底）にあります。
adapterの役割は、一方のfamilyの残差を他方の空間へ変換し、異なるfamilyのblockを
その位置から動かせるようにすることです。

## 難しさ

2つの凍結モデルは互換性のない残差空間を使います。有用なinterfaceは、
他方のfamilyが推論を続けるのに十分な情報を保持しつつ、親モデルの間で新しいモデルを丸ごと
学習しているだけにならない程度に小さくなければなりません。

本プロジェクトはこれを**shared-latent adapter**で再検討します。両familyがencode/decodeする
単一latent spaceを全層同時に学習し、alignment R²だけではなく、合成が実際に機能するかを問います。

## 4経路の命名

公開表記は2×2のexecution-path行列です。**A = RWKV**、**B = Tulu-Pythia**とし、
1文字目がprefix family、2文字目がsuffix/read-out familyを示します。

| prefix ↓ / suffix → | A（RWKV） | B（Pythia） |
|---|---|---|
| A（RWKV） | `NinaXander-AA@L`（同family対照） | `NinaXander-AB@L`（cross-family chimera） |
| B（Pythia） | `NinaXander-BA@L`（cross-family chimera） | `NinaXander-BB@L`（同family対照） |

`@L`はprefixがblock Lまで実行され、read-out/suffixがblock L+1から始まることを意味します。
CSV metadataでは`A_to_A`、`A_to_B`、`B_to_B`、`B_to_A`を使います。
対角cellの`NinaXander-AA@L`と`NinaXander-BB@L`は命名規則を流用した同family対照にすぎず、
上で定義したNinaXanderモデル（必ずfamily境界をまたぐ合成モデル）ではありません。

## 現時点での率直な主張

1. 両familyがencodeできるshared latentは**存在し**、全層を同時に使って学習できます。
   ただし品質はraw correlationではなく**centered** alignmentで測る必要があります。
   non-affine LayerNormのためraw corrはlossの代数的な言い換えとなり、optimizerは情報を持たない
   token非依存common modeにより自動的に満たせるからです。詳細は
   [02-findings.md](02-findings.md)を参照してください。
2. shared latentを使うと、**単一境界chimeraは機能**します。劣化はしますが崩壊しません。
   これは以前のevolutionary experimentで起きた破局とは異なる結果です。
3. 残差interfaceは**lossy**ですが、現行7B対照だけではtranslationが唯一のcostだとは特定できません。
   観測gapにはsame-family reconstruction（encodeとdecodeの往復）項とcross-family-substitution項が
   含まれ、後者には2つのprefix候補間の能力差とtranslation errorが混在します。7Bでは対応read-outが
   **A→B 0.559 < B→B 0.711**、**B→A 0.590 < A→A 0.695**であり、
   全測定層で成り立ちます。したがって高いlatent alignment（ρ_ctr 0.901）のもとでも
   same-family reconstructionの段階ですでに誤差が残り、cross read-outはそれを下回ります。
   その誤差がencoder、latent正規化、decoder、学習時noiseのいずれに由来するかは切り分けていません。
   項目5のdomain shiftと合わせると、障害は不完全なalignment、reconstruction error、domain shiftです。
   self-mapを制約する要因は未解決です。latentはresidualと同じ幅で、non-affine LNが落とすのは
   tokenごとの平均とnormの2自由度だけなので、widthやrankが強い制約とは考えにくいです。
   noise σがB→B 0.711のceilingを定めているかは不明であり、σ sweepで検証できます。adapterはどちらの
   cross-family pathでも**near-linearではありません**。最良affine imitationが説明するのは
   AB出力の86.6%、BA出力の88.0%で、direct affine mapは0.487 / 0.535です。
   同時に学習したこのadapterのresidual-block branchをα=0にすると、両read-outは負のR²へ崩壊し、
   両pathで下流精度も崩壊します。ただしこれはnon-linearityが一般に必要であることを示しません。
   BAの精度では、別途fitしたaffine対照とadapterの差は小さく（−3.4〜+2.0点）、adapterが一貫して上回るわけではありません。
4. どちらのcross-family pathも、全test setで**より強い**Pythia親を上回りません。
   AB@4はSciQ 70.9 / ARC 59.6で、RWKVを上回るのはSciQだけ、ARCでは下回ります。
   BA@4は69.2 / 62.2で、両taskともRWKVと統計的に同等です。深いBA switchは
   同層ABより大幅に強く（L=16/24で+8.9〜+18.8 point）、depth効果は経路依存です。
   same-family reconstruction対照（AA/BB）はencodeとdecodeの往復による損失を切り分けます。この損失は
   BBでは全switch層の両taskで、AAではL=16とL=24の両taskで下流精度に有意に現れます。ただしcross pathとの
   gapには前半親モデルの能力とcross-family translationが混在するため、純粋なtranslation error推定ではありません。
5. 実測上の最良memory点は`NinaXander-BA@4`です。Transformer block 5個、
   **KV 84.375%削減**、ARC 62.2、SciQ 69.2です。2 task平均65.70はAB@4の65.25を
   0.45 point上回るだけで、独立deployment-selection splitなしに、同じ報告test上の
   2 cross-family path × 4 switchからpost-hocに選びました。物理pruned BA modelは
   14.11 GiBで、unpruned pathを正確に再現します。ただしpure RWKVのKVは依然ゼロであり、
   対応Pythia early exitはweightが少ないため、この合成はtrade-offであって、
   常にPareto-optimalなarchitectureではありません。domain shiftも残ります。
   BA@4はWikiText context-2048 perplexityをAB@4の83.4から34.8へ改善しますが、
   より良い親モデルは8.27です。

## 対象モデルpair（構築済み、adapter収束済み、chimera測定済み）

以前使ったbase 130M modelはそもそも会話できないため不可能だった、
「chimeraは*会話*能力を保持するか」を問うため、本プロジェクトは共有GPT-NeoX tokenizerを使う
pure-non-Transformerの**instruct** pairを採用します。

| | RWKV-4-Raven-7B | open-instruct-pythia-6.9b-tulu |
|---|---|---|
| architecture | pure RNN（attentionなし） | Transformer |
| block数 | 32 | 32 |
| width | 4096 | 4096 |
| tokenizer | GPT-NeoX-20B | GPT-NeoX-20B |
| tuning | instruct（Alpaca/ShareGPT） | instruct（AllenAI Tulu SFT） |

当初予定していたDolly-v2-7bはHFから削除されたため、
`allenai/open-instruct-pythia-6.9b-tulu`（Pythia-6.9b + Tulu SFT）を
instruct Transformerとして代用しました。両モデルはAlpaca instruction text上で整列させます。

同じblock数とtokenizerはshared-latent methodの**必須条件**です。残差を対応token位置でpairにするため
tokenizationが同一でなければならず、block *j* ↔ block *j*には同じdepthが必要だからです。
architectureをまたぐinstruct pairの多くはtokenizerが互換でないため、この制約がpair構築を難しくしました。
RWKV-RavenはBlinkDL `.pth`として配布され、同梱していない一回限りのhelperでHF形式へ変換し、
transformers 5.x向けにpatchしました。詳細は[03-roadmap.md](03-roadmap.md)を参照してください。
