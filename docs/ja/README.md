# NinaXander

[English](../README.md) | 日本語 | [简体中文](../zh-CN/README.md)

> **NinaXanderは本プロジェクトが生成する合成chimera言語モデル群の名称です。**
> 各モデルは、異なるarchitecture familyの凍結blockを1つの学習済みshared-latent adapterで接続したものです。
> 個々のモデルは`NinaXander-BA@4`（Pythia→RWKV、L=4）のようにcross-family pathとswitch層で指定します。
> 「NinaXander」は常にこれらの合成モデルを指し、研究分野全体を指すものではありません。

異なるarchitecture family（RNN/SSM ↔ Transformer）の凍結済みpretrained blockを、
学習した**shared-latent adapter**を介して合成し、**chimera**言語モデル群を構築します。

adapterは手段であって目的ではありません。目標は、実際に*機能する*chimeraです。
残差alignment R²はそのproxyにすぎません。

## 文書

| ファイル | 内容 |
|---|---|
| [00-overview.md](00-overview.md) | NinaXanderの定義、主張、対象モデルpair |
| [01-method.md](01-method.md) | shared-latent adapter、そのloss、chimera構築法 |
| [02-findings.md](02-findings.md) | runまで追跡可能な確立済み結果を、誇張せず記述 |
| [03-roadmap.md](03-roadmap.md) | 完了事項、未解決問題、既知の失敗経路 |
| [../paper/](../../paper/) | 論文（`paper_ja.tex`） |

## 現状の要約

対象pairは**RWKV-4-Raven-7B（pure RNN）↔ open-instruct-pythia-6.9b-tulu（Transformer）**です。
ともに32 block、d4096、GPT-NeoX tokenizerであり、Alpaca上で整列させました。
latent幅z=4096（圧縮なし）、σ=0.4、268.5M parameterの単一shared-latent adapterを
**415k step**（job 1832 → 1837）学習し、forward hookで**全32 block**の残差を取得しました。
held-outの**centered** ρ_ctrは0.901、AA/AB/BB/BA read-outは
0.695/0.559/0.711/0.590です。両cross read-outは**全32層でsame-family reconstructionを下回ります**
（A→BはB→Bを、B→AはA→Aを下回ります）。高いlatent alignmentのもとでもsame-family reconstructionの
段階ですでに誤差が残り、その誤差がencoder、latent正規化、decoder、学習時noiseのいずれに由来するかは
切り分けていません。圧縮をなくしてもどちらのsame-family reconstructionもnear-losslessにならず、
non-affine LNが落とすのはtokenごとの2自由度だけなのでwidthやrankが強い制約とは考えにくいものの、
self-mapを制約する要因は未解決です。noise σがceilingを定めているかは不明であり、
σ sweepで検証できます。

adapterはどちらの方向でも**near-linearではありません**。AB出力の13.4%、BA出力の12.0%が
不可約なnon-linear成分です。同時に学習したこのadapterのresidual-block branchを無効にすると
（α=0。ABはE_AとD_Bの、BAはE_BとD_Aのblockを通ります）両read-outと下流精度が崩壊しますが、
これはnon-linearityが一般に必要であることを示しません。BAの精度では、別途fitしたaffine対照と
adapterの差は小さく（−3.4〜+2.0点）、adapterが一貫して上回るわけではありません。
7B chimeraは実在し、**会話できます**。両方とも“What is the capital of France?”へ回答しますが、
残りの固定promptでは事実またはformatが劣化します。**全test set**
（SciQ N=1000、ARC-Easy N=2376）でAB@4は70.9/59.6となり、RWKVを上回るのはSciQだけです。
BA@4は69.2/62.2で、両taskともRWKVと統計的に同等です。両方のcross-family pathの全構成が、
より強いPythia親を下回ります。BA@16/@24は同層ABより8.9〜18.8 point高く、
depth効果が経路依存であることを示します。

実測release既定値は`NinaXander-BA@4`です。Transformer blockを5つだけ保持し、
KVを**84.375%**削減しますが、2 task平均65.70はAB@4を0.45 point上回るだけです。
これは独立したdeployment selection splitによる検証ではなく、報告test上で
2 cross-family path × 4 switchからpost-hocに選んだ結果です。same-family reconstruction対照（AA/BB）が
切り分けるのはencodeとdecodeの往復による損失であり、どちらの方向でもそれらとcross pathとの差には
前半親モデルの能力とtranslationの両方が含まれ、純粋なtranslation errorではありません。
interfaceは依然として**domain-specific**です。BA@4はWikiText context-2048 perplexityを
AB@4の83.4から34.8へ改善しますが、より良い親モデルは8.27です。

中心的な未解決問題は、translationをnear-losslessにできるかです。現在の対照だけでは
chimeraの損失全体をcross-map R²へ帰属できず、encodeとdecodeの往復による損失と選択した
前半親モデルの能力も寄与します。

## 再現

プログラムは`../experiments/`、役割別launcherは`../experiments/slurm/`にあります。
`../reproduce.sh check|tables|map|paper|gpu-eval`で検証・集計・投入を行います。実験出力は
`../artifacts/metrics/raw/`へ書き、`../artifacts/metrics/tables/`の論文用CSVへ正規化します。
報告runでは初期probeにV100-32GB 1台、学習・評価にV100-16GB 4台のnodeを使いました。
model-parallel evaluator（`*_mp.py`）はpairを16 GiB GPU 2枚へ分割します。Python、partition、
CPU割当、GPU request、log pathはGit管理外の`.env`で設定し、公開launcherにサイト固有pathや
cluster名は含めません。
