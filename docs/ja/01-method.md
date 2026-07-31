# 01 · 手法

[English](../01-method.md) | 日本語 | [简体中文](../zh-CN/01-method.md)

## 残差pair（データ）

tokenizerを共有する2つの凍結モデルA・Bへ**同じtext**を入力します。token位置*t*、block index
*j*における各モデルのblock *j*出力を取り、**indexと位置を一致させたpair**にします。

残差は`hidden_states`から読むのではなく、**全blockのforward hook**で取得します。
`hidden_states`はfamily間で統一された量ではなく、最後のentryはblock出力ですらありません。

- RNN/SSM model（RWKV、Mamba）はblock loopの*内部*でappendするため、
  `hidden_states[j]`はblock *j*の出力で、`ln_out`が別の最終entryとして追加されます。
- Transformer（GPTNeoX）はembeddingを先頭へ追加するため、block *j*は
  `hidden_states[j+1]`にあります。さらに最後のentryは
  `final_layer_norm(block_{L-1})`であり、最終blockのraw出力は公開されません。
  normはtokenごとの平均とscaleを捨てるためinvertibleでもありません。

したがって報告adapterはforward-hook captureを使います。両familyの全層が同種の量となり、
layer 0を含む32 blockすべてをraw block出力同士でpairにします。hookが処理し、仮定ではなく
**gateで検証**すべきhazardが2つあります。RWKVはblock returnの*後*に
`config.rescale_every` blockごとにstreamを半分へscaleするためhook側も再現する必要があり、
`RwkvBlock`はtuple、`GPTNeoXLayer`はbare tensorを返します。

現在のextractorは、同じprompt/token/layer indexから両cacheを構築し、hook取得値を
Hugging Face forward出力に対してgateします。

## shared-latent adapter

各モデルに1つのencoderと1つのdecoderを置きます。両encoderは**全層で共有される単一latent**
`z ∈ ℝ^d_z`へ写像します（layer conditioningはなく、同じE_Aがlayer 1とlayer 23を処理します）。

```
z_A = LN(E_A(r_A)),   z_B = LN(E_B(r_B))          LN = non-affine LayerNorm
r̂_A = D_A(z_A + ξ),  r̂_B = D_B(z_B + ξ)          ξ ~ N(0, σ² I)
L = ‖r̂_A − r_A‖² + ‖r̂_B − r_B‖² + λ · ‖z_A − z_B‖²        λ = 1
```

各項はsquared normではなくper-element mean（`F.mse_loss`）です。adapterへ入れる前に、
残差を次元ごとに標準化します。

**どちらのcross-mapもcross-family targetへfitしません。**
`A→B := D_B(E_A(r_A))`に対して、`r_A`はBのreconstruction lossへ現れません。
`B→A := D_A(E_B(r_B))`も対称的に同様であり、直接学習しません。ただしこれは
「unsupervised」ではありません。alignment項`λ‖z_A − z_B‖²`はsupervised bridgeと同じ
位置対応済みcross-family pairを消費し、pairing情報を与えます。正確な主張はより限定的で、
**どちらの経路にもcross-reconstruction targetは不要**というものです。

### 本質的な2つの設計

- **zへのnon-affine LayerNorm。** これがないとalign項はscale-degenerateになります。
  optimizerが`z`を縮小しdecoderが再増幅するだけで、何もalignされていないのにalign → 0となります。
  LNは`‖z‖²`を固定します。
- **decode前のnoise ξ（σ > 0）。** LNはzの*norm*を固定しますが、token非依存common modeと
  token変動signalの配分は固定しません。報告32層modelではσ=0.4です。現在、対応するσ sweepはなく、
  0.4が最適とも、noiseが最終reconstruction ceilingの原因だとも主張しません。

任意の`lam_cross > 0`により、supervised cross項
`‖D_B(z_A) − r_B‖² + ‖D_A(z_B) − r_A‖²`を追加できます。報告checkpointでは使用せず、
現行結果は`lam_cross = 0`です。

## 4つの実行経路

全体を通して**A = RWKV**、**B = Tulu-Pythia**とします。経路名はprefix、suffix/read-outの順に
`AA`、`AB`、`BB`、`BA`と記録します。公開model名は`NinaXander-XY@L`、
構造化dataでは`X_to_Y`です。

両cross-family pathはlayer *L*でfamilyを**一度だけ**切り替え、直接実行して測定します。

| 経路 | prefix block | 1回の変換 | suffix block + head | 保持Pythia block |
|---|---|---|---|---:|
| `NinaXander-AB@L` | RWKV `0..L` | A空間 → B空間 | Pythia `L+1..31` | `31−L` |
| `NinaXander-BA@L` | Pythia `0..L` | B空間 → A空間 | RWKV `L+1..31` | `L+1` |

```
h_A = A.hidden_states[L]                                  # RWKV block Lの出力（Aは0..LのL+1 blockを実行）
h_B = unstd_B( D_B( E_A( std_A(h_A) ) ) )                # A空間 → B空間、変換は1回
inject h_B as the input to B.block_{L+1}                  # Bは31-L block + headを実行

h_B = B.hidden_states[L+1]                                # Pythia block Lの出力
h_A = unstd_A( D_A( E_B( std_B(h_B) ) ) )                # B空間 → A空間、変換は1回
inject h_A as the input to A.block_{L+1}                  # RWKVは31-L block + headを実行
```

分割はL / 32−Lではなく**L+1 / 31−L**です。どちらのprefixもすでに*L+1* blockを実行しているため、
AB@16はRWKV 17 + Pythia 15 block、BA@16はPythia 17 + RWKV 15 blockです。
ABの残りPythia blockは、GPTNeoXのpartial rotaryとcausal maskを手作業で再実装せず、
注入層の入力を差し替えるpre-hookを用いた**HF自身のforward**で実行します。

BAもrepresentation R²から推測せず直接評価します。この通常とは異なる経路を3つの独立gateで守ります。
RWKV自身のpost-L residualをHF pre-hookで再注入するとpure-RWKV logitsを復元すること、
prefix-freeなdirect RWKV suffixでも同じlogitsを復元すること、物理的にpruneした
Pythia-front/RWKV-back実装がunpruned hook pathと一致することです。全gateを`1e-4`未満に保ちます。
direct suffixはfull QA、generation、long-context CEで使い、捨てるRWKV prefixをarmごとに再計算しません。
unpruned serving benchmarkは意図的にfull re-forward実装を測り、pruned benchmarkは
deploy可能なblock-truncated pathを測ります。

QAのanswer optionでは常に標準RWKV batch size 1を使います。adapter、affine、A→A-control armを
連結するとlogitsが相対`1.8e-3`変化し、equivalence gateで不採用になりました。
独立batch-one armは、全switchでsequential/concurrent gateに合格した後に限り別CUDA streamで実行できます。
これによりbatch-one kernel pathを保ちながら、数値的に異なるbatched resultを避けます。

双方向checkpoint自体は変更しません。BAだけを実行する場合、到達不能なE_A/D_B moduleはCPU上に
置いたままにできます。QAではA→A decoder-loss対照にE_Aを使うため保持しますが、
D_Bは全BA armで到達不能です。定量的BA比較はすべて、held-out A→Bで選ばれた同じ
公開step-415,000 checkpointを固定し、B→AやBA QAを見てからcheckpointを再選択しません。

残る2 cell、`NinaXander-AA@L`と`NinaXander-BB@L`は同family reconstruction対照です。
同一境界で同じencoder/decoder interfaceを使いますが、model familyは変更しません。
4 cellすべてを`paper_four_path_qa_accuracy.csv`へまとめて報告します。

現在の構築はfamily境界1つ、したがってtranslation 1回です。複数境界の合成は報告実験の範囲外です。

## 評価指標

- cross-map `A→B`、`B→A`とself-map `A→A`、`B→B`の**held-out per-dim R²**。
  held-outは分離したcorpus halfで、現行extractorがsplitを固定します。
- **centered alignment** ρと**token-varying fraction**
  f = tr(Cov_t(z)) / E‖z‖²。`align = 2·f·(1 − ρ)`です。
  ρとfを報告し、構造上`1 − align/2`に等しい**raw corr(z_A, z_B)は決して報告しません**。
- functional testとしての**chimera CE / perplexity**と決定論的greedy generation。
  4 promptは英語のまま逐語的に与え、temperature 0（argmax）、最大40 new token、
  EOSが出た場合だけ早期終了します。

## 報告してはいけないもの（本sessionで反映済みの修正）

- shared spaceの証拠としてのraw `corr(z_A, z_B)`。これはtraining lossの言い換えであり、
  情報量ゼロでも最適値へ到達できます。
- `A→B`または`B→A`を「emergent」「unsupervised」とする表現。
  「どちらの経路にもcross-reconstruction targetは不要」と記述します。
- 単一の「supervised ceiling比%」。報告checkpointには対応するmulti-seed supervised対照がありません。
- `A→A`/`B→B`を`B→A`/`A→B`の*ceiling*とすること、またはどちらかのgapを
  「sharingの代価」とすること。targetが異なる空間にあり、真の参照ceilingは
  直接superviseしたlayer別bridgeです。

## adapterを軽量にすべき理由（本質的）

NinaXanderは、異種modelの対応層が*単純な*変換後にsemantic spaceを共有するという仮説を検証します。
したがってadapter capacityはoverfitting上の懸念だけでなく、方法論上の制約です。
重いadapter（deep/wide MLPやlarge ResNet）は2空間間の任意mapを近似できるため、
重いadapterで高いAB/BA read-outを得てもshared spaceの証拠にはなりません。
仮説を支持するのは2空間をalignする**軽量・low-capacity** mapだけです。最も明確なcapacity対照は
任意correspondenceをfitできない*linear* mapであり、それでもAB 0.487、BA 0.535へ到達します。
この対応linear-reachability測定がshared-space claimを支えます。

報告latentは非圧縮（`z = d = 4096`）です。そのためlightweightという議論はwidthではなく
*capacity*に関するものです。adapterは2つのlinear mapの間にzero-initialized residual blockを
1つ置き、厳密なlinear mapとして開始して必要なnon-linearityだけを成長させます。
268.5M parameterは14.2B frozen weightの1.9%です。cross-readoutはparameter countとarchitectureと
合わせて両方向を報告すべきで、対応して制約されたmapである場合にだけ証拠になります。

学習*後*にどの程度linearに近いかは測定可能で、結果は「near-linear」という枠組みを撤回させます。
同一の実残差row上で、最良affine imitationが説明するadapter出力はAB **86.6%**、BA **88.0%**です。
したがって13.4% / 12.0%は不可約なnon-linear成分です。同じ学習済みresidual-MLP branchを除くと
両cross read-outは崩壊します（AB 0.56→−0.41、BA 0.58→−0.30）。
よってadapterはlow-*capacity*と記述し、「near-linear」とは決して記述しません。
