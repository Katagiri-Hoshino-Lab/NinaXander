# NinaXander：異なるモデル族の凍結言語モデルを合成する

[English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

**「NinaXander：共有潜在表現による異種モデル族の凍結言語モデル合成――成立条件と限界」**のcode、構造化結果data、
run log、日本語論文（[`paper/paper_ja.pdf`](paper/paper_ja.pdf)）を収録しています。

**一文で。** 異なるarchitecture familyに属する2つの*凍結*7B言語モデル、
RWKV-4-Raven-7B（pure RNN）とopen-instruct-pythia-6.9b-tulu（Transformer）は、
対応条件下の対応層でtoken-level representation spaceを共有します。小さな学習済みadapterは
その対応を十分に回収し、残差変換1回で接続した実動する**AB/BA chimera**を構築できます。
経路は重要で、深い境界ではPythia-front → RWKV-backが大幅に強くなります。
@4の2 task平均はAB 65.25%、BA 65.70%で、BAが保持するTransformer blockは5個だけです。
chimeraは会話できますが事実性が劣化し、adapterは強くdomain-specificです。

> **NinaXander**という名称は『鋼の錬金術師』のchimeraに由来します。融合は成立し、その存在は機能します
> （言葉を話します）が、元の存在には及びません。本研究のchimeraも同様に、動作はするものの強い方の親には
> 届きません。この名称はその対応になぞらえたものです。

> 範囲：model pair 1組、corpus 1つ（Alpaca）、seed 1つ。論文の§Limitationsを参照してください。
> 公開結果contractに含まれるのは、現行32層RWKV-Raven/Tulu-Pythia実験だけです。

## 主要数値

すべて現行adapterによる値です。**z=4096（圧縮なし）、268.5M parameter、σ=0.4、
32層、415,000 step**（job 1832 → 1837）で、forward hookにより全blockの残差を取得しました。
held-outの最深点（step 415,000）：

| | |
|---|---|
| centered latent alignment ρ_ctr | **0.901** |
| 4経路read-out AA / AB / BB / BA | 0.695 / **0.559** / **0.711** / 0.590 |
| 最良direct affine cross-map AB / BA | 0.487 / 0.535（不可約non-linear出力：13.4% / 12.0%） |
| SciQ acc_norm（N=1000）：Pythia / AB@4 / BA@4 / RWKV | 81.0 / 70.9 / **69.2** / 67.0 |
| ARC-Easy acc_norm（N=2376）：Pythia / AB@4 / BA@4 / RWKV | 65.3 / 59.6 / **62.2** / 61.8 |
| @4で削除したKV：AB（RWKV→Pythia）/ BA（Pythia→RWKV） | 15.625% / **84.375%** |

公開結果では対称な**4経路表記**を使います。1文字目がprefix family、
2文字目がsuffix/read-out familyで、`@L`は切替層です。

| prefix ↓ / suffix → | A = RWKV | B = Pythia |
|---|---:|---:|
| **A = RWKV** | `NinaXander-AA@L`（同family対照） | `NinaXander-AB@L`（chimera） |
| **B = Pythia** | `NinaXander-BA@L`（chimera） | `NinaXander-BB@L`（同family対照） |

`L=4`の全量normalized accuracy：

| task | AA | AB | BB | BA |
|---|---:|---:|---:|---:|
| ARC-Easy | 62.8 | 59.6 | 62.2 | 62.2 |
| SciQ | 68.5 | 70.9 | 74.8 | 69.2 |

結果を支える4つの知見：

1. **両cross-readoutは対応するself-reconstructionを下回ります。**
   ρ_ctr 0.901 > B→B 0.711 > A→B 0.559、および
   ρ_ctr 0.901 > A→A 0.695 > B→A 0.590です。この不等式は32層すべてで成り立ちます。
   latent間の一致はdecoderが残差を再構築する能力よりはるかに高いため、scarce resourceは
   alignmentではなく*reconstruction*です。圧縮をなくしても（z = d = 4096）self-mapは
   1に近づかず、制約要因は未解決です。以前のdraftの1/(1+σ²) ≈ 0.862という
   「noise ceiling」は誤りで、*noisy*-eval値でした。clean zで評価する場合のceilingは約0.98で、
   B→B 0.711は大幅に下回ります。noiseはbinding constraintではなく、原因特定にはσ sweepが必要です。
2. **どちらのcross-family pathも「near-linear」ではありません。**
   最良affine imitationが説明するadapter出力はAB 86.6%、BA 88.0%で、残る13.4% / 12.0%は
   不可約non-linearです。direct affine mapはAB 0.487、BA 0.535へ到達します。
   同じ学習済みresidual-MLP branchを除くと、対応read-outは−0.41、−0.30へ崩壊します。
3. **両実行pathとも生成できますが、事実性が劣化します。** 4 promptを英語のまま逐語的に与え、
   決定論的greedy argmax（temperature 0）、最大40 new token、EOS時だけ早期終了でdecodeします。
   論文用sampleは共通frontier点`L=4`を使います。両pathとも“Paris”へ回答し、
   primary-colors回答を正しく開始しますが、水のcompletionには失敗します。ABはblack holeを
   1文で正しく定義し、BAは色の対話へ逸脱し、black holeを循環的に定義します。
   全量対応test setでは全AB/BA switchがguessingを上回り、より強いPythia親を下回ります。
   AB@4はSciQだけでRWKVを上回り（+3.9、p=0.019）、ARCでは下回ります
   （−2.2、p=0.042）。RWKVを有意に上回るBA switchはありません。対称なBB/AA対照は、
   gapが純粋なtranslation errorではないことも示します。ABはBBを下回る一方、
   深いBAはprefix親も変わるためAAを上回る場合があります。interfaceは両pathで
   **domain-specific**です。context 2048のWikiText perplexityはAB@4 83.4、
   BA@4 34.8で、より良い親は8.27です。
4. **ABとBAのdepth profileは異なります。** BA@16/@24は同層ABを8.9〜18.8 point上回り、
   BA@4はTransformer KVを84.375%削除しながら62.2/69.2を得ます。それでも両taskでPythiaを下回り、
   RWKVを有意には上回りません。2 task平均65.70はAB@4より0.45 point高いだけです。
   公開既定値への選択は、報告した2 test上で2 cross-family path × 4 switchを順位付けした
   **post-hoc**なもので、独立deployment-selection splitはありません。
   BA@4はWikiTextのdamageを減らしますが、context 2048のperplexityは34.8で、
   より良い親の8.27には及びません。

対応する同一adapter介入も対称に報告します。各entryは
full-adapter → residual-MLP-disabledのnormalized accuracy（%）です。

| L | ARC-Easy AB | ARC-Easy BA | SciQ AB | SciQ BA |
|---:|---:|---:|---:|---:|
| 4 | 59.6 → 31.9 | 62.2 → 31.8 | 70.9 → 31.3 | 69.2 → 32.6 |
| 8 | 59.0 → 29.9 | 57.3 → 35.5 | 67.7 → 31.4 | 64.6 → 37.0 |
| 16 | 48.6 → 28.5 | 57.5 → 43.7 | 54.5 → 31.7 | 68.0 → 47.3 |
| 24 | 47.0 → 34.8 | 59.6 → 48.4 | 51.1 → 40.0 | 69.9 → 49.1 |

16個すべての項目対応比較でMcNemar p<2e-9です。したがって学習済みnonlinear branchは
AB/BA両方で本質的であり、一方のpathから推測した効果ではありません。

## クイックスタート

```bash
pip install -r requirements.txt      # torch、transformers==5.2.0、datasets
cp .env.example .env                 # Python、model、SLURM設定を編集
./reproduce.sh slurm-config          # 解決済みの非secret設定を表示

./reproduce.sh check     # 可搬なsourceと既存構造化dataを検証（数秒、GPU不要）
./reproduce.sh tables    # raw JSON/CSV成果物から論文用CSVを再構築
./reproduce.sh map       # 全論文用CSV、行数、説明を一覧表示
./reproduce.sh paper     # paper/paper_ja.pdfを再構築
./reproduce.sh hf-check  # 3つのHub packageをstatic/checkpoint検証
./reproduce.sh hf-gpu-check  # 方向固定GPU smoke test 2件を投入
./reproduce.sh gpu-eval  # 対応するAB/BA非学習評価を投入
```

`check`、`tables`、`map`、`paper`、`hf-check`はCPUだけで実行できます。
`hf-gpu-check`と評価programには2つの凍結7B親モデル、最終adapter、GPUが必要です。
`gpu-eval`は学習を投入しません。

SLURM launcherには、ローカルcheckout path、partition、output path、account、QoS、GRES、
CPU数を含めません。これらはGit管理外の`.env`から読み、
`experiments/slurm/submit.sh`が注入します。`#SBATCH`行はshellが`.env`をsourceする前に
解析され、directive内の変数は確実に展開されないためです。設定変数と直接投入例は
[`experiments/slurm/README.ja.md`](experiments/slurm/README.ja.md)を参照してください。

## 結果data contract

各評価はmachine-readable dataを直接出力します。入れ子/項目別dataはJSONに保持し、
論文用の全数値関係をCSVへ正規化します。

- `artifacts/metrics/raw/`：実験ごとのJSON/CSV bundleと、paired-test/generation CSV。
- generation rowは`decoding=greedy_argmax`、`temperature=0`、
  `max_new_tokens=40`、`seed=0`を記録し、prompt自体も逐語的に保存します。
- `artifacts/metrics/raw/training_curve.csv`：公開417点training trajectory。
  ローカルscheduler logは不要です。
- `artifacts/metrics/tables/`：論文の報告値監査に使う、安定した結合済みCSV table。
- `artifacts/metrics/tables/MANIFEST.csv`：table名、schema、行数、説明。
- `artifacts/metrics/tables/paper_result_coverage.csv`：tableごとのAA/AB/BB/BA数。
  公開cross-family結果系列は必ず対応AB/BAを含みます。truncated Pythia baselineは
  `B_early_exit`として別に数え、BB adapter self-mapにはしません。
  4経路complete flagは、AA/BB self-map対照が方法論上適用可能な結果系列にだけ設定します。
- `tools/build_result_tables.py`：決定論的raw-to-table変換。
- `tools/validate_results.py`：schema、行数、JSON、manifest、release metadata、
  対応18-arm AB/BA QA、両pathのboundary gate（BAのdirect-suffix/concurrency gateを含む）、
  pruning/generation gate、正確なequal-memory depth、serving/KV整合性、対応raw-CSV schema一致、
  path間parent-item整合性を検証。

AB/BAどちらのend-to-end測定もrepresentation R²から推測しません。各pathを直接実行し、
対応する`a_to_b_*` / `b_to_a_*` raw名で出力します。標準4経路QA matrixは
`paper_four_path_qa_accuracy.csv`です。残る対応AB/BA評価はすべて
`paper_cross_family_*`へ統合し、別々の`producer_path`来歴field、実際の
`execution_path` columnを持ち、parent rowを重複させません。正規化公開tableを
AB-onlyとBA-onlyのcopyへ分けることはありません。

評価後は`./reproduce.sh tables && ./reproduce.sh check`を実行してください。
runtime logは来歴と診断用であり、論文数値のdata sourceではありません。

## Hugging Face配布package

軽量なadapter-only Hub公開候補はローカルの
`release/huggingface/ninaxander-raven7b-tulu69-adapter`
にstagingされます（Git管理外。`tools/export_hf_adapter.py`で生成）。FP32 adapterと層別統計をSafeTensorsで収録し、standalone loader、
gate付きchimera generation例、親成果物fingerprint、model card、該当research-only licenseを含みます。
3.1 GiB training checkpointのoptimizer/scheduler stateとローカル絶対pathは除外しています。

完全model（同様に`tools/export_hf_chimera.py`でローカルにstagingし、Git管理外）は
明示的な実行経路ごとに分離します。

- `release/huggingface/ninaxander-tulu69-to-raven7b-best`
  はPythia→RWKV、実測最良の既定値L=4です。Pythia block `0..4`、変換1回、
  RWKV block `5..31`を実行します。
- `release/huggingface/ninaxander-raven7b-to-tulu69-best`
  はRWKV→Pythia、実測最良の既定値L=4です。RWKV block `0..4`、変換1回、
  Pythia block `5..31`を実行します。

各packageは約28 GiBのoffline bundleで、正確なfp16両親、tokenizer、
step-415,000 adapter、全層standardization tensor、明示的switch-plan tensorを含みます。
経路は固定され、L=8/16/24は同一経路内比較に限り選択できます。両L=4既定値は、
独立selection splitなしに報告test上でpost-hocに選んだものです。

3 packageすべてが`SNAP_L32_final.pt`とのtensor完全一致検査に合格します。
完全bundleはさらに`AutoModelForCausalLM` load、parent-reproduction injection gate、
2 GPU generationへ対応します。各packageに英語・日本語・簡体字中国語model cardがあります。
完全static release監査は`./reproduce.sh hf-check`で実行してください。

private Hub snapshot（アクセス許可が必要）：
[collection](https://huggingface.co/collections/kotama7/ninaxander-private-inference-releases-6a6b02ec40c18d6b41ed040a)、
[adapter](https://huggingface.co/kotama7/ninaxander-raven7b-tulu69-adapter)、
[Pythia→RWKV best](https://huggingface.co/kotama7/ninaxander-tulu69-to-raven7b-best)、
[RWKV→Pythia best](https://huggingface.co/kotama7/ninaxander-raven7b-to-tulu69-best)。
軽量・完全packageの対応する決定論的exporterは`tools/export_hf_adapter.py`と
`tools/export_hf_chimera.py`です。documentationだけを変更した後は、
`tools/update_hf_release_checksums.py`が共有hard-link親shardを一度だけhashし、
3 packageのintegrity manifestを更新します。

元のBlinkDL Raven `.pth`の正確なfilename/revisionがローカルHF変換時に保持されなかったため、
3 packageともrelease candidateです。最終公開前にこのprovenanceを回収してください。

## directory構成

```
src/ninaxander/       再利用可能adapter、RWKV-suffix runtime、result-I/O package
.env.example          ローカルPython/model/SLURM設定の公開template
experiments/          現行32層training/evaluation entry point
  slurm/              役割別launcher：train、evaluate、aggregate
tools/                table builder、validator、Hub exporter
artifacts/
  checkpoints/        学習済みadapter（ローカル・Git管理外）
  metrics/raw/        実験の直接出力
  metrics/tables/     正規化済み論文用CSV
  logs/               ローカル・Git管理外のSLURM/evaluation診断
  runs/               構造化submission record（ローカル・Git管理外）
paper/                論文sourceとbuild済みPDF
docs/                 project-level method、finding、roadmap
release/huggingface/  推論専用Hugging Face package 3件（exporter出力・Git管理外）
models/               ローカル凍結親weight（Git管理外）
```

## 特定数値の再現

`./reproduce.sh map`は正規化済みCSV tableを一覧表示します。たとえば
`paper_four_path_qa_accuracy.csv`はAA/AB/BB/BA matrix、
`paper_cross_family_qa_accuracy.csv`はAB/BAのparent/chimera/control統合row、
`paper_cross_family_paired_statistics.csv`は両pathおよび同一項目比較のMcNemar testと
paired-bootstrap interval、`paper_cross_family_linearity.csv`は対応するAB/BA non-linearity分析、
`paper_cross_family_mlp_intervention.csv`は同一adapter alpha介入を保持します。
そのtask-level raw bundleも対応AB/BA summaryと項目別outcomeを含みます。
`paper_cross_family_domain_shift.csv`はin-domain/out-of-domain CE studyの統合表です。

`raw/cross_family_runtime_smoke.{json,csv}`は全4 switchでGPU検証済みAB/BA boundary gateを記録します。
標準QA programは`experiments/a_to_b_chimera_bench_full_mp.py`と
`experiments/b_to_a_chimera_bench_full_mp.py`、adapter学習は`experiments/adapter7b.py`です。

## hardware / environment

- **GPU必須。** 報告clusterでは初期probeにV100-32GB 1台、学習・評価に
  V100-16GB 4台のnodeを使いました。pairのresident memoryは28.9 GiB
  （RWKV 14 GiB + Pythia 13 GiB、fp16）なので、16 GiB GPU上の評価では各pairを2 GPUへ分割します
  （`*_mp.py`：RWKVは`cuda:0`、Pythiaは`cuda:1`、adapterはpathのsuffix modelと同じdevice）。
  BA evaluatorは到達可能なE_B/D_A halfだけをGPU上に置き、BA QAはAA decoder-loss対照用に
  E_Aも保持します。cluster partition/GRES名を公開scriptへ埋め込みません。
- **walltime。** adapter学習は12 h上限を超えるため1回resumeします。
  `continue_adapter.sbatch`がoptimizer、scheduler、global stepを復元します
  （runは2 cycle合計416,000 stepまで走行し、報告値はheld-out A→B最良のstep 415,000 checkpointを用います）。
- data（Alpaca、ARC-Easy、SciQ）は`datasets`で随時downloadし、
  `~/.cache/huggingface`へcacheします。
- `models/`以下のstandalone親copyはローカル用でGit管理外です。
  `release/huggingface/*-best/`の方向別候補2件は正確なfp16親を意図的にbundleし、
  source projectとは別repositoryへuploadします。RWKV-RavenのupstreamはBlinkDL `.pth`で、
  HF形式へ変換しました。両親はGPT-NeoX-20B tokenizerを共有します
  （必須条件。`docs/ja/00-overview.md`参照）。
