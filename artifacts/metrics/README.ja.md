# 評価指標データ

[English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

評価指標パイプラインは2層構成です。

`raw/`は実験プログラムが書き出します。入れ子構造を含む実験は`name.json`と矩形の
`name.csv`を併記し、もともと表形式の実験はCSVを直接出力します。対応のある検定を再現できるよう、
項目ごとの正誤はJSONに保持します。

`raw/training_curve.csv`は公開学習曲線です。現行の学習と同じ追記専用schemaを使います。
最初の417個の履歴点は、正式なスケジューラログから一度だけ移行したため、公開checkoutで
ローカルログを必要としません。

`tables/`は決定論的に再構築されます。

```bash
./reproduce.sh tables
```

`tables/MANIFEST.csv`は、論文用の全tableについて行数とschemaを記録します。
`tools/validate_results.py`は、JSONの可読性、CSV headerと非空性、manifestの網羅性、
期待行数、最終表現tableとHugging Face配布metadataの整合性を検証します。ABとBAの検証では、
明示的な経路metadata、ARC-Easy/SciQの全項目数、および同一の18個のQA構成を要求します。
内訳は親モデル2、自己経路4、横断経路4、同一アダプターのalpha-zero介入4、affine対照4です。
両経路とも項目配列の整列と経路固有の境界gateが`1e-4`未満であることを検証します。
BAは独自のbatch-1 RWKV suffix runtimeを使うため、direct-suffix gateとconcurrent-arm gateも記録します。
ABとBAのQA runで2つの純粋な親モデルの項目別結果配列が完全一致することも求め、
dataset順序と採点の対称性を保護します。等メモリ対照では、ABに27/23/15/7、
BAに5/9/17/25個のPythia blockを使い、両経路で全項目配列を保持します。長文contextのwindow数、
serving構成集合、生成gate、実測KV層数・削減率も検査します。serving cellには有限の正値、
または明示的に構造化されたruntime errorだけを許可します。checkpoint step、線形fitの行数とridge、
context window数、concurrency cutoff、serving反復数、KV probe長、最終checkpointと生成の同一性は、
非形式的なログmetadataではなくsemantic contractの一部です。

長文contextのaffine対照は、各評価domainの先頭40,000 tokenという互いに分離された領域で
fitし直します。したがってWikiText affine armはdomain適応済みoracleであり、
Alpacaでfitしたmapをdomain外へ適用したものではありません。
過去のAB長文context/serving JSON 4件は、構造化run metadataの導入前に作成されました。
測定cellは変更せず、標準invocationから回収したfieldには
`legacy_metadata_enrichment`を明記しています。保存されていなかったruntime gate値は、
再構成せず未記録として列挙します。

`cross_family_runtime_smoke`は、AB/BAの全4切替点について、親再現gate、有限logit、
出力token検査を記録します。BAはRWKV suffixがbatchに敏感なため、direct-vs-hook、
独立CUDA stream、fused batch-3診断も記録します。fused受理flagは実測`1e-4`閾値と
一致しなければなりませんが、不採用は正当です。報告QAではfused optionを使わず、
標準RWKV経路は独立したbatch-1 callのままです。

対応のあるQA、MLP介入、等メモリ、同一層AB/BA比較は
`tables/paper_cross_family_paired_statistics.csv`に統合します。AB/BAのQAはどちらも、
task固有の最良切替を検定前に選ばず、4切替すべてを両親と比較します。全producerとtable builderは
`src/ninaxander/paired.py`を使い、そこでexact binomial tailをlog空間で評価します。
検証はunderflowした`mcnemar_p=0`を拒否します。

`raw/cross_family_mlp_representation.json`は、ABとBAそれぞれ25個の対応する介入cellを含みます。
task単位の`raw/cross_family_mlp_intervention_{arc,sciq}.json` bundleも対称であり、
各経路のsweepをalpha=0/1の8 QA構成および項目別結果と結合します。
`tools/complete_cross_family_mlp_bundle.py`は、標準AB/BA QA evaluatorの完了後に
この無損失joinを行います。旧式の一方向`linearity_L32.*`と`mlp_ablation_*.*` bundleは
検証で拒否され、結果contractには含まれません。過去の`mlp_items_*`と
`paired_statistics_mlp.csv`中間物は標準QA/task bundleへ一度統合され、その後は拒否されます。

標準`paper_four_path_qa_accuracy.csv` tableは、
2 task × 4切替層 × 4経路（AA、AB、BB、BA）を正確に含みます。raw成果物に残る
`a_to_b_`と`b_to_a_`のproducer prefixは、来歴情報としてのみ使用します。公開tableは
`producer_path`と実際の`execution_path`を分離し、同一の親行を重複排除して、
対応するAB/BA条件を交互に配置します。特にPythiaだけの早期終了は、
横断経路evaluatorが要求したというだけでAB/BAやBB adapter経路とはせず、
`B_early_exit`に分類します。

方向ごとに対応するCSVは、列schemaも完全一致しなければなりません。normalizerは、
旧generation、KV、長文context、serving sidecarを共有AB/BA code pathで書き直してから検証します。
`paper_result_coverage.csv`は公開結果系列をすべて監査し、対応するBAのないAB-only系列を拒否します。
paired tableは経路を独立標本として扱わず、全標準切替点で同一項目のBA対AB比較を含みます。

`raw/four_path_layer_metrics.{json,csv}`も、32層すべてのAA、AB、BB、BAを1つのbundleに記録します。
名称は意図的に経路中立であり、旧BA catch-up名称は廃止されています。
