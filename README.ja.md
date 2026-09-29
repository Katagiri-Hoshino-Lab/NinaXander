# NinaXander — 論文の参照用リポジトリ

[English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

本リポジトリは論文「NinaXander：共有潜在表現による異種モデル族の凍結言語モデル合成――成立条件と限界」
（[`paper/paper_ja.pdf`](paper/paper_ja.pdf)、[`paper/paper_en.pdf`](paper/paper_en.pdf)）の付随成果物である。
論文の読者が、報告された任意の数値の根拠データを特定し、生の評価出力から論文用の表を再生成し、両者を突き合わせる
検査を再実行できるようにすることを目的とする。汎用ライブラリではない。

本研究は、異なる系統の 7B 指示追従モデル 2 つ——線形注意で再帰的な推論形式を持つ RWKV-4-Raven-7B と、Transformer
である open-instruct-pythia-6.9b-tulu——を凍結し、対応層の残差ペアで単一の共有潜在アダプタを学習する。合成は、一方の
親の最初の `L+1` ブロックを走らせ、残差を一度だけ翻訳し、他方の親の残りのブロックを走らせることで行い、同じアダプタで
双方向を評価する。

## 主張から数値の根拠にたどる

論文に印字された数値はすべて正規化された CSV 表から導かれ、`tools/check_paper_claims.py` が再計算して不一致なら
ビルドを落とす。主張からデータへは次のようにたどる。

| 論文中の主張 | `artifacts/metrics/tables/` 配下の表 |
|---|---|
| 潜在整合、四経路の読み出し | `paper_representation.csv` |
| 層別および全層対応 | `paper_layer_metrics.csv`、`paper_layer_correspondence.csv` |
| 四経路の多肢選択正答率 | `paper_four_path_qa_accuracy.csv` |
| 境界ごとのキメラとアフィン対照 | `paper_cross_family_qa_accuracy.csv` |
| 有意性検定とブートストラップ区間 | `paper_cross_family_paired_statistics.csv` |
| アダプタの非線形性、α 介入 | `paper_cross_family_linearity.csv`、`paper_cross_family_mlp_intervention.csv` |
| KV キャッシュと正答率のトレードオフ | `paper_cross_family_memory_accuracy.csv` |
| 領域内と領域外の perplexity | `paper_cross_family_domain_shift.csv` |
| 生成例 | `paper_cross_family_generation_samples.csv` |
| 学習曲線 | `paper_training_curve.csv` |
| 評価集合への露出の分析 | `artifacts/metrics/raw/eval_leakage.json` |

`./reproduce.sh map` はすべての表をスキーマ・行数・説明とともに出力する。
`artifacts/metrics/raw/` には未変換の実験出力があり、ペア化検定が用いる項目単位の正誤記録もここに含まれる。

## 検査の実行

```bash
pip install -r requirements.txt
cp .env.example .env                 # Python・モデル・SLURM のローカル設定

./reproduce.sh check     # 構造、データスキーマ、論文とデータの一致（CPU、数秒）
./reproduce.sh tables    # 生出力から論文用 CSV 表を再生成
./reproduce.sh map       # 各表をスキーマと来歴つきで一覧
./reproduce.sh paper     # PDF を再ビルド
```

`check` は 4 つのゲートを走らせる。`validate_results.py`（スキーマ、行数、経路間の整合）、
`check_paper_claims.py`（見出し数値をすべて表と照合）、`check_en_number_parity.py`（英語版と日本語版が同じ数値を
含むこと）、`validate_markdown_languages.py` である。いずれも CPU のみで、モデル重みを必要としない。

`tools/check_eval_leakage.py` は、報告した評価集合が Transformer 側の親の指示チューニング混合データにどれだけ
含まれるかを定量化する。項目単位の記録から該当項目を除いた正答率を再計算するので、こちらも GPU を必要としない。

## 実験の再現

評価と学習の入口は、凍結した 2 つの 7B 親モデルを GPU 上に必要とする。

```bash
./reproduce.sh gpu-eval        # 評価を投入（学習は決して投入しない）
./reproduce.sh hf-check        # 3 つのリリース資材の静的検証
./reproduce.sh hf-gpu-check    # 方向固定パッケージの 2GPU スモークテスト
```

親モデルは fp16 で 28.9 GiB 常駐するため、`*_mp.py` の評価器は各ペアを 2 枚のカードに分割する。SLURM のランチャには
チェックアウトパス・パーティション・アカウント・QoS・GRES を一切含めない。これらは gitignore された `.env` から
`experiments/slurm/submit.sh` 経由で注入される。`#SBATCH` 指示行はシェルが `.env` を読む前に解釈されるためである。
[`experiments/slurm/README.md`](experiments/slurm/README.md) を参照。

アダプタの学習は `experiments/adapter7b.py` である。12 時間制限を超えるため一度再開し、optimizer・scheduler・
global step を復元する。報告するチェックポイントは 416,000 反復の走行のうち、保留 RWKV→Pythia 読み出しが最良となる
step 415,000 である。

## Hugging Face 配布物

推論用の 3 パッケージを `tools/export_hf_adapter.py` と `tools/export_hf_chimera.py` が
`release/huggingface/`（gitignore 済み）に生成し、公開の Hub リポジトリで配布している。

| パッケージ | 内容 |
|---|---|
| [`ninaxander-raven7b-tulu69-adapter`](https://huggingface.co/Katagiri-Hoshino-Lab/ninaxander-raven7b-tulu69-adapter) | アダプタと層別統計のみ、単体ローダ付き |
| [`ninaxander-tulu69-to-raven7b-best`](https://huggingface.co/Katagiri-Hoshino-Lab/ninaxander-tulu69-to-raven7b-best) | Pythia→RWKV の `L=4`、fp16 の両親を同梱 |
| [`ninaxander-raven7b-to-tulu69-best`](https://huggingface.co/Katagiri-Hoshino-Lab/ninaxander-raven7b-to-tulu69-best) | RWKV→Pythia の `L=4`、fp16 の両親を同梱 |

これらのリポジトリは公開しており、
[コレクション](https://huggingface.co/collections/Katagiri-Hoshino-Lab/ninaxander-inference-releases-6a9010088ba6e224ea2ea8dd)
にまとめてある。`release/huggingface/REMOTE_RELEASES.csv` に、各リポジトリの検証済みリビジョン・ファイル数・
完全性の状態を記録している。方向固定パッケージの既定値 `L=4` は報告テスト集合上の事後選択であり、独立した選択分割を
持たない。他の境界も、そのパッケージ自身の方向内でなら選択できる。

ライセンスはより制約の強い親（AI2 AI Model License、非商用研究限定）に従い、各パッケージは英語・日本語・簡体字中国語の
モデルカードを備える。

## 主張の射程

論文の主張は意図的に限定されており、本リポジトリがそれを広げることはない。対応関係は、トークナイザ・幅・層数・基盤の
事前学習コーパスを共有する 1 組のモデル対で測ったものである。学習は乱数種 1 つ、指示形式の 1 コーパスによる。切替層は
報告するのと同じテスト集合上で選んでいる。合成モデルは強い方の親を超えず、翻訳は領域外で大きく劣化する。何が示され
何が示されていないかは論文の限界節に述べてある。ここにある数値を再利用する前に、そちらを読まれたい。

## 構成

```
paper/                論文ソースとビルド済み PDF
src/ninaxander/       アダプタ、RWKV 後段ランタイム、結果入出力
experiments/          学習・評価の入口
  slurm/              役割別ランチャ
tools/                表生成、検証、エクスポータ、漏洩検査
artifacts/
  metrics/raw/        実験の直接出力、項目単位の記録
  metrics/tables/     正規化された論文用 CSV
  checkpoints/        学習済みアダプタ（gitignore）
  logs/, runs/        診断と投入記録（gitignore）
docs/                 手法、知見、ロードマップ
release/huggingface/  エクスポータ出力（gitignore）
models/               凍結した親モデルの重み（gitignore）
```

本リポジトリのすべての Markdown 文書は英語・日本語・簡体字中国語で存在し、`validate_markdown_languages.py` が
三者の構造一致を強制する。
