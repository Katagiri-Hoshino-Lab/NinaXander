# 成果物

[English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

生成された研究成果物とローカルの研究成果物は、役割ごとに分けています。

- `checkpoints/`：アダプターのチェックポイント。ローカル専用で、directory全体がGit管理外です。
- `metrics/raw/`：実験が直接出力するデータ。JSONには入れ子構造と項目単位の記録を保持し、
  併設するCSVには矩形の測定値を格納します。
- `metrics/tables/`：`tools/build_result_tables.py`が決定論的に生成する、論文用CSVファイル。
- `logs/slurm/`：ローカルのスケジューラ標準出力と過去の診断情報。
- `logs/evaluation/`：現行の評価プログラムごとのローカル診断ログ。
- `runs/`：`reproduce.sh gpu-eval`が生成する構造化された投入記録。
- `figures/`：必要に応じて生成される図の素材。

ログはGitの対象外であり、正式な数値データセットではありません。公開実験記録には
`metrics/raw/`を、報告値の関係には`metrics/tables/`を使用してください。特に
`metrics/raw/training_curve.csv`により、サイト固有の学習ログへ依存する必要がなくなっています。

`metrics/tables/paper_result_coverage.csv`は、機械可読な対称性監査です。
公開するすべてのモデル族横断結果には、対応するAB行とBA行が必要です。標準QA行列にはさらに、
各タスク・各切替層のAAとBBも含まれます。Pythiaの早期終了は`B_early_exit`に分類し、
BBアダプター対照として誤集計しません。AA/BB対照が方法論上適用できない測定系列では、
対応するAB/BA実験を未完了と誤表示せず、`four_path_complete`を空欄にします。
