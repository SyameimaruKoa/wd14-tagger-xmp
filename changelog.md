# 実装履歴

## 2026-05-15
- `avif` 画像フォーマットへの対応を追加しました。
  - `embed_tags_universal.py`: `VALID_EXTS` に `.avif` を追加。PillowがAVIFを読み込めるように `pillow_avif` のインポート処理を追記。
  - `run_tagger.ps1`, `run_tagger.sh`: 環境構築時のpipインストール対象に `pillow-avif-plugin` を追加。
- パッケージインストール処理をスマート化（リファクタリング）
  - 共通ライブラリを `requirements.txt` に分離。
  - カスタムインストールロジック（`pip list` との突き合わせなど）を廃止し、pip標準の依存関係解決と `-r requirements.txt` を活用する形に変更。
- GPUの遊休時間を減らすため、バッチ推論と前処理並列化を追加しました。
  - `embed_tags_universal.py`: `--batch-size` と `--io-workers` を追加し、ローカル推論時にまとめて処理できるように変更。
  - `run_tagger.ps1`, `run_tagger.sh`, `README.md`: 新しいオプションを追記。
