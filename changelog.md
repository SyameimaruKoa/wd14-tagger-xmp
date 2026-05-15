# 実装履歴

## 2026-05-15
- `avif` 画像フォーマットへの対応を追加しました。
  - `embed_tags_universal.py`: `VALID_EXTS` に `.avif` を追加。PillowがAVIFを読み込めるように `pillow_avif` のインポート処理を追記。
  - `run_tagger.ps1`: 環境構築時のpipインストール対象に `pillow-avif-plugin` を追加（Windows環境用）。
  - `run_tagger.sh`: 環境構築時のpipインストール対象に `pillow-avif-plugin` を追加（Linux環境用）。
