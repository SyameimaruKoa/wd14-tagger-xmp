# 実装履歴

## 2026-09-20

- DBV4 fullモデルファミリーへの推論エンジン移行。
  - DBV4用 Model Profile を追加し、lightweight / balanced / high / large / ultra を共通インターフェースで切り替え可能にした。
  - 初期デフォルトを animetimm/convformer_s36.dbv4-full の balanced に設定。
  - model.onnx / selected_tags.csv / preprocess.json / categories.json / thresholds.csv をモデルmetadataとして一体管理。
  - selected_tags.csv の tag-specific best_threshold を標準thresholdとして使用し、--thresh は明示overrideとして分離。
  - General / Character / Rating をmetadataから解釈し、WD14 V3の固定rating位置依存を廃止。
  - preprocess.json に基づく PadToSize / Resize / CenterCrop / Tensor化 / ImageNet Normalize のモデル固有前処理を追加。
  - Server / Client間で model_id / metadata_version / output_size / protocol version を検証。
  - Client側はモデル本体をダウンロードせず、metadataのみを取得してサーバー推論結果を利用する構成に変更。

- R-00 / R-15 / R-17 / R-18体系をDBV4へ接続。
  - R-15_0〜R-15_4、R-17_0〜R-17_4 の5段階体系を維持。
  - R-15 / R-17のseverity計算をDBV4 rating scoreへ接続。
  - DBV4 scoreをWD14 V3 scoreと同一視せず、severity calibrationを独立レイヤーとして保持。

- XMP / 再整理 / reportをDBV4対応。
  - XMPへ dbv4_model marker と4 ratingのraw score / percentageを保存。
  - 同一DBV4 model markerと4 raw scoreが揃っている場合、再推論せずrating再計算・整理できるよう変更。
  - 旧WD14 scoreだけが存在する場合はDBV4推論へフォールバック。
  - HTML reportをR-15/R-17の5段階ラベルへ対応。

- Windows / Linux / Colabの実行経路をDBV4向けに更新。
  - --model-profile をPowerShell / Bash wrapperへ追加。
  - --thresh を明示指定した場合のみDBV4標準thresholdをoverride。
  - ColabサーバーをDBV4サーバーへ更新。

- DBV4 metadata / preprocessing / input layout / output probability変換の単体テストを追加。



## 2026-09-19

- GPU/OpenVINOの内部デバッグログを通常実行時には抑制し、`run_tagger.sh --debug` 指定時のみ有効化。
  - `Inference successful` や `Model is fully supported on OpenVINO` などのOpenVINO内部診断をデバッグ用途として保持。
  - 通常実行では関連するデバッグ環境変数を解除し、NVIDIA / Intel / AMD のGPUプロバイダ診断を不用意に消さない方針を維持。

- R-15 / R-17 5段階分類の実行時エラーを修正。
  - `rating_sublevel_thresholds_5way` は5段階を4つの境界で分割するため、必要なthreshold数を4へ修正。
  - 10段階時代のR-15 / R-17説明を5段階（`_0`〜`_4`）へ整理。
- Linux + Intel OpenVINO GPU実行を明示化。
  - Intel iGPU向けの既定OpenVINOデバイスを `GPU.0` に設定。
  - OpenVINOへの要求デバイスを起動時に表示。
  - `sess.get_provider_options()` の値を実効デバイス判定に使用しないよう修正。

## 2026-09-17

- Sensitive 分割デフォルトの変更およびタグ書き込み・再推論判定の改善
  - `embed_tags_universal.py` / `README.md`: `sensitive_split_mode` のデフォルト値を 6（6分割モード）に変更。
  - `embed_tags_universal.py`: 際どさ・レーティングに関する全4項目（`general`, `sensitive`, `questionable`, `explicit`）の RAWスコア（`<rating>_score:0.XXXX`）および割合タグ（`<rating>:XX.X%`）をメタデータに一括記録するよう拡張。
  - `embed_tags_universal.py`: 4分類すべてのRAWスコアがメタデータに揃っている場合、AI推論を完全スキップしながらモデル閾値・分割モード変更に応じた全レーティングの瞬時再判定を行うロジックを実装。
  - `embed_tags_universal.py`: RAWスコア未存在ファイルの推論判定を改善。旧タグによるスキップ・フォールバックを完全廃止し、RAWスコアが無いファイルは全件確実に再推論を実行するよう修正（ファイルごとのスキップ/推論の不整合を解消）。
  - `embed_tags_universal.py`: 推論実行時のメタデータタグ書き込みロジックを修正。既存タグの有無にかかわらず、推論によって生成された最新タグ（新レーティング、RAWスコア、割合タグ）を確実に書き込むように変更（XnView 等での RAW スコア視認性を確保し、再整理の推論スキップを保証）。
  - `embed_tags_universal.py`: 既存タグ更新時に、ユーザー定義の一般タグを破壊しないよう、旧レーティング・旧スコアタグのみを除外して既存タグをマージ保持する安全処理を追加。

## 2026-09-15

- Sensitive 2/4/6分割モードおよびRAWスコア記録機能の実装
  - `embed_tags_universal.py`: `determine_sensitive_level()` 関数を追加し、`sensitive_split_mode` 設定 (2/4/6) に応じて Sensitive 強度を `sensitive_mild`/`sensitive_high` (2分割) または `sensitive_lvl1`〜`lvl4` (4分割) または `sensitive_lvl1`〜`lvl6` (6分割) へ振り分けるロジックを実装。
  - `embed_tags_universal.py`: `format_score_tags()` 関数を追加。`sensitive_score:0.XXXX` 形式の RAW スコアタグと `sensitive:XX.X%` 形式の割合タグを XMP に記録する。
  - `embed_tags_universal.py`: `--organize` 時の再整理ロジックを改修。`sensitive_score:` タグが存在する場合は RAW スコアから再計算してスキップ（AI再推論なし）。旧形式タグ (`sensitive_lvlX`, `sensitive_mild`, `sensitive_high`) のみの場合は AI 推論を再実行（4分類への縮退を防止）。
  - `embed_tags_universal.py`: CLI 引数 `--sensitive-split-mode <2|4|6>`、`--record-ratio`、`--no-record-ratio` を追加。
  - `embed_tags_universal.py`: `RATING_TAGS` に `sensitive_lvl1`〜`lvl6` を追加。`DEFAULT_CONFIG` に `sensitive_split_mode`、`sensitive_split_thresholds_4way`、`sensitive_split_thresholds_6way`、`record_rating_percentages`、`record_raw_score` を追加。
  - `make_report.py`: Sensitive 分割レベル (lvl1〜lvl6, mild, high) に対応したバッジスタイルを追加。カードにスコア情報 (`sensitive:XX.X%`) を表示。
  - `run_tagger.ps1`: `SensitiveSplitMode`、`RecordRatio`、`NoRecordRatio` パラメータを追加し、PyArgs へのパススルーを実装。
  - `run_tagger.sh`: `--sensitive-split-mode`、`--record-ratio`、`--no-record-ratio` オプションを追加。
  - `README.md`: 新設定項目 (`sensitive_split_mode` 等) と `folder_names` の `sensitive_lvl1`〜`lvl6` を追記。

- Google Colab サーバーでの Google ドライブ Tailscale 接続情報永続化 & シークレット不要再接続モードの実装
  - `run_colab_server.ipynb`: 「すべてのセルを実行 (Run All)」による全自動起動に対応。
  - 最上部に設定用フォームセルを追加し、Google ドライブ保存、ホスト名、リセット設定等を一括管理可能に変更。
  - ノートブック内のすべてのコードセルをフォーム形式で折りたたみ、画面をすっきり整理。
  - 最後の完全ログアウトセルを見直し、Google ドライブ保存モード有効時はログアウトを安全に自動スキップするガード処理を追加（次回のシークレット不要再接続を保護）。
  - `embed_tags_universal.py` / `README.md`: ドキュメントおよび接続先リストの更新。

## 2026-08-15

- Nintendo Switch (Tegra X1 / Switchroot L4T) および ARM64 環境への対応
  - `run_tagger.sh`: Tegra SoC (`/dev/nvhost-gpu`, `/etc/nv_tegra_release`, `/usr/lib/aarch64-linux-gnu/tegra` 等) の自動検出を追加し、Nintendo Switch や Jetson 等の統合 GPU 環境で NVIDIA GPU を正しく認識するように改善。
  - `run_tagger.sh`: Tegra L4T ドライバからの CUDA バージョン判定ロジックを追加。
  - `run_tagger.sh`: ARM64 (aarch64) 環境下での依存ライブラリインストール処理を最適化し、x86_64 専用パッケージによるエラーを防止。
  - `run_tagger.sh`: Tegra ドライバ格納パス (`/usr/lib/aarch64-linux-gnu/tegra`) を `LD_LIBRARY_PATH` に自動追加。
  - `embed_tags_universal.py`: ARM64 / Tegra 等で利用可能な GPU プロバイダが存在しない場合に、分かりやすい案内を出力して CPU (ARM NEON) モードへフォールバックする機能を追加。
  - `README.md`: Switchroot Ubuntu 24.04 の仕様・制限事項（CUDA 10.0 ランタイムあり、CUDA コンパイラ・cuDNN なし）と、メイン PC と連携した高速 GPU クライアント/サーバー利用方法をドキュメント化。

## 2026-08-01

- NVIDIA GPU (CUDA / TensorRT) 高速化機能の追加および動作の最適化
  - `run_tagger.sh`: システムの CUDA バージョンに合わせた `nvidia-*` ランタイムパッケージおよび ONNX Runtime 互換の `tensorrt<11` (10.x系) パッケージの自動検出・自動インストールロジックを追加。
  - `run_tagger.sh`: `LD_LIBRARY_PATH` の動的生成ロジックを改善し、`site-packages` 内の NVIDIA / TensorRT ライブラリディレクトリを正確に追加して共有ライブラリのロードエラーを解決。
  - `embed_tags_universal.py`: コンパイルを伴う ExecutionProvider (`TensorrtExecutionProvider` 等) 使用時の注意事項出力機能を追加。
  - `embed_tags_universal.py`: 実効バッチサイズに対応したダミーテンソルによる事前ウォームアップ推論（エンジン自動構築）を追加し、ウォームアップ所要時間をログ出力。
  - `embed_tags_universal.py`: 1枚目（初回バッチ）の処理時間に関する外れ値判定ロジックを組み込み、コンパイルによる遅延を選択的に除外した実効推論速度の報告、および1枚目処理時間・全体の詳細サマリー出力を追加。

## 2026-05-15 (Update 2)

- デフォルトモデルを最新のV3系 (`SmilingWolf/wd-swinv2-tagger-v3`) に変更しました。
  - `embed_tags_universal.py` の `DEFAULT_CONFIG` を更新。
- `README.md` に現在利用可能な SmilingWolf 氏のモデル一覧（V3、Large V3、V2）とそれぞれの特徴を追加しました。
- READMEの推奨モデルの記述を、最新モデルに合わせて更新しました。

## 2026-05-15

- `avif` 画像フォーマットへの対応を追加しました。
  - `embed_tags_universal.py`: `VALID_EXTS` に `.avif` を追加。PillowがAVIFを読み込めるように `pillow_avif` のインポート処理を追記。
  - `run_tagger.ps1`, `run_tagger.sh`: 環境構築時のpipインストール対象に `pillow-avif-plugin` を追加。
- パッケージインストール処理をスマート化（リファクタリング）
  - 共通ライブラリを `requirements.txt` に分離。
  - カスタムインストールロジック（`pip list` との突き合わせなど）を廃止し、pip標準の依存関係解決と `-r requirements.txt` を活用する形に変更。
- GPUの遊休時間を減らすため、バッチ推論と前処理並列化を追加しました。
  - `embed_tags_universal.py`: `--batch-size` と `--io-workers` を追加し、ローカル推論時にまとめて処理できるように変更。デフォルトでバッチサイズ 4、IO ワーカー自動計算で並列化を有効化。モデルがバッチ非対応の場合は自動で 1 にフォールバック。
  - `run_tagger.ps1`, `run_tagger.sh`, `README.md`: 新しいオプションを追記し、デフォルト値とフォールバックの説明を追加。
- モデル/タグの切り替え機能を追加しました。
  - `embed_tags_universal.py`: `--model-repo`, `--model-file`, `--tags-file` を追加し、config.json と CLI から切り替え可能に変更。
  - `run_tagger.ps1`, `run_tagger.sh`, `README.md`, `config.json`: 新しい項目とヘルプを追記。
- README に推奨モデルを追記しました。
- README にバッチ推論対応モデル（swinv2）の推奨を追記しました。
- デフォルトモデルを `SmilingWolf/wd-v1-4-swinv2-tagger-v2` に変更しました。
