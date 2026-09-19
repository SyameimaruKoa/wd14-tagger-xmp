# DBV4 Tagger Universal

AnimeTimm / DeepGHS系の DBV4 fullモデルをONNX Runtimeで実行し、画像タグ付け・XMP保存・フォルダ整理・HTMLレポート・推論サーバー/クライアントを行うツールじゃ。

Windows (PowerShell) と Linux (Bash) に対応しており、DBV4のモデル・前処理・ラベルカテゴリ・タグ単位thresholdをモデルプロファイルから動的に扱う構成になっておるぞ。

## 特徴

- DBV4 fullの大規模マルチラベル出力をmetadataベースで復号
- selected_tags.csv の best_threshold をタグ単位で適用
- preprocess.json に従ったモデル固有前処理
- General / Character / Rating をmetadataから分離
- R-00 / R-15_0〜R-15_4 / R-17_0〜R-17_4 / R-18 を維持
- DBV4 rating scoreをXMPへ保存し、再整理時の再推論を省略
- 軽量〜超大型モデルを同じインターフェースで切り替え
- CPU / NVIDIA / Intel OpenVINO / AMD系 / DirectML の既存実行経路を維持
- Server / Client間でmodel ID・metadata version・output sizeを検証
- 既存のユーザータグを保持
- HTMLレポートとフォルダ整理を維持

## DBV4モデルプロファイル

| Profile | Repository | 目安 |
| --- | --- | --- |
| lightweight | animetimm/mobilenetv4_conv_small.dbv4-full | 軽量 |
| balanced | animetimm/convformer_s36.dbv4-full | 初期デフォルト |
| high | animetimm/swinv2_base_window8_256.dbv4-full | 高性能 |
| large | animetimm/eva02_large_patch14_448.dbv4-full | 大型 |
| ultra | animetimm/convnextv2_huge.dbv4-full | 超大型 |

balanced はDBV4移行時点の初期デフォルトであり、実環境ベンチマークによる最終選定とは分離しておる。

各プロファイルはモデル・タグCSV・前処理定義・カテゴリ定義・threshold定義を一つの論理単位として扱う。将来DBV4モデルを追加する場合も、このプロファイルへ定義を追加すれば共通推論経路を変更せず切り替えられる設計じゃ。

## DBV4出力

DBV4ではモデル出力を固定の先頭4要素として扱わず、selected_tags.csv とmetadataを基準に解釈する。

~~~text
raw output
  ↓
label metadata
  ↓
General / Character / Rating
  ↓
tag-specific best_threshold
  ↓
最終タグ
~~~

Ratingもmetadataから general / sensitive / questionable / explicit の4ラベルを取得する。DBV4のscore分布をWD14 V3のscore分布と同一視しないため、R-15 / R-17用のseverity計算は別レイヤーとして扱うぞ。

### Tag threshold

通常時は各タグの best_threshold を使う。

~~~text
prediction[tag] >= tag.best_threshold
~~~

--thresh を明示した場合のみ、全タグへその値をoverrideとして適用するのじゃ。

## R-00 / R-15 / R-17 / R-18

R-15 / R-17はDBV4公式ratingではなく、このプロジェクト独自のseverity尺度じゃ。

~~~text
R-00
R-15_0
R-15_1
R-15_2
R-15_3
R-15_4
R-17_0
R-17_1
R-17_2
R-17_3
R-17_4
R-18
~~~

Sensitive帯とQuestionable帯は同じ連続severity軸上に置き、R-15_4 → R-17_0 と連続する。

R-00はGeneral thresholdの安全弁、R-18はExplicit側の基本判定として維持し、Sensitive / Questionable の内部を5段階へ分割する。

--sensitive-split-mode は旧CLI互換のため残しておるが、DBV4では5段階固定のため非推奨じゃ。

## XMP

既存の XMP:Subject 運用とExifToolを維持する。

DBV4推論時には、4 rating scoreに加えて以下のmarkerを保存する。

~~~text
dbv4_model:<repository-id>
general_score:0.XXXX
sensitive_score:0.XXXX
questionable_score:0.XXXX
explicit_score:0.XXXX
~~~

この4つのscoreとDBV4 model markerが揃っている場合、--organize では再推論せずratingを再計算できる。旧WD14 scoreだけが残っている場合はDBV4推論を実行するぞ。

## Windows / PowerShell

初回セットアップ：

~~~powershell
.\run_tagger.ps1
~~~

通常実行：

~~~powershell
.\run_tagger.ps1 -Path "C:\Images" -Gpu
~~~

モデルプロファイル指定：

~~~powershell
.\run_tagger.ps1 -Path "C:\Images" -Gpu -ModelProfile high
~~~

フォルダ整理のみ：

~~~powershell
.\run_tagger.ps1 -Path "C:\Images" -Organize
~~~

タグ付け＋整理：

~~~powershell
.\run_tagger.ps1 -Path "C:\Images" -Tag -Organize
~~~

推論サーバー：

~~~powershell
.\run_tagger.ps1 -Server -Gpu -ModelProfile balanced
~~~

クライアント：

~~~powershell
.\run_tagger.ps1 -Client -HostIP "192.168.1.10" -Path "C:\Images" -Organize
~~~

## Linux / Bash

初回セットアップ：

~~~bash
./run_tagger.sh
~~~

通常実行：

~~~bash
./run_tagger.sh -g -p /path/to/images
~~~

モデルプロファイル指定：

~~~bash
./run_tagger.sh -g --model-profile high -p /path/to/images
~~~

フォルダ整理：

~~~bash
./run_tagger.sh --organize -p /path/to/images
~~~

推論サーバー：

~~~bash
./run_tagger.sh --server -g --model-profile balanced
~~~

クライアント：

~~~bash
./run_tagger.sh --client --host 192.168.1.10 -p /path/to/images --organize
~~~

Intel OpenVINOを使用する場合は既存の --gpu 経路を維持し、openvino_gpu_device に使用デバイスを指定できるぞ。

## Google Colab

run_colab_server.ipynb はDBV4サーバーを起動する構成へ更新されておる。

<a href="https://colab.research.google.com/github/SyameimaruKoa/wd14-tagger-xmp/blob/main/run_colab_server.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

初期状態では balanced プロファイルを使用する。ローカル側は通常のClientモードで接続できるぞ。

## 設定ファイル

初回実行時に config.json が自動生成される。既存の設定へ不足項目を追記する方式なので、旧設定ファイルが残っていても破壊しない。

主要項目：

| Key | 初期値 | 説明 |
| --- | --- | --- |
| model_profile | "balanced" | 使用するDBV4プロファイル |
| server_hosts | ["localhost", "google-colab", "100.xxx.xxx.xxx"] | Client接続先 |
| server_port | 5000 | Server/Clientポート |
| client_timeout | 15 | Client timeout秒 |
| openvino_gpu_device | "GPU.0" | Intel OpenVINOデバイス |
| general_threshold | 0.40 | R-00安全弁 |
| rating_sublevel_thresholds_5way | [0.20,0.40,0.60,0.80] | R-15/R-17の5段階境界 |
| record_rating_percentages | true | rating割合をXMPへ記録 |
| record_raw_score | true | raw rating scoreをXMPへ記録 |
| folder_names | 下記 | 整理先フォルダ名 |

プロファイル自体も model_profiles へ追加・上書きできるぞ。

## Server / Clientプロトコル

Serverは次の情報をJSONで返す。

~~~json
{
    "protocol": 1,
    "model_id": "animetimm/convformer_s36.dbv4-full",
    "profile": "balanced",
    "metadata_version": "xxxxxxxxxxxxxxxx",
    "output_size": 12476,
    "rating_labels": [
        "general",
        "sensitive",
        "questionable",
        "explicit"
    ],
    "probabilities": []
}
~~~

Clientは model_id、metadata_version、output_size を検証してから結果を利用する。異なるDBV4モデルやmetadataを接続した場合はエラーとして停止するのじゃ。

## テスト

DBV4 metadata / preprocessing / input layout / output probability変換の単体テストを実行できる。

~~~bash
python -m unittest discover -s tests -p "test_*.py"
~~~

実モデルのCPU/GPU速度比較やR-15/R-17キャリブレーションは、実行環境ごとのベンチマーク工程として別途評価する。

## ライセンス

DBV4モデル自体のライセンスはモデルごとに異なるため、使用するprofileのモデルカードを確認すること。

モデルファイルはリポジトリへ同梱せず、Hugging Faceから実行時に取得する方式を基本とする。

## 対応範囲

- Windows / Linux
- CPU
- NVIDIA CUDA / TensorRT
- Intel OpenVINO
- AMD ROCm / MIGraphX
- Server / Client
- Batch inference
- Nintendo Switch / ARM64向け既存クライアント経路
- XMP / ExifTool
- HTML report
