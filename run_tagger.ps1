<#
.SYNOPSIS
    イラスト分類AI(WD14 Tagger)の環境構築および実行を行うラッパースクリプトじゃ。
    
.DESCRIPTION
    Pythonの仮想環境(venv)を自動で作成・有効化し、必要なライブラリをインストールした後、
    タグ付けスクリプト(embed_tags.py)を実行する。
    指定したフォルダ内のWebPファイルを検索して処理するぞ。

.PARAMETER Path
    処理対象のパス（ファイルまたはフォルダ）。ワイルドカード使用可。
    デフォルトは現在のフォルダの "*.webp"。

.PARAMETER Thresh
    タグ付けの信頼度しきい値（0.0〜1.0）。デフォルトは 0.35。

.EXAMPLE
    .\run_tagger.ps1
    現在のフォルダにある全ての .webp ファイルを処理する。

.EXAMPLE
    .\run_tagger.ps1 -Path "C:\Images\*.webp"
    指定したパスの画像を処理する。

.EXAMPLE
    .\run_tagger.ps1 -h
    ヘルプを表示する。
#>

param (
    [string]$Path = "*.webp",
    [float]$Thresh = 0.35,
    [switch]$Help
)

# ヘルプ表示関数
function Show-Help {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  AIタグ付け 自動実行ツール (WD14 Tagger)" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "使い方:"
    Write-Host "  .\run_tagger.ps1 [オプション]"
    Write-Host ""
    Write-Host "オプション:"
    Write-Host "  -Path <文字列>   処理対象のファイルパス（ワイルドカード可）。"
    Write-Host "                   デフォルト: *.webp"
    Write-Host "  -Thresh <数値>   判定のしきい値 (0.0 - 1.0)。"
    Write-Host "                   デフォルト: 0.35"
    Write-Host "  -h, --help       このヘルプを表示する。"
    Write-Host ""
    Write-Host "例:"
    Write-Host "  .\run_tagger.ps1"
    Write-Host "  .\run_tagger.ps1 -Path 'C:\MyImages\*.webp'"
    Write-Host ""
}

# 引数チェック (-h / --help)
if ($Help -or ($args -contains '-h') -or ($args -contains '--help')) {
    Show-Help
    exit
}

# --- 設定 ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "embed_tags.py"
$VenvDir = Join-Path $ScriptDir "venv"
$Requirements = @("onnxruntime", "pillow", "huggingface_hub", "numpy", "tqdm")

# --- メイン処理 ---
Write-Host "[INFO] 開始するぞ..." -ForegroundColor Green

# 1. Pythonスクリプトがあるか確認
if (-not (Test-Path $PythonScript)) {
    Write-Host "[ERROR] '$PythonScript' が見つからぬ！同じフォルダに置くのじゃ。" -ForegroundColor Red
    exit 1
}

# 2. 仮想環境(venv)の確認と作成
if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO] 仮想環境(venv)を作成中..." -ForegroundColor Yellow
    try {
        python -m venv $VenvDir
    }
    catch {
        Write-Host "[ERROR] venvの作成に失敗した。Pythonは入っておるか？" -ForegroundColor Red
        exit 1
    }
}

# 3. 仮想環境の有効化とライブラリインストール
Write-Host "[INFO] 環境を確認中..." -ForegroundColor Cyan
# Activateスクリプトのパス解決 (Windows)
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"

if (-not (Test-Path $ActivateScript)) {
    Write-Host "[ERROR] 仮想環境が壊れているようじゃ。venvフォルダを削除してやり直せ。" -ForegroundColor Red
    exit 1
}

# 現在のプロセス内でActivateするのではなく、pythonのフルパスを使って実行するスタイルをとる
# (PowerShellのExecutionPolicyに阻まれるのを防ぐため)
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

# ライブラリのインストール確認 (tqdmが入っているかで簡易判定)
$PipCheck = & $VenvPip list
if ($PipCheck -notmatch "tqdm") {
    Write-Host "[INFO] 必要なライブラリをインストールしておる。初回は時間がかかるぞ..." -ForegroundColor Yellow
    & $VenvPip install $Requirements | Out-Null
}

# 4. ExifToolの存在確認 (簡易)
if (-not (Get-Command "exiftool" -ErrorAction SilentlyContinue)) {
    $LocalExifTool = Join-Path $ScriptDir "exiftool.exe"
    if (-not (Test-Path $LocalExifTool)) {
        Write-Host "[WARN] 'exiftool' が見つからぬ！ PATHに通すか、このフォルダに置くのじゃ。" -ForegroundColor Magenta
        Write-Host "       (動作しない可能性があるぞ)"
    }
}

# 5. Pythonスクリプトの実行
Write-Host "[INFO] AIタグ付けを実行開始じゃ！" -ForegroundColor Green
Write-Host "       対象: $Path"
Write-Host "       しきい値: $Thresh"
Write-Host ""

# 実行
& $VenvPython $PythonScript $Path --thresh $Thresh

Write-Host ""
Write-Host "[INFO] 全て完了じゃ。悪かったな、待たせて。" -ForegroundColor Green

# 終了時にポーズ（ダブルクリック実行などでウィンドウがすぐ消えないように）
if ($Host.Name -eq "ConsoleHost" -and -not $PSCmdlet.MyInvocation.BoundParameters["Help"]) {
    Read-Host "Enterキーを押して終了せよ"
}