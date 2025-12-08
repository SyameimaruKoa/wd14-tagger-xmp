<#
.SYNOPSIS
    イラスト分類AI(WD14 Tagger)の統合実行ツールじゃ。CPUとAMD GPUの両方に対応しておる。
    
.DESCRIPTION
    Pythonの仮想環境を自動で構築し、タグ付けスクリプトを実行する。
    CPU用とAMD用で仮想環境を分離し、ライブラリの競合を防いでおる。

.PARAMETER Path
    処理対象のパス（ファイルまたはフォルダ）。ワイルドカード使用可。
    デフォルトは "*.webp"。

.PARAMETER Thresh
    タグ付けの信頼度しきい値（0.0〜1.0）。デフォルトは 0.35。

.PARAMETER Amd
    [スイッチ] これを指定すると AMD GPU (DirectML) モードで動作する。
    専用のスクリプト(embed_tags_amd.py)と専用の環境(venv_amd)が使用される。

.EXAMPLE
    .\run_tagger.ps1
    CPUモードで実行する（デフォルト）。

.EXAMPLE
    .\run_tagger.ps1 -Amd
    AMD GPUモードで実行する。

.EXAMPLE
    .\run_tagger.ps1 -Path "C:\Data\*.webp" -Thresh 0.5 -Amd
    画像パスとしきい値を指定してAMDモードで実行。
#>

[CmdletBinding()]
param (
    [string]$Path = "*.webp",
    [float]$Thresh = 0.35,
    [switch]$Amd,
    [switch]$Help
)

# ヘルプ表示関数
function Show-Help {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  AIタグ付け 統合実行ツール (CPU / AMD)" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "概要:"
    Write-Host "  WD14 Taggerを使って画像にタグ付けを行う。CPUとAMD GPUを切り替え可能じゃ。"
    Write-Host ""
    Write-Host "使い方:"
    Write-Host "  .\run_tagger.ps1 [オプション]"
    Write-Host ""
    Write-Host "オプション:"
    Write-Host "  -Path <文字列>   処理対象のファイルパス（ワイルドカード可）。"
    Write-Host "                   デフォルト: *.webp"
    Write-Host "  -Thresh <数値>   判定のしきい値 (0.0 - 1.0)。"
    Write-Host "                   デフォルト: 0.35"
    Write-Host "  -Amd             AMD GPU (DirectML) を使用するスイッチ。"
    Write-Host "                   ※ これを付けると 'venv_amd' 環境と 'embed_tags_amd.py' を使う。"
    Write-Host "  -h, --help       このヘルプを表示する。"
    Write-Host ""
}

# 引数チェック (-h / --help)
# CmdletBindingを使った場合、標準の -Verbose 等も使えるようになるが、
# 手動の -Help チェックも残しておく
if ($Help -or ($args -contains '-h') -or ($args -contains '--help')) {
    Show-Help
    exit
}

# --- 設定の切り替えロジック ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($Amd) {
    # AMDモードの設定
    $ModeName = "AMD GPU (DirectML)"
    $PythonScript = Join-Path $ScriptDir "embed_tags_amd.py"
    $VenvDir = Join-Path $ScriptDir "venv_amd"
    $Requirements = @("onnxruntime-directml", "pillow", "huggingface_hub", "numpy", "tqdm")
} else {
    # CPUモードの設定
    $ModeName = "CPU (Standard)"
    $PythonScript = Join-Path $ScriptDir "embed_tags.py"
    $VenvDir = Join-Path $ScriptDir "venv"
    $Requirements = @("onnxruntime", "pillow", "huggingface_hub", "numpy", "tqdm")
}

# --- メイン処理 ---
Write-Host "[INFO] モード: $ModeName で開始するぞ..." -ForegroundColor Green

# 1. スクリプトの存在確認
if (-not (Test-Path $PythonScript)) {
    Write-Host "[ERROR] スクリプト '$PythonScript' が見つからぬ！" -ForegroundColor Red
    if ($Amd) {
        Write-Host "        AMDモードゆえ 'embed_tags_amd.py' が必要じゃ。"
    } else {
        Write-Host "        通常モードゆえ 'embed_tags.py' が必要じゃ。"
    }
    exit 1
}

# 2. 仮想環境(venv)の確認と作成
if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO] 仮想環境フォルダ($($VenvDir | Split-Path -Leaf))を作成中..." -ForegroundColor Yellow
    try {
        python -m venv $VenvDir
    } catch {
        Write-Host "[ERROR] venvの作成に失敗した。Pythonのパスは通っておるか？" -ForegroundColor Red
        exit 1
    }
}

# 3. 仮想環境の有効化とライブラリインストール
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Host "[ERROR] 仮想環境が壊れているようじゃ。フォルダ($VenvDir)を削除してやり直せ。" -ForegroundColor Red
    exit 1
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

# ライブラリ簡易チェック
$PipCheck = & $VenvPip list
$NeedInstall = $false

if ($Amd) {
    if ($PipCheck -notmatch "onnxruntime-directml") { $NeedInstall = $true }
} else {
    if ($PipCheck -notmatch "onnxruntime") { $NeedInstall = $true }
}
if ($PipCheck -notmatch "tqdm") { $NeedInstall = $true }

if ($NeedInstall) {
    Write-Host "[INFO] 必要なライブラリをインストール中... ($ModeName 用)" -ForegroundColor Yellow
    & $VenvPip install $Requirements | Out-Null
}

# 4. ExifToolの存在確認
if (-not (Get-Command "exiftool" -ErrorAction SilentlyContinue)) {
    $LocalExifTool = Join-Path $ScriptDir "exiftool.exe"
    if (-not (Test-Path $LocalExifTool)) {
        Write-Host "[WARN] 'exiftool' が見つからぬ！ PATHに通すか、このフォルダに置くのじゃ。" -ForegroundColor Magenta
    }
}

# 5. Pythonスクリプトの実行
Write-Host "[INFO] 実行開始じゃ！" -ForegroundColor Green
Write-Host "       対象: $Path"
Write-Host "       環境: $($VenvDir | Split-Path -Leaf)"
Write-Host ""

# 実行
& $VenvPython $PythonScript $Path --thresh $Thresh

Write-Host ""
Write-Host "[INFO] 完了じゃ。" -ForegroundColor Green

# 終了待機 (ヘルプ表示時は上でexitしているので、ここに到達するのは実行後のみ)
if ($Host.Name -eq "ConsoleHost") {
    Read-Host "Enterキーを押して終了せよ"
}