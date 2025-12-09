<#
.SYNOPSIS
    イラスト分類AI(WD14 Tagger) 統合実行ツール (Universal版)
    Windows / Linux 対応。レジューム機能付き。

.DESCRIPTION
    Python仮想環境を自動構築し、画像をタグ付けする。
    -Resume をつけると、前回処理したファイルをスキップする。
    OSを自動判定し、WindowsならDirectML(AMD)、LinuxならCPU環境を構築する。

.PARAMETER Path
    対象ファイルパス。デフォルト "*.webp"。

.PARAMETER Thresh
    しきい値。デフォルト 0.35。

.PARAMETER Amd
    [Windows専用] AMD GPU (DirectML) を使用する。Linuxでは無視される。

.PARAMETER Resume
    [New!] 履歴ファイル(processed_history.txt)を参照し、処理済みのファイルをスキップする。

.EXAMPLE
    .\run_tagger.ps1 -Resume
    前回の続きから実行する。

.EXAMPLE
    .\run_tagger.ps1 -Amd -Resume
    WindowsでAMDを使って、続きから実行する。
#>

[CmdletBinding()]
param (
    [string]$Path = "*.webp",
    [float]$Thresh = 0.35,
    [switch]$Amd,
    [switch]$Resume,
    [switch]$Help
)

# OS判定
$IsWindows = $true
if ($PSVersionTable.PSVersion.Major -ge 6) {
    if ($IsWindows -and $IsLinux) { $IsWindows = $false } # 変数スコープ等の都合
    if ([System.OperatingSystem]::IsLinux()) { $IsWindows = $false }
}

function Show-Help {
    Write-Host "=== AI Tagging Tool (Universal) ===" -ForegroundColor Cyan
    Write-Host "Usage: .\run_tagger.ps1 [-Path] [-Thresh] [-Amd] [-Resume]"
    Write-Host "  -Resume : Skip already processed files."
    Write-Host "  -Amd    : Use DirectML (Windows only)."
    Write-Host ""
}

if ($Help) { Show-Help; exit }

# --- 設定 ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "embed_tags_universal.py"

# OSによる仮想環境名の切り替え
if ($IsWindows -and $Amd) {
    $VenvDir = Join-Path $ScriptDir "venv_amd"
    $Requirements = @("onnxruntime-directml", "pillow", "huggingface_hub", "numpy", "tqdm")
} else {
    $VenvDir = Join-Path $ScriptDir "venv_std"
    $Requirements = @("onnxruntime", "pillow", "huggingface_hub", "numpy", "tqdm")
    if ($Amd) { Write-Host "[WARN] Linux detected. AMD(DirectML) is disabled. Using CPU mode." -ForegroundColor Yellow }
}

# --- 実行開始 ---
Write-Host "[INFO] OS: $(if($IsWindows){'Windows'}else{'Linux'}) / Mode: $(if($Amd -and $IsWindows){'AMD'}else{'CPU'})" -ForegroundColor Green

# 1. Pythonスクリプト確認
if (-not (Test-Path $PythonScript)) {
    Write-Host "[ERROR] '$PythonScript' not found." -ForegroundColor Red
    exit 1
}

# 2. venv作成
if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO] Creating venv at $VenvDir ..." -ForegroundColor Yellow
    if ($IsWindows) { python -m venv $VenvDir } else { python3 -m venv $VenvDir }
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Failed to create venv." -ForegroundColor Red; exit 1 }
}

# 3. パス解決
if ($IsWindows) {
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $VenvPip = Join-Path $VenvDir "Scripts\pip.exe"
} else {
    $VenvPython = Join-Path $VenvDir "bin/python"
    $VenvPip = Join-Path $VenvDir "bin/pip"
}

# 4. ライブラリ確認
$PipCheck = & $VenvPip list
$NeedInstall = $false
if ($IsWindows -and $Amd) {
    if ($PipCheck -notmatch "onnxruntime-directml") { $NeedInstall = $true }
} else {
    if ($PipCheck -notmatch "onnxruntime") { $NeedInstall = $true }
}
if ($PipCheck -notmatch "tqdm") { $NeedInstall = $true }

if ($NeedInstall) {
    Write-Host "[INFO] Installing requirements..." -ForegroundColor Yellow
    & $VenvPip install $Requirements | Out-Null
}

# 5. ExifTool確認
if (-not (Get-Command "exiftool" -ErrorAction SilentlyContinue)) {
    if ($IsWindows) {
        $LocalExifTool = Join-Path $ScriptDir "exiftool.exe"
        if (-not (Test-Path $LocalExifTool)) {
            Write-Host "[WARN] 'exiftool.exe' not found in folder or PATH!" -ForegroundColor Magenta
        }
    } else {
        Write-Host "[WARN] 'exiftool' command not found! Run: sudo apt install libimage-exiftool-perl" -ForegroundColor Magenta
    }
}

# 6. 実行
Write-Host "[INFO] Running Tagger..." -ForegroundColor Green
$PyArgs = @($PythonScript, $Path, "--thresh", $Thresh)

# ★ここを修正したぞ！ (--amd ではなく --gpu を渡す)
if ($IsWindows -and $Amd) { $PyArgs += "--gpu" }

if ($Resume) { $PyArgs += "--resume" }

& $VenvPython @PyArgs

Write-Host "[INFO] Done." -ForegroundColor Green
