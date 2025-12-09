<#
.SYNOPSIS
    イラスト分類AI(WD14 Tagger) 統合実行ツール (Universal版)
    Windows / Linux 対応。再帰スキャン対応。
    デフォルトで「タグ付きファイル」はスキップ(レジューム)します。

.DESCRIPTION
    Python仮想環境を自動構築し、画像をタグ付けする。
    ファイルに既にタグがある場合はスキップします。
    すべて再解析したい場合は -Force を付けてください。

.PARAMETER Path
    対象ファイルパス。フォルダを指定すると再帰的に処理します。
    デフォルト "*.webp"。

.PARAMETER Thresh
    しきい値。デフォルト 0.35。

.PARAMETER Amd
    [Windows専用] AMD GPU (DirectML) を使用する。

.PARAMETER Force
    [New!] 既にタグが付いているファイルも強制的に再解析・上書きします。

.EXAMPLE
    .\run_tagger.ps1
    現在のフォルダを処理（タグ付きはスキップ）。

.EXAMPLE
    .\run_tagger.ps1 -Path "C:\Images" -Amd -Force
    指定フォルダをAMD GPUで処理し、全ファイルを強制上書きする。
#>

[CmdletBinding()]
param (
    [string]$Path = "*.webp",
    [float]$Thresh = 0.35,
    [switch]$Amd,
    [switch]$Force,
    [switch]$Help
)

# OS判定
$IsWindows = $true
if ($PSVersionTable.PSVersion.Major -ge 6) {
    if ($IsWindows -and $IsLinux) { $IsWindows = $false }
    if ([System.OperatingSystem]::IsLinux()) { $IsWindows = $false }
}

function Show-Help {
    Write-Host "=== AI Tagging Tool (Universal) ===" -ForegroundColor Cyan
    Write-Host "Usage: .\run_tagger.ps1 [-Path] [-Thresh] [-Amd] [-Force]"
    Write-Host "  Default behavior is RESUME (skip tagged files)."
    Write-Host "  -Force  : Force re-process all files."
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

if ($IsWindows -and $Amd) { $PyArgs += "--gpu" }
if ($Force) { $PyArgs += "--force" }

& $VenvPython @PyArgs

Write-Host "[INFO] Done." -ForegroundColor Green
if ($Host.Name -eq "ConsoleHost") {
    Read-Host "Enter to exit"
}