<#
.SYNOPSIS
    WD14 Tagger Universal Wrapper (Standalone / Server / Client)

.DESCRIPTION
    - Standalone (Default): Process files locally.
    - Server: Start GPU inference server.
    - Client: Send files to server for processing.

.PARAMETER Path
    Target files (Standalone/Client).

.PARAMETER Server
    Start as Server mode.

.PARAMETER Client
    Start as Client mode.

.PARAMETER ServerAddr
    Server IP Address (for Client mode). Renamed from HostIP to avoid conflict with Help.

.PARAMETER Port
    Server Port (Default: 5000).

.PARAMETER Gpu
    Use GPU (DirectML/CUDA/ROCm/OpenVINO).

.PARAMETER Force
    Force overwrite existing tags.
#>

[CmdletBinding()]
param (
    [string]$Path = "*.webp",
    [float]$Thresh = 0.35,
    [switch]$Server,
    [switch]$Client,
    [string]$ServerAddr = "localhost",
    [int]$Port = 5000,
    [switch]$Gpu,
    [switch]$Force,
    [Alias('h')] # ★これでもう迷わせない
    [switch]$Help
)

function Show-Help {
    Write-Host "=== WD14 Tagger Universal ===" -ForegroundColor Cyan
    Write-Host "Usage:"
    Write-Host "  Standalone : .\run_tagger.ps1 -Path 'C:\Imgs' -Gpu"
    Write-Host "  Server     : .\run_tagger.ps1 -Server -Gpu"
    Write-Host "  Client     : .\run_tagger.ps1 -Client -Path 'C:\Imgs' -ServerAddr '192.168.x.x'"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -h, -Help  : Show this help"
    Write-Host ""
}

if ($Help) { Show-Help; exit }

# --- 設定 ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "embed_tags_universal.py"

# OS判定
$IsWindows = $true
if ($PSVersionTable.PSVersion.Major -ge 6) {
    if ([System.OperatingSystem]::IsLinux()) { $IsWindows = $false }
}

# 仮想環境設定
if ($IsWindows -and $Gpu) {
    $VenvDir = Join-Path $ScriptDir "venv_gpu"
    $Requirements = @("onnxruntime-directml", "pillow", "huggingface_hub", "numpy", "tqdm")
} else {
    $VenvDir = Join-Path $ScriptDir "venv_std"
    $Requirements = @("onnxruntime", "pillow", "huggingface_hub", "numpy", "tqdm")
}

# --- 実行開始 ---
$Mode = "standalone"
if ($Server) { $Mode = "server" }
if ($Client) { $Mode = "client" }

if ($Mode -ne "server") {
    Write-Host "[INFO] Mode: $Mode"
}

# 1. venv作成
if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO] Creating venv at $VenvDir ..." -ForegroundColor Yellow
    if ($IsWindows) { python -m venv $VenvDir } else { python3 -m venv $VenvDir }
}

# 2. Activate & Install
if ($IsWindows) {
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $VenvPip = Join-Path $VenvDir "Scripts\pip.exe"
} else {
    $VenvPython = Join-Path $VenvDir "bin/python"
    $VenvPip = Join-Path $VenvDir "bin/pip"
}

$PipCheck = & $VenvPip list
if ($PipCheck -notmatch "tqdm") {
    Write-Host "[INFO] Installing requirements..." -ForegroundColor Yellow
    & $VenvPip install $Requirements | Out-Null
}

# 3. ExifTool
if (-not (Get-Command "exiftool" -ErrorAction SilentlyContinue)) {
    if ($IsWindows) {
        if (-not (Test-Path (Join-Path $ScriptDir "exiftool.exe"))) {
            Write-Host "[WARN] exiftool.exe missing!" -ForegroundColor Magenta
        }
    }
}

# 4. 実行引数構築
$PyArgs = @($PythonScript, "--mode", $Mode)

if ($Mode -eq "server") {
    $PyArgs += ("--port", $Port)
    if ($Gpu) { $PyArgs += "--gpu" }
}
elseif ($Mode -eq "client") {
    $PyArgs += ($Path, "--host", $ServerAddr, "--port", $Port, "--thresh", $Thresh)
    if ($Force) { $PyArgs += "--force" }
}
else {
    # Standalone
    $PyArgs += ($Path, "--thresh", $Thresh)
    if ($Gpu) { $PyArgs += "--gpu" }
    if ($Force) { $PyArgs += "--force" }
}

Write-Host "[INFO] Starting Python ($Mode)..." -ForegroundColor Green
& $VenvPython @PyArgs