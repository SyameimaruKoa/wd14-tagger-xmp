<#
.SYNOPSIS
    WD14 Tagger Universal Wrapper (Standalone / Server / Client)

.DESCRIPTION
    - Standalone (Default): Process files locally.
    - Server: Start GPU inference server.
    - Client: Send files to server for processing.

.PARAMETER Path
    Target files (Standalone/Client).

.PARAMETER Thresh
    Threshold for tag confidence (Default: 0.35).

.PARAMETER RatingThresh
    Threshold for non-general ratings (sensitive/questionable/explicit).
    If the confidence is lower than this value, it defaults to 'general'.
    Example: 0.5 (Only classify as sensitive if confidence > 0.5).

.PARAMETER Server
    Start as Server mode.

.PARAMETER Client
    Start as Client mode.

.PARAMETER ServerAddr
    Server IP Address (for Client mode).

.PARAMETER Port
    Server Port (Default: 5000).

.PARAMETER Gpu
    Use GPU (DirectML/CUDA/ROCm/OpenVINO).

.PARAMETER Force
    Force overwrite existing tags.

.PARAMETER Organize
    Move files to folders (general/sensitive/questionable/explicit) based on rating.

.EXAMPLE
    .\run_tagger.ps1 -Path 'C:\Images' -Gpu -Organize -RatingThresh 0.5
    Move files to rating folders. 'sensitive' is only chosen if confidence > 0.5.
#>

[CmdletBinding()]
param (
    [string]$Path = "*.webp",
    [float]$Thresh = 0.35,
    [float]$RatingThresh,
    [switch]$Server,
    [switch]$Client,
    [string]$ServerAddr = "localhost",
    [int]$Port = 5000,
    [switch]$Gpu,
    [switch]$Force,
    [switch]$Organize,
    [Alias('h')]
    [switch]$Help
)

#region Help Function
function Show-Help {
    Write-Host "=== WD14 Tagger Universal ===" -ForegroundColor Cyan
    Write-Host "Usage:"
    Write-Host "  Standalone : .\run_tagger.ps1 -Path 'C:\Imgs' -Gpu -Organize [-RatingThresh 0.5]"
    Write-Host "  Server     : .\run_tagger.ps1 -Server -Gpu"
    Write-Host "  Client     : .\run_tagger.ps1 -Client -Path 'C:\Imgs' -ServerAddr '192.168.x.x'"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Organize     : Move files to folders based on rating."
    Write-Host "  -RatingThresh : Min confidence for sensitive/questionable/explicit (e.g., 0.5)."
    Write-Host "  -h, -Help     : Show this help"
    Write-Host ""
}

if ($Help) { Show-Help; exit }
#endregion

#region Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "embed_tags_universal.py"

# OS Check
$IsWindows = $true
if ($PSVersionTable.PSVersion.Major -ge 6) {
    if ([System.OperatingSystem]::IsLinux()) { $IsWindows = $false }
}

# Virtual Env Config
if ($IsWindows -and $Gpu) {
    $VenvDir = Join-Path $ScriptDir "venv_gpu"
    $Requirements = @("onnxruntime-directml", "pillow", "huggingface_hub", "numpy", "tqdm")
} else {
    $VenvDir = Join-Path $ScriptDir "venv_std"
    $Requirements = @("onnxruntime", "pillow", "huggingface_hub", "numpy", "tqdm")
}
#endregion

#region Execution Logic
$Mode = "standalone"
if ($Server) { $Mode = "server" }
if ($Client) { $Mode = "client" }

if ($Mode -ne "server") {
    Write-Host "[INFO] Mode: $Mode"
}

# 1. Create venv
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

# 3. ExifTool Check
if (-not (Get-Command "exiftool" -ErrorAction SilentlyContinue)) {
    if ($IsWindows) {
        if (-not (Test-Path (Join-Path $ScriptDir "exiftool.exe"))) {
            Write-Host "[WARN] exiftool.exe missing!" -ForegroundColor Magenta
        }
    }
}

# 4. Build Arguments
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
    if ($Organize) { $PyArgs += "--organize" }
    if ($PSBoundParameters.ContainsKey('RatingThresh')) {
        $PyArgs += ("--rating-thresh", $RatingThresh)
    }
}

Write-Host "[INFO] Starting Python ($Mode)..." -ForegroundColor Green
& $VenvPython @PyArgs
#endregion