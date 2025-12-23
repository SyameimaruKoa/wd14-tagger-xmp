<#
.SYNOPSIS
    WD14 Tagger Universal ラッパー (スタンドアロン / サーバー / クライアント)

.DESCRIPTION
    画像認識AI (WD14 Tagger) を使用して画像のタグ付けや整理を行うスクリプトじゃ。
    以下の3つのモードで動作するぞ。
    1. スタンドアロン (通常): その場で画像を読み込んで処理する。
    2. サーバー: GPUを使って待機し、送られてきた画像を処理する。
    3. クライアント: 別のPC（サーバー）に画像を投げて処理してもらう。

.PARAMETER Path
    【対象パス】 (文字列)
    処理対象のファイルパス、またはフォルダパスを指定する。
    ワイルドカードも使用可能じゃ。
    例: 'C:\Images' や '*.png' など。
    デフォルト: "*.webp"

.PARAMETER Thresh
    【タグ確信度閾値】 (0.0〜1.0)
    タグとして採用するための最低ラインじゃ。
    AIが「このタグである確率」がこの値を超えたものだけが書き込まれる。
    値を上げると精度は上がるがタグ数は減る。
    デフォルト: 0.35

.PARAMETER RatingThresh
    【レーティング判定閾値】 (0.0〜1.0)
    これを指定すると、センシティブ判定の基準を厳しくできる。
    「sensitive」「questionable」「explicit」の確信度がこの値を超えない限り、
    強制的に「general (全年齢)」として扱われるようになる。
    ※これを指定すると、既存タグがあっても強制的にAI解析が走るぞ。
    例: 0.5 (確信度50%を超えない限りセンシティブ扱いしない)

.PARAMETER Server
    【サーバーモード】 (スイッチ)
    これを付けると推論サーバーとして起動する。
    入力待ち状態になり、クライアントからのリクエストを処理する。

.PARAMETER Client
    【クライアントモード】 (スイッチ)
    これを付けるとクライアントとして動作し、
    指定したサーバーへ画像を送信して処理させる。

.PARAMETER ServerAddr
    【サーバーアドレス】 (文字列)
    クライアントモードで接続する先のIPアドレスじゃ。
    デフォルト: "localhost"

.PARAMETER Port
    【ポート番号】 (整数)
    サーバーとの通信に使用するポート番号じゃ。
    デフォルト: 5000

.PARAMETER Gpu
    【GPU使用】 (スイッチ)
    ONにすると GPU (DirectML / CUDA / ROCm / OpenVINO) を使用して高速化する。
    指定しない場合は CPU でのんびり処理する。

.PARAMETER Force
    【強制上書き】 (スイッチ)
    ファイルに既にタグ情報 (XMP) が埋め込まれていても、無視して再解析・上書き保存する。
    指定しない場合、タグ付きファイルはスキップされる（RatingThresh指定時を除く）。

.PARAMETER Organize
    【フォルダ振り分け】 (スイッチ)
    判定されたレーティングに基づいて、ファイルを自動的にフォルダへ移動させる。
    移動先: general, sensitive, questionable, explicit
    ※タグ付けと同時に整理したい時に使うのじゃ。

.EXAMPLE
    .\run_tagger.ps1 -Path 'C:\Images' -Gpu -Organize -RatingThresh 0.5
    GPUを使って C:\Images 内の画像を処理し、
    確信度0.5を基準にフォルダ分けを行う（それ以下ならgeneralへ）。
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
    Write-Host "=== WD14 Tagger Universal (日本語ヘルプ) ===" -ForegroundColor Cyan
    Write-Host "使い方:"
    Write-Host "  通常実行   : .\run_tagger.ps1 -Path 'フォルダパス' -Gpu -Organize [-RatingThresh 0.5]"
    Write-Host "  サーバー   : .\run_tagger.ps1 -Server -Gpu"
    Write-Host "  クライアント: .\run_tagger.ps1 -Client -Path 'フォルダパス' -ServerAddr '192.168.x.x'"
    Write-Host ""
    Write-Host "オプション一覧:"
    Write-Host "  -Path          : 処理する画像やフォルダのパス (既定: *.webp)"
    Write-Host "  -Gpu           : GPUを使って高速化する"
    Write-Host "  -Organize      : レーティング(general/sensitive等)別にフォルダ分けする"
    Write-Host "  -Thresh        : タグ付けの確信度閾値 (既定: 0.35)"
    Write-Host "  -RatingThresh  : センシティブ判定の閾値。これ以下の確信度はgeneral扱いにする (例: 0.5)"
    Write-Host "  -Force         : 既存タグがあっても強制的に上書き・再解析する"
    Write-Host "  -Server        : サーバーモードで起動"
    Write-Host "  -Client        : クライアントモードで起動"
    Write-Host "  -ServerAddr    : 接続先サーバーのIPアドレス"
    Write-Host "  -Port          : 通信ポート (既定: 5000)"
    Write-Host "  -h, -Help      : このヘルプを表示する"
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
    Write-Host "[INFO] モード: $Mode"
}

# 1. Create venv
if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO] 仮想環境を作成中: $VenvDir ..." -ForegroundColor Yellow
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
    Write-Host "[INFO] 必要なパッケージをインストール中..." -ForegroundColor Yellow
    & $VenvPip install $Requirements | Out-Null
}

# 3. ExifTool Check
if (-not (Get-Command "exiftool" -ErrorAction SilentlyContinue)) {
    if ($IsWindows) {
        if (-not (Test-Path (Join-Path $ScriptDir "exiftool.exe"))) {
            Write-Host "[WARN] exiftool.exe が見つかりません！タグの書き込みに失敗する可能性があります。" -ForegroundColor Magenta
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

Write-Host "[INFO] Pythonスクリプトを開始 ($Mode)..." -ForegroundColor Green
& $VenvPython @PyArgs
#endregion