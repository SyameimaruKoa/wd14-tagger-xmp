<#
.SYNOPSIS
    WD14 Tagger Universal Wrapper (日本語版)

.DESCRIPTION
    画像認識AI (WD14 Tagger) を使用して画像のタグ付けや整理を行うスクリプトじゃ。
    引数なしで実行すると「環境構築モード」として動作し、セットアップのみを行って終了する。
    事故防止のため、処理を行いたい場合は必ず -Path などを指定するのじゃ。

    【主な実行モード】
    1. Standalone (通常): その場で画像を処理する。
    2. Server: GPU推論サーバーとして待機する。
    3. Client: サーバーに画像を投げる。

.PARAMETER Path
    【対象パス】 (文字列)
    処理対象の画像ファイル、またはフォルダパス。

.PARAMETER Organize
    【整理モード】 (スイッチ)
    タグ付けを行わず、フォルダ振り分けのみを行うモードじゃ。

.PARAMETER Tag
    【タグ付け有効化】 (スイッチ)
    -Organize と併用する際に、「整理もしつつタグ付けもしたい」場合に指定する。

.PARAMETER NoReport
    【レポートなし】 (スイッチ)
    HTMLレポートの作成をスキップする。

.PARAMETER Recursive
    【再帰検索】 (スイッチ)
    サブフォルダも検索対象にする。

.PARAMETER NoRecursive
    【再帰なし】 (スイッチ)
    サブフォルダを検索しない。

.PARAMETER Thresh
    【タグ採用閾値】 (数値)
    タグ付けを採用する確率の閾値。

.PARAMETER Gpu
    【GPU使用】 (スイッチ)
    GPUを使用して高速化する。

.PARAMETER BatchSize
    【バッチサイズ】 (数値)
    推論をまとめて行う枚数。GPUが暇な場合に効果的。

.PARAMETER IoWorkers
    【前処理ワーカー数】 (数値)
    画像の読み込み・前処理を並列化するワーカー数。

.PARAMETER Force
    【強制実行】 (スイッチ)
    既存タグがあっても強制的に再解析・上書きする。

.PARAMETER Server
    【サーバーモード】 (スイッチ)
    推論サーバーとして起動する。

.PARAMETER Client
    【クライアントモード】 (スイッチ)
    クライアントとして動作し、指定したサーバーへ画像を送信する。

.PARAMETER HostIP
    【ホストIP】 (文字列)
    サーバーのIPアドレス。

.PARAMETER Port
    【ポート】 (数値)
    ポート番号。

.PARAMETER RatingThresh
    【R指定閾値】 (数値)
    [旧機能] R指定タグ合計値による閾値判定。

.PARAMETER IgnoreSensitive
    【Sensitive無視】 (スイッチ)
    [旧機能] SensitiveをGeneralとして扱う。

.PARAMETER Help
    【ヘルプ表示】 (スイッチ)
    このヘルプを表示する。

.PARAMETER RemainingArgs
    未定義の引数（--helpなど）を捕捉するための内部パラメータ。

.EXAMPLE
    # 初回セットアップ (何もしない)
    .\run_tagger.ps1

    # 通常実行 (タグ付け＋レポート)
    .\run_tagger.ps1 -Path "C:\Images" -Gpu

    # フォルダ整理のみ (タグ付けなし)
    .\run_tagger.ps1 -Path "C:\Images" -Organize

    # 全部入り (タグ付け＋整理＋レポート)
    .\run_tagger.ps1 -Path "C:\Images" -Tag -Organize -Gpu
#>

[CmdletBinding()]
param (
    [string]$Path,
    [switch]$Organize,
    [switch]$Tag,
    [switch]$NoReport,
    [switch]$Recursive,
    [switch]$NoRecursive,
    [float]$Thresh = 0.35,
    [switch]$Gpu,
    [int]$BatchSize,
    [int]$IoWorkers,
    [switch]$Force,
    [switch]$Server,
    [switch]$Client,
    [string]$HostIP,
    [int]$Port,
    
    # Old params
    [float]$RatingThresh,
    [switch]$IgnoreSensitive,
    
    [Alias('h')]
    [switch]$Help,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

#region Help Function
function Show-Help {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
}

if ($Help -or ($RemainingArgs -contains '--help')) { Show-Help; exit }
#endregion

#region Environment Setup
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "embed_tags_universal.py"

$IsWindows = $true
if ($PSVersionTable.PSVersion.Major -ge 6) {
    if ([System.OperatingSystem]::IsLinux()) { $IsWindows = $false }
}

function Prepare-Environment {
    param ([bool]$UseGpu)
    if ($IsWindows -and $UseGpu) {
        $EnvName = "GPU (DirectML)"
        $TargetVenv = Join-Path $ScriptDir "venv_gpu"
        $OnnxPackage = "onnxruntime-directml"
    }
    else {
        $EnvName = "Standard (CPU)"
        $TargetVenv = Join-Path $ScriptDir "venv_std"
        $OnnxPackage = "onnxruntime"
    }
    
    Write-Host "[INFO] 環境確認: $EnvName" -ForegroundColor Cyan
    if (-not (Test-Path $TargetVenv)) {
        Write-Host "  -> 仮想環境を作成中..." -ForegroundColor Yellow
        if ($IsWindows) { python -m venv $TargetVenv } else { python3 -m venv $TargetVenv }
    }
    
    if ($IsWindows) {
        $Bin = Join-Path $TargetVenv "Scripts"
        $PyEx = Join-Path $Bin "python.exe"
        $PipEx = Join-Path $Bin "pip.exe"
    }
    else {
        $Bin = Join-Path $TargetVenv "bin"
        $PyEx = Join-Path $Bin "python"
        $PipEx = Join-Path $Bin "pip"
    }

    Write-Host "  -> ライブラリの確認・インストールを行います..." -ForegroundColor Yellow
    $ReqFile = Join-Path $ScriptDir "requirements.txt"
    & $PipEx install -r $ReqFile $OnnxPackage -q | Out-Null
    
    return $PyEx
}
#endregion

#region Main Logic

# 引数が一つもない場合はセットアップモード
if ($PSBoundParameters.Count -eq 0 -and (-not $RemainingArgs)) {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "   WD14 Tagger Universal - Setup Mode" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "引数が指定されなかったため、環境構築のみを行いました。"
    Write-Host "画像処理を行うには -Path オプションなどを指定してください。"
    Write-Host "使い方がわからない場合は -Help を参照するのじゃ。"
    
    # Config生成のために一度CPU環境で実行
    $Py = Prepare-Environment -UseGpu $false
    & $Py $PythonScript --gen-config
    exit
}

# 環境準備
$VenvPython = Prepare-Environment -UseGpu $Gpu

# Python引数構築
$PyArgs = @($PythonScript)

# モード設定
if ($Server) { $PyArgs += ("--mode", "server") }
elseif ($Client) { $PyArgs += ("--mode", "client") }
else { $PyArgs += ("--mode", "standalone") }

# アクション設定
# Organize指定時 -> デフォルトでNo-Tag扱いになる。Tag指定があればタグも有効。
if ($Organize) {
    $PyArgs += "--organize"
    if (-not $Tag) { $PyArgs += "--no-tag" }
}
else {
    # 通常モード -> Tag指定は不要(デフォルトON)。No-Tag指定があれば...無いので実装不要
    # もし将来的に「タグなし・整理なし・レポートのみ」をするなら --no-tag 引数が必要だが
    # 今回のPSラッパーでは Organize がスイッチになっているため自動制御する
}

if ($NoReport) { $PyArgs += "--no-report" }

# 再帰設定
if ($Recursive) { $PyArgs += "--recursive" }
if ($NoRecursive) { $PyArgs += "--no-recursive" }

# その他パラメータ
if ($Thresh -ne 0.35) { $PyArgs += ("--thresh", $Thresh) }
if ($Gpu) { $PyArgs += "--gpu" }
if ($PSBoundParameters.ContainsKey('BatchSize')) { $PyArgs += ("--batch-size", $BatchSize) }
if ($PSBoundParameters.ContainsKey('IoWorkers')) { $PyArgs += ("--io-workers", $IoWorkers) }
if ($Force) { $PyArgs += "--force" }

if ($HostIP) { $PyArgs += ("--host", $HostIP) }
if ($Port) { $PyArgs += ("--port", $Port) }

# Old Params
if ($PSBoundParameters.ContainsKey('RatingThresh')) { $PyArgs += ("--rating-thresh", $RatingThresh) }
if ($IgnoreSensitive) { $PyArgs += "--ignore-sensitive" }

# 最後にパス
if ($Path) { $PyArgs += $Path }

# 実行
Write-Host "[INFO] Pythonスクリプトを実行..." -ForegroundColor Green
& $VenvPython @PyArgs

#endregion