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

.PARAMETER Pixiv
    【Pixiv整理モード】 (スイッチ)
    Pixiv専用の整理を行う。再帰的にタグ付けし、R15.5以下は移動せず、R17以上を親フォルダに階層を維持して移動する。

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
    推論をまとめて行う枚数（デフォルト: 4）。モデル非対応時は自動で 1 になる。

.PARAMETER IoWorkers
    【前処理ワーカー数】 (数値)
    画像の読み込み・前処理を並列化するワーカー数（デフォルト: 自動）。

.PARAMETER ModelRepo
    【モデルリポジトリ】 (文字列)
    HuggingFaceのモデル/タグのリポジトリID。

.PARAMETER ModelFile
    【モデルファイル】 (文字列)
    モデルファイル名またはローカルパス。

.PARAMETER TagsFile
    【タグCSV】 (文字列)
    タグCSVファイル名またはローカルパス。

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
    [string]$ModelRepo,
    [string]$ModelFile,
    [string]$TagsFile,
    [switch]$Force,
    [switch]$Server,
    [switch]$Client,
    [string]$HostIP,
    [int]$Port,
    [switch]$Pixiv,
    
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
    Write-Host "WD14 Tagger Universal (日本語ヘルプ)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "使い方: .\run_tagger.ps1 [オプション] [パス]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  引数なしで実行すると「環境構築モード」となり、セットアップのみを行います。" -ForegroundColor Gray
    Write-Host ""
    Write-Host "主なオプション:" -ForegroundColor Yellow
    Write-Host "    -Path <path>          処理対象ファイル/フォルダ"
    Write-Host "    -Gpu                  GPUを使用する（Windows: DirectML）"
    Write-Host "    -Organize             フォルダ整理のみ行う（タグ付けOFF）"
    Write-Host "    -Tag                  タグ付けも行う（-Organize併用時）"
    Write-Host "    -Pixiv                Pixiv整理モード（R17以上を親フォルダへ移動）"
    Write-Host "    -NoReport             レポート作成なし"
    Write-Host "    -Recursive            再帰検索ON"
    Write-Host "    -NoRecursive          再帰検索OFF"
    Write-Host "    -Thresh <0.0-1.0>     タグ採用確率の閾値（デフォルト: 0.35）"
    Write-Host "    -BatchSize <n>        推論バッチサイズ"
    Write-Host "    -IoWorkers <n>        前処理の並列ワーカー数"
    Write-Host "    -ModelRepo <repo>     モデル/タグのHFリポジトリID"
    Write-Host "    -ModelFile <file>     モデルファイル名またはパス"
    Write-Host "    -TagsFile <file>      タグCSVファイル名またはパス"
    Write-Host "    -Force                既存タグがあっても強制的に再解析・上書き"
    Write-Host "    -Server               サーバーモード（推論待機）"
    Write-Host "    -Client               クライアントモード"
    Write-Host "    -HostIP <ip>          サーバーのIPアドレス"
    Write-Host "    -Port <port>          ポート番号"
    Write-Host "    -Help (-h, --help)    このヘルプを表示"
    Write-Host ""
    Write-Host "実行例:" -ForegroundColor Yellow
    Write-Host "    # 初回セットアップ（何もしない）"
    Write-Host "    .\run_tagger.ps1"
    Write-Host ""
    Write-Host "    # 通常実行（タグ付け＋レポート＋GPU）"
    Write-Host "    .\run_tagger.ps1 -Path C:\Images -Gpu"
    Write-Host ""
    Write-Host "    # フォルダ整理のみ（タグ付けなし）"
    Write-Host "    .\run_tagger.ps1 -Path C:\Images -Organize"
    Write-Host ""
    Write-Host "    # 全部入り（タグ付け＋整理＋レポート＋GPU）"
    Write-Host "    .\run_tagger.ps1 -Path C:\Images -Organize -Tag -Gpu"
    Write-Host ""
}

if ($Help -or ($RemainingArgs -contains '--help') -or ($RemainingArgs -contains '-h')) { Show-Help; exit }
#endregion

#region Environment Setup
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "embed_tags_universal.py"

$IsWindowsOS = $true
if ($PSVersionTable.PSVersion.Major -ge 6) {
    if ([System.OperatingSystem]::IsLinux()) { $IsWindowsOS = $false }
}

function Prepare-Environment {
    param ([bool]$UseGpu, [bool]$IsClient)
    
    if ($IsClient) {
        if (Test-Path (Join-Path $ScriptDir "venv_gpu")) {
            $EnvName = "Client (流用 venv_gpu)"
            $TargetVenv = Join-Path $ScriptDir "venv_gpu"
            $OnnxPackage = "onnxruntime-directml"
        }
        elseif (Test-Path (Join-Path $ScriptDir "venv_std")) {
            $EnvName = "Client (流用 venv_std)"
            $TargetVenv = Join-Path $ScriptDir "venv_std"
            $OnnxPackage = "onnxruntime"
        }
        else {
            $EnvName = "Client (軽量)"
            $TargetVenv = Join-Path $ScriptDir "venv_client"
            $OnnxPackage = ""
        }
    } else {
        if ($IsWindowsOS -and $UseGpu) {
            $EnvName = "GPU (DirectML)"
            $TargetVenv = Join-Path $ScriptDir "venv_gpu"
            $OnnxPackage = "onnxruntime-directml"
        }
        elseif (-not $UseGpu -and (Test-Path (Join-Path $ScriptDir "venv_gpu"))) {
            $EnvName = "CPU (流用 venv_gpu)"
            $TargetVenv = Join-Path $ScriptDir "venv_gpu"
            $OnnxPackage = "onnxruntime-directml"
        }
        else {
            $EnvName = "Standard (CPU)"
            $TargetVenv = Join-Path $ScriptDir "venv_std"
            $OnnxPackage = "onnxruntime"
        }
    }
    
    Write-Host "[INFO] 環境確認: $EnvName" -ForegroundColor Cyan
    if (-not (Test-Path $TargetVenv)) {
        Write-Host "  -> 仮想環境を作成中..." -ForegroundColor Yellow
        if ($IsWindowsOS) { python -m venv $TargetVenv } else { python3 -m venv $TargetVenv }
    }
    
    if ($IsWindowsOS) {
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
    if ($OnnxPackage) {
        & $PipEx install -r $ReqFile $OnnxPackage -q | Out-Null
    } else {
        & $PipEx install -r $ReqFile -q | Out-Null
    }
    
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
    $Py = Prepare-Environment -UseGpu $false -IsClient $false
    & $Py $PythonScript --gen-config
    exit
}

# 環境準備
$VenvPython = Prepare-Environment -UseGpu $Gpu -IsClient $Client

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
    if (-not $Tag -and -not $Pixiv) { $PyArgs += "--no-tag" }
}
if ($Pixiv) {
    $PyArgs += "--pixiv"
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
if ($PSBoundParameters.ContainsKey('ModelRepo')) { $PyArgs += ("--model-repo", $ModelRepo) }
if ($PSBoundParameters.ContainsKey('ModelFile')) { $PyArgs += ("--model-file", $ModelFile) }
if ($PSBoundParameters.ContainsKey('TagsFile')) { $PyArgs += ("--tags-file", $TagsFile) }
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