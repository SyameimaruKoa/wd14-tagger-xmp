<#
.SYNOPSIS
    WD14 Tagger Universal Wrapper (Standalone / Server / Client)

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

.PARAMETER IgnoreSensitive
    【センシティブ無視】 (スイッチ)
    これをONにすると、「sensitive (軽度の性的描写)」と判定されたものを
    強制的に「general (全年齢)」として扱う。
    「R-15程度なら一般向け」という豪快なそなたのための機能じゃ。

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
    デフォルト: "localhost" (config.jsonで変更可)

.PARAMETER Port
    【ポート番号】 (整数)
    サーバーとの通信に使用するポート番号じゃ。
    デフォルト: 5000 (config.jsonで変更可)

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

.PARAMETER Setup
    【環境構築モード】 (スイッチ)
    Pythonスクリプトを実行せず、仮想環境の作成とライブラリインストールのみを行う。
    初回導入時や、ライブラリを更新したい時に使うがよい。
    ※この時 config.json も生成されるぞ。

.PARAMETER All
    【全環境作成】 (スイッチ)
    -Setup と組み合わせて使う。
    CPU環境(venv_std)とGPU環境(venv_gpu)の両方をまとめて作成・更新する。

.PARAMETER Report
    【強制レポート作成】 (スイッチ)
    解析処理(Python)をスキップし、既存のログファイル(report_log.json)からHTMLレポートのみを生成する。
    ※通常は解析後に自動でレポートが作られるため、このオプションは不要じゃ。
    手動でレポートだけ作り直したい時専用じゃ。

.EXAMPLE
    .\run_tagger.ps1 -Setup -All
    CPU用とGPU用の仮想環境を両方作成し、config.jsonを生成して終了する。
#>

[CmdletBinding()]
param (
    [string]$Path = "*.webp",
    [float]$Thresh = 0.35,
    [float]$RatingThresh,
    [switch]$IgnoreSensitive,
    [switch]$Server,
    [switch]$Client,
    [string]$ServerAddr,
    [int]$Port,
    [switch]$Gpu,
    [switch]$Force,
    [switch]$Organize,
    [switch]$Setup,
    [switch]$All,
    [switch]$Report,
    [Alias('h')]
    [switch]$Help
)

#region Help Function
function Show-Help {
    Write-Host "=== WD14 Tagger Universal (日本語ヘルプ) ===" -ForegroundColor Cyan
    Write-Host "使い方:"
    Write-Host "  通常実行   : .\run_tagger.ps1 -Path 'フォルダパス' -Gpu -Organize"
    Write-Host "  環境構築   : .\run_tagger.ps1 -Setup [-All] [-Gpu]"
    Write-Host "  サーバー   : .\run_tagger.ps1 -Server -Gpu"
    Write-Host "  クライアント: .\run_tagger.ps1 -Client -Path 'フォルダパス'"
    Write-Host ""
    Write-Host "オプション一覧:"
    Write-Host "  -Path             : 処理対象パス (既定: *.webp)"
    Write-Host "  -Gpu              : GPUを使用"
    Write-Host "  -Organize         : フォルダ振り分けモード"
    Write-Host "  -Report           : 強制レポート作成 (解析スキップ)"
    Write-Host "  -Setup            : 環境構築のみ実行"
    Write-Host "  -All              : Setup時に全環境(CPU/GPU)を作成"
    Write-Host "  -Thresh           : タグ確信度閾値 (0.35)"
    Write-Host "  -RatingThresh     : センシティブ判定閾値 (例: 0.5)"
    Write-Host "  -IgnoreSensitive  : sensitiveをgeneral扱いにする"
    Write-Host "  -Force            : 強制再解析"
    Write-Host "  -Server           : サーバーモード"
    Write-Host "  -Client           : クライアントモード"
    Write-Host "  -h, -Help         : ヘルプ表示"
    Write-Host ""
}

if ($Help) { Show-Help; exit }
#endregion

#region Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "embed_tags_universal.py"
$ReportScript = Join-Path $ScriptDir "make_report.py"
$LogFile = Join-Path (Get-Location) "report_log.json"

# OS Check
$IsWindows = $true
if ($PSVersionTable.PSVersion.Major -ge 6) {
    if ([System.OperatingSystem]::IsLinux()) { $IsWindows = $false }
}
#endregion

#region Environment Setup Function
function Prepare-Environment {
    param (
        [bool]$UseGpu
    )

    if ($IsWindows -and $UseGpu) {
        $EnvName = "GPU (DirectML)"
        $TargetVenv = Join-Path $ScriptDir "venv_gpu"
        $Requirements = @("onnxruntime-directml", "pillow", "huggingface_hub", "numpy", "tqdm")
    } else {
        $EnvName = "Standard (CPU)"
        $TargetVenv = Join-Path $ScriptDir "venv_std"
        $Requirements = @("onnxruntime", "pillow", "huggingface_hub", "numpy", "tqdm")
    }

    Write-Host "[INFO] 環境確認: $EnvName ($TargetVenv)" -ForegroundColor Cyan

    # 1. Create venv
    if (-not (Test-Path $TargetVenv)) {
        Write-Host "  -> 仮想環境を作成中..." -ForegroundColor Yellow
        if ($IsWindows) { python -m venv $TargetVenv } else { python3 -m venv $TargetVenv }
    }

    # Paths
    if ($IsWindows) {
        $Bin = Join-Path $TargetVenv "Scripts"
        $PyEx = Join-Path $Bin "python.exe"
        $PipEx = Join-Path $Bin "pip.exe"
    } else {
        $Bin = Join-Path $TargetVenv "bin"
        $PyEx = Join-Path $Bin "python"
        $PipEx = Join-Path $Bin "pip"
    }

    # 2. Update Pip (Silence warnings)
    Write-Host "  -> pipを更新確認中..."
    & $PyEx -m pip install --upgrade pip | Out-Null

    # 3. Install Requirements
    $Installed = & $PipEx list
    $NeedsInstall = $false
    foreach ($req in $Requirements) {
        if ($Installed -notmatch $req) { $NeedsInstall = $true; break }
    }

    if ($NeedsInstall) {
        Write-Host "  -> ライブラリをインストール中..." -ForegroundColor Yellow
        & $PipEx install $Requirements | Out-Null
    } else {
        Write-Host "  -> ライブラリは最新じゃ。" -ForegroundColor Gray
    }

    return $PyEx
}
#endregion

#region Execution Logic

# Setup Mode
if ($Setup) {
    if ($All) {
        $Py = Prepare-Environment -UseGpu $false
        & $Py $PythonScript --gen-config # Python側でConfig生成
        
        if ($IsWindows) { 
            $Py = Prepare-Environment -UseGpu $true
            & $Py $PythonScript --gen-config 
        }
    } else {
        $Py = Prepare-Environment -UseGpu $Gpu
        & $Py $PythonScript --gen-config
    }
    
    Write-Host "`n[INFO] 環境構築完了じゃ。" -ForegroundColor Green
    exit
}

# Normal Execution
$Mode = "standalone"
if ($Server) { $Mode = "server" }
if ($Client) { $Mode = "client" }

if ($Mode -ne "server") {
    Write-Host "[INFO] モード: $Mode"
}

# ExifTool Check (Serverモードならチェックしない)
if ($Mode -ne "server") {
    if (-not (Get-Command "exiftool" -ErrorAction SilentlyContinue)) {
        if ($IsWindows) {
            if (-not (Test-Path (Join-Path $ScriptDir "exiftool.exe"))) {
                Write-Host "[WARN] exiftool.exe が見つかりません！タグの書き込みに失敗する可能性があります。" -ForegroundColor Magenta
            }
        }
    }
}

# Prepare Env
$VenvPython = Prepare-Environment -UseGpu $Gpu

# ★ ReportOnlyモード判定
if ($Report) {
    # 強制レポートモード: 解析をスキップして make_report.py だけ走らせる
    if (Test-Path $LogFile) {
        Write-Host "`n[INFO] 既存ログからレポートを作成中..." -ForegroundColor Cyan
        & $VenvPython $ReportScript
    } else {
        Write-Host "[WARN] レポートログ ($LogFile) が見つからぬ。まずは通常実行してログを作るのじゃ。" -ForegroundColor Red
    }
    exit
}

# --- 通常の解析処理 ---

# Build Arguments
$PyArgs = @($PythonScript, "--mode", $Mode, "--save-report") # ★常にレポート用ログを保存させる

if ($Mode -eq "server") {
    if ($Port) { $PyArgs += ("--port", $Port) }
    if ($Gpu) { $PyArgs += "--gpu" }
}
elseif ($Mode -eq "client") {
    $PyArgs += ($Path, "--thresh", $Thresh)
    if ($ServerAddr) { $PyArgs += ("--host", $ServerAddr) }
    if ($Port) { $PyArgs += ("--port", $Port) }
    if ($Force) { $PyArgs += "--force" }
    
    # Client Mode Features
    if ($Organize) { $PyArgs += "--organize" }
    if ($IgnoreSensitive) { $PyArgs += "--ignore-sensitive" }
    if ($PSBoundParameters.ContainsKey('RatingThresh')) {
        $PyArgs += ("--rating-thresh", $RatingThresh)
    }
}
else {
    # Standalone
    $PyArgs += ($Path, "--thresh", $Thresh)
    if ($Gpu) { $PyArgs += "--gpu" }
    if ($Force) { $PyArgs += "--force" }
    if ($Organize) { $PyArgs += "--organize" }
    if ($IgnoreSensitive) { $PyArgs += "--ignore-sensitive" }
    if ($PSBoundParameters.ContainsKey('RatingThresh')) {
        $PyArgs += ("--rating-thresh", $RatingThresh)
    }
    
    if ($ServerAddr) { $PyArgs += ("--host", $ServerAddr) }
    if ($Port) { $PyArgs += ("--port", $Port) }
}

Write-Host "[INFO] Pythonスクリプトを開始 ($Mode)..." -ForegroundColor Green
& $VenvPython @PyArgs

# ★ 自動レポート生成 (ログがあれば)
if (Test-Path $LogFile) {
    Write-Host "`n[INFO] レポートHTMLを作成中..." -ForegroundColor Cyan
    & $VenvPython $ReportScript
    # 削除は make_report.py 側で行うため、ここでは呼ばなくてよい
}

echo "[INFO] Done."
#endregion