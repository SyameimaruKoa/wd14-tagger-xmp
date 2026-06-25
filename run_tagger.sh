#!/bin/bash

# 下にヘルプがあるぞ

# ==========================================
# WD14 Tagger Universal (Bash Wrapper)
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/embed_tags_universal.py"

# デフォルト値
USE_GPU=0
FORCE_TYPE="auto" # auto, nvidia, intel, amd
PY_ARGS=()
DO_ORGANIZE=0
DO_TAG=0
DO_PIXIV=0
IS_CLIENT=0

show_help() {
    echo "WD14 Tagger Universal (日本語ヘルプ)"
    echo ""
    echo "使い方: ./run_tagger.sh [オプション] [パス]"
    echo ""
    echo "  引数なしで実行すると「環境構築モード」となり、セットアップのみを行います。"
    echo ""
    echo "主なオプション:"
    echo "    -p, --path <path>   処理対象ファイル/フォルダ"
    echo "    -g, --gpu           GPUを使用する（自動判別）"
    echo "    --force-intel       Intel GPUを強制的に使用する"
    echo "    --force-nvidia      NVIDIA GPUを強制的に使用する"
    echo "    --force-amd         AMD GPUを強制的に使用する"
    echo "    --organize          フォルダ整理のみ行う（タグ付けOFF）"
    echo "    --tag               タグ付けも行う（--organize併用時）"
    echo "    --pixiv             Pixiv整理モード（R17以上を親フォルダへ移動）"
    echo "    --no-report         レポート作成なし"
    echo "    --recursive         再帰検索ON"
    echo "    --no-recursive      再帰検索OFF"
    echo "    --batch-size <n>    推論バッチサイズ（デフォルト: 4 / 非対応時は 1）"
    echo "    --io-workers <n>    前処理の並列ワーカー数（デフォルト: 自動）"
    echo "    --model-repo <repo> モデル/タグのHFリポジトリID"
    echo "    --model-file <file> モデルファイル名またはパス"
    echo "    --tags-file <file>  タグCSVファイル名またはパス"
    echo "    -f, --force         既存タグがあっても強制的に再解析・上書きする"
    echo "    --server            サーバーモード"
    echo "    --client            クライアントモード"
    echo "    -H, --host <ip>     サーバーのIPアドレス"
    echo "    -P, --port <port>   ポート番号"
    echo "    -h, --help          ヘルプ表示"
    echo ""
}

# --- GPU検出関数 ---
detect_gpu_vendor() {
    # lspciの結果からVGA/3Dコントローラを探す
    local lspci_out=$(lspci | grep -E "VGA|3D|Display" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$lspci_out" == *"nvidia"* ]]; then
        echo "nvidia"
    elif [[ "$lspci_out" == *"intel"* ]]; then
        echo "intel"
    elif [[ "$lspci_out" == *"amd"* ]] || [[ "$lspci_out" == *"advanced micro devices"* ]] || [[ "$lspci_out" == *"radeon"* ]]; then
        # 内蔵GPUアーキテクチャの互換性チェック
        if command -v rocminfo >/dev/null 2>&1; then
            local arch=$(rocminfo | grep -o "gfx[0-9a-f]\+" | head -n 1)
            if [ "$arch" = "gfx90c" ]; then
                echo "amd_unsupported"
                return
            fi
        fi
        echo "amd"
    else
        echo "none"
    fi
}

# --- 環境セットアップ ---
setup_env() {
    local backend=$1 # nvidia, intel, amd, cpu, client
    local is_client=$2
    local venv_name=""
    
    if [ "$is_client" = "1" ]; then
        if [ -d "$SCRIPT_DIR/venv_gpu" ]; then
            venv_name="venv_gpu"
            backend="nvidia"
        elif [ -d "$SCRIPT_DIR/venv_intel" ]; then
            venv_name="venv_intel"
            backend="intel"
        elif [ -d "$SCRIPT_DIR/venv_amd" ]; then
            venv_name="venv_amd"
            backend="amd"
        elif [ -d "$SCRIPT_DIR/venv_std" ]; then
            venv_name="venv_std"
            backend="cpu"
        else
            venv_name="venv_client"
            backend="client"
        fi
    elif [ "$backend" = "cpu" ]; then
        if [ -d "$SCRIPT_DIR/venv_gpu" ]; then
            venv_name="venv_gpu"
            backend="nvidia"
        elif [ -d "$SCRIPT_DIR/venv_intel" ]; then
            venv_name="venv_intel"
            backend="intel"
        elif [ -d "$SCRIPT_DIR/venv_amd" ]; then
            venv_name="venv_amd"
            backend="amd"
        else
            venv_name="venv_std"
        fi
    else
        if [ "$backend" = "nvidia" ]; then
            venv_name="venv_gpu"
        elif [ "$backend" = "intel" ]; then
            venv_name="venv_intel"
        elif [ "$backend" = "amd" ]; then
            venv_name="venv_amd"
        fi
    fi

    VENV_DIR="$SCRIPT_DIR/$venv_name"
    
    if [ ! -d "$VENV_DIR" ]; then
        echo "[INFO] 仮想環境を作成中 ($venv_name)..."
        python3 -m venv "$VENV_DIR"
    fi
    
    PIP_CMD="$VENV_DIR/bin/pip"
    
    # 必要なパッケージのインストール
    echo "[INFO] 依存ライブラリを確認・インストール中 ($backend)..."
    REQ_FILE="$SCRIPT_DIR/requirements.txt"
    
    # 失敗したときにすぐ止まるようにエラー処理を追加じゃ
    if [ "$backend" = "client" ]; then
        $PIP_CMD install -r "$REQ_FILE" || { echo "[ERROR] ライブラリのインストールに失敗しました。"; exit 1; }
    elif [ "$backend" = "nvidia" ]; then
        $PIP_CMD install -r "$REQ_FILE" onnxruntime-gpu || { echo "[ERROR] ライブラリのインストールに失敗しました。"; exit 1; }
    elif [ "$backend" = "intel" ]; then
        $PIP_CMD install -r "$REQ_FILE" onnxruntime-openvino || { echo "[ERROR] ライブラリのインストールに失敗しました。"; exit 1; }
    elif [ "$backend" = "amd" ]; then
        # AMD用 ROCm対応パッケージは「onnxruntime-migraphx」に変更されておるのじゃ
        $PIP_CMD install -r "$REQ_FILE" onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/ -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ || { echo "[ERROR] ライブラリのインストールに失敗しました。"; exit 1; }
        
        # Ubuntu等の新しいLinux環境では、実行可能スタックのフラグが原因でロードエラーになるため解除するのじゃ
        SO_FILE=$(find "$VENV_DIR" -name "onnxruntime_pybind11_state.so" | head -n 1)
        if [ -n "$SO_FILE" ]; then
            if command -v execstack >/dev/null 2>&1; then
                execstack -c "$SO_FILE"
            elif command -v patchelf >/dev/null 2>&1; then
                patchelf --clear-execstack "$SO_FILE"
            else
                echo "[ERROR] セキュリティ制約の回避に必要な execstack または patchelf が見つからぬ！"
                echo "        Ubuntu 24.04等では古いexecstackは削除されておるため、以下のコマンドで patchelf をインストールしてから再度実行するのじゃ。"
                echo "        sudo apt install patchelf"
                exit 1
            fi
        fi
    else
        # CPU用
        $PIP_CMD install -r "$REQ_FILE" onnxruntime || { echo "[ERROR] ライブラリのインストールに失敗しました。"; exit 1; }
    fi
}

# 引数なしチェック
if [ $# -eq 0 ]; then
    echo "=========================================="
    echo "   WD14 Tagger Universal - Setup Mode"
    echo "=========================================="
    echo "引数が指定されなかったため、環境構築のみを行います。"
    # CPUのみ作っておく（クライアントフラグ0）
    setup_env "cpu" "0"
    "$SCRIPT_DIR/venv_std/bin/python" "$PYTHON_SCRIPT" --gen-config
    echo "[INFO] セットアップ完了。GPU環境は --gpu 指定時に構築されます。"
    exit 0
fi

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --server) PY_ARGS+=("--mode" "server"); shift ;;
        --client) PY_ARGS+=("--mode" "client"); IS_CLIENT=1; shift ;;
        --organize) DO_ORGANIZE=1; shift ;;
        --tag) DO_TAG=1; shift ;; 
        --pixiv) PY_ARGS+=("--pixiv"); DO_PIXIV=1; shift ;;
        --no-report) PY_ARGS+=("--no-report"); shift ;;
        --recursive) PY_ARGS+=("--recursive"); shift ;;
        --no-recursive) PY_ARGS+=("--no-recursive"); shift ;;
        --batch-size) PY_ARGS+=("--batch-size" "$2"); shift 2 ;;
        --io-workers) PY_ARGS+=("--io-workers" "$2"); shift 2 ;;
        --model-repo) PY_ARGS+=("--model-repo" "$2"); shift 2 ;;
        --model-file) PY_ARGS+=("--model-file" "$2"); shift 2 ;;
        --tags-file) PY_ARGS+=("--tags-file" "$2"); shift 2 ;;
        -g|--gpu) USE_GPU=1; shift ;;
        --force-intel) USE_GPU=1; FORCE_TYPE="intel"; shift ;;
        --force-nvidia) USE_GPU=1; FORCE_TYPE="nvidia"; shift ;;
        --force-amd) USE_GPU=1; FORCE_TYPE="amd"; shift ;;
        -f|--force) PY_ARGS+=("--force"); shift ;;
        -p|--path) PY_ARGS+=("$2"); shift 2 ;;
        -H|--host) PY_ARGS+=("--host" "$2"); shift 2 ;;
        -P|--port) PY_ARGS+=("--port" "$2"); shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) PY_ARGS+=("$1"); shift ;;
    esac
done

# アクションロジック構築
if [ $DO_ORGANIZE -eq 1 ] || [ $DO_PIXIV -eq 1 ]; then
    if [[ ! " ${PY_ARGS[*]} " =~ " --organize " ]]; then
        PY_ARGS+=("--organize")
    fi
    if [ $DO_TAG -eq 0 ] && [ $DO_PIXIV -eq 0 ]; then
        PY_ARGS+=("--no-tag")
    fi
fi

# GPUモード決定ロジック
BACKEND_MODE="cpu"
if [ $USE_GPU -eq 1 ]; then
    if [ "$FORCE_TYPE" = "intel" ]; then
        echo "[INFO] Intel GPU モードを強制使用します。"
        BACKEND_MODE="intel"
    elif [ "$FORCE_TYPE" = "nvidia" ]; then
        echo "[INFO] NVIDIA GPU モードを強制使用します。"
        BACKEND_MODE="nvidia"
    elif [ "$FORCE_TYPE" = "amd" ]; then
        echo "[WARN] AMD GPU モードを強制使用しますが、アーキテクチャ未サポートによるコアダンプの危険性があります。"
        BACKEND_MODE="amd"
    else
        # 自動判別
        DETECTED=$(detect_gpu_vendor)
        if [ "$DETECTED" = "nvidia" ]; then
            echo "[INFO] NVIDIA GPU を検出しました。CUDAモードで実行します。"
            BACKEND_MODE="nvidia"
        elif [ "$DETECTED" = "intel" ]; then
            echo "[INFO] Intel GPU を検出しました。OpenVINOモードで実行します。"
            BACKEND_MODE="intel"
        elif [ "$DETECTED" = "amd" ]; then
            echo "[INFO] AMD GPU を検出しました。ROCmモードで実行します。"
            BACKEND_MODE="amd"
        elif [ "$DETECTED" = "amd_unsupported" ]; then
            echo "[WARN] サポート外のAMD内蔵GPU(gfx90c等)を検出しました。コアダンプ回避のため、安全なCPUモードで実行します。"
            BACKEND_MODE="cpu"
        else
            echo "[WARN] GPUが見つからない、または判別できませんでした。CPUモードで実行します。"
            BACKEND_MODE="cpu"
        fi
    fi
    # Python側には --gpu フラグを渡す（Python側でプロバイダを総当たりさせるため）
    if [ "$BACKEND_MODE" != "cpu" ]; then
        PY_ARGS+=("--gpu")
    fi
fi

# CPUモード時はスレッドプールのデッドロックを防ぐため並列前処理を強制OFFにするのじゃ
if [ "$BACKEND_MODE" = "cpu" ]; then
    PY_ARGS+=("--io-workers" "0")
fi

setup_env "$BACKEND_MODE" "$IS_CLIENT"

echo "[INFO] Pythonスクリプトを実行 ($BACKEND_MODE)..."
"$VENV_DIR/bin/python" "$PYTHON_SCRIPT" "${PY_ARGS[@]}"