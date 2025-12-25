#!/bin/bash

# ==========================================
# WD14 Tagger Universal (Bash Wrapper)
# ==========================================

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PYTHON_SCRIPT="$SCRIPT_DIR/embed_tags_universal.py"

# デフォルト値
USE_GPU=0
FORCE_TYPE="auto" # auto, nvidia, intel
PY_ARGS=()
DO_ORGANIZE=0
DO_TAG=0

show_help() {
    echo "WD14 Tagger Universal (日本語ヘルプ)"
    echo ""
    echo "使い方: ./run_tagger.sh [オプション] [パス]"
    echo ""
    echo "  引数なしで実行すると「環境構築モード」となり、セットアップのみを行います。"
    echo ""
    echo "主なオプション:"
    echo "  -p, --path <path>   処理対象ファイル/フォルダ"
    echo "  -g, --gpu           GPUを使用する（自動判別）"
    echo "  --force-intel       Intel GPUを強制的に使用する"
    echo "  --force-nvidia      NVIDIA GPUを強制的に使用する"
    echo "  --organize          フォルダ整理のみ行う（タグ付けOFF）"
    echo "  --tag               タグ付けも行う（--organize併用時）"
    echo "  --no-report         レポート作成なし"
    echo "  --recursive         再帰検索ON"
    echo "  --server            サーバーモード"
    echo "  --client            クライアントモード"
    echo "  -h, --help          ヘルプ表示"
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
    else
        echo "none"
    fi
}

# --- 環境セットアップ ---
setup_env() {
    local backend=$1 # nvidia, intel, cpu
    local venv_name="venv_std"
    
    if [ "$backend" = "nvidia" ]; then
        venv_name="venv_gpu"
    elif [ "$backend" = "intel" ]; then
        venv_name="venv_intel"
    fi

    VENV_DIR="$SCRIPT_DIR/$venv_name"
    
    if [ ! -d "$VENV_DIR" ]; then
        echo "[INFO] 仮想環境を作成中 ($venv_name)..."
        python3 -m venv "$VENV_DIR"
    fi
    
    PIP_CMD="$VENV_DIR/bin/pip"
    
    # 必要なパッケージのインストール
    # 既にインストール済みか簡易チェックしてもいいが、pipは既存ならスキップしてくれるのでそのまま叩く
    echo "[INFO] 依存ライブラリを確認中 ($backend)..."
    
    if [ "$backend" = "nvidia" ]; then
        $PIP_CMD install onnxruntime-gpu pillow huggingface_hub numpy tqdm > /dev/null 2>&1
    elif [ "$backend" = "intel" ]; then
        # Intel用 OpenVINO EP
        $PIP_CMD install onnxruntime-openvino pillow huggingface_hub numpy tqdm > /dev/null 2>&1
    else
        # CPU用
        $PIP_CMD install onnxruntime pillow huggingface_hub numpy tqdm > /dev/null 2>&1
    fi
}

# 引数なしチェック
if [ $# -eq 0 ]; then
    echo "=========================================="
    echo "   WD14 Tagger Universal - Setup Mode"
    echo "=========================================="
    echo "引数が指定されなかったため、環境構築のみを行います。"
    # CPU, NVIDIA, Intel それぞれの環境を作っておく
    setup_env "cpu"
    # GPU環境等は実機に合わせて作るのが無難じゃが、ここではCPUだけ作って終わる
    "$SCRIPT_DIR/venv_std/bin/python" "$PYTHON_SCRIPT" --gen-config
    echo "[INFO] セットアップ完了。GPU環境は --gpu 指定時に構築されます。"
    exit 0
fi

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --server) PY_ARGS+=("--mode" "server"); shift ;;
        --client) PY_ARGS+=("--mode" "client"); shift ;;
        --organize) DO_ORGANIZE=1; shift ;;
        --tag) DO_TAG=1; shift ;; 
        --no-report) PY_ARGS+=("--no-report"); shift ;;
        --recursive) PY_ARGS+=("--recursive"); shift ;;
        --no-recursive) PY_ARGS+=("--no-recursive"); shift ;;
        -g|--gpu) USE_GPU=1; shift ;;
        --force-intel) USE_GPU=1; FORCE_TYPE="intel"; shift ;;
        --force-nvidia) USE_GPU=1; FORCE_TYPE="nvidia"; shift ;;
        -f|--force) PY_ARGS+=("--force"); shift ;;
        -p|--path) PY_ARGS+=("$2"); shift 2 ;;
        -H|--host) PY_ARGS+=("--host" "$2"); shift 2 ;;
        -P|--port) PY_ARGS+=("--port" "$2"); shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) PY_ARGS+=("$1"); shift ;;
    esac
done

# アクションロジック構築
if [ $DO_ORGANIZE -eq 1 ]; then
    PY_ARGS+=("--organize")
    if [ $DO_TAG -eq 0 ]; then
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
    else
        # 自動判別
        DETECTED=$(detect_gpu_vendor)
        if [ "$DETECTED" = "nvidia" ]; then
            echo "[INFO] NVIDIA GPU を検出しました。CUDAモードで実行します。"
            BACKEND_MODE="nvidia"
        elif [ "$DETECTED" = "intel" ]; then
            echo "[INFO] Intel GPU を検出しました。OpenVINOモードで実行します。"
            BACKEND_MODE="intel"
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

setup_env "$BACKEND_MODE"

echo "[INFO] Pythonスクリプトを実行 ($BACKEND_MODE)..."
"$VENV_DIR/bin/python" "$PYTHON_SCRIPT" "${PY_ARGS[@]}"