#!/bin/bash

# ==========================================
# WD14 Tagger Universal (Bash Wrapper)
# ==========================================

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PYTHON_SCRIPT="$SCRIPT_DIR/embed_tags_universal.py"

# デフォルト設定
MODE="standalone"
THRESH=0.35
USE_GPU=0
FORCE_MODE=0
HOST_IP="localhost"
PORT=5000
declare -a TARGET_FILES=()

# ヘルプ表示関数 (今回はちゃんと定義するぞ)
show_help() {
    echo "Usage: ./run_tagger.sh [OPTIONS] [PATH...]"
    echo ""
    echo "Modes:"
    echo "  (default)           Standalone mode (process locally)"
    echo "  --server            Start GPU Server mode"
    echo "  --client            Start Client mode"
    echo ""
    echo "Options:"
    echo "  -p, --path <path>   Target file/folder"
    echo "  -t, --thresh <val>  Threshold (default: 0.35)"
    echo "  -g, --gpu           Use GPU (ROCm on Linux / DirectML on Windows)"
    echo "  -f, --force         Force overwrite existing tags"
    echo "  -H, --host <ip>     Server IP (Client mode)"
    echo "  -P, --port <val>    Server Port (default: 5000)"
    echo "  -h, --help          Show this help"
    echo ""
}

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            MODE="server"
            shift
            ;;
        --client)
            MODE="client"
            shift
            ;;
        -H|--host)
            HOST_IP="$2"
            shift 2
            ;;
        -P|--port)
            PORT="$2"
            shift 2
            ;;
        -p|--path)
            if [[ "$2" != -* ]] && [[ -n "$2" ]]; then
                TARGET_FILES+=("$2")
                shift 2
            else
                shift
            fi
            ;;
        -t|--thresh)
            THRESH="$2"
            shift 2
            ;;
        -g|--gpu|-gpu)
            USE_GPU=1
            shift
            ;;
        -f|--force|--force)
            FORCE_MODE=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # オプション以外の引数はファイルパスとして扱う
            TARGET_FILES+=("$1")
            shift
            ;;
    esac
done

# デフォルトファイル設定 (サーバーモード以外でファイル指定がない場合)
if [ ${#TARGET_FILES[@]} -eq 0 ] && [ "$MODE" != "server" ]; then
    TARGET_FILES=("*.webp")
fi

echo "----------------------------------------"
echo " Mode: $MODE"
if [ "$MODE" != "server" ]; then
    echo " Target: ${#TARGET_FILES[@]} files/patterns"
fi
echo " GPU: $(if [ $USE_GPU -eq 1 ]; then echo 'ON'; else echo 'OFF'; fi)"
echo "----------------------------------------"

# 1. venv作成
# Linuxでは python3-venv が入っていないとここでコケる
if [ $USE_GPU -eq 1 ]; then
    VENV_DIR="$SCRIPT_DIR/venv_gpu"
else
    VENV_DIR="$SCRIPT_DIR/venv_std"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment at $VENV_DIR ..."
    # エラーハンドリングを追加
    if ! python3 -m venv "$VENV_DIR"; then
        echo ""
        echo "[ERROR] Failed to create venv."
        echo "It seems 'python3-venv' is missing."
        echo "Please run: sudo apt install python3-venv"
        echo ""
        exit 1
    fi
fi

# 2. Activate
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "[ERROR] Activation script not found at $VENV_DIR/bin/activate"
    echo "Please delete the '$VENV_DIR' folder and try again."
    exit 1
fi

# 3. Requirements
PIP_CMD="$VENV_DIR/bin/pip"
INSTALLED_PKGS=$($PIP_CMD list)
NEEDS_INSTALL=0

if [[ $INSTALLED_PKGS != *"tqdm"* ]]; then NEEDS_INSTALL=1; fi

if [ $USE_GPU -eq 1 ]; then
    if [[ $INSTALLED_PKGS != *"onnxruntime-rocm"* ]] && [[ $INSTALLED_PKGS != *"onnxruntime-gpu"* ]]; then
        NEEDS_INSTALL=1
    fi
else
    if [[ $INSTALLED_PKGS != *"onnxruntime"* ]]; then NEEDS_INSTALL=1; fi
fi

if [ $NEEDS_INSTALL -eq 1 ]; then
    echo "[INFO] Installing requirements..."
    $PIP_CMD install --upgrade pip
    $PIP_CMD install pillow huggingface_hub numpy tqdm

    if [ $USE_GPU -eq 1 ]; then
        echo "[INFO] Attempting to install ROCm runtime..."
        # ROCmのインストールを試みる
        if ! $PIP_CMD install onnxruntime-rocm; then
             echo "[WARN] 'onnxruntime-rocm' install failed. Trying standard 'onnxruntime'..."
             $PIP_CMD install onnxruntime
        fi
    else
        $PIP_CMD install onnxruntime
    fi
fi

# 4. 実行
ARGS="--mode $MODE"

if [ "$MODE" == "server" ]; then
    ARGS="$ARGS --port $PORT"
    if [ $USE_GPU -eq 1 ]; then ARGS="$ARGS --gpu"; fi
    
    echo "[INFO] Starting Server..."
    python3 "$PYTHON_SCRIPT" $ARGS

elif [ "$MODE" == "client" ]; then
    ARGS="$ARGS --host $HOST_IP --port $PORT --thresh $THRESH"
    if [ $FORCE_MODE -eq 1 ]; then ARGS="$ARGS --force"; fi
    
    echo "[INFO] Starting Client..."
    python3 "$PYTHON_SCRIPT" "${TARGET_FILES[@]}" $ARGS

else
    # Standalone
    ARGS="$ARGS --thresh $THRESH"
    if [ $USE_GPU -eq 1 ]; then ARGS="$ARGS --gpu"; fi
    if [ $FORCE_MODE -eq 1 ]; then ARGS="$ARGS --force"; fi
    
    echo "[INFO] Starting Tagger (Standalone)..."
    python3 "$PYTHON_SCRIPT" "${TARGET_FILES[@]}" $ARGS
fi

echo "[INFO] Done."