#!/bin/bash

# ==========================================
# WD14 Tagger Universal (Bash Wrapper)
# Auto-detects NVIDIA/AMD for Linux
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

show_help() {
    echo "Usage: ./run_tagger.sh [OPTIONS] [PATH...]"
    echo ""
    echo "Modes:"
    echo "  (default)           Standalone mode"
    echo "  --server            Start GPU Server mode"
    echo "  --client            Start Client mode"
    echo ""
    echo "Options:"
    echo "  -p, --path <path>   Target file/folder"
    echo "  -t, --thresh <val>  Threshold (default: 0.35)"
    echo "  -g, --gpu           Use GPU (Auto-detect CUDA/ROCm/DirectML)"
    echo "  -f, --force         Force overwrite"
    echo "  -H, --host <ip>     Server IP"
    echo "  -P, --port <val>    Server Port"
    echo "  -h, --help          Show this help"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --server) MODE="server"; shift ;;
        --client) MODE="client"; shift ;;
        -H|--host) HOST_IP="$2"; shift 2 ;;
        -P|--port) PORT="$2"; shift 2 ;;
        -p|--path) 
            if [[ "$2" != -* ]] && [[ -n "$2" ]]; then TARGET_FILES+=("$2"); shift 2; else shift; fi ;;
        -t|--thresh) THRESH="$2"; shift 2 ;;
        -g|--gpu|-gpu) USE_GPU=1; shift ;;
        -f|--force|--force) FORCE_MODE=1; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) TARGET_FILES+=("$1"); shift ;;
    esac
done

if [ ${#TARGET_FILES[@]} -eq 0 ] && [ "$MODE" != "server" ]; then
    TARGET_FILES=("*.webp")
fi

echo "----------------------------------------"
echo " Mode: $MODE"
echo " GPU: $(if [ $USE_GPU -eq 1 ]; then echo 'ON'; else echo 'OFF'; fi)"
echo "----------------------------------------"

# 1. venv作成
if [ $USE_GPU -eq 1 ]; then
    VENV_DIR="$SCRIPT_DIR/venv_gpu"
else
    VENV_DIR="$SCRIPT_DIR/venv_std"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment at $VENV_DIR ..."
    if ! python3 -m venv "$VENV_DIR"; then
        echo "[ERROR] Failed to create venv. Try: sudo apt install python3-venv"
        exit 1
    fi
fi

# 2. Activate
source "$VENV_DIR/bin/activate"

# 3. Requirements & GPU Logic
PIP_CMD="$VENV_DIR/bin/pip"
INSTALLED_PKGS=$($PIP_CMD list)
NEEDS_INSTALL=0

if [[ $INSTALLED_PKGS != *"tqdm"* ]]; then NEEDS_INSTALL=1; fi

# GPUライブラリのチェック
if [ $USE_GPU -eq 1 ]; then
    # 既に何らかのGPU版が入っていればOKとする
    if [[ $INSTALLED_PKGS != *"onnxruntime-gpu"* ]] && \
       [[ $INSTALLED_PKGS != *"onnxruntime-rocm"* ]] && \
       [[ $INSTALLED_PKGS != *"onnxruntime-openvino"* ]]; then
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
        # ★ここが改良点：GPUメーカー自動判別★
        if command -v nvidia-smi &> /dev/null; then
            echo "[INFO] NVIDIA GPU detected. Installing 'onnxruntime-gpu' (CUDA)..."
            $PIP_CMD install onnxruntime-gpu
        elif command -v rocminfo &> /dev/null; then
            echo "[INFO] AMD GPU detected. Installing 'onnxruntime-rocm'..."
            if ! $PIP_CMD install onnxruntime-rocm; then
                 echo "[WARN] ROCm install failed. Fallback to standard onnxruntime."
                 $PIP_CMD install onnxruntime
            fi
        else
            echo "[WARN] No specific GPU tool found (nvidia-smi/rocminfo missing)."
            echo "[INFO] Installing standard 'onnxruntime-gpu' hoping for the best..."
            $PIP_CMD install onnxruntime-gpu
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
    python3 "$PYTHON_SCRIPT" $ARGS
elif [ "$MODE" == "client" ]; then
    ARGS="$ARGS --host $HOST_IP --port $PORT --thresh $THRESH"
    if [ $FORCE_MODE -eq 1 ]; then ARGS="$ARGS --force"; fi
    python3 "$PYTHON_SCRIPT" "${TARGET_FILES[@]}" $ARGS
else
    ARGS="$ARGS --thresh $THRESH"
    if [ $USE_GPU -eq 1 ]; then ARGS="$ARGS --gpu"; fi
    if [ $FORCE_MODE -eq 1 ]; then ARGS="$ARGS --force"; fi
    python3 "$PYTHON_SCRIPT" "${TARGET_FILES[@]}" $ARGS
fi

echo "[INFO] Done."