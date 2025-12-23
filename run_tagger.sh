#!/bin/bash

# ==========================================
# WD14 Tagger Universal (Bash Wrapper)
# ==========================================

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PYTHON_SCRIPT="$SCRIPT_DIR/embed_tags_universal.py"

MODE="standalone"
THRESH=0.35
RATING_THRESH=""
USE_GPU=0
FORCE_MODE=0
ORGANIZE_MODE=0
HOST_IP="localhost"
PORT=5000
declare -a TARGET_FILES=()

show_help() {
    echo "Usage: ./run_tagger.sh [OPTIONS] [PATH...]"
    echo ""
    echo "Modes:"
    echo "  --server            Start GPU Server mode"
    echo "  --client            Start Client mode"
    echo ""
    echo "Options:"
    echo "  -g, --gpu           Use GPU (Auto-detect)"
    echo "  -p, --path <path>   Target file/folder"
    echo "  -H, --host <ip>     Server IP"
    echo "  --organize          Move files to folders based on rating"
    echo "  --rating-thresh <v> Min confidence for sensitive/questionable/explicit"
    echo "  -f, --force         Force overwrite tags"
    echo "  -h, --help          Show help"
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
        --rating-thresh) RATING_THRESH="$2"; shift 2 ;;
        -g|--gpu|-gpu) USE_GPU=1; shift ;;
        -f|--force|--force) FORCE_MODE=1; shift ;;
        --organize) ORGANIZE_MODE=1; shift ;;
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

# 0. ExifToolチェック & 自動インストール (Linux)
if [ "$(uname)" == "Linux" ]; then
    if ! command -v exiftool &> /dev/null; then
        echo "[INFO] ExifTool not found. Checking installation method..."
        if command -v apt-get &> /dev/null; then
            echo "[INFO] Installing libimage-exiftool-perl via apt..."
            sudo apt-get update && sudo apt-get install -y libimage-exiftool-perl
        else
            echo "[WARN] Automatic installation not supported on this OS. Please install ExifTool manually."
        fi
    fi
fi

# 1. venv作成
if [ $USE_GPU -eq 1 ]; then
    VENV_DIR="$SCRIPT_DIR/venv_gpu"
else
    VENV_DIR="$SCRIPT_DIR/venv_std"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment..."
    if ! python3 -m venv "$VENV_DIR"; then
        echo "[ERROR] 'python3-venv' is missing. Run: sudo apt install python3-venv"
        exit 1
    fi
fi

source "$VENV_DIR/bin/activate"
PIP_CMD="$VENV_DIR/bin/pip"
INSTALLED_PKGS=$($PIP_CMD list)
NEEDS_INSTALL=0

if [[ $INSTALLED_PKGS != *"tqdm"* ]]; then NEEDS_INSTALL=1; fi
if [ $USE_GPU -eq 1 ]; then
    if [[ $INSTALLED_PKGS != *"onnxruntime-gpu"* ]] && [[ $INSTALLED_PKGS != *"onnxruntime-rocm"* ]]; then
        NEEDS_INSTALL=1
    fi
    # NVIDIAチェック
    if command -v nvidia-smi &> /dev/null; then
        if [[ $INSTALLED_PKGS != *"nvidia-cuda-runtime-cu12"* ]]; then NEEDS_INSTALL=1; fi
    fi
else
    if [[ $INSTALLED_PKGS != *"onnxruntime"* ]]; then NEEDS_INSTALL=1; fi
fi

# 2. パッケージインストール
if [ $NEEDS_INSTALL -eq 1 ]; then
    echo "[INFO] Installing requirements..."
    $PIP_CMD install --upgrade pip
    $PIP_CMD install pillow huggingface_hub numpy tqdm

    if [ $USE_GPU -eq 1 ]; then
        if command -v nvidia-smi &> /dev/null; then
            echo "[INFO] NVIDIA GPU detected."
            $PIP_CMD install onnxruntime-gpu
            echo "[INFO] Installing NVIDIA CUDA libraries..."
            $PIP_CMD install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 nvidia-cufft-cu12
        elif command -v rocminfo &> /dev/null; then
            echo "[INFO] AMD GPU detected."
            if ! $PIP_CMD install onnxruntime-rocm; then
                 $PIP_CMD install onnxruntime
            fi
        else
            echo "[WARN] No GPU tool detected. Installing onnxruntime-gpu anyway..."
            $PIP_CMD install onnxruntime-gpu
        fi
    else
        $PIP_CMD install onnxruntime
    fi
fi

# 3. LD_LIBRARY_PATHの自動補正 (NVIDIAの場合)
if [ $USE_GPU -eq 1 ] && command -v nvidia-smi &> /dev/null; then
    SITE_PACKAGES=$($VENV_DIR/bin/python3 -c "import site; print(site.getsitepackages()[0])")
    for lib_dir in $SITE_PACKAGES/nvidia/*/lib; do
        if [ -d "$lib_dir" ]; then
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$lib_dir"
        fi
    done
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
    if [ $ORGANIZE_MODE -eq 1 ]; then ARGS="$ARGS --organize"; fi
    if [ -n "$RATING_THRESH" ]; then ARGS="$ARGS --rating-thresh $RATING_THRESH"; fi
    python3 "$PYTHON_SCRIPT" "${TARGET_FILES[@]}" $ARGS
fi

echo "[INFO] Done."