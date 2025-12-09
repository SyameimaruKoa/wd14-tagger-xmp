#!/bin/bash

# ==========================================
# AI Tagging Tool for Linux (Bash)
# Default behavior: Skip tagged files (Resume)
# ==========================================

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PYTHON_SCRIPT="$SCRIPT_DIR/embed_tags_universal.py"

# デフォルト設定
THRESH=0.35
USE_GPU=0
FORCE_MODE=0
declare -a TARGET_FILES=()

# ヘルプ表示
show_help() {
    echo "Usage: ./run_tagger.sh [OPTIONS] [PATH...]"
    echo ""
    echo "Options:"
    echo "  -p, --path <path>   Target file/folder"
    echo "  -t, --thresh <val>  Threshold (default: 0.35)"
    echo "  -g, --gpu           Use GPU (ROCm on Linux)"
    echo "  -f, --force         Force re-process all files"
    echo "  -h, --help          Show this help"
    echo ""
}

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
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
        -f|--force|-force)
            FORCE_MODE=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            TARGET_FILES+=("$1")
            shift
            ;;
    esac
done

if [ ${#TARGET_FILES[@]} -eq 0 ]; then
    TARGET_FILES=("*.webp")
fi

echo "----------------------------------------"
echo " Target Files: ${#TARGET_FILES[@]} args passed"
echo " GPU Mode: $(if [ $USE_GPU -eq 1 ]; then echo 'ON (ROCm)'; else echo 'OFF (CPU)'; fi)"
echo " Force Mode: $(if [ $FORCE_MODE -eq 1 ]; then echo 'ON'; else echo 'OFF (Skip tagged)'; fi)"
echo "----------------------------------------"

# 1. ExifTool
if ! command -v exiftool &> /dev/null; then
    echo "[ERROR] exiftool not found!"
    exit 1
fi

# 2. venv
if [ $USE_GPU -eq 1 ]; then
    VENV_DIR="$SCRIPT_DIR/venv_rocm"
else
    VENV_DIR="$SCRIPT_DIR/venv_std"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# 3. Activate
source "$VENV_DIR/bin/activate"

# 4. Requirements
PIP_CMD="$VENV_DIR/bin/pip"
INSTALLED_PKGS=$($PIP_CMD list)
NEEDS_INSTALL=0

if [[ $INSTALLED_PKGS != *"tqdm"* ]]; then NEEDS_INSTALL=1; fi

if [ $USE_GPU -eq 1 ]; then
    if [[ $INSTALLED_PKGS != *"onnxruntime-rocm"* ]]; then NEEDS_INSTALL=1; fi
else
    if [[ $INSTALLED_PKGS != *"onnxruntime"* ]]; then NEEDS_INSTALL=1; fi
fi

if [ $NEEDS_INSTALL -eq 1 ]; then
    echo "[INFO] Installing requirements..."
    $PIP_CMD install --upgrade pip
    $PIP_CMD install pillow huggingface_hub numpy tqdm
    if [ $USE_GPU -eq 1 ]; then
        if ! $PIP_CMD install onnxruntime-rocm; then
             $PIP_CMD install onnxruntime
        fi
    else
        $PIP_CMD install onnxruntime
    fi
fi

# 5. Run
ARGS="--thresh $THRESH"
if [ $USE_GPU -eq 1 ]; then ARGS="$ARGS --gpu"; fi
if [ $FORCE_MODE -eq 1 ]; then ARGS="$ARGS --force"; fi

echo "[INFO] Starting Tagger..."
python3 "$PYTHON_SCRIPT" "${TARGET_FILES[@]}" $ARGS

echo "[INFO] Done."