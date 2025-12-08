#!/bin/bash

# ==========================================
# AI Tagging Tool for Linux (Bash)
# ==========================================

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PYTHON_SCRIPT="$SCRIPT_DIR/embed_tags_universal.py"

# デフォルト設定
TARGET_PATH="*.webp"
THRESH=0.35
USE_GPU=0
RESUME=0

# ヘルプ表示
show_help() {
    echo "Usage: ./run_tagger.sh [OPTIONS] [PATH]"
    echo ""
    echo "Options:"
    echo "  -p, --path <path>   Target file/folder (default: *.webp)"
    echo "  -t, --thresh <val>  Threshold (default: 0.35)"
    echo "  --gpu               Use GPU (ROCm on Linux)"
    echo "  --resume            Skip processed files"
    echo "  -h, --help          Show this help"
    echo ""
}

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            TARGET_PATH="$2"
            shift 2
            ;;
        -t|--thresh)
            THRESH="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=1
            shift
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # 最後の引数がパスの場合のフォールバック
            if [[ "$1" == *.webp ]] || [[ "$1" == *.jpg ]] || [[ "$1" == *.png ]] || [[ -d "$1" ]]; then
                TARGET_PATH="$1"
            fi
            shift
            ;;
    esac
done

echo "----------------------------------------"
echo " Target: $TARGET_PATH"
echo " GPU Mode: $(if [ $USE_GPU -eq 1 ]; then echo 'ON (ROCm)'; else echo 'OFF (CPU)'; fi)"
echo "----------------------------------------"

# 1. ExifToolのチェック
if ! command -v exiftool &> /dev/null; then
    echo "[ERROR] exiftool not found!"
    echo "Please install it: sudo apt install libimage-exiftool-perl"
    exit 1
fi

# 2. 仮想環境のセットアップ
# GPU(ROCm)なら 'venv_rocm'、CPUなら 'venv_std' に分ける
if [ $USE_GPU -eq 1 ]; then
    VENV_DIR="$SCRIPT_DIR/venv_rocm"
else
    VENV_DIR="$SCRIPT_DIR/venv_std"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# 3. 仮想環境有効化
source "$VENV_DIR/bin/activate"

# 4. パッケージインストール
# GPUモードなら onnxruntime-rocm を入れる
# ※ROCm版はPyPIにある場合と、AMDのリポジトリが必要な場合がある。
#   ここでは標準的なインストールを試みる。
PIP_CMD="$VENV_DIR/bin/pip"
INSTALLED_PKGS=$($PIP_CMD list)

NEEDS_INSTALL=0

if [[ $INSTALLED_PKGS != *"tqdm"* ]]; then
    NEEDS_INSTALL=1
fi

if [ $USE_GPU -eq 1 ]; then
    if [[ $INSTALLED_PKGS != *"onnxruntime-rocm"* ]]; then
        NEEDS_INSTALL=1
    fi
else
    if [[ $INSTALLED_PKGS != *"onnxruntime"* ]]; then
        NEEDS_INSTALL=1
    fi
fi

if [ $NEEDS_INSTALL -eq 1 ]; then
    echo "[INFO] Installing requirements..."
    $PIP_CMD install --upgrade pip
    $PIP_CMD install pillow huggingface_hub numpy tqdm

    if [ $USE_GPU -eq 1 ]; then
        echo "[INFO] Installing onnxruntime-rocm..."
        # 注意: 環境によっては --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-x.x/ が必要かもしれん
        # とりあえず標準で試すが、失敗したら手動で入れてくれというスタンスじゃ
        if ! $PIP_CMD install onnxruntime-rocm; then
             echo "[WARN] 'onnxruntime-rocm' installation failed via standard pip."
             echo "Trying to install standard 'onnxruntime' as fallback, or check AMD documentation."
             $PIP_CMD install onnxruntime
        fi
    else
        $PIP_CMD install onnxruntime
    fi
fi

# 5. 実行
ARGS="$TARGET_PATH --thresh $THRESH"
if [ $USE_GPU -eq 1 ]; then
    ARGS="$ARGS --gpu"
fi
if [ $RESUME -eq 1 ]; then
    ARGS="$ARGS --resume"
fi

echo "[INFO] Starting Tagger..."
python3 "$PYTHON_SCRIPT" $ARGS

echo "[INFO] Done."