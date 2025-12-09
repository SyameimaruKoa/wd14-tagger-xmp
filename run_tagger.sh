#!/bin/bash

# ==========================================
# AI Tagging Tool for Linux (Bash) - Fixed
# ==========================================

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PYTHON_SCRIPT="$SCRIPT_DIR/embed_tags_universal.py"

# デフォルト設定
THRESH=0.35
USE_GPU=0
RESUME=0
# ファイルリストを格納する配列
declare -a TARGET_FILES=()

# ヘルプ表示
show_help() {
    echo "Usage: ./run_tagger.sh [OPTIONS] [PATH...]"
    echo ""
    echo "Options:"
    echo "  -p, --path <path>   Target file/folder (wildcards allowed)"
    echo "  -t, --thresh <val>  Threshold (default: 0.35)"
    echo "  -g, --gpu           Use GPU (ROCm on Linux)"
    echo "  -r, --resume        Skip processed files"
    echo "  -h, --help          Show this help"
    echo ""
}

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            # 次の引数がオプションでなければ追加
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
        -g|--gpu|-gpu) # -gpu も許容
            USE_GPU=1
            shift
            ;;
        -r|--resume|-resume) # -resume も許容
            RESUME=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            # オプション以外の引数はすべてファイルパスとして配列に追加
            TARGET_FILES+=("$1")
            shift
            ;;
    esac
done

# もし配列が空ならデフォルト(*.webp)を設定
if [ ${#TARGET_FILES[@]} -eq 0 ]; then
    TARGET_FILES=("*.webp")
fi

echo "----------------------------------------"
echo " Target Files: ${#TARGET_FILES[@]} args passed"
echo " GPU Mode: $(if [ $USE_GPU -eq 1 ]; then echo 'ON (ROCm)'; else echo 'OFF (CPU)'; fi)"
echo "----------------------------------------"

# 1. ExifToolのチェック
if ! command -v exiftool &> /dev/null; then
    echo "[ERROR] exiftool not found!"
    echo "Please install it: sudo apt install libimage-exiftool-perl"
    exit 1
fi

# 2. 仮想環境のセットアップ
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
        echo "[INFO] Installing onnxruntime-rocm..."
        # 標準インストール試行
        if ! $PIP_CMD install onnxruntime-rocm; then
             echo "[WARN] 'onnxruntime-rocm' install failed via standard pip."
             echo "Falling back to CPU 'onnxruntime'."
             $PIP_CMD install onnxruntime
        fi
    else
        $PIP_CMD install onnxruntime
    fi
fi

# 5. 実行
# 配列をそのままPythonに渡す
ARGS="--thresh $THRESH"
if [ $USE_GPU -eq 1 ]; then ARGS="$ARGS --gpu"; fi
if [ $RESUME -eq 1 ]; then ARGS="$ARGS --resume"; fi

echo "[INFO] Starting Tagger..."
# "${TARGET_FILES[@]}" で配列を展開して渡す
python3 "$PYTHON_SCRIPT" "${TARGET_FILES[@]}" $ARGS

echo "[INFO] Done."