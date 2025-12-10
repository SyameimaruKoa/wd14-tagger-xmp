#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
PYTHON_SCRIPT="$SCRIPT_DIR/embed_tags_universal.py"

MODE="standalone"
THRESH=0.35
USE_GPU=0
FORCE_MODE=0
HOST_IP="localhost"
PORT=5000
declare -a TARGET_FILES=()

# ヘルプ省略... (引数解析へ)

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
        -f|--force) FORCE_MODE=1; shift ;;
        *) TARGET_FILES+=("$1"); shift ;;
    esac
done

# デフォルトファイル
if [ ${#TARGET_FILES[@]} -eq 0 ] && [ "$MODE" != "server" ]; then
    TARGET_FILES=("*.webp")
fi

# 1. venv作成 (GPUなら venv_gpu, なければ venv_std)
# ※NVIDIA等を使う場合、ユーザーが自分で venv_gpu/bin/pip install onnxruntime-gpu する必要がある
if [ $USE_GPU -eq 1 ]; then VENV_DIR="$SCRIPT_DIR/venv_gpu"; else VENV_DIR="$SCRIPT_DIR/venv_std"; fi
if [ ! -d "$VENV_DIR" ]; then python3 -m venv "$VENV_DIR"; fi

source "$VENV_DIR/bin/activate"
PIP_CMD="$VENV_DIR/bin/pip"

# 簡易依存チェック
INSTALLED=$($PIP_CMD list)
if [[ $INSTALLED != *"tqdm"* ]]; then
    $PIP_CMD install pillow huggingface_hub numpy tqdm
    if [ $USE_GPU -eq 1 ]; then
        # 自動でROCmを入れようとするが、NVIDIAの人はここで手動インストールが必要かもしれん
        if ! $PIP_CMD install onnxruntime-rocm; then $PIP_CMD install onnxruntime; fi
    else
        $PIP_CMD install onnxruntime
    fi
fi

# 2. 実行
ARGS="--mode $MODE"
if [ "$MODE" == "server" ]; then
    ARGS="$ARGS --port $PORT"
    if [ $USE_GPU -eq 1 ]; then ARGS="$ARGS --gpu"; fi
elif [ "$MODE" == "client" ]; then
    ARGS="$ARGS --host $HOST_IP --port $PORT --thresh $THRESH"
    if [ $FORCE_MODE -eq 1 ]; then ARGS="$ARGS --force"; fi
    # クライアントモードはファイルリストを最後に渡す
    python3 "$PYTHON_SCRIPT" "${TARGET_FILES[@]}" $ARGS
else
    # Standalone
    ARGS="$ARGS --thresh $THRESH"
    if [ $USE_GPU -eq 1 ]; then ARGS="$ARGS --gpu"; fi
    if [ $FORCE_MODE -eq 1 ]; then ARGS="$ARGS --force"; fi
    python3 "$PYTHON_SCRIPT" "${TARGET_FILES[@]}" $ARGS
fi