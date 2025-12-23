#!/bin/bash

# ==========================================
# WD14 Tagger Universal (Bash Wrapper)
# ==========================================

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PYTHON_SCRIPT="$SCRIPT_DIR/embed_tags_universal.py"

# デフォルト値
USE_GPU=0
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
    echo "  -g, --gpu           GPUを使用する"
    echo "  --organize          フォルダ整理のみ行う（タグ付けOFF）"
    echo "  --tag               タグ付けも行う（--organize併用時）"
    echo "  --no-report         レポート作成なし"
    echo "  --recursive         再帰検索ON"
    echo "  --server            サーバーモード"
    echo "  --client            クライアントモード"
    echo "  -h, --help          ヘルプ表示"
    echo ""
}

# 引数なしチェック
if [ $# -eq 0 ]; then
    echo "=========================================="
    echo "   WD14 Tagger Universal - Setup Mode"
    echo "=========================================="
    echo "引数が指定されなかったため、環境構築のみを行います。"
    setup_env 0
    "$VENV_DIR/bin/python" "$PYTHON_SCRIPT" --gen-config
    echo "[INFO] セットアップ完了。"
    exit 0
fi

setup_env() {
    local use_gpu=$1
    local venv_name="venv_std"
    if [ $use_gpu -eq 1 ]; then venv_name="venv_gpu"; fi
    VENV_DIR="$SCRIPT_DIR/$venv_name"
    if [ ! -d "$VENV_DIR" ]; then
        echo "[INFO] 仮想環境を作成中 ($venv_name)..."
        python3 -m venv "$VENV_DIR"
    fi
    PIP_CMD="$VENV_DIR/bin/pip"
    if [ $use_gpu -eq 1 ]; then
        $PIP_CMD install onnxruntime-gpu pillow huggingface_hub numpy tqdm > /dev/null 2>&1
    else
        $PIP_CMD install onnxruntime pillow huggingface_hub numpy tqdm > /dev/null 2>&1
    fi
}

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
        -g|--gpu) USE_GPU=1; PY_ARGS+=("--gpu"); shift ;;
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
# 通常モード（Organizeなし）ならデフォルトでタグ付けONなので何もしなくて良い

setup_env $USE_GPU

echo "[INFO] Pythonスクリプトを実行..."
"$VENV_DIR/bin/python" "$PYTHON_SCRIPT" "${PY_ARGS[@]}"