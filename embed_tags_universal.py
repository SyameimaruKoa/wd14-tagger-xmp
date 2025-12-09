import argparse
import csv
import os
import sys
import subprocess
import glob
import numpy as np
import uuid
import shutil
import platform
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download

# 進捗バー用
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# --- OS判定と設定 ---
SYSTEM_OS = platform.system()
IS_WINDOWS = (SYSTEM_OS == 'Windows')
IS_LINUX = (SYSTEM_OS == 'Linux')

if IS_WINDOWS:
    EXIFTOOL_CMD = "exiftool"
    FS_ENCODING = 'cp932' 
else:
    EXIFTOOL_CMD = "exiftool"
    FS_ENCODING = 'utf-8'

# 対象とする拡張子 (小文字)
VALID_EXTS = ('.webp', '.jpg', '.jpeg', '.png', '.bmp')

def preprocess(image, size=448):
    image = image.convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    img_np = np.array(image).astype(np.float32)
    img_np = img_np[:, :, ::-1]
    img_np = np.expand_dims(img_np, 0)
    return img_np

def load_model_and_tags(use_gpu=False):
    repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
    tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
    tags = []
    with open(tags_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        tags = [row[1] for row in reader]
    
    providers = ['CPUExecutionProvider']
    if use_gpu:
        if IS_WINDOWS:
            providers.insert(0, 'DmlExecutionProvider')
            print(f"[INFO] GPU Mode: DirectML (Windows)")
        elif IS_LINUX:
            providers.insert(0, 'ROCMExecutionProvider')
            print(f"[INFO] GPU Mode: ROCm (Linux)")
    
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    
    try:
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    except Exception as e:
        print(f"[WARN] Failed to load GPU provider. Fallback to CPU. Error: {e}")
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

    return sess, tags

def has_xmp_tags(image_path):
    """
    ExifToolを使ってXMPタグ(Subject)が既に存在するか確認する。
    戻り値: True (タグあり), False (タグなし)
    """
    try:
        # -s3: 値のみ出力, -fast: 高速読み込み
        cmd = [EXIFTOOL_CMD, "-XMP:Subject", "-s3", "-fast", image_path]
        
        # Windowsでのウィンドウポップアップ抑止など
        startupinfo = None
        if IS_WINDOWS:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding=FS_ENCODING, 
            errors='ignore',
            startupinfo=startupinfo
        )
        
        # 出力があればタグが存在する
        if result.returncode == 0 and result.stdout.strip():
            return True
        return False
        
    except Exception:
        return False

def write_xmp_passthrough_safe(image_path, tags_list):
    if not tags_list: return False
    tags_str = ", ".join(tags_list)
    abs_path = os.path.abspath(image_path)
    dir_name = os.path.dirname(abs_path)
    _, ext = os.path.splitext(abs_path)
    temp_name = f"temp_{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(dir_name, temp_name)
    success = False

    try:
        os.rename(abs_path, temp_path)
        cmd = [
            EXIFTOOL_CMD, "-overwrite_original", "-P", "-m", "-sep", ", ", 
            f"-XMP:Subject={tags_str}", temp_path 
        ]
        
        startupinfo = None
        if IS_WINDOWS:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding=FS_ENCODING, 
            errors='ignore',
            startupinfo=startupinfo
        )
        
        if result.returncode != 0:
            tqdm.write(f"ExifTool Error ({os.path.basename(image_path)}): {result.stderr.strip()}")
        else:
            success = True
    except OSError as e:
        tqdm.write(f"Rename Error: {e}")
        return False
    except Exception as e:
        tqdm.write(f"Error: {e}")
        return False
    finally:
        if os.path.exists(temp_path):
            try:
                os.rename(temp_path, abs_path)
            except OSError:
                tqdm.write(f"CRITICAL: Failed to restore {temp_path}")
                success = False
    return success

def collect_images(path_args):
    collected = []
    for p in path_args:
        if '*' in p or '?' in p:
            candidates = glob.glob(p, recursive=True)
        else:
            candidates = [p]

        for candidate in candidates:
            if os.path.isdir(candidate):
                print(f"[INFO] Scanning directory: {candidate}")
                for root, _, files in os.walk(candidate):
                    for f in files:
                        if f.lower().endswith(VALID_EXTS):
                            collected.append(os.path.join(root, f))
            elif os.path.isfile(candidate):
                if candidate.lower().endswith(VALID_EXTS):
                    collected.append(candidate)
    
    return sorted(list(set(collected)))

def main():
    parser = argparse.ArgumentParser(description="Auto-tagging Universal (Check Tags & Recursive).")
    parser.add_argument("images", nargs='*', help="List of image paths or directories.")
    parser.add_argument("--thresh", type=float, default=0.35)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force overwrite even if tags exist")
    args = parser.parse_args()

    if not args.images:
        parser.print_help()
        sys.exit(0)

    # 画像収集
    target_files = collect_images(args.images)
    
    if not target_files:
        print("No files found to process.")
        return

    print("Loading model...")
    try:
        sess, tags = load_model_and_tags(use_gpu=args.gpu)
    except Exception as e:
        print(f"[Error] Model load failed: {e}")
        sys.exit(1)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    print(f"Start processing {len(target_files)} images...")
    
    # スキップカウンター
    skipped_count = 0
    processed_count = 0

    # 進捗バーの設定
    pbar = tqdm(target_files, unit="img", ncols=80)
    
    for img_path in pbar:
        try:
            # --- タグチェック処理 (レジューム判定) ---
            if not args.force:
                # 強制モードでなければ、タグがあるかチェック
                if has_xmp_tags(img_path):
                    skipped_count += 1
                    # 進捗バーの表示を更新してスキップ
                    # pbar.set_description(f"Skip: {os.path.basename(img_path)[:15]}...")
                    continue
            
            # --- ここからAI解析 ---
            pil_image = Image.open(img_path)
            img_input = preprocess(pil_image)
            probs = sess.run([label_name], {input_name: img_input})[0][0]
            
            detected_tags = []
            for i, p in enumerate(probs):
                if p > args.thresh:
                    detected_tags.append(tags[i])
            
            # タグがあれば書き込み
            if detected_tags:
                if write_xmp_passthrough_safe(img_path, detected_tags):
                    processed_count += 1
            
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)
        except Exception as e:
            tqdm.write(f"Error {os.path.basename(img_path)}: {e}")
    
    print(f"\n[Done] Processed: {processed_count}, Skipped (Already Tagged): {skipped_count}")

if __name__ == "__main__":
    main()