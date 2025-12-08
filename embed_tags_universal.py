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
SYSTEM_OS = platform.system()  # 'Windows' or 'Linux'
IS_WINDOWS = (SYSTEM_OS == 'Windows')
IS_LINUX = (SYSTEM_OS == 'Linux')

# ExifToolコマンド設定
if IS_WINDOWS:
    # WindowsならカレントかPATHから
    EXIFTOOL_CMD = "exiftool"
    FS_ENCODING = 'cp932' 
else:
    # LinuxならPATHに入っている前提
    EXIFTOOL_CMD = "exiftool"
    FS_ENCODING = 'utf-8'

# 履歴ファイル名
HISTORY_FILE = "processed_history.txt"

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
    
    # プロバイダ設定
    providers = ['CPUExecutionProvider']
    
    if use_gpu:
        if IS_WINDOWS:
            # Windows + GPU = DirectML
            providers.insert(0, 'DmlExecutionProvider')
            print(f"[INFO] GPU Mode: DirectML (Windows)")
        elif IS_LINUX:
            # Linux + GPU = ROCm
            # ※ROCm環境が整っていないとエラーになる可能性があるが、その場合はCPUにフォールバックされることが多い
            providers.insert(0, 'ROCMExecutionProvider')
            print(f"[INFO] GPU Mode: ROCm (Linux)")
    
    # ログ抑制
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    
    try:
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        print(f"[INFO] Active Providers: {sess.get_providers()}")
    except Exception as e:
        print(f"[WARN] Failed to load requested provider. Error: {e}")
        print("[WARN] Falling back to CPU.")
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

    return sess, tags

def load_history():
    processed = set()
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                processed.add(line.strip())
    return processed

def append_history(file_path):
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"{file_path}\n")
    except Exception:
        pass

def write_xmp_passthrough_safe(image_path, tags_list):
    if not tags_list:
        return False

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
            EXIFTOOL_CMD,
            "-overwrite_original",
            "-P",
            "-m",
            "-sep", ", ", 
            f"-XMP:Subject={tags_str}",
            temp_path 
        ]

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding=FS_ENCODING,
            errors='ignore'
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

def main():
    parser = argparse.ArgumentParser(
        description="Auto-tagging Universal (ROCm/DirectML supported).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("images", nargs='*', help="List of image paths.")
    parser.add_argument("--thresh", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU (DirectML on Win, ROCm on Linux)")
    parser.add_argument("--resume", action="store_true", help="Skip files listed in history")
    args = parser.parse_args()

    if not args.images:
        parser.print_help()
        sys.exit(0)

    processed_files = set()
    if args.resume:
        processed_files = load_history()
        print(f"[INFO] Resuming... {len(processed_files)} processed files loaded.")

    target_files = []
    for path_str in args.images:
        expanded = glob.glob(path_str)
        if expanded:
            target_files.extend(expanded)
        else:
            target_files.append(path_str)

    target_files = sorted(list(set(target_files)))
    
    if args.resume:
        before_count = len(target_files)
        target_files = [f for f in target_files if f not in processed_files]
        skipped_count = before_count - len(target_files)
        if skipped_count > 0:
            print(f"[INFO] Skipped {skipped_count} files.")

    if not target_files:
        print("No new files found.")
        return

    print("Loading model...")
    try:
        # --gpu フラグがついていればGPUモードを試行
        sess, tags = load_model_and_tags(use_gpu=args.gpu)
    except Exception as e:
        print(f"\n[Error] Model load failed: {e}")
        sys.exit(1)

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    print(f"Start processing {len(target_files)} images...")
    
    for img_path in tqdm(target_files, unit="img", ncols=80):
        if not os.path.exists(img_path) or not os.path.isfile(img_path):
            continue
        
        try:
            pil_image = Image.open(img_path)
            img_input = preprocess(pil_image)
            probs = sess.run([label_name], {input_name: img_input})[0][0]
            
            detected_tags = []
            for i, p in enumerate(probs):
                if p > args.thresh:
                    detected_tags.append(tags[i])
            
            # タグ書き込み
            if detected_tags:
                if write_xmp_passthrough_safe(img_path, detected_tags):
                    if args.resume: append_history(img_path)
            else:
                if args.resume: append_history(img_path)

        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)
        except Exception as e:
            tqdm.write(f"Error {os.path.basename(img_path)}: {e}")

if __name__ == "__main__":
    main()