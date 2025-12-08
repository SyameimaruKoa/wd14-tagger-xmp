import argparse
import csv
import os
import sys
import subprocess
import glob
import numpy as np
import uuid
import shutil
import unicodedata
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download

# 進捗バー用
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# --- 設定 ---
EXIFTOOL_CMD = "exiftool"

def preprocess(image, size=448):
    image = image.convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    img_np = np.array(image).astype(np.float32)
    img_np = img_np[:, :, ::-1]
    img_np = np.expand_dims(img_np, 0)
    return img_np

def load_model_and_tags():
    repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
    tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
    tags = []
    with open(tags_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        tags = [row[1] for row in reader]
    
    # CPU版の設定
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return sess, tags

def write_xmp_passthrough_safe(image_path, tags_list):
    if not tags_list:
        return

    tags_str = ", ".join(tags_list)
    abs_path = os.path.abspath(image_path)
    dir_name = os.path.dirname(abs_path)
    
    # ★修正点：元のファイルの拡張子を取得して一時ファイルに使う
    _, ext = os.path.splitext(abs_path)
    
    temp_name = f"temp_{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(dir_name, temp_name)

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
            encoding='cp932',
            errors='ignore'
        )
        
        if result.returncode != 0:
            tqdm.write(f"ExifTool Error ({os.path.basename(image_path)}): {result.stderr.strip()}")

    except OSError as e:
        tqdm.write(f"Rename Error: {e}")
        return
    except Exception as e:
        tqdm.write(f"Error: {e}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.rename(temp_path, abs_path)
            except OSError:
                tqdm.write(f"CRITICAL: Failed to restore {temp_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Auto-tagging for CPU - Multi-format support.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("images", nargs='*', help="List of image paths (wildcards supported).")
    parser.add_argument("--thresh", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    if not args.images:
        parser.print_help()
        sys.exit(0)

    target_files = []
    for path_str in args.images:
        expanded = glob.glob(path_str)
        if expanded:
            target_files.extend(expanded)
        else:
            target_files.append(path_str)

    target_files = sorted(list(set(target_files)))
    if not target_files:
        print("No files found.")
        return

    print("Loading model (CPU)... (Please wait)")
    sess, tags = load_model_and_tags()
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
            
            if detected_tags:
                write_xmp_passthrough_safe(img_path, detected_tags)

        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)
        except Exception as e:
            tqdm.write(f"Error {os.path.basename(img_path)}: {e}")

if __name__ == "__main__":
    main()