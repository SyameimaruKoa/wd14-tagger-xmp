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
import json
import io
import socket
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download

# サーバー/クライアント用
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import urllib.error

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

HISTORY_FILE = "processed_history.txt"
VALID_EXTS = ('.webp', '.jpg', '.jpeg', '.png', '.bmp')

# --- モデル読み込み関数 ---
def load_model_and_tags(use_gpu=False):
    repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
    tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
    tags = []
    with open(tags_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        tags = [row[1] for row in reader]
    
    # プロバイダの優先順位設定
    # 利用可能なものを自動で上から試行する
    providers = []
    if use_gpu:
        if IS_WINDOWS:
            # Windows: DirectML(AMD/Intel), CUDA(NVIDIA), OpenVINO(Intel)
            providers.extend(['DmlExecutionProvider', 'CUDAExecutionProvider', 'OpenVINOExecutionProvider'])
        elif IS_LINUX:
            # Linux: ROCm(AMD), CUDA(NVIDIA), OpenVINO(Intel)
            providers.extend(['ROCMExecutionProvider', 'CUDAExecutionProvider', 'OpenVINOExecutionProvider'])
    
    # 最後に必ずCPUを入れる
    providers.append('CPUExecutionProvider')
    
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    
    print(f"[INFO] Attempting providers: {providers}")
    try:
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        print(f"[INFO] Active Providers: {sess.get_providers()}")
    except Exception as e:
        print(f"[WARN] Failed to load requested provider. Error: {e}")
        print("[INFO] Fallback to CPU.")
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

    return sess, tags, input_name_cache, label_name_cache

# キャッシュ用グローバル変数
sess_global = None
tags_global = None
input_name_cache = None
label_name_cache = None

def init_global_model(use_gpu):
    global sess_global, tags_global, input_name_cache, label_name_cache
    if sess_global is None:
        sess_global, tags_global, _, _ = load_model_and_tags(use_gpu)
        input_name_cache = sess_global.get_inputs()[0].name
        label_name_cache = sess_global.get_outputs()[0].name

def preprocess(image, size=448):
    image = image.convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    img_np = np.array(image).astype(np.float32)
    img_np = img_np[:, :, ::-1]
    img_np = np.expand_dims(img_np, 0)
    return img_np

# --- 共通ツール関数 ---
def has_xmp_tags(image_path):
    try:
        cmd = [EXIFTOOL_CMD, "-XMP:Subject", "-s3", "-fast", image_path]
        startupinfo = None
        if IS_WINDOWS:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding=FS_ENCODING, errors='ignore', startupinfo=startupinfo
        )
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
            cmd, capture_output=True, text=True, encoding=FS_ENCODING, errors='ignore', startupinfo=startupinfo
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

# --- サーバーモード用クラス ---
class TagServerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # 画像処理
            img = Image.open(io.BytesIO(post_data))
            img_input = preprocess(img)
            
            # グローバルモデル使用
            probs = sess_global.run([label_name_cache], {input_name_cache: img_input})[0][0]
            
            response_data = json.dumps(probs.astype(float).tolist())
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(response_data.encode('utf-8'))
            
            # ログ表示 (進捗バーがないのでprintで良い)
            print(f"[Request] From {self.client_address[0]} - {content_length} bytes processed")

        except Exception as e:
            print(f"[Error] Processing request: {e}")
            self.send_response(500)
            self.end_headers()

def run_server(port, use_gpu):
    init_global_model(use_gpu)
    server_address = ('0.0.0.0', port) # Tailscale含む全インターフェースで待機
    httpd = HTTPServer(server_address, TagServerHandler)
    
    # IP表示
    hostname = socket.gethostname()
    print(f"\n[INFO] Server running on Port {port}")
    print(f"[INFO] Hostname: {hostname}")
    print(f"[INFO] Ready to accept requests from Tailscale/LAN.")
    print(f"[INFO] Press Ctrl+C to stop.\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped.")

# --- クライアントモード用関数 ---
def run_client(target_files, host, port, thresh, force):
    url = f"http://{host}:{port}"
    print(f"[INFO] Connecting to Server: {url}")
    
    # タグリスト取得（サーバーからではなくローカル定義を使用）
    # ※サーバー・クライアント間でselected_tags.csvが一致している前提
    repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
    tags = []
    with open(tags_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        tags = [row[1] for row in reader]

    processed_count = 0
    skipped_count = 0
    
    pbar = tqdm(target_files, unit="img", ncols=80)
    for img_path in pbar:
        try:
            if not force:
                if has_xmp_tags(img_path):
                    skipped_count += 1
                    continue

            # ファイル読み込み
            with open(img_path, 'rb') as f:
                img_data = f.read()

            # 送信
            req = urllib.request.Request(url, data=img_data, method='POST')
            req.add_header('Content-Type', 'application/octet-stream')
            
            with urllib.request.urlopen(req) as res:
                if res.status != 200:
                    tqdm.write(f"Server Error: {res.status}")
                    continue
                response_body = res.read()
                probs = json.loads(response_body.decode('utf-8'))

            # タグ判定
            detected_tags = []
            for i, p in enumerate(probs):
                if p > thresh:
                    detected_tags.append(tags[i])
            
            if detected_tags:
                if write_xmp_passthrough_safe(img_path, detected_tags):
                    processed_count += 1
        
        except urllib.error.URLError as e:
            tqdm.write(f"Connection Error: {e}")
            break
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)
        except Exception as e:
            tqdm.write(f"Error {os.path.basename(img_path)}: {e}")
            
    print(f"\n[Done] Processed: {processed_count}, Skipped: {skipped_count}")


# --- メイン処理 ---
def main():
    parser = argparse.ArgumentParser(description="WD14 Tagger Universal (Standalone / Server / Client)")
    
    # モード選択
    parser.add_argument("--mode", choices=['standalone', 'server', 'client'], default='standalone', help="Operation mode")
    
    # 共通オプション
    parser.add_argument("images", nargs='*', help="Images (Standalone/Client only)")
    parser.add_argument("--thresh", type=float, default=0.35)
    parser.add_argument("--gpu", action="store_true", help="Enable GPU (Standalone/Server only)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing tags (Standalone/Client only)")
    
    # サーバー・クライアント用オプション
    parser.add_argument("--host", default="localhost", help="Server IP (Client mode)")
    parser.add_argument("--port", type=int, default=5000, help="Server Port")
    
    args = parser.parse_args()

    # モード別分岐
    if args.mode == 'server':
        # サーバー起動 (画像引数は無視)
        run_server(args.port, args.gpu)

    elif args.mode == 'client':
        if not args.images:
            print("Error: No images specified for client mode.")
            return
        files = collect_images(args.images)
        run_client(files, args.host, args.port, args.thresh, args.force)

    else:
        # Standalone (デフォルト)
        if not args.images:
            parser.print_help()
            return
        
        files = collect_images(args.images)
        if not files:
            print("No files found.")
            return

        print("Loading model...")
        init_global_model(args.gpu)
        
        processed = 0
        skipped = 0
        pbar = tqdm(files, unit="img", ncols=80)
        
        for img_path in pbar:
            try:
                if not args.force:
                    if has_xmp_tags(img_path):
                        skipped += 1
                        continue
                
                pil_image = Image.open(img_path)
                img_input = preprocess(pil_image)
                probs = sess_global.run([label_name_cache], {input_name_cache: img_input})[0][0]
                
                detected_tags = []
                for i, p in enumerate(probs):
                    if p > args.thresh:
                        detected_tags.append(tags_global[i])
                
                if detected_tags:
                    if write_xmp_passthrough_safe(img_path, detected_tags):
                        processed += 1
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                tqdm.write(f"Error {os.path.basename(img_path)}: {e}")
        
        print(f"\n[Done] Processed: {processed}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
