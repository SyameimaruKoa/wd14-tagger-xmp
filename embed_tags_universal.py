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
import time
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import urllib.error

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

SYSTEM_OS = platform.system()
IS_WINDOWS = (SYSTEM_OS == 'Windows')
IS_LINUX = (SYSTEM_OS == 'Linux')

if IS_WINDOWS:
    EXIFTOOL_CMD = "exiftool"
    FS_ENCODING = 'utf-8' 
else:
    EXIFTOOL_CMD = "exiftool"
    FS_ENCODING = 'utf-8'

VALID_EXTS = ('.webp', '.jpg', '.jpeg', '.png', '.bmp')

# WD14 Rating Tags (Indices 0-3)
RATING_TAGS = ['general', 'sensitive', 'questionable', 'explicit']

# ==========================================
# ★ コンフィグ管理 (config.json) ★
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")

DEFAULT_CONFIG = {
    "server_host": "localhost",
    "server_port": 5000,
    "sensitive_split_threshold": 0.50,
    "folder_names": {
        "general": "R-00",
        "sensitive_mild": "R-15_0",
        "sensitive_high": "R-15_5",
        "questionable": "R-17",
        "explicit": "R-18"
    }
}

def load_config():
    """config.jsonを読み込む。なければデフォルトを作成する。"""
    config = DEFAULT_CONFIG.copy()
    
    if not os.path.exists(CONFIG_FILE):
        print(f"[INFO] Creating default config file: {CONFIG_FILE}")
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Failed to create config file: {e}")
    else:
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # トップレベルのキーを更新
                for key in ["server_host", "server_port", "sensitive_split_threshold"]:
                    if key in user_config:
                        config[key] = user_config[key]
                # フォルダ名は辞書なのでupdate
                if "folder_names" in user_config:
                    config["folder_names"].update(user_config["folder_names"])
                
                print(f"[INFO] Loaded config from {CONFIG_FILE}")
        except Exception as e:
            print(f"[WARN] Failed to load config file: {e}. Using defaults.")
    
    return config

# グローバル設定として保持
APP_CONFIG = load_config()

# ★ ANSIカラー定義 ★
class Colors:
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    GREY = '\033[90m' # 暗い灰色（空のバー用）
    RESET = '\033[0m'

def get_bar(prob, color, width=5):
    """確率に応じたカラーバーを生成する"""
    fill_len = int(prob * width)
    empty_len = width - fill_len
    # 色付きブロック + 灰色ブロック + リセット
    return f"{color}{'█' * fill_len}{Colors.GREY}{'░' * empty_len}{Colors.RESET}"

class ExifToolWrapper:
    def __init__(self, cmd=EXIFTOOL_CMD):
        self.cmd = cmd
        self.process = None
        self.running = False

    def start(self):
        if self.running: return
        try:
            startupinfo = None
            if IS_WINDOWS:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            self.process = subprocess.Popen(
                [self.cmd, "-stay_open", "True", "-@", "-", "-common_args", "-charset", "filename=utf8"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, 
                startupinfo=startupinfo
            )
            self.running = True
        except Exception as e:
            print(f"[ERROR] Failed to start ExifTool: {e}")
            self.running = False

    def stop(self):
        if not self.running: return
        try:
            self.process.stdin.write(b"-stay_open\nFalse\n")
            self.process.stdin.flush()
            self.process.wait(timeout=2)
        except Exception:
            if self.process:
                self.process.kill()
        self.running = False

    def execute(self, args):
        if not self.running:
            self.start()
            if not self.running: return ""

        try:
            for arg in args:
                self.process.stdin.write(arg.encode('utf-8') + b"\n")
            
            self.process.stdin.write(b"-execute\n")
            self.process.stdin.flush()

            output_lines = []
            while True:
                line = self.process.stdout.readline()
                if not line: break 
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str == "{ready}":
                    break
                output_lines.append(line_str)
            
            return "\n".join(output_lines)

        except Exception as e:
            print(f"[Error] ExifTool communication: {e}")
            self.stop()
            return ""

    def get_tags(self, path):
        res = self.execute(["-XMP:Subject", "-s3", "-sep", ", ", "-fast", path])
        if res:
            return [t.strip() for t in res.split(',')]
        return []

    def write_tags(self, path, tags):
        if not tags: return False
        tags_str = ", ".join(tags)
        
        res = self.execute([
            "-overwrite_original", "-P", "-m", "-sep", ", ",
            f"-XMP:Subject={tags_str}",
            path
        ])
        return "image files updated" in res

et_wrapper = ExifToolWrapper()

def get_ip_addresses():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
        ips.append(f"LAN: {lan_ip}")
    except Exception:
        pass

    if IS_LINUX:
        try:
            cmd = ["ip", "-4", "addr", "show"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            for line in res.stdout.split('\n'):
                if "inet" in line and "100." in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        ts_ip = parts[1].split('/')[0]
                        if ts_ip.startswith("100."):
                            ips.append(f"Tailscale: {ts_ip}")
        except Exception:
            pass
    return ips

def load_model_and_tags(use_gpu=False):
    repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
    tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
    tags = []
    with open(tags_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        tags = [row[1] for row in reader]
    
    providers = []
    if use_gpu:
        if IS_WINDOWS:
            providers.extend(['DmlExecutionProvider', 'CUDAExecutionProvider'])
        elif IS_LINUX:
            providers.extend(['CUDAExecutionProvider', 'ROCMExecutionProvider'])
    
    providers.append('CPUExecutionProvider')
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    
    print(f"[INFO] Attempting providers: {providers}")
    try:
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        print(f"[INFO] Active Providers: {sess.get_providers()}")
    except Exception as e:
        print(f"[WARN] Failed to load GPU provider. Error: {e}")
        print("[INFO] Fallback to CPU.")
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

    return sess, tags, sess.get_inputs()[0].name, sess.get_outputs()[0].name

sess_global = None
tags_global = None
input_name_cache = None
label_name_cache = None
req_count = 0

def init_global_model(use_gpu):
    global sess_global, tags_global, input_name_cache, label_name_cache
    if sess_global is None:
        sess_global, tags_global, input_name_cache, label_name_cache = load_model_and_tags(use_gpu)

def preprocess(image, size=448):
    image = image.convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    img_np = np.array(image).astype(np.float32)
    img_np = img_np[:, :, ::-1]
    img_np = np.expand_dims(img_np, 0)
    return img_np

def organize_file(file_path, rating):
    """Moves the file to a subfolder based on rating."""
    if not rating:
        return False
    
    folder_mapping = APP_CONFIG.get("folder_names", {})
    folder_name = folder_mapping.get(rating, rating)

    try:
        abs_path = os.path.abspath(file_path)
        dir_name = os.path.dirname(abs_path)
        file_name = os.path.basename(abs_path)

        target_dir = os.path.join(dir_name, folder_name)
        
        if os.path.abspath(dir_name) == os.path.abspath(target_dir):
            return False

        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, file_name)
        
        if os.path.exists(target_path):
            base, ext = os.path.splitext(file_name)
            target_path = os.path.join(target_dir, f"{base}_{uuid.uuid4().hex[:6]}{ext}")

        shutil.move(abs_path, target_path)
        return True
    except Exception as e:
        tqdm.write(f"[Warn] Failed to move {file_path}: {e}")
        return False

def collect_images(path_args, recursive=True):
    collected = []
    for p in path_args:
        if '*' in p or '?' in p:
            candidates = glob.glob(p, recursive=True)
        else:
            candidates = [p]
        
        for candidate in candidates:
            if os.path.isdir(candidate):
                print(f"[INFO] Scanning directory (Recursive={recursive}): {candidate}")
                if recursive:
                    for root, _, files in os.walk(candidate):
                        for f in files:
                            if f.lower().endswith(VALID_EXTS):
                                collected.append(os.path.join(root, f))
                else:
                    try:
                        for f in os.listdir(candidate):
                            full_path = os.path.join(candidate, f)
                            if os.path.isfile(full_path) and f.lower().endswith(VALID_EXTS):
                                collected.append(full_path)
                    except OSError:
                        pass
            elif os.path.isfile(candidate):
                if candidate.lower().endswith(VALID_EXTS):
                    collected.append(candidate)
    return sorted(list(set(collected)))

def calculate_rating(probs, tags, rating_thresh, split_thresh, ignore_sensitive, fname_disp=""):
    """
    推論結果(probs)からレーティングを決定し、ログを出力する。
    """
    rating_probs = probs[:4]
    
    def fmt_prob(p):
        val = p * 100
        if val >= 100: return "100.0%"
        return f"{val:04.1f}%"

    if fname_disp:
        # Gen:Green, Sen:Yellow, Que:Magenta, Exp:Red
        b_gen = get_bar(rating_probs[0], Colors.GREEN)
        b_sen = get_bar(rating_probs[1], Colors.YELLOW)
        b_que = get_bar(rating_probs[2], Colors.MAGENTA)
        b_exp = get_bar(rating_probs[3], Colors.RED)

        tqdm.write(f"[{fname_disp}] "
                   f"Gen:{b_gen} {fmt_prob(rating_probs[0])} "
                   f"Sen:{b_sen} {fmt_prob(rating_probs[1])} "
                   f"Que:{b_que} {fmt_prob(rating_probs[2])} "
                   f"Exp:{b_exp} {fmt_prob(rating_probs[3])}")

    if rating_thresh is not None:
        nsfw_probs = rating_probs[1:]
        max_nsfw_idx = np.argmax(nsfw_probs)
        max_nsfw_prob = nsfw_probs[max_nsfw_idx]
        
        if max_nsfw_prob > rating_thresh:
            rating_idx = max_nsfw_idx + 1
        else:
            rating_idx = 0
    else:
        rating_idx = np.argmax(rating_probs)
    
    rating = tags[rating_idx]

    if rating == 'sensitive':
        if rating_probs[1] < split_thresh:
            rating = 'sensitive_mild'
        else:
            rating = 'sensitive_high'
    
    if (rating == 'sensitive' or rating == 'sensitive_mild' or rating == 'sensitive_high') and ignore_sensitive:
        rating = 'general'

    return rating

class TagServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        global req_count
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            img = Image.open(io.BytesIO(post_data))
            img_input = preprocess(img)
            probs = sess_global.run([label_name_cache], {input_name_cache: img_input})[0][0]
            response_data = json.dumps(probs.astype(float).tolist())
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(response_data.encode('utf-8'))
            req_count += 1
            print(f"[INFO] Req #{req_count} | {self.client_address[0]} | {content_length} bytes")
        except Exception as e:
            print(f"[Error] Processing request: {e}")
            self.send_response(500)
            self.end_headers()

def run_server(port, use_gpu):
    init_global_model(use_gpu)
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, TagServerHandler)
    
    print(f"\n[INFO] Server running on Port {port}")
    ips = get_ip_addresses()
    for ip in ips:
        print(f"[INFO] {ip}")
    
    print(f"[INFO] Ready to accept requests.")
    print(f"[INFO] Press Ctrl+C to stop.\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped.")

def run_client(target_files, host, port, thresh, force, args):
    url = f"http://{host}:{port}"
    print(f"[INFO] Connecting to Server: {url}")
    
    et_wrapper.start()

    repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
    tags = []
    with open(tags_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        tags = [row[1] for row in reader]

    split_thresh = APP_CONFIG.get("sensitive_split_threshold", 0.50)

    processed_count = 0
    skipped_count = 0
    organized_count = 0
    
    pbar = tqdm(target_files, unit="img", ncols=80)
    for img_path in pbar:
        try:
            rating = None
            existing_tags = []
            
            if not force:
                existing_tags = et_wrapper.get_tags(img_path)
            
            need_inference = True
            if existing_tags and not force:
                if args.rating_thresh is not None:
                    need_inference = True
                elif args.organize:
                    found_ratings = [t for t in existing_tags if t in RATING_TAGS]
                    if found_ratings:
                        rating = found_ratings[0]
                        if rating == 'sensitive':
                            need_inference = True
                        else:
                            need_inference = False
                    else:
                        need_inference = True
                else:
                    need_inference = False
            
            detected_tags = []
            probs = None

            if need_inference:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                req = urllib.request.Request(url, data=img_data, method='POST')
                req.add_header('Content-Type', 'application/octet-stream')
                with urllib.request.urlopen(req) as res:
                    if res.status != 200:
                        tqdm.write(f"Server Error: {res.status}")
                        continue
                    response_body = res.read()
                    probs = np.array(json.loads(response_body.decode('utf-8')))
                
                fname_disp = os.path.basename(img_path)
                if len(fname_disp) > 20: fname_disp = fname_disp[:17] + "..."
                
                rating = calculate_rating(
                    probs, tags, 
                    args.rating_thresh, 
                    split_thresh, 
                    args.ignore_sensitive, 
                    fname_disp
                )

                for i, p in enumerate(probs):
                    if p > thresh:
                        detected_tags.append(tags[i])
                
                should_write = False
                if not existing_tags: should_write = True
                if force: should_write = True
                
                if should_write and detected_tags:
                    if et_wrapper.write_tags(img_path, detected_tags):
                        processed_count += 1
                elif not should_write:
                    skipped_count += 1
            else:
                skipped_count += 1

            if args.organize and rating:
                if organize_file(img_path, rating):
                    organized_count += 1

        except urllib.error.URLError as e:
            tqdm.write(f"Connection Error: {e}")
            break
        except KeyboardInterrupt:
            print("\nAborted.")
            et_wrapper.stop()
            sys.exit(0)
        except Exception as e:
            tqdm.write(f"Error {os.path.basename(img_path)}: {e}")
    
    et_wrapper.stop()
    print(f"\n[Done] Processed: {processed_count}, Skipped: {skipped_count}, Organized: {organized_count}")

def main():
    parser = argparse.ArgumentParser(description="WD14 Tagger Universal")
    parser.add_argument("--mode", choices=['standalone', 'server', 'client'], default='standalone')
    parser.add_argument("images", nargs='*')
    parser.add_argument("--thresh", type=float, default=0.35)
    parser.add_argument("--rating-thresh", type=float, default=None, help="Threshold for non-general rating")
    parser.add_argument("--ignore-sensitive", action="store_true", help="Treat sensitive as general")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--organize", action="store_true", help="Move images to folders based on rating")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gen-config", action="store_true", help="Generate config.json and exit")
    
    args = parser.parse_args()

    # Windowsのコマンドプロンプト用にANSIエスケープシーケンスを有効化
    if IS_WINDOWS:
        os.system('')

    if args.gen_config:
        sys.exit(0)

    if args.host is None:
        args.host = APP_CONFIG.get("server_host", "localhost")
    if args.port is None:
        args.port = APP_CONFIG.get("server_port", 5000)

    split_thresh = APP_CONFIG.get("sensitive_split_threshold", 0.50)
    use_recursive = not args.organize

    if args.mode == 'server':
        run_server(args.port, args.gpu)
    elif args.mode == 'client':
        if not args.images:
            print("Error: No images specified for client mode.")
            return
        files = collect_images(args.images, recursive=use_recursive)
        run_client(files, args.host, args.port, args.thresh, args.force, args)
    else:
        if not args.images:
            parser.print_help()
            return
        files = collect_images(args.images, recursive=use_recursive)
        if not files:
            print("No files found.")
            return
        print("Loading model...")
        init_global_model(args.gpu)
        et_wrapper.start()
        
        processed = 0
        skipped = 0
        organized = 0
        
        pbar = tqdm(files, unit="img", ncols=80)
        for img_path in pbar:
            try:
                rating = None
                existing_tags = []
                
                existing_tags = et_wrapper.get_tags(img_path)
                
                need_inference = True
                
                if existing_tags and not args.force:
                    if args.rating_thresh is not None:
                        need_inference = True
                    elif args.organize:
                        found_ratings = [t for t in existing_tags if t in RATING_TAGS]
                        if found_ratings:
                            rating = found_ratings[0]
                            if rating == 'sensitive': 
                                need_inference = True 
                            else:
                                need_inference = False
                        else:
                            need_inference = True
                    else:
                        need_inference = False
                
                detected_tags = []
                
                if need_inference:
                    pil_image = Image.open(img_path)
                    img_input = preprocess(pil_image)
                    probs = sess_global.run([label_name_cache], {input_name_cache: img_input})[0][0]
                    
                    fname_disp = os.path.basename(img_path)
                    if len(fname_disp) > 20: fname_disp = fname_disp[:17] + "..."
                    
                    rating = calculate_rating(
                        probs, tags_global, 
                        args.rating_thresh, 
                        split_thresh, 
                        args.ignore_sensitive, 
                        fname_disp
                    )

                    for i, p in enumerate(probs):
                        if p > args.thresh:
                            detected_tags.append(tags_global[i])
                    
                    should_write = False
                    if not existing_tags: should_write = True
                    if args.force: should_write = True
                    
                    if should_write and detected_tags:
                        if et_wrapper.write_tags(img_path, detected_tags):
                            processed += 1
                    elif not should_write:
                        skipped += 1
                else:
                    skipped += 1
                
                if args.organize and rating:
                    if organize_file(img_path, rating):
                        organized += 1

            except KeyboardInterrupt:
                print("\n[INFO] Stopping...")
                et_wrapper.stop()
                sys.exit(0)
            except Exception as e:
                tqdm.write(f"Error {os.path.basename(img_path)}: {e}")
        
        et_wrapper.stop()
        print(f"\n[Done] Processed (Tagged): {processed}, Skipped: {skipped}, Organized: {organized}")

if __name__ == "__main__":
    main()