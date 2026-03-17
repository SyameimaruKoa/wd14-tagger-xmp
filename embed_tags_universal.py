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
import warnings
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import urllib.error

try:
    import make_report
except ImportError:
    make_report = None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")
SYSTEM_OS = platform.system()
IS_WINDOWS = SYSTEM_OS == "Windows"
IS_LINUX = SYSTEM_OS == "Linux"
EXIFTOOL_CMD = "exiftool"
VALID_EXTS = (".webp", ".jpg", ".jpeg", ".png", ".bmp")
RATING_TAGS = ["general", "sensitive", "questionable", "explicit"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
REPORT_LOG_FILE = os.path.join(os.getcwd(), "report_log.json")
DEFAULT_CONFIG = {
    "server_hosts": ["localhost", "100,xxx,xxx,xxx"],
    "server_port": 5000,
    "client_timeout": 15,
    "sensitive_split_threshold": 0.50,
    "general_threshold": 0.40,
    "folder_names": {
        "general": "R-00",
        "sensitive_mild": "R-15_0",
        "sensitive_high": "R-15_5",
        "questionable": "R-17",
        "explicit": "R-18",
    },
}


def merge_defaults(target, source):
    has_change = False
    for k, v in source.items():
        if k not in target:
            target[k] = v
            has_change = True
        elif isinstance(v, dict) and isinstance(target.get(k), dict):
            if merge_defaults(target[k], v):
                has_change = True
    return has_change


def load_config():
    config = DEFAULT_CONFIG.copy()
    if not os.path.exists(CONFIG_FILE):
        print(f"[INFO] コンフィグファイルを生成しました: {CONFIG_FILE}")
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] コンフィグファイルの作成に失敗しました: {e}")
    else:
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            if merge_defaults(user_config, DEFAULT_CONFIG):
                print(
                    f"[INFO] コンフィグファイルを更新しました（不足項目を追加）: {CONFIG_FILE}"
                )
                try:
                    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                        json.dump(user_config, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    print(f"[WARN] コンフィグファイルの更新保存に失敗しました: {e}")
            config = user_config
            print(f"[INFO] コンフィグを読み込みました: {CONFIG_FILE}")
        except Exception as e:
            print(
                f"[WARN] コンフィグの読み込みに失敗しました: {e}. デフォルト値を使用します。"
            )
    return config


APP_CONFIG = load_config()


class Colors:
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    GREY = "\033[90m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


def get_bar(prob, color, width=5):
    fill_len = int(prob * width)
    empty_len = width - fill_len
    return f"{color}{'█' * fill_len}{Colors.GREY}{'░' * empty_len}{Colors.RESET}"


REPORT_DATA = []


class ExifToolWrapper:
    def __init__(self, cmd=EXIFTOOL_CMD):
        self.cmd = cmd
        self.process = None
        self.running = False

    def start(self):
        if self.running:
            return
        try:
            startupinfo = None
            if IS_WINDOWS:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            self.process = subprocess.Popen(
                [
                    self.cmd,
                    "-stay_open",
                    "True",
                    "-@",
                    "-",
                    "-common_args",
                    "-charset",
                    "filename=utf8",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
            )
            self.running = True
        except Exception as e:
            print(f"[ERROR] ExifToolの起動に失敗しました: {e}")
            self.running = False

    def stop(self):
        if not self.running:
            return
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
            if not self.running:
                return ""
        try:
            for arg in args:
                self.process.stdin.write(arg.encode("utf-8") + b"\n")
            self.process.stdin.write(b"-execute\n")
            self.process.stdin.flush()
            output_lines = []
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="ignore").strip()
                if line_str == "{ready}":
                    break
                output_lines.append(line_str)
            return "\n".join(output_lines)
        except Exception as e:
            print(f"[Error] ExifTool通信エラー: {e}")
            self.stop()
            return ""

    def get_tags(self, path):
        res = self.execute(["-XMP:Subject", "-s3", "-sep", ", ", "-fast", path])
        if res:
            return [t.strip() for t in res.split(",")]
        return []

    def write_tags(self, path, tags):
        if not tags:
            return False
        tags_str = ", ".join(tags)
        res = self.execute(
            [
                "-overwrite_original",
                "-P",
                "-m",
                "-sep",
                ", ",
                f"-XMP:Subject={tags_str}",
                path,
            ]
        )
        return "image files updated" in res


et_wrapper = ExifToolWrapper()


def load_model_and_tags(use_gpu=False):
    repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
    tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
    tags = []
    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        tags = [row[1] for row in reader]
    providers = []
    if use_gpu:
        available_providers = ort.get_available_providers()
        desired_providers = []
        if IS_WINDOWS:
            desired_providers.extend(["DmlExecutionProvider", "CUDAExecutionProvider"])
        elif IS_LINUX:
            desired_providers.extend(
                [
                    "OpenVINOExecutionProvider",
                    "CUDAExecutionProvider",
                    "ROCMExecutionProvider",
                ]
            )
        for p in desired_providers:
            if p in available_providers:
                providers.append(p)
    providers.append("CPUExecutionProvider")
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    print(f"[INFO] 試行プロバイダ: {providers}")
    try:
        sess = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        print(f"[INFO] アクティブプロバイダ: {sess.get_providers()}")
    except Exception as e:
        print(f"[WARN] GPUプロバイダのロードに失敗しました: {e}")
        print("[INFO] CPUモードに切り替えます。")
        sess = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
    return sess, tags, sess.get_inputs()[0].name, sess.get_outputs()[0].name


sess_global = None
tags_global = None
input_name_cache = None
label_name_cache = None


def init_global_model(use_gpu):
    global sess_global, tags_global, input_name_cache, label_name_cache
    if sess_global is None:
        sess_global, tags_global, input_name_cache, label_name_cache = (
            load_model_and_tags(use_gpu)
        )


def preprocess(image, size=448):
    image = image.convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    img_np = np.array(image).astype(np.float32)
    img_np = img_np[:, :, ::-1]
    img_np = np.expand_dims(img_np, 0)
    return img_np


def organize_file(file_path, rating):
    folder_mapping = APP_CONFIG.get("folder_names", {})
    folder_name = folder_mapping.get(rating, rating)
    try:
        abs_path = os.path.abspath(file_path)
        dir_name = os.path.dirname(abs_path)
        file_name = os.path.basename(abs_path)
        target_dir = os.path.join(dir_name, folder_name)
        if os.path.abspath(dir_name) == os.path.abspath(target_dir):
            return False, abs_path
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, file_name)
        if os.path.exists(target_path):
            base, ext = os.path.splitext(file_name)
            target_path = os.path.join(
                target_dir, f"{base}_{uuid.uuid4().hex[:6]}{ext}"
            )
        shutil.move(abs_path, target_path)
        return True, target_path
    except Exception as e:
        tqdm.write(f"[Warn] 移動失敗 {file_path}: {e}")
        return False, file_path


def collect_images(path_args, recursive=True):
    collected = []
    for p in path_args:
        if "*" in p or "?" in p:
            candidates = glob.glob(p, recursive=recursive)
        else:
            candidates = [p]
        for candidate in candidates:
            if os.path.isdir(candidate):
                print(
                    f"[INFO] ディレクトリをスキャン中 (再帰={recursive}): {candidate}"
                )
                if recursive:
                    for root, _, files in os.walk(candidate):
                        for f in files:
                            if f.lower().endswith(VALID_EXTS):
                                collected.append(os.path.join(root, f))
                else:
                    try:
                        for f in os.listdir(candidate):
                            full_path = os.path.join(candidate, f)
                            if os.path.isfile(full_path) and f.lower().endswith(
                                VALID_EXTS
                            ):
                                collected.append(full_path)
                    except OSError:
                        pass
            elif os.path.isfile(candidate):
                if candidate.lower().endswith(VALID_EXTS):
                    collected.append(candidate)
    return sorted(list(set(collected)))


def calculate_rating(
    probs,
    tags,
    rating_thresh,
    split_thresh,
    ignore_sensitive,
    gen_thresh,
    fname_disp="",
):
    rating_probs = probs[:4]
    if fname_disp:

        def fmt_prob(p):
            val = p * 100
            if val >= 100:
                return "100.0%"
            return f"{val:04.1f}%"

        b_gen = get_bar(rating_probs[0], Colors.GREEN)
        b_sen = get_bar(rating_probs[1], Colors.YELLOW)
        b_que = get_bar(rating_probs[2], Colors.MAGENTA)
        b_exp = get_bar(rating_probs[3], Colors.RED)
        tqdm.write(
            f"[{fname_disp}] Gen:{b_gen}{fmt_prob(rating_probs[0])} Sen:{b_sen}{fmt_prob(rating_probs[1])} Que:{b_que}{fmt_prob(rating_probs[2])} Exp:{b_exp}{fmt_prob(rating_probs[3])}",
            end="",
        )
    if rating_probs[0] >= gen_thresh:
        rating_idx = 0
    else:
        if rating_thresh is not None:
            nsfw_sum = np.sum(rating_probs[1:])
            if nsfw_sum > rating_thresh:
                nsfw_probs = rating_probs[1:]
                max_nsfw_idx = np.argmax(nsfw_probs)
                rating_idx = max_nsfw_idx + 1
            else:
                rating_idx = 0
        else:
            rating_idx = np.argmax(rating_probs)
    rating = tags[rating_idx]
    if rating == "sensitive":
        if rating_probs[1] < split_thresh:
            rating = "sensitive_mild"
        else:
            rating = "sensitive_high"
    if (
        rating == "sensitive"
        or rating == "sensitive_mild"
        or rating == "sensitive_high"
    ) and ignore_sensitive:
        rating = "general"
    if fname_disp:
        folder_mapping = APP_CONFIG.get("folder_names", {})
        folder_name = folder_mapping.get(rating, rating)
        res_color = Colors.CYAN
        if rating == "explicit":
            res_color = Colors.RED
        elif rating == "questionable":
            res_color = Colors.MAGENTA
        elif "sensitive" in rating:
            res_color = Colors.YELLOW
        elif rating == "general":
            res_color = Colors.GREEN
        tqdm.write(f" => {res_color}[{folder_name}]{Colors.RESET}")
    return rating


class TagServerHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            img = Image.open(io.BytesIO(post_data))
            img_input = preprocess(img)
            probs = sess_global.run([label_name_cache], {input_name_cache: img_input})[
                0
            ][0]
            response_data = json.dumps(probs.astype(float).tolist())
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(response_data.encode("utf-8"))
        except Exception:
            self.send_response(500)
            self.end_headers()


def run_server(port, use_gpu):
    init_global_model(use_gpu)
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, TagServerHandler)
    print(f"\n[INFO] サーバー稼働中 Port: {port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def process_images(args):
    host = args.host
    port = args.port
    is_client = args.mode == "client"
    split_thresh = APP_CONFIG.get("sensitive_split_threshold", 0.50)
    gen_thresh = APP_CONFIG.get("general_threshold", 0.40)
    client_timeout = APP_CONFIG.get("client_timeout", 15)
    if not is_client:
        print("[INFO] モデルをロード中...")
        init_global_model(args.gpu)
    server_url = f"http://{host}:{port}"
    if is_client:
        print(f"[INFO] サーバーに接続: {server_url} (Timeout: {client_timeout}s)")
    if not args.no_tag:
        et_wrapper.start()
    tags = tags_global
    if is_client:
        repo_id = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
        tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
        with open(tags_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            tags = [row[1] for row in reader]
    use_recursive = True
    if args.recursive is not None:
        use_recursive = args.recursive
    else:
        if args.organize:
            use_recursive = False
        else:
            use_recursive = True
    target_files = collect_images(args.images, recursive=use_recursive)
    if not target_files:
        print("[WARN] 対象ファイルが見つかりません。")
        if not args.no_tag:
            et_wrapper.stop()
        return
    processed_count = 0
    skipped_count = 0
    organized_count = 0
    pbar = tqdm(target_files, unit="img", ncols=80)
    for img_path in pbar:
        try:
            rating = None
            existing_tags = []
            need_inference = True
            if not args.no_tag or args.organize:
                if et_wrapper.running:
                    existing_tags = et_wrapper.get_tags(img_path)
            if args.force:
                need_inference = True
            elif existing_tags:
                if args.rating_thresh is not None:
                    need_inference = True
                else:
                    if args.organize:
                        found_ratings = [t for t in existing_tags if t in RATING_TAGS]
                        if found_ratings:
                            rating = found_ratings[0]
                            need_inference = False
                        else:
                            need_inference = True
                    else:
                        need_inference = False
            probs = None
            detected_tags = []
            final_path = img_path
            if need_inference:
                if is_client:
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                    req = urllib.request.Request(
                        server_url, data=img_data, method="POST"
                    )
                    req.add_header("Content-Type", "application/octet-stream")
                    with urllib.request.urlopen(req, timeout=client_timeout) as res:
                        if res.status != 200:
                            tqdm.write(f"Server Error: {res.status}")
                            continue
                        response_body = res.read()
                        probs = np.array(json.loads(response_body.decode("utf-8")))
                else:
                    pil_image = Image.open(img_path)
                    img_input = preprocess(pil_image)
                    probs = sess_global.run(
                        [label_name_cache], {input_name_cache: img_input}
                    )[0][0]
                fname_disp = os.path.basename(img_path)
                if len(fname_disp) > 20:
                    fname_disp = fname_disp[:17] + "..."
                rating = calculate_rating(
                    probs,
                    tags,
                    args.rating_thresh,
                    split_thresh,
                    args.ignore_sensitive,
                    gen_thresh,
                    fname_disp,
                )
                for i, p in enumerate(probs):
                    if p > args.thresh:
                        detected_tags.append(tags[i])
            if not args.no_tag:
                should_write = False
                if args.force:
                    should_write = True
                elif not existing_tags:
                    should_write = True
                if should_write and detected_tags:
                    if et_wrapper.write_tags(img_path, detected_tags):
                        processed_count += 1
                elif not should_write:
                    skipped_count += 1
            else:
                if need_inference:
                    pass
                skipped_count += 1
            if args.organize and rating:
                moved, new_path = organize_file(img_path, rating)
                if moved:
                    organized_count += 1
                    final_path = new_path
            if not args.no_report and probs is not None:
                REPORT_DATA.append(
                    {
                        "path": os.path.abspath(final_path),
                        "rating": rating,
                        "probs": probs[:4].tolist(),
                    }
                )
        except (urllib.error.URLError, socket.timeout) as e:
            tqdm.write(f"接続エラー(タイムアウト含む): {e}")
            break
        except KeyboardInterrupt:
            print("\n[INFO] 中断されました。")
            if not args.no_tag:
                et_wrapper.stop()
            sys.exit(0)
        except Exception as e:
            tqdm.write(f"エラー {os.path.basename(img_path)}: {e}")
    if not args.no_tag:
        et_wrapper.stop()
    print(
        f"\n[完了] タグ付け: {processed_count}, スキップ: {skipped_count}, 整理: {organized_count}"
    )
    if not args.no_report:
        if REPORT_DATA:
            with open(REPORT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(REPORT_DATA, f, ensure_ascii=False)
            print(f"[INFO] レポート用ログを保存: {REPORT_LOG_FILE}")
            if make_report:
                print("[INFO] HTMLレポートを生成中...")
                make_report.make_report()
            else:
                print(
                    "[WARN] make_report モジュールが見つからないため、HTML生成をスキップします。"
                )
        else:
            print("[INFO] レポート対象データがありませんでした。")


def main():
    parser = argparse.ArgumentParser(
        description="WD14 Tagger Universal (日本語版)", add_help=False
    )
    parser.add_argument("images", nargs="*", help="処理対象の画像またはフォルダパス")
    mode_group = parser.add_argument_group("実行モード")
    mode_group.add_argument(
        "--mode",
        choices=["standalone", "server", "client"],
        default="standalone",
        help="動作モード (standalone: 通常, server: 待機, client: 送信)",
    )
    action_group = parser.add_argument_group("アクション設定")
    action_group.add_argument(
        "--no-tag", action="store_true", help="タグ付け処理を行わない"
    )
    action_group.add_argument(
        "--organize",
        action="store_true",
        help="レーティングに基づいてフォルダ振り分けを行う",
    )
    action_group.add_argument(
        "--no-report", action="store_true", help="HTMLレポートを作成しない"
    )
    conf_group = parser.add_argument_group("判定・システム設定")
    conf_group.add_argument(
        "--thresh", type=float, default=0.35, help="タグ採用の確信度閾値 (0.35)"
    )
    conf_group.add_argument("--gpu", action="store_true", help="GPUを使用する")
    conf_group.add_argument(
        "--force",
        action="store_true",
        help="既存タグがあっても強制的に再解析・上書きする",
    )
    conf_group.add_argument(
        "--recursive",
        action="store_const",
        const=True,
        default=None,
        help="サブフォルダも再帰的に検索する (指定優先)",
    )
    conf_group.add_argument(
        "--no-recursive",
        action="store_const",
        const=False,
        dest="recursive",
        help="サブフォルダは検索しない (指定優先)",
    )
    net_group = parser.add_argument_group("ネットワーク設定")
    net_group.add_argument("--host", default=None, help="サーバーIPアドレス")
    net_group.add_argument("--port", type=int, default=None, help="ポート番号")
    misc_group = parser.add_argument_group("その他・旧機能")
    misc_group.add_argument(
        "--rating-thresh",
        type=float,
        default=None,
        help="[Old] R指定タグ合計値による閾値判定",
    )
    misc_group.add_argument(
        "--ignore-sensitive",
        action="store_true",
        help="[Old] SensitiveをGeneralとして扱う",
    )
    misc_group.add_argument(
        "--gen-config", action="store_true", help="設定ファイル生成のみ実行"
    )
    misc_group.add_argument("-h", "--help", action="help", help="ヘルプを表示")
    args = parser.parse_args()
    if IS_WINDOWS:
        os.system("")
    if args.gen_config:
        load_config()
        sys.exit(0)
    if args.host is None:
        saved_hosts = APP_CONFIG.get(
            "server_hosts", APP_CONFIG.get("server_host", ["localhost"])
        )
        if isinstance(saved_hosts, str):
            saved_hosts = [saved_hosts]
        if len(saved_hosts) == 1:
            args.host = saved_hosts[0]
        elif len(saved_hosts) > 1:
            print(
                f"{Colors.CYAN}[INFO] 接続先サーバーが複数登録されています。{Colors.RESET}"
            )
            for i, h in enumerate(saved_hosts):
                print(f"  {i+1}: {h}")
            while True:
                choice = input("接続先を選択してください: ")
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(saved_hosts):
                        args.host = saved_hosts[idx]
                        break
                    else:
                        print("無効な番号じゃ。正しい数値を入力するのじゃ。")
                except ValueError:
                    print("数値を入力するのじゃ。")
        else:
            args.host = "localhost"
    if args.port is None:
        args.port = APP_CONFIG.get("server_port", 5000)
    if args.mode == "server":
        run_server(args.port, args.gpu)
    else:
        if not args.images:
            print(
                f"{Colors.YELLOW}[案内] 画像ファイルまたはフォルダを指定してください。{Colors.RESET}"
            )
            parser.print_help()
            return
        process_images(args)


if __name__ == "__main__":
    main()
