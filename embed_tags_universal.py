import argparse, csv, os, sys, subprocess, glob, uuid, shutil, platform, json, io, socket, warnings, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnxruntime as ort
from PIL import Image
try:
    import pillow_avif
except ImportError:
    pass
from huggingface_hub import hf_hub_download
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    import make_report
except ImportError:
    make_report = None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")
SYSTEM_OS, IS_WINDOWS, IS_LINUX = (
    platform.system(),
    platform.system() == "Windows",
    platform.system() == "Linux",
)
EXIFTOOL_CMD = "exiftool"
VALID_EXTS = (".webp", ".jpg", ".jpeg", ".png", ".bmp", ".avif")
RATING_TAGS = [
    "general",
    "sensitive",
    "sensitive_mild",
    "sensitive_high",
    "questionable",
    "explicit",
]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
REPORT_LOG_FILE = os.path.join(os.getcwd(), "report_log.json")
COMPARE_REPORT_JSON = os.path.join(os.getcwd(), "compare_report.json")
COMPARE_REPORT_CSV = os.path.join(os.getcwd(), "compare_report.csv")
DEFAULT_CONFIG = {
    "model_repo": "SmilingWolf/wd-swinv2-tagger-v3",
    "model_file": "model.onnx",
    "tags_file": "selected_tags.csv",
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
            target[k], has_change = v, True
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
            config, print_str = (
                user_config,
                f"[INFO] コンフィグを読み込みました: {CONFIG_FILE}",
            )
            print(print_str)
        except Exception as e:
            print(
                f"[WARN] コンフィグの読み込みに失敗しました: {e}. デフォルト値を使用します。"
            )
    return config


APP_CONFIG = load_config()


class Colors:
    GREEN, YELLOW, MAGENTA, RED, GREY, CYAN, RESET = (
        "\033[32m",
        "\033[33m",
        "\033[35m",
        "\033[31m",
        "\033[90m",
        "\033[36m",
        "\033[0m",
    )


def get_bar(prob, color, width=5):
    fill_len = int(prob * width)
    return (
        f"{color}{'█' * fill_len}{Colors.GREY}{'░' * (width - fill_len)}{Colors.RESET}"
    )


REPORT_DATA = []
COMPARE_DATA = []


class ExifToolWrapper:
    def __init__(self, cmd=EXIFTOOL_CMD):
        self.cmd, self.process, self.running = cmd, None, False

    def start(self):
        if self.running:
            return
        try:
            startupinfo = subprocess.STARTUPINFO() if IS_WINDOWS else None
            if IS_WINDOWS:
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
        return [t.strip() for t in res.split(",")] if res else []

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


def get_batch_limit(session):
    try:
        shape = session.get_inputs()[0].shape
    except Exception:
        return None
    if not shape:
        return None
    batch_dim = shape[0]
    if isinstance(batch_dim, (int, np.integer)):
        return int(batch_dim)
    return None


def resolve_local_path(filename):
    if not filename:
        return None
    candidates = []
    if os.path.isabs(filename):
        candidates.append(filename)
    else:
        candidates.append(os.path.join(os.getcwd(), filename))
        candidates.append(os.path.join(SCRIPT_DIR, filename))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def resolve_hf_or_local(repo_id, filename, label):
    local_path = resolve_local_path(filename)
    if local_path:
        return local_path
    if os.path.isabs(filename or ""):
        raise FileNotFoundError(f"{label}が見つかりません: {filename}")
    if not repo_id:
        raise ValueError(f"{label}の取得先リポジトリIDが未指定です。")
    return hf_hub_download(repo_id=repo_id, filename=filename)


def load_tags_from_path(tags_path):
    with open(tags_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[1] for row in reader]


def load_model_and_tags(use_gpu=False, model_repo=None, model_file=None, tags_file=None):
    global MODEL_BATCH_LIMIT
    repo_id = model_repo or DEFAULT_CONFIG.get("model_repo")
    model_file = model_file or DEFAULT_CONFIG.get("model_file")
    tags_file = tags_file or DEFAULT_CONFIG.get("tags_file")
    try:
        model_path = resolve_hf_or_local(repo_id, model_file, "モデル")
        tags_path = resolve_hf_or_local(repo_id, tags_file, "タグCSV")
    except Exception as e:
        print(f"[ERROR] モデル/タグの読み込みに失敗しました: {e}")
        sys.exit(1)
    tags = load_tags_from_path(tags_path)
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
                    "MIGraphXExecutionProvider",
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
        print(
            f"[WARN] GPUプロバイダのロードに失敗しました: {e}\n[INFO] CPUモードに切り替えます。"
        )
        sess = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
    MODEL_BATCH_LIMIT = get_batch_limit(sess)
    return sess, tags, sess.get_inputs()[0].name, sess.get_outputs()[0].name


sess_global, tags_global, input_name_cache, label_name_cache = None, None, None, None
MODEL_BATCH_LIMIT = None


def init_global_model(use_gpu, model_repo, model_file, tags_file):
    global sess_global, tags_global, input_name_cache, label_name_cache
    if sess_global is None:
        sess_global, tags_global, input_name_cache, label_name_cache = (
            load_model_and_tags(use_gpu, model_repo, model_file, tags_file)
        )


def preprocess(image, size=448):
    image = image.convert("RGB").resize((size, size), Image.BICUBIC)
    img_np = np.array(image).astype(np.float32)[:, :, ::-1]
    return np.expand_dims(img_np, 0)


def load_and_preprocess(path):
    try:
        with Image.open(path) as img:
            return preprocess(img), None
    except Exception as e:
        return None, e


def organize_file(file_path, rating):
    folder_mapping = APP_CONFIG.get("folder_names", {})
    folder_name = folder_mapping.get(rating)
    if folder_name is None:
        base = rating.split("_")[0] if isinstance(rating, str) and "_" in rating else rating
        folder_name = folder_mapping.get(base, rating)
    try:
        abs_path = os.path.abspath(file_path)
        dir_name, file_name = os.path.dirname(abs_path), os.path.basename(abs_path)
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
        candidates = glob.glob(p, recursive=recursive) if "*" in p or "?" in p else [p]
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
            elif os.path.isfile(candidate) and candidate.lower().endswith(VALID_EXTS):
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
        fmt_prob = lambda p: "100.0%" if p * 100 >= 100 else f"{p * 100:04.1f}%"
        b_gen, b_sen, b_que, b_exp = (
            get_bar(rating_probs[0], Colors.GREEN),
            get_bar(rating_probs[1], Colors.YELLOW),
            get_bar(rating_probs[2], Colors.MAGENTA),
            get_bar(rating_probs[3], Colors.RED),
        )
        tqdm.write(
            f"[{fname_disp}] Gen:{b_gen}{fmt_prob(rating_probs[0])} Sen:{b_sen}{fmt_prob(rating_probs[1])} Que:{b_que}{fmt_prob(rating_probs[2])} Exp:{b_exp}{fmt_prob(rating_probs[3])}",
            end="",
        )
    if rating_probs[0] >= gen_thresh:
        rating_idx = 0
    else:
        if rating_thresh is not None:
            rating_idx = (
                np.argmax(rating_probs[1:]) + 1
                if np.sum(rating_probs[1:]) > rating_thresh
                else 0
            )
        else:
            rating_idx = np.argmax(rating_probs)
    rating = tags[rating_idx]
    if rating == "sensitive":
        rating = (
            "sensitive_mild" if rating_probs[1] < split_thresh else "sensitive_high"
        )
    if rating in ["sensitive", "sensitive_mild", "sensitive_high"] and ignore_sensitive:
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


def run_server(port, use_gpu, model_repo, model_file, tags_file):
    init_global_model(use_gpu, model_repo, model_file, tags_file)
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, TagServerHandler)
    print(f"\n[INFO] サーバー稼働中 Port: {port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def process_images(args):
    host, port, is_client = args.host, args.port, args.mode == "client"
    split_thresh, gen_thresh, client_timeout = (
        APP_CONFIG.get("sensitive_split_threshold", 0.50),
        APP_CONFIG.get("general_threshold", 0.40),
        APP_CONFIG.get("client_timeout", 15),
    )
    if not is_client:
        print("[INFO] モデルをロード中...")
        init_global_model(args.gpu, args.model_repo, args.model_file, args.tags_file)
    server_url = f"http://{host}:{port}"
    if is_client:
        print(f"[INFO] サーバーに接続: {server_url} (Timeout: {client_timeout}s)")
    if not args.no_tag:
        et_wrapper.start()
    tags = tags_global
    if is_client:
        try:
            tags_path = resolve_hf_or_local(args.model_repo, args.tags_file, "タグCSV")
        except Exception as e:
            print(f"[ERROR] タグCSVの読み込みに失敗しました: {e}")
            if not args.no_tag:
                et_wrapper.stop()
            return
        tags = load_tags_from_path(tags_path)
    use_recursive = args.recursive if args.recursive is not None else not args.organize
    target_files = collect_images(args.images, recursive=use_recursive)
    if not target_files:
        print("[WARN] 対象ファイルが見つかりません。")
        if not args.no_tag:
            et_wrapper.stop()
        return
    batch_size = args.batch_size if args.batch_size > 0 else 1
    if is_client and batch_size > 1:
        print("[WARN] クライアントモードではバッチ推論を使用できません。")
        batch_size = 1
    if (not is_client) and MODEL_BATCH_LIMIT is not None:
        if MODEL_BATCH_LIMIT <= 1 and batch_size > 1:
            print(
                "[WARN] このモデルはバッチ推論に非対応のため、batch-size を 1 に変更します。"
            )
            batch_size = 1
        elif batch_size > MODEL_BATCH_LIMIT:
            print(
                f"[WARN] batch-size がモデル上限({MODEL_BATCH_LIMIT})を超えているため、{MODEL_BATCH_LIMIT} に変更します。"
            )
            batch_size = MODEL_BATCH_LIMIT
    io_workers = args.io_workers
    if io_workers == -1:
        io_workers = max(2, min(4, (os.cpu_count() or 1) // 2)) if batch_size > 1 else 0
    elif io_workers is not None and io_workers < 0:
        io_workers = 0
    if (not is_client) and batch_size > 1:
        print(f"[INFO] バッチ推論: {batch_size} / IOワーカー: {io_workers}")
    processed_count, skipped_count, organized_count = 0, 0, 0
    pbar = tqdm(total=len(target_files), unit="img", ncols=80)
    executor = (
        ThreadPoolExecutor(max_workers=io_workers)
        if (not is_client and batch_size > 1 and io_workers > 0)
        else None
    )

    def finalize_result(img_path, existing_tags, detected_tags, rating, probs):
        nonlocal processed_count, skipped_count, organized_count
        final_path = img_path
        if not args.no_tag:
            should_write = True if args.force or not existing_tags else False
            if should_write and detected_tags:
                if et_wrapper.write_tags(img_path, detected_tags):
                    processed_count += 1
            elif not should_write:
                skipped_count += 1
        else:
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
        pbar.update(1)

    def handle_inference_result(item, probs):
        fname_disp = os.path.basename(item["path"])
        fname_disp = fname_disp[:17] + "..." if len(fname_disp) > 20 else fname_disp
        rating = calculate_rating(
            probs,
            tags,
            args.rating_thresh,
            split_thresh,
            args.ignore_sensitive,
            gen_thresh,
            fname_disp,
        )
        detected_tags = []
        for i, p in enumerate(probs):
            if p > args.thresh:
                detected_tags.append(tags[i])
        finalize_result(item["path"], item["existing_tags"], detected_tags, rating, probs)

    def run_batch(batch_items):
        if not batch_items:
            return
        paths = [item["path"] for item in batch_items]
        if executor:
            results = list(executor.map(load_and_preprocess, paths))
        else:
            results = [load_and_preprocess(p) for p in paths]
        valid_items, inputs = [], []
        for item, (img_input, err) in zip(batch_items, results):
            if err is not None:
                tqdm.write(f"エラー {os.path.basename(item['path'])}: {err}")
                pbar.update(1)
                continue
            valid_items.append(item)
            inputs.append(img_input)
        if not valid_items:
            return
        batch_input = np.concatenate(inputs, axis=0) if len(inputs) > 1 else inputs[0]
        try:
            batch_probs = sess_global.run(
                [label_name_cache], {input_name_cache: batch_input}
            )[0]
        except Exception as e:
            tqdm.write(f"[WARN] バッチ推論に失敗: {e} -> 1枚ずつに切り替えます。")
            for item, img_input in zip(valid_items, inputs):
                try:
                    probs = sess_global.run(
                        [label_name_cache], {input_name_cache: img_input}
                    )[0][0]
                    handle_inference_result(item, probs)
                except Exception as e2:
                    tqdm.write(f"エラー {os.path.basename(item['path'])}: {e2}")
                    pbar.update(1)
            return
        if len(valid_items) == 1:
            probs_list = (
                [batch_probs[0]] if getattr(batch_probs, "ndim", 1) > 1 else [batch_probs]
            )
        else:
            probs_list = batch_probs
        for item, probs in zip(valid_items, probs_list):
            handle_inference_result(item, probs)

    pending = []
    aborted = False
    try:
        for img_path in target_files:
            tqdm.write(f"[DEBUG] 読込開始: {os.path.basename(img_path)}")
            try:
                rating, existing_tags, need_inference = None, [], True
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
                                rating, need_inference = found_ratings[0], False
                            else:
                                need_inference = True
                        else:
                            need_inference = False
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
                                pbar.update(1)
                                continue
                            probs = np.array(json.loads(res.read().decode("utf-8")))
                        handle_inference_result(
                            {"path": img_path, "existing_tags": existing_tags}, probs
                        )
                    else:
                        pending.append({"path": img_path, "existing_tags": existing_tags})
                        if len(pending) >= batch_size:
                            run_batch(pending[:batch_size])
                            pending = pending[batch_size:]
                else:
                    finalize_result(img_path, existing_tags, [], rating, None)
            except (urllib.error.URLError, socket.timeout) as e:
                tqdm.write(f"接続エラー(タイムアウト含む): {e}")
                aborted = True
                break
            except KeyboardInterrupt:
                print("\n[INFO] 中断されました。")
                aborted = True
                sys.exit(0)
            except Exception as e:
                tqdm.write(f"エラー {os.path.basename(img_path)}: {e}")
                pbar.update(1)
        if not aborted and (not is_client) and pending:
            run_batch(pending)
    finally:
        if executor:
            executor.shutdown(wait=True)
        if not args.no_tag:
            et_wrapper.stop()
        pbar.close()
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
    parser.add_argument_group("実行モード").add_argument(
        "--mode",
        choices=["standalone", "server", "client"],
        default="standalone",
        help="動作モード",
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
        "--thresh", type=float, default=0.35, help="タグ採用の確信度閾値"
    )
    conf_group.add_argument("--gpu", action="store_true", help="GPUを使用する")
    conf_group.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="推論バッチサイズ（ローカル時のみ、デフォルト: 4 / 非対応時は自動で 1）",
    )
    conf_group.add_argument(
        "--io-workers",
        type=int,
        default=-1,
        help="前処理の並列ワーカー数（デフォルト: -1=自動）",
    )
    conf_group.add_argument("--force", action="store_true", help="強制再解析")
    conf_group.add_argument(
        "--recursive", action="store_const", const=True, default=None, help="再帰検索ON"
    )
    conf_group.add_argument(
        "--no-recursive",
        action="store_const",
        const=False,
        dest="recursive",
        help="再帰検索OFF",
    )
    model_group = parser.add_argument_group("モデル設定")
    model_group.add_argument(
        "--model-repo",
        default=None,
        help="モデル/タグのHFリポジトリID（未指定時はconfig.jsonのmodel_repo）",
    )
    model_group.add_argument(
        "--model-file",
        default=None,
        help="モデルファイル名またはパス（未指定時はconfig.jsonのmodel_file）",
    )
    model_group.add_argument(
        "--tags-file",
        default=None,
        help="タグCSVファイル名またはパス（未指定時はconfig.jsonのtags_file）",
    )
    net_group = parser.add_argument_group("ネットワーク設定")
    net_group.add_argument("--host", default=None, help="サーバーIPアドレス")
    net_group.add_argument("--port", type=int, default=None, help="ポート番号")
    misc_group = parser.add_argument_group("その他・旧機能")
    misc_group.add_argument("--rating-thresh", type=float, default=None)
    misc_group.add_argument("--ignore-sensitive", action="store_true")
    misc_group.add_argument("--gen-config", action="store_true")
    misc_group.add_argument("-h", "--help", action="help", help="ヘルプを表示")
    args = parser.parse_args()
    if IS_WINDOWS:
        os.system("")
    if args.gen_config:
        load_config()
        sys.exit(0)
    if args.model_repo is None:
        args.model_repo = APP_CONFIG.get(
            "model_repo", DEFAULT_CONFIG.get("model_repo")
        )
    if args.model_file is None:
        args.model_file = APP_CONFIG.get(
            "model_file", DEFAULT_CONFIG.get("model_file")
        )
    if args.tags_file is None:
        args.tags_file = APP_CONFIG.get("tags_file", DEFAULT_CONFIG.get("tags_file"))
    if args.host is None:
        if args.mode == "client":
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
        else:
            args.host = "localhost"
    if args.port is None:
        args.port = APP_CONFIG.get("server_port", 5000)
    if args.mode == "server":
        run_server(args.port, args.gpu, args.model_repo, args.model_file, args.tags_file)
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