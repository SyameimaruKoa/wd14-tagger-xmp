import argparse
import csv
import glob
import io
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pillow_avif = None

try:
    import make_report
except ImportError:
    make_report = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **kwargs: iterable

from dbv4 import (
    DBV4Metadata,
    DBV4Preprocessor,
    DBV4_RATING_NAMES,
    MODEL_PROFILES,
    adapt_input_layout,
    get_model_profile,
    infer_output_to_probabilities,
)

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")

SYSTEM_OS = platform.system()
IS_WINDOWS = SYSTEM_OS == "Windows"
IS_LINUX = SYSTEM_OS == "Linux"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")
REPORT_LOG_FILE = os.path.join(os.getcwd(), "report_log.json")
VALID_EXTS = (".webp", ".jpg", ".jpeg", ".png", ".bmp", ".avif")

RATING_TAGS = {
    "general",
    "sensitive",
    "questionable",
    "explicit",
    *(f"sensitive_{i}" for i in range(10)),
    *(f"questionable_{i}" for i in range(10)),
    "sensitive_mild",
    "sensitive_high",
    *(f"sensitive_lvl{i}" for i in range(1, 7)),
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_profile": "balanced",
    "model_profiles": MODEL_PROFILES,
    "server_hosts": ["localhost", "google-colab", "100.xxx.xxx.xxx"],
    "server_port": 5000,
    "client_timeout": 15,
    "openvino_gpu_device": "GPU.0",
    "general_threshold": 0.40,
    "rating_sublevel_thresholds_5way": [0.20, 0.40, 0.60, 0.80],
    "rating_severity_sensitive_upper_reference": 25.0,
    "rating_severity_questionable_upper_reference": 40.0,
    "record_rating_percentages": True,
    "record_raw_score": True,
    "raw_score_format": "{rating}_score:{raw_score:.4f}",
    "percentage_format": "{rating}:{percentage}%",
    "folder_names": {
        "general": "R-00",
        "sensitive_0": "R-15_0",
        "sensitive_1": "R-15_1",
        "sensitive_2": "R-15_2",
        "sensitive_3": "R-15_3",
        "sensitive_4": "R-15_4",
        "questionable_0": "R-17_0",
        "questionable_1": "R-17_1",
        "questionable_2": "R-17_2",
        "questionable_3": "R-17_3",
        "questionable_4": "R-17_4",
        "explicit": "R-18",
    },
}

APP_CONFIG: Dict[str, Any] = {}


class Colors:
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    GREY = "\033[90m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


def safe_write(message: str, end: str = "\n") -> None:
    try:
        tqdm.write(message, end=end)
    except Exception:
        try:
            sys.stdout.write(message + end)
            sys.stdout.flush()
        except Exception:
            pass


def merge_defaults(target: Dict[str, Any], source: Dict[str, Any]) -> bool:
    changed = False
    for key, value in source.items():
        if key not in target:
            target[key] = json.loads(json.dumps(value))
            changed = True
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            changed = merge_defaults(target[key], value) or changed
    return changed


def load_config() -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"[INFO] DBV4用コンフィグを生成しました: {CONFIG_FILE}")
        return config
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        if not isinstance(user_config, dict):
            raise ValueError("config.json のルートがオブジェクトではありません")
        if merge_defaults(user_config, DEFAULT_CONFIG):
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(user_config, f, indent=4, ensure_ascii=False)
            print(f"[INFO] DBV4設定を追記しました: {CONFIG_FILE}")
        return user_config
    except Exception as exc:
        print(f"[WARN] config.jsonを読み込めないためデフォルト設定を使用します: {exc}")
        return config


APP_CONFIG = load_config()


def get_bar(probability: float, color: str, width: int = 5) -> str:
    filled = max(0, min(width, int(probability * width)))
    return f"{color}{'█' * filled}{Colors.GREY}{'░' * (width - filled)}{Colors.RESET}"


def resolve_exiftool_cmd(command: str) -> str:
    resolved = shutil.which(command) if not os.path.isabs(command) else command
    return resolved or command


class ExifToolWrapper:
    def __init__(self, command: str = "exiftool") -> None:
        self.raw_command = command
        self.command = resolve_exiftool_cmd(command)
        self.process = None
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        try:
            startupinfo = subprocess.STARTUPINFO() if IS_WINDOWS else None
            if IS_WINDOWS:
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            self.process = subprocess.Popen(
                [
                    self.command,
                    "-stay_open",
                    "True",
                    "-@",
                    "-",
                    "-common_args",
                    "-charset",
                    "filename=utf8",
                    "-lang",
                    "en",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
            )
            self.running = True
        except Exception as exc:
            print(f"[ERROR] ExifToolの起動に失敗しました: {exc}")

    def stop(self) -> None:
        if not self.running:
            return
        try:
            if self.process and self.process.stdin:
                self.process.stdin.write(b"-stay_open\nFalse\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            if self.process:
                try:
                    self.process.wait(timeout=2)
                except Exception:
                    self.process.kill()
        except Exception:
            pass
        finally:
            self.process = None
            self.running = False

    def execute(self, arguments: Sequence[str]) -> str:
        if not self.running:
            self.start()
        if not self.running or not self.process or not self.process.stdin or not self.process.stdout:
            return ""
        try:
            for argument in arguments:
                self.process.stdin.write(str(argument).encode("utf-8") + b"\n")
            self.process.stdin.write(b"-execute\n")
            self.process.stdin.flush()
            output = []
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore").strip()
                if text == "{ready}":
                    break
                output.append(text)
            return "\n".join(output)
        except Exception as exc:
            safe_write(f"[ERROR] ExifTool通信エラー: {exc}")
            self.stop()
            return ""

    def get_tags(self, path: str) -> List[str]:
        output = self.execute(["-XMP:Subject", "-s3", "-sep", ", ", "-fast", path])
        if not output or any(token in output for token in ("Error", "Warning", "File not found")):
            return []
        return [tag.strip() for tag in output.split(",") if tag.strip()]

    def write_tags(self, path: str, tags: Sequence[str]) -> bool:
        if not tags:
            return False
        output = self.execute(
            [
                "-overwrite_original",
                "-P",
                "-m",
                "-sep",
                ", ",
                f"-XMP:Subject={', '.join(tags)}",
                path,
            ]
        )
        if "image files updated" in output:
            return True
        safe_write(f"[WARN] タグ書き込み失敗 ({os.path.basename(path)}): {output}")
        return False

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


et_wrapper = ExifToolWrapper()


def build_providers(use_gpu: bool) -> List[Any]:
    if not use_gpu:
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    candidates: List[Any] = []
    if IS_WINDOWS:
        candidates = [
            "DmlExecutionProvider",
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
        ]
    elif IS_LINUX:
        candidates = [
            ("OpenVINOExecutionProvider", {"device_type": APP_CONFIG.get("openvino_gpu_device", "GPU.0")}),
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "MIGraphXExecutionProvider",
        ]
    providers: List[Any] = []
    for candidate in candidates:
        name = candidate[0] if isinstance(candidate, tuple) else candidate
        if name in available:
            providers.append(candidate)
    providers.append("CPUExecutionProvider")
    return providers


class RuntimeModel:
    def __init__(
        self,
        metadata: DBV4Metadata,
        preprocessor: DBV4Preprocessor,
        session: ort.InferenceSession,
    ) -> None:
        self.metadata = metadata
        self.preprocessor = preprocessor
        self.session = session
        input_meta = session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape
        self.output_name = session.get_outputs()[0].name

    @property
    def batch_limit(self) -> Optional[int]:
        if not self.input_shape:
            return None
        value = self.input_shape[0]
        return int(value) if isinstance(value, (int, np.integer)) else None

    def preprocess_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        batch = np.stack([self.preprocessor(image) for image in images], axis=0).astype(np.float32)
        return adapt_input_layout(batch, self.input_shape)

    def predict_images(self, images: Sequence[Image.Image]) -> List[np.ndarray]:
        if not images:
            return []
        raw = self.session.run(
            [self.output_name],
            {self.input_name: self.preprocess_batch(images)},
        )[0]
        raw = np.asarray(raw)
        if raw.ndim == 1:
            raw = raw[None, :]
        return [infer_output_to_probabilities(row) for row in raw]


def load_runtime_model(
    use_gpu: bool,
    profile_name: str,
    model_repo: Optional[str] = None,
    model_file: Optional[str] = None,
    tags_file: Optional[str] = None,
) -> RuntimeModel:
    profiles = APP_CONFIG.get("model_profiles", MODEL_PROFILES)
    profile = get_model_profile(profile_name, profiles)
    metadata = DBV4Metadata.load(
        profile,
        base_dir=SCRIPT_DIR,
        model_repo_override=model_repo,
        model_file_override=model_file,
        tags_file_override=tags_file,
        load_model=True,
    )
    preprocessor = DBV4Preprocessor.from_metadata(metadata)
    providers = build_providers(use_gpu)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    try:
        session = ort.InferenceSession(
            metadata.model_path,
            sess_options=session_options,
            providers=providers,
        )
    except Exception as exc:
        if use_gpu:
            raise RuntimeError(f"DBV4 GPUプロバイダの初期化に失敗しました: {exc}") from exc
        session = ort.InferenceSession(
            metadata.model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    active = session.get_providers()
    print(f"[INFO] DBV4モデル: {metadata.repo_id} ({metadata.profile_name})")
    print(f"[INFO] ラベル数: {metadata.label_count}")
    print(f"[INFO] metadata version: {metadata.metadata_version}")
    print(f"[INFO] 入力 shape: {session.get_inputs()[0].shape}")
    print(f"[INFO] アクティブプロバイダ: {active}")
    if use_gpu:
        active_names = {item[0] if isinstance(item, tuple) else item for item in active}
        candidate_names = {
            item[0] if isinstance(item, tuple) else item
            for item in providers
            if (item[0] if isinstance(item, tuple) else item) != "CPUExecutionProvider"
        }
        if not active_names.intersection(candidate_names):
            raise RuntimeError("DBV4モデルでGPU Execution Providerを有効化できませんでした。")
    return RuntimeModel(metadata, preprocessor, session)


def warmup_runtime(runtime: RuntimeModel, batch_size: int) -> float:
    active = runtime.session.get_providers()
    compiling = {
        "TensorrtExecutionProvider",
        "OpenVINOExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider",
    }
    if not any(
        (provider[0] if isinstance(provider, tuple) else provider) in compiling
        for provider in active
    ):
        return 0.0
    print(f"[INFO] コンパイル系EPのウォームアップ推論を実行します (batch={batch_size})...")
    started = time.time()
    image = Image.new("RGB", (512, 512), (0, 0, 0))
    runtime.predict_images([image] * max(1, batch_size))
    if batch_size > 1:
        runtime.predict_images([image])
    elapsed = time.time() - started
    print(f"[INFO] ウォームアップ完了 (所要時間: {elapsed:.2f}秒)。エンジンの準備が整いました。")
    return elapsed


def calculate_rating_severity(
    scores: Dict[str, float],
    base_rating: str,
    sensitive_reference: float,
    questionable_reference: float,
) -> float:
    values = [max(1e-6, min(1.0 - 1e-6, float(scores[name]))) for name in DBV4_RATING_NAMES]
    gen, sen, que, exp = values

    def pairwise_position(upper: float, lower: float) -> float:
        upper_logit = np.log(upper / (1.0 - upper))
        lower_logit = np.log(lower / (1.0 - lower))
        return float(1.0 / (1.0 + np.exp(-(upper_logit - lower_logit))))

    def score_position(score: float, reference_percent: float) -> float:
        return float(np.clip(np.log1p(score * 100.0) / np.log1p(reference_percent), 0.0, 1.0))

    if base_rating == "sensitive":
        lower = pairwise_position(sen, gen)
        upper = score_position(que, sensitive_reference)
        band = 1
    elif base_rating == "questionable":
        lower = pairwise_position(que, sen)
        upper = score_position(exp, questionable_reference)
        band = 2
    else:
        return 0.0 if base_rating == "general" else 1.0

    return float(np.clip((band + np.sqrt(lower * upper)) / 4.0, 0.0, 1.0))


def determine_rating_sublevel(base_rating: str, severity: float) -> str:
    if base_rating not in ("sensitive", "questionable"):
        return base_rating
    thresholds = APP_CONFIG.get("rating_sublevel_thresholds_5way", [0.20, 0.40, 0.60, 0.80])
    if len(thresholds) != 4:
        raise ValueError("rating_sublevel_thresholds_5way must contain exactly 4 values")
    band = 1 if base_rating == "sensitive" else 2
    local = float(np.clip(severity * 4.0 - band, 0.0, np.nextafter(1.0, 0.0)))
    for index, threshold in enumerate(thresholds):
        if local < float(threshold):
            return f"{base_rating}_{index}"
    return f"{base_rating}_4"


def calculate_rating(
    metadata: DBV4Metadata,
    probabilities: Sequence[float],
    rating_thresh: Optional[float],
    ignore_sensitive: bool,
    general_threshold: float,
    fname_disp: str = "",
) -> str:
    scores = metadata.get_rating_scores(probabilities)
    if scores["general"] >= general_threshold:
        base = "general"
    elif rating_thresh is not None:
        non_general = sum(scores[name] for name in DBV4_RATING_NAMES[1:])
        base = max(DBV4_RATING_NAMES[1:], key=scores.get) if non_general > rating_thresh else "general"
    else:
        base = max(DBV4_RATING_NAMES, key=scores.get)

    rating = base
    if base in ("sensitive", "questionable"):
        rating = determine_rating_sublevel(
            base,
            calculate_rating_severity(
                scores,
                base,
                float(APP_CONFIG.get("rating_severity_sensitive_upper_reference", 25.0)),
                float(APP_CONFIG.get("rating_severity_questionable_upper_reference", 40.0)),
            ),
        )
    if ignore_sensitive and rating.startswith("sensitive_"):
        rating = "general"

    if fname_disp:
        values = [scores[name] for name in DBV4_RATING_NAMES]
        bars = [
            get_bar(values[0], Colors.GREEN),
            get_bar(values[1], Colors.YELLOW),
            get_bar(values[2], Colors.MAGENTA),
            get_bar(values[3], Colors.RED),
        ]
        folder = APP_CONFIG.get("folder_names", {}).get(rating, rating)
        color = Colors.CYAN
        if rating == "general":
            color = Colors.GREEN
        elif rating.startswith("sensitive"):
            color = Colors.YELLOW
        elif rating.startswith("questionable"):
            color = Colors.MAGENTA
        elif rating == "explicit":
            color = Colors.RED
        safe_write(
            f"[{fname_disp}] Gen:{bars[0]}{values[0] * 100:04.1f}% "
            f"Sen:{bars[1]}{values[1] * 100:04.1f}% "
            f"Que:{bars[2]}{values[2] * 100:04.1f}% "
            f"Exp:{bars[3]}{values[3] * 100:04.1f}% "
            f"=> {color}[{folder}]{Colors.RESET}"
        )
    return rating


def format_score_tags(metadata: DBV4Metadata, probabilities: Sequence[float]) -> List[str]:
    scores = metadata.get_rating_scores(probabilities)
    result: List[str] = []
    raw_enabled = bool(APP_CONFIG.get("record_raw_score", True))
    pct_enabled = bool(APP_CONFIG.get("record_rating_percentages", True))
    raw_format = APP_CONFIG.get("raw_score_format", "{rating}_score:{raw_score:.4f}")
    pct_format = APP_CONFIG.get("percentage_format", "{rating}:{percentage}%")
    result.append(metadata.rating_tags_marker)
    for name in DBV4_RATING_NAMES:
        score = scores[name]
        if raw_enabled:
            try:
                result.append(raw_format.format(rating=name, cat=name, raw_score=score))
            except Exception:
                result.append(f"{name}_score:{score:.4f}")
        if pct_enabled:
            percentage = f"{score * 100:.1f}"
            try:
                result.append(pct_format.format(rating=name, cat=name, percentage=percentage))
            except Exception:
                result.append(f"{name}:{percentage}%")
    return result


def is_score_tag(tag: str) -> bool:
    return bool(
        re.match(r"^(general|sensitive|questionable|explicit)_score:[0-9.]+$", tag)
        or re.match(r"^(general|sensitive|questionable|explicit):[0-9.]+%$", tag)
    )


def extract_raw_rating_scores(
    metadata: DBV4Metadata,
    tags: Sequence[str],
) -> Optional[List[float]]:
    if metadata.rating_tags_marker not in tags:
        return None
    scores: Dict[str, float] = {}
    pattern = re.compile(r"^(general|sensitive|questionable|explicit)_score:([0-9.]+)$")
    for tag in tags:
        match = pattern.match(tag.strip())
        if match:
            try:
                scores[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
    if any(name not in scores for name in DBV4_RATING_NAMES):
        return None
    return [scores[name] for name in DBV4_RATING_NAMES]


def preserve_existing_tags(tags: Sequence[str]) -> List[str]:
    return [
        tag.strip()
        for tag in tags
        if tag.strip()
        and tag.strip() not in RATING_TAGS
        and not is_score_tag(tag.strip())
        and not tag.strip().startswith("dbv4_model:")
    ]


def organize_file(
    file_path: str,
    rating: str,
    is_pixiv: bool = False,
    base_dirs: Optional[Sequence[str]] = None,
) -> Tuple[bool, str]:
    mapping = APP_CONFIG.get("folder_names", {})
    folder_name = mapping.get(rating, rating)
    if is_pixiv and (rating == "general" or rating.startswith("sensitive_")):
        return False, file_path
    try:
        source = os.path.abspath(file_path)
        source_dir = os.path.dirname(source)
        filename = os.path.basename(source)
        target_dir = os.path.join(source_dir, folder_name)
        target_path = os.path.join(target_dir, filename)
        if is_pixiv and base_dirs:
            matches = [base for base in base_dirs if source.startswith(base)]
            if matches:
                base = max(matches, key=len)
                relative = os.path.relpath(source, base)
                target_path = os.path.join(
                    os.path.dirname(base),
                    folder_name,
                    relative,
                )
                target_dir = os.path.dirname(target_path)
        if os.path.abspath(source_dir) == os.path.abspath(target_dir):
            return False, source
        os.makedirs(target_dir, exist_ok=True)
        if os.path.exists(target_path):
            stem, ext = os.path.splitext(filename)
            target_path = os.path.join(target_dir, f"{stem}_{uuid.uuid4().hex[:6]}{ext}")
        shutil.move(source, target_path)
        return True, target_path
    except Exception as exc:
        safe_write(f"[WARN] 移動失敗 {file_path}: {exc}")
        return False, file_path


def collect_images(paths: Sequence[str], recursive: bool = True) -> List[str]:
    collected: List[str] = []
    for raw_path in paths:
        candidates = glob.glob(raw_path, recursive=recursive) if "*" in raw_path or "?" in raw_path else [raw_path]
        for candidate in candidates:
            if os.path.isdir(candidate):
                if recursive:
                    for root, _, files in os.walk(candidate):
                        collected.extend(
                            os.path.join(root, name)
                            for name in files
                            if name.lower().endswith(VALID_EXTS)
                        )
                else:
                    try:
                        collected.extend(
                            os.path.join(candidate, name)
                            for name in os.listdir(candidate)
                            if os.path.isfile(os.path.join(candidate, name))
                            and name.lower().endswith(VALID_EXTS)
                        )
                    except OSError:
                        pass
            elif os.path.isfile(candidate) and candidate.lower().endswith(VALID_EXTS):
                collected.append(candidate)
    return sorted(set(collected))


class TagServerHandler(BaseHTTPRequestHandler):
    runtime: Optional[RuntimeModel] = None

    def do_POST(self) -> None:
        try:
            if not self.runtime:
                raise RuntimeError("DBV4 runtimeが初期化されていません。")
            length = int(self.headers.get("Content-Length", "0"))
            image = Image.open(io.BytesIO(self.rfile.read(length))).convert("RGB")
            probabilities = self.runtime.predict_images([image])[0]
            payload = {
                **self.runtime.metadata.summary(),
                "probabilities": probabilities.astype(float).tolist(),
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(args: argparse.Namespace) -> None:
    runtime = load_runtime_model(
        args.gpu,
        args.model_profile,
        args.model_repo,
        args.model_file,
        args.tags_file,
    )
    TagServerHandler.runtime = runtime
    server = HTTPServer(("0.0.0.0", args.port), TagServerHandler)
    print(f"\n[INFO] DBV4推論サーバー稼働中 Port: {args.port}")
    print(f"[INFO] model_id={runtime.metadata.repo_id}")
    print(f"[INFO] output_size={runtime.metadata.label_count}")
    print(f"[INFO] metadata_version={runtime.metadata.metadata_version}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def load_client_metadata(args: argparse.Namespace) -> DBV4Metadata:
    profiles = APP_CONFIG.get("model_profiles", MODEL_PROFILES)
    profile = get_model_profile(args.model_profile, profiles)
    return DBV4Metadata.load(
        profile,
        base_dir=SCRIPT_DIR,
        model_repo_override=args.model_repo,
        model_file_override=args.model_file,
        tags_file_override=args.tags_file,
        load_model=False,
    )


def client_predict(
    server_url: str,
    image_path: str,
    metadata: DBV4Metadata,
    timeout: int,
) -> np.ndarray:
    with open(image_path, "rb") as f:
        data = f.read()
    request = urllib.request.Request(server_url, data=data, method="POST")
    request.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, list):
        probabilities = np.asarray(payload, dtype=np.float32)
    elif isinstance(payload, dict):
        if payload.get("protocol") != 1:
            raise RuntimeError("サーバーのDBV4 protocol versionが不一致です。")
        if payload.get("model_id") != metadata.repo_id:
            raise RuntimeError("サーバーとクライアントのmodel_idが不一致です。")
        if int(payload.get("output_size", -1)) != metadata.label_count:
            raise RuntimeError("サーバーとクライアントのoutput sizeが不一致です。")
        if payload.get("metadata_version") != metadata.metadata_version:
            raise RuntimeError("サーバーとクライアントのmetadata versionが不一致です。")
        probabilities = np.asarray(payload.get("probabilities", []), dtype=np.float32)
    else:
        raise RuntimeError("サーバー応答形式が不正です。")
    if probabilities.shape[0] != metadata.label_count:
        raise RuntimeError("DBV4 output sizeがmetadataと一致しません。")
    return probabilities


def process_images(args: argparse.Namespace) -> None:
    is_client = args.mode == "client"
    metadata = load_client_metadata(args) if is_client else None
    runtime: Optional[RuntimeModel] = None
    if not is_client:
        runtime = load_runtime_model(
            args.gpu,
            args.model_profile,
            args.model_repo,
            args.model_file,
            args.tags_file,
        )
        metadata = runtime.metadata

    assert metadata is not None
    need_exiftool = (not args.no_tag) or args.organize
    if need_exiftool:
        et_wrapper.start()

    if args.record_ratio is not None:
        APP_CONFIG["record_raw_score"] = args.record_ratio
        APP_CONFIG["record_rating_percentages"] = args.record_ratio

    recursive = args.recursive if args.recursive is not None else not args.organize
    target_files = collect_images(args.images, recursive)
    if not target_files:
        print("[WARN] 対象ファイルが見つかりません。")
        if need_exiftool:
            et_wrapper.stop()
        return

    base_dirs = []
    for raw_path in args.images:
        base = os.path.abspath(raw_path.split("*")[0].split("?")[0])
        if not os.path.isdir(base):
            base = os.path.dirname(base)
        base_dirs.append(base)

    batch_size = 1 if is_client else max(1, args.batch_size)
    if runtime and runtime.batch_limit is not None:
        batch_size = min(batch_size, max(1, runtime.batch_limit))
    io_workers = args.io_workers
    if io_workers < 0:
        io_workers = max(2, min(4, (os.cpu_count() or 1) // 2)) if batch_size > 1 else 0

    if runtime and batch_size > 1:
        print(f"[INFO] DBV4 batch-size={batch_size}, io-workers={io_workers}")

    warmup_time = 0.0
    if runtime:
        try:
            warmup_time = warmup_runtime(runtime, batch_size)
        except Exception as exc:
            print(f"[WARN] ウォームアップ推論をスキップしました: {exc}")

    processed = organized = 0
    inferred = skipped = 0
    inferred_time = skipped_time = 0.0
    executor = (
        ThreadPoolExecutor(max_workers=io_workers)
        if runtime and batch_size > 1 and io_workers > 0
        else None
    )
    pending: List[Dict[str, Any]] = []
    report_data: List[Dict[str, Any]] = []
    progress = tqdm(total=len(target_files), unit="img", dynamic_ncols=True)

    def finalize(
        path: str,
        existing_tags: Sequence[str],
        detected_tags: Sequence[str],
        rating: str,
        probabilities: Optional[np.ndarray],
    ) -> None:
        nonlocal processed, organized
        final_path = path
        if detected_tags and (not args.no_tag or args.organize):
            if et_wrapper.write_tags(path, detected_tags):
                processed += 1
        if args.organize:
            moved, new_path = organize_file(path, rating, args.pixiv, base_dirs)
            if moved:
                organized += 1
                final_path = new_path
        if probabilities is not None and not args.no_report:
            report_data.append(
                {
                    "path": os.path.abspath(final_path),
                    "rating": rating,
                    "probs": [
                        metadata.get_rating_scores(probabilities)[name]
                        for name in DBV4_RATING_NAMES
                    ],
                    "model_id": metadata.repo_id,
                    "metadata_version": metadata.metadata_version,
                }
            )
        progress.update(1)

    def decode_and_finalize(item: Dict[str, Any], probabilities: np.ndarray) -> None:
        display = os.path.basename(item["path"])
        if len(display) > 20:
            display = display[:17] + "..."
        rating = calculate_rating(
            metadata,
            probabilities,
            args.rating_thresh,
            args.ignore_sensitive,
            float(APP_CONFIG.get("general_threshold", 0.40)),
            display,
        )
        tags = [rating]
        tags.extend(format_score_tags(metadata, probabilities))
        tags.extend(metadata.decode_tags(probabilities, args.thresh))
        for old_tag in preserve_existing_tags(item.get("existing_tags", [])):
            if old_tag not in tags:
                tags.append(old_tag)
        finalize(item["path"], item.get("existing_tags", []), tags, rating, probabilities)

    def load_image(path: str) -> Tuple[Optional[Image.Image], Optional[Exception]]:
        try:
            with Image.open(path) as image:
                return image.convert("RGB"), None
        except Exception as exc:
            return None, exc

    def run_batch(items: Sequence[Dict[str, Any]]) -> None:
        nonlocal inferred, inferred_time
        if not runtime or not items:
            return
        started = time.time()
        if executor:
            loaded = list(executor.map(lambda item: load_image(item["path"]), items))
        else:
            loaded = [load_image(item["path"]) for item in items]
        valid_items: List[Dict[str, Any]] = []
        images: List[Image.Image] = []
        for item, (image, error) in zip(items, loaded):
            if error or image is None:
                safe_write(f"エラー {os.path.basename(item['path'])}: {error}")
                progress.update(1)
                continue
            valid_items.append(item)
            images.append(image)
        if not valid_items:
            return
        try:
            predictions = runtime.predict_images(images)
            elapsed = time.time() - started
            inferred += len(valid_items)
            inferred_time += elapsed
            for item, prediction in zip(valid_items, predictions):
                decode_and_finalize(item, prediction)
        except Exception as exc:
            safe_write(f"[WARN] DBV4バッチ推論に失敗: {exc} -> 1枚ずつに切り替えます。")
            for item, image in zip(valid_items, images):
                single_started = time.time()
                try:
                    prediction = runtime.predict_images([image])[0]
                    inferred += 1
                    inferred_time += time.time() - single_started
                    decode_and_finalize(item, prediction)
                except Exception as single_exc:
                    safe_write(f"エラー {os.path.basename(item['path'])}: {single_exc}")
                    progress.update(1)

    try:
        for image_path in target_files:
            started = time.time()
            existing_tags = et_wrapper.get_tags(image_path) if need_exiftool else []
            raw_scores = extract_raw_rating_scores(metadata, existing_tags) if not args.force else None

            if raw_scores is not None and args.rating_thresh is None:
                probabilities = np.zeros(metadata.label_count, dtype=np.float32)
                for name, score in zip(DBV4_RATING_NAMES, raw_scores):
                    probabilities[metadata.rating_indices[name]] = score
                rating = calculate_rating(
                    metadata,
                    probabilities,
                    None,
                    args.ignore_sensitive,
                    float(APP_CONFIG.get("general_threshold", 0.40)),
                )
                skipped += 1
                skipped_time += time.time() - started
                finalize(image_path, existing_tags, [], rating, None)
                continue

            item = {"path": image_path, "existing_tags": existing_tags}
            if is_client:
                try:
                    prediction = client_predict(
                        f"http://{args.host}:{args.port}",
                        image_path,
                        metadata,
                        int(APP_CONFIG.get("client_timeout", 15)),
                    )
                    inferred += 1
                    inferred_time += time.time() - started
                    decode_and_finalize(item, prediction)
                except (urllib.error.URLError, socket.timeout) as exc:
                    safe_write(f"接続エラー(タイムアウト含む): {exc}")
                    break
            else:
                pending.append(item)
                if len(pending) >= batch_size:
                    run_batch(pending[:batch_size])
                    pending = pending[batch_size:]
    finally:
        if runtime and pending:
            run_batch(pending)
        if executor:
            executor.shutdown(wait=True)
        if need_exiftool:
            et_wrapper.stop()
        progress.close()

    print("\n[完了] 処理結果サマリー:")
    infer_speed = inferred / inferred_time if inferred_time else 0.0
    skip_speed = skipped / skipped_time if skipped_time else 0.0
    print(f"  ・推論実行ファイル (DBV4 AI演算あり): {inferred} 枚 | {infer_speed:.2f} img/s")
    print(f"  ・演算スキップファイル (DBV4 score): {skipped} 枚 | {skip_speed:.2f} img/s")
    if warmup_time > 0:
        print(f"  ・ウォームアップ: {warmup_time:.2f} 秒")
    print(f"  ・詳細: タグ書き込み {processed} 枚, 整理移動 {organized} 枚")

    if report_data and not args.no_report:
        with open(REPORT_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False)
        if make_report:
            make_report.make_report()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DBV4 Tagger Universal (日本語版)")
    parser.add_argument("images", nargs="*", help="処理対象の画像またはフォルダパス")
    parser.add_argument("--mode", choices=["standalone", "server", "client"], default="standalone")
    parser.add_argument("--no-tag", action="store_true", help="タグ付け処理を行わない")
    parser.add_argument("--organize", action="store_true", help="レーティングに基づきフォルダ整理を行う")
    parser.add_argument("--pixiv", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="DBV4のtag best_thresholdを上書きする明示的な閾値",
    )
    parser.add_argument("--gpu", action="store_true", help="GPUを使用する")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--recursive", action="store_const", const=True, default=None)
    parser.add_argument("--no-recursive", action="store_const", const=False, dest="recursive")
    parser.add_argument(
        "--model-profile",
        choices=list(MODEL_PROFILES.keys()),
        default=None,
        help="DBV4モデルプロファイル",
    )
    parser.add_argument("--model-repo", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tags-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--sensitive-split-mode", choices=[2, 4, 6], type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--record-ratio", action="store_true", default=None)
    parser.add_argument("--no-record-ratio", action="store_false", dest="record_ratio")
    parser.add_argument("--rating-thresh", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ignore-sensitive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gen-config", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.gen_config:
        load_config()
        return

    profiles = APP_CONFIG.get("model_profiles", MODEL_PROFILES)
    if args.model_profile is None:
        args.model_profile = APP_CONFIG.get("model_profile", "balanced")
    if args.model_profile not in profiles:
        raise SystemExit(f"[ERROR] 不明なDBV4モデルプロファイルです: {args.model_profile}")

    if args.sensitive_split_mode is not None:
        print("[WARN] --sensitive-split-mode はDBV4移行後は非推奨です。R-15/R-17は5段階固定で処理します。")

    if args.record_ratio is not None:
        APP_CONFIG["record_raw_score"] = bool(args.record_ratio)
        APP_CONFIG["record_rating_percentages"] = bool(args.record_ratio)

    if args.host is None:
        if args.mode == "client":
            hosts = APP_CONFIG.get("server_hosts", ["localhost"])
            hosts = [hosts] if isinstance(hosts, str) else hosts
            if len(hosts) == 1:
                args.host = hosts[0]
            else:
                print(f"{Colors.CYAN}[INFO] 接続先サーバーを選択してください。{Colors.RESET}")
                for index, host in enumerate(hosts, 1):
                    print(f"  {index}: {host}")
                while True:
                    try:
                        index = int(input("番号: ")) - 1
                        if 0 <= index < len(hosts):
                            args.host = hosts[index]
                            break
                    except ValueError:
                        pass
        else:
            args.host = "localhost"
    if args.port is None:
        args.port = int(APP_CONFIG.get("server_port", 5000))

    if args.pixiv:
        args.organize = True
        args.recursive = True

    if args.mode == "server":
        run_server(args)
        return
    if not args.images:
        print("[案内] 画像ファイルまたはフォルダを指定してください。")
        create_parser().print_help()
        return
    process_images(args)


if __name__ == "__main__":
    main()
