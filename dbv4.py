import csv
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


DBV4_RATING_NAMES = ("general", "sensitive", "questionable", "explicit")
DBV4_CATEGORY_IDS = {"general": 0, "character": 4, "rating": 9}

MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "lightweight": {
        "repo_id": "animetimm/mobilenetv4_conv_small.dbv4-full",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
    },
    "balanced": {
        "repo_id": "animetimm/convformer_s36.dbv4-full",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
    },
    "high": {
        "repo_id": "animetimm/swinv2_base_window8_256.dbv4-full",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
    },
    "large": {
        "repo_id": "animetimm/eva02_large_patch14_448.dbv4-full",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
    },
    "ultra": {
        "repo_id": "animetimm/convnextv2_huge.dbv4-full",
        "model_file": "model.onnx",
        "tags_file": "selected_tags.csv",
        "preprocess_file": "preprocess.json",
        "categories_file": "categories.json",
        "thresholds_file": "thresholds.csv",
    },
}


def clone_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(profile))


def get_model_profile(name: str, profiles: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    source = profiles or MODEL_PROFILES
    if name not in source:
        raise KeyError(f"未知のDBV4モデルプロファイルです: {name}")
    result = clone_profile(source[name])
    result["profile_name"] = name
    return result


def _local_path(filename: Optional[str], base_dir: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    candidates = []
    if os.path.isabs(filename):
        candidates.append(filename)
    if base_dir:
        candidates.append(os.path.join(base_dir, filename))
    candidates.append(os.path.abspath(filename))
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def resolve_model_artifact(
    repo_id: Optional[str],
    filename: str,
    base_dir: Optional[str] = None,
    required: bool = True,
) -> Optional[str]:
    path = _local_path(filename, base_dir)
    if path:
        return path
    if not repo_id:
        if required:
            raise FileNotFoundError(f"DBV4アーティファクトが見つかりません: {filename}")
        return None
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    except Exception:
        if required:
            raise
        return None


def _load_categories(path: Optional[str]) -> Dict[int, str]:
    categories = {value: key for key, value in DBV4_CATEGORY_IDS.items()}
    if not path:
        return categories
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and "category" in item and "name" in item:
                    categories[int(item["category"])] = str(item["name"])
    except (OSError, TypeError, ValueError):
        pass
    return categories


def _load_category_thresholds(path: Optional[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not path:
        return result
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("name")
                value = row.get("threshold")
                if name and value not in (None, ""):
                    result[str(name)] = float(value)
    except (OSError, TypeError, ValueError):
        pass
    return result


def _read_selected_tags(
    path: str,
    categories: Dict[int, str],
    category_thresholds: Dict[str, float],
) -> Tuple[List[str], List[str], Dict[str, float]]:
    labels: List[str] = []
    category_names: List[str] = []
    thresholds: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError("selected_tags.csv に name 列がありません。")
        for row in reader:
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            try:
                category_id = int(row.get("category", 0))
            except (TypeError, ValueError):
                category_id = 0
            category = categories.get(category_id, str(category_id))
            raw_threshold = row.get("best_threshold")
            try:
                threshold = float(raw_threshold) if raw_threshold not in (None, "") else None
            except (TypeError, ValueError):
                threshold = None
            if threshold is None:
                threshold = float(category_thresholds.get(category, 0.35))
            labels.append(name)
            category_names.append(category)
            thresholds[name] = float(threshold)
    return labels, category_names, thresholds


def _metadata_hash(paths: Iterable[Optional[str]]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path or not os.path.exists(path):
            digest.update(b"<missing>")
            continue
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class DBV4Metadata:
    repo_id: str
    profile_name: str
    model_path: str
    tags_path: str
    preprocess_path: str
    categories_path: Optional[str]
    thresholds_path: Optional[str]
    labels: Tuple[str, ...]
    categories: Tuple[str, ...]
    tag_thresholds: Dict[str, float]
    rating_indices: Dict[str, int]
    metadata_version: str

    @classmethod
    def load(
        cls,
        profile: Dict[str, Any],
        base_dir: Optional[str] = None,
        model_repo_override: Optional[str] = None,
        model_file_override: Optional[str] = None,
        tags_file_override: Optional[str] = None,
        load_model: bool = True,
    ) -> "DBV4Metadata":
        repo_id = model_repo_override or profile["repo_id"]
        model_path = resolve_model_artifact(
            repo_id,
            model_file_override or profile.get("model_file", "model.onnx"),
            base_dir,
            True,
        )
        tags_path = resolve_model_artifact(
            repo_id,
            tags_file_override or profile.get("tags_file", "selected_tags.csv"),
            base_dir,
            True,
        )
        preprocess_path = resolve_model_artifact(
            repo_id,
            profile.get("preprocess_file", "preprocess.json"),
            base_dir,
            True,
        )
        categories_path = resolve_model_artifact(
            repo_id,
            profile.get("categories_file", "categories.json"),
            base_dir,
            False,
        )
        thresholds_path = resolve_model_artifact(
            repo_id,
            profile.get("thresholds_file", "thresholds.csv"),
            base_dir,
            False,
        )

        category_map = _load_categories(categories_path)
        category_thresholds = _load_category_thresholds(thresholds_path)
        labels, categories, thresholds = _read_selected_tags(
            tags_path,
            category_map,
            category_thresholds,
        )
        rating_indices = {
            label: index
            for index, (label, category) in enumerate(zip(labels, categories))
            if category == "rating" and label in DBV4_RATING_NAMES
        }
        missing = [name for name in DBV4_RATING_NAMES if name not in rating_indices]
        if missing:
            raise ValueError("DBV4 rating metadataが不足しています: " + ", ".join(missing))

        return cls(
            repo_id=repo_id,
            profile_name=str(profile.get("profile_name", "custom")),
            model_path=model_path,
            tags_path=tags_path,
            preprocess_path=preprocess_path,
            categories_path=categories_path,
            thresholds_path=thresholds_path,
            labels=tuple(labels),
            categories=tuple(categories),
            tag_thresholds=thresholds,
            rating_indices=rating_indices,
            metadata_version=_metadata_hash(
                [tags_path, preprocess_path, categories_path, thresholds_path]
            ),
        )

    @property
    def label_count(self) -> int:
        return len(self.labels)

    @property
    def rating_tags_marker(self) -> str:
        return f"dbv4_model:{self.repo_id}"

    def threshold_for(self, label: str, override: Optional[float] = None) -> float:
        return float(override) if override is not None else float(self.tag_thresholds.get(label, 0.35))

    def get_rating_scores(self, probabilities: Sequence[float]) -> Dict[str, float]:
        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if len(values) != self.label_count:
            raise ValueError(
                f"DBV4 output size mismatch: model={len(values)}, metadata={self.label_count}"
            )
        return {
            name: float(values[index])
            for name, index in self.rating_indices.items()
        }

    def decode_tags(
        self,
        probabilities: Sequence[float],
        threshold_override: Optional[float] = None,
    ) -> List[str]:
        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if len(values) != self.label_count:
            raise ValueError(
                f"DBV4 output size mismatch: model={len(values)}, metadata={self.label_count}"
            )
        return [
            label
            for index, (label, category) in enumerate(zip(self.labels, self.categories))
            if float(values[index]) >= self.threshold_for(label, threshold_override)
            and category != "rating"
        ]

    def summary(self) -> Dict[str, Any]:
        return {
            "protocol": 1,
            "model_id": self.repo_id,
            "profile": self.profile_name,
            "metadata_version": self.metadata_version,
            "output_size": self.label_count,
            "rating_labels": list(DBV4_RATING_NAMES),
        }


class DBV4Preprocessor:
    def __init__(self, payload: Dict[str, Any]):
        self.steps = self._steps(payload)

    @classmethod
    def from_metadata(cls, metadata: DBV4Metadata) -> "DBV4Preprocessor":
        with open(metadata.preprocess_path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    @staticmethod
    def _steps(payload: Dict[str, Any]) -> List[Any]:
        value = payload.get("test", payload)
        if isinstance(value, dict):
            value = value.get("transforms", value.get("steps", []))
        if not isinstance(value, list):
            raise ValueError("preprocess.json の test 定義を解釈できません。")
        return value

    @staticmethod
    def _step(step: Any) -> Tuple[str, Dict[str, Any]]:
        if isinstance(step, str):
            return step.lower(), {}
        if not isinstance(step, dict):
            raise ValueError(f"未知のpreprocess step形式です: {step!r}")
        name = str(step.get("name") or step.get("type") or step.get("transform") or "").lower()
        params = step.get("params") or step.get("kwargs") or {}
        return name, params if isinstance(params, dict) else {}

    @staticmethod
    def _size(value: Any) -> Tuple[int, int]:
        if isinstance(value, (int, float)):
            size = int(value)
            return size, size
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
        raise ValueError(f"画像サイズを解釈できません: {value!r}")

    @staticmethod
    def _color(value: Any) -> Tuple[int, int, int]:
        colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "grey": (128, 128, 128),
        }
        if isinstance(value, str) and value.lower() in colors:
            return colors[value.lower()]
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return (int(value[0]),) * 3
            if len(value) == 3:
                return tuple(int(v) for v in value)
        if isinstance(value, (int, float)):
            return (int(value),) * 3
        return (255, 255, 255)

    @staticmethod
    def _resampling(value: Any) -> int:
        table = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
            "box": Image.Resampling.BOX,
            "hamming": Image.Resampling.HAMMING,
        }
        return table.get(str(value or "bicubic").lower(), Image.Resampling.BICUBIC)

    def __call__(self, image: Image.Image) -> np.ndarray:
        current = image.convert("RGB")
        normalize_steps: List[Dict[str, Any]] = []
        for raw_step in self.steps:
            name, params = self._step(raw_step)
            if name in {"padtosize", "pad_to_size"}:
                target_h, target_w = self._size(params.get("size"))
                width, height = current.size
                if width < target_w or height < target_h:
                    canvas = Image.new(
                        "RGB",
                        (max(width, target_w), max(height, target_h)),
                        self._color(params.get("background_color", "white")),
                    )
                    canvas.paste(
                        current,
                        (
                            (canvas.width - width) // 2,
                            (canvas.height - height) // 2,
                        ),
                    )
                    current = canvas
            elif name == "resize":
                value = params.get("size")
                resample = self._resampling(params.get("interpolation"))
                if isinstance(value, (int, float)):
                    size = int(value)
                    width, height = current.size
                    if width == height:
                        target = (size, size)
                    elif width < height:
                        target = (size, max(1, round(height * size / width)))
                    else:
                        target = (max(1, round(width * size / height)), size)
                else:
                    h, w = self._size(value)
                    target = (w, h)
                current = current.resize(target, resample)
            elif name in {"centercrop", "center_crop"}:
                h, w = self._size(params.get("size"))
                width, height = current.size
                left = max(0, (width - w) // 2)
                top = max(0, (height - h) // 2)
                current = current.crop((left, top, left + w, top + h))
            elif name in {
                "maybetotensor",
                "totensor",
                "convertimagedtype",
                "convert_image_dtype",
                "identity",
                "",
            }:
                continue
            elif name == "normalize":
                normalize_steps.append(params)
            else:
                raise ValueError(f"未対応のDBV4前処理です: {name}")

        array = np.asarray(current, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        for params in normalize_steps:
            mean = np.asarray(
                params.get("mean", [0.485, 0.456, 0.406]),
                dtype=np.float32,
            ).reshape(3, 1, 1)
            std = np.asarray(
                params.get("std", [0.229, 0.224, 0.225]),
                dtype=np.float32,
            ).reshape(3, 1, 1)
            array = (array - mean) / std
        return array.astype(np.float32)


def infer_output_to_probabilities(raw_output: Any) -> np.ndarray:
    values = np.asarray(raw_output, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("DBV4 outputに非有限値があります。")
    if values.size == 0:
        raise ValueError("DBV4 outputが空です。")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        values = 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))
    return np.clip(values, 0.0, 1.0)


def detect_input_layout(input_shape: Sequence[Any]) -> str:
    if len(input_shape) != 4:
        raise ValueError(f"4次元入力を想定しています: {input_shape}")
    if input_shape[1] == 3:
        return "NCHW"
    if input_shape[-1] == 3:
        return "NHWC"
    return "NCHW"


def adapt_input_layout(batch_nchw: np.ndarray, input_shape: Sequence[Any]) -> np.ndarray:
    if detect_input_layout(input_shape) == "NHWC":
        return np.transpose(batch_nchw, (0, 2, 3, 1)).astype(np.float32)
    return batch_nchw.astype(np.float32)
