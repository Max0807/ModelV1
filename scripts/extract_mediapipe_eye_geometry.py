"""Extract label-free 15-point MediaPipe geometry for each anatomical eye.

Each eye uses ten eyelid/canthus points and five refined iris points.  The NPZ
keeps every selected dataset row (including failures) so ``sample_id`` remains
the authoritative join key.  Coordinates are stored in source-image pixels,
eye-crop-normalized coordinates, and a mirrored anatomical eye-local frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.depth_prior.dataset_assets import resolve_image_asset_path
from modelv1.data.eye_geometry_pseudo_labels import (
    EYE_GEOMETRY_PSEUDO_LABEL_SCHEMA_VERSION,
    EYE_PSEUDO_GEOMETRY_FEATURE_NAMES,
    EYE_PSEUDO_IRIS_CENTER_INDEX,
    EYE_PSEUDO_POINT_COUNT,
    EYE_PSEUDO_POINT_NAMES,
)
from modelv1.processed_artifacts import canonical_dataset_id


SCHEMA_VERSION = EYE_GEOMETRY_PSEUDO_LABEL_SCHEMA_VERSION
POINT_COUNT_PER_EYE = EYE_PSEUDO_POINT_COUNT
EYE_POINT_NAMES = EYE_PSEUDO_POINT_NAMES

# MediaPipe's 473 group lies on the subject's anatomical left eye (image
# right), while the 468 group lies on the anatomical right eye (image left).
# Both index lists are ordered outer canthus -> upper lid -> inner canthus ->
# lower lid -> outer canthus, followed by iris centre and four rim points.
LEFT_EYE_MEDIAPIPE_INDICES = (
    263,
    466,
    388,
    386,
    384,
    362,
    398,
    373,
    374,
    380,
    473,
    474,
    475,
    476,
    477,
)  # 左眼15关键点
RIGHT_EYE_MEDIAPIPE_INDICES = (
    33,
    246,
    161,
    159,
    157,
    133,
    155,
    153,
    145,
    163,
    468,
    469,
    470,
    471,
    472,
)  # 右眼15关键点
GEOMETRY_FEATURE_NAMES = EYE_PSEUDO_GEOMETRY_FEATURE_NAMES
GEOMETRY_FEATURE_COUNT = len(GEOMETRY_FEATURE_NAMES)


@dataclass(frozen=True)
class EyeFit:
    crop_normalized: np.ndarray
    local: np.ndarray
    point_valid_mask: np.ndarray
    features: np.ndarray
    valid: bool
    invalid_reason: str


def parse_args() -> argparse.Namespace:
    """解析独立 Eye15 提取工具的输入 CSV、输出路径和质量阈值。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Numeric suffix, e.g. 11, or dataset_dual_rigid_body_11.",
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed_v2_160",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Default: <root>/dataset<ID>/shared/modelv1_dataset<ID>.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the default root-level mediapipe_eye_geometry_v1_dataset<ID>.npz.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional data_collection root used when paths saved in the CSV are stale.",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-canthus-distance-px", type=float, default=8.0)
    parser.add_argument("--min-crop-inside-ratio", type=float, default=0.8)
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.25,
        help="Heuristic validity threshold; points are retained even below it.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_id = canonical_dataset_id(args.dataset_id)
    suffix = f"dataset{dataset_id}"
    root = args.preprocessed_root.resolve()
    args.dataset_id = dataset_id
    args.dataset_name = f"dataset_dual_rigid_body_{dataset_id}"
    args.csv = (
        args.csv.resolve()
        if args.csv is not None
        else root / suffix / "shared" / f"modelv1_dataset{dataset_id}.csv"
    )
    args.output = (
        args.output.resolve()
        if args.output is not None
        else root / f"mediapipe_eye_geometry_v1_{suffix}.npz"
    )
    args.report = (
        args.report.resolve()
        if args.report is not None
        else root / f"mediapipe_eye_geometry_v1_{suffix}_report.json"
    )
    return args


def validate_args(args: argparse.Namespace) -> None:
    """校验输入文件、阈值范围、处理数量和输出覆盖策略。"""

    if not args.csv.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {args.csv}")
    if not 0.0 <= args.min_detection_confidence <= 1.0:
        raise ValueError("--min-detection-confidence must lie in [0, 1].")
    if args.min_canthus_distance_px <= 0.0:
        raise ValueError("--min-canthus-distance-px must be positive.")
    if not 0.0 <= args.min_crop_inside_ratio <= 1.0:
        raise ValueError("--min-crop-inside-ratio must lie in [0, 1].")
    if not 0.0 <= args.min_quality_score <= 1.0:
        raise ValueError("--min-quality-score must lie in [0, 1].")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    existing = [path for path in (args.output, args.report) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing}. Use --overwrite to replace it."
        )


def read_rows(
    path: Path,
    *,
    dataset_name: str,
    limit: int | None,
) -> list[dict[str, str]]:
    """读取指定数据集的样本，验证必需列与 sample_id 唯一性并应用数量限制。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset CSV has no header: {path}")
        required = {
            "sample_id",
            "dataset",
            "image_name",
            "source_dataset_dir",
            "source_image_path",
            "left_eye_bbox_x",
            "left_eye_bbox_y",
            "left_eye_bbox_w",
            "left_eye_bbox_h",
            "right_eye_bbox_x",
            "right_eye_bbox_y",
            "right_eye_bbox_w",
            "right_eye_bbox_h",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset CSV is missing columns: {sorted(missing)}")
        rows = [row for row in reader if row["dataset"] == dataset_name]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows selected for dataset {dataset_name!r}.")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Selected dataset rows contain duplicate sample_id values.")
    return rows


def eye_bbox(row: Mapping[str, str], side: str) -> tuple[float, float, float, float]:
    """从样本行读取指定解剖侧眼框，并验证 xywh 数值有限且宽高为正。"""

    values = tuple(float(row[f"{side}_eye_bbox_{axis}"]) for axis in ("x", "y", "w", "h"))
    if not all(math.isfinite(value) for value in values) or values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"invalid_{side}_eye_bbox")
    return values


def extract_selected_points(
    landmarks: Sequence[Any],
    indices: Sequence[int],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """按固定 Eye15 索引从 refined FaceMesh 中抽点并转换为原图像素坐标。"""

    if len(indices) != POINT_COUNT_PER_EYE:
        raise ValueError(f"Expected {POINT_COUNT_PER_EYE} landmark indices.")
    if len(landmarks) <= max(indices):
        raise ValueError(
            "MediaPipe result has no refined iris landmarks; ensure "
            "refine_landmarks=True."
        )
    return np.asarray(
        [[float(landmarks[index].x * width), float(landmarks[index].y * height)] for index in indices],
        dtype=np.float32,
    )


def _soft_range_score(value: float, low: float, high: float, margin: float) -> float:
    """将数值映射为区间内为 1、区间外按 margin 线性衰减的质量分数。"""

    if low <= value <= high:
        return 1.0
    if value < low:
        return float(np.clip(1.0 - (low - value) / margin, 0.0, 1.0))
    return float(np.clip(1.0 - (value - high) / margin, 0.0, 1.0))


def fit_eye_geometry(
    points_px: np.ndarray,
    *,
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    min_canthus_distance_px: float,
    min_crop_inside_ratio: float,
    min_quality_score: float,
) -> EyeFit:
    """建立眼部局部坐标，拟合虹膜/眼睑特征并依据几何阈值生成有效性结果。"""

    points = np.asarray(points_px, dtype=np.float64)
    if points.shape != (POINT_COUNT_PER_EYE, 2):
        raise ValueError(
            f"points_px must have shape ({POINT_COUNT_PER_EYE}, 2), got {points.shape}."
        )
    if not np.isfinite(points).all():
        raise ValueError("eye_points_non_finite")
    x, y, width, height = bbox
    crop = np.column_stack(((points[:, 0] - x) / width, (points[:, 1] - y) / height))
    in_image = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < image_width)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < image_height)
    )
    in_crop = (
        (crop[:, 0] >= 0.0)
        & (crop[:, 0] <= 1.0)
        & (crop[:, 1] >= 0.0)
        & (crop[:, 1] <= 1.0)
    )

    outer = points[0]
    inner = points[5]
    horizontal = inner - outer
    canthus_distance = float(np.linalg.norm(horizontal))
    if canthus_distance <= 1e-8:
        raise ValueError("canthus_distance_zero")
    horizontal /= canthus_distance
    upper_mean = points[1:5].mean(axis=0)
    lower_mean = points[6:10].mean(axis=0)
    vertical_hint = lower_mean - upper_mean
    vertical = vertical_hint - np.dot(vertical_hint, horizontal) * horizontal
    vertical_norm = float(np.linalg.norm(vertical))
    if vertical_norm <= 1e-8:
        vertical = np.asarray([-horizontal[1], horizontal[0]], dtype=np.float64)
        if np.dot(vertical, vertical_hint) < 0.0:
            vertical = -vertical
    else:
        vertical /= vertical_norm
    origin = 0.5 * (outer + inner)
    relative = points - origin
    local = np.column_stack(
        (relative @ horizontal / canthus_distance, relative @ vertical / canthus_distance)
    )

    iris_center = local[EYE_PSEUDO_IRIS_CENTER_INDEX]
    iris_rim = local[EYE_PSEUDO_IRIS_CENTER_INDEX + 1 :]
    fitted_center = iris_rim.mean(axis=0)
    radii = np.linalg.norm(iris_rim - fitted_center, axis=1)
    iris_radius = float(radii.mean())
    radius_cv = float(radii.std() / max(iris_radius, 1e-8))
    center_fit_offset = float(np.linalg.norm(iris_center - fitted_center))
    eyelid_opening = float(local[6:10, 1].mean() - local[1:5, 1].mean())
    image_ratio = float(in_image.mean())
    crop_ratio = float(in_crop.mean())

    circle_score = math.exp(-4.0 * radius_cv)
    center_score = math.exp(-2.0 * center_fit_offset / max(iris_radius, 1e-8))
    radius_score = _soft_range_score(iris_radius, 0.035, 0.30, 0.10)
    opening_score = _soft_range_score(eyelid_opening, 0.025, 0.60, 0.10)
    factors = np.asarray(
        [image_ratio, crop_ratio, circle_score, center_score, radius_score, opening_score],
        dtype=np.float64,
    ).clip(0.0, 1.0)
    quality = float(np.prod(np.maximum(factors, 1e-8)) ** (1.0 / len(factors)))
    features = np.asarray(
        [
            iris_center[0],
            iris_center[1],
            iris_radius,
            radius_cv,
            center_fit_offset,
            eyelid_opening,
            image_ratio,
            crop_ratio,
            canthus_distance,
            quality,
        ],
        dtype=np.float32,
    )

    invalid_reasons: list[str] = []
    if canthus_distance < min_canthus_distance_px:
        invalid_reasons.append("canthus_too_small")
    if image_ratio < 1.0:
        invalid_reasons.append("point_outside_image")
    if crop_ratio < min_crop_inside_ratio:
        invalid_reasons.append("points_outside_eye_crop")
    if not 0.01 <= iris_radius <= 0.5:
        invalid_reasons.append("implausible_iris_radius")
    if eyelid_opening <= 0.0:
        invalid_reasons.append("nonpositive_eyelid_opening")
    if quality < min_quality_score:
        invalid_reasons.append("quality_below_threshold")
    return EyeFit(
        crop_normalized=crop.astype(np.float32),
        local=local.astype(np.float32),
        point_valid_mask=(in_image & in_crop),
        features=features,
        valid=not invalid_reasons,
        invalid_reason=";".join(invalid_reasons),
    )


def _empty_arrays(count: int) -> dict[str, np.ndarray]:
    """按样本数预分配 Eye15 坐标、几何特征、质量分数和各级 mask。"""

    point_shape = (count, POINT_COUNT_PER_EYE, 2)
    return {
        "image_size_px": np.zeros((count, 2), dtype=np.int32),  # [N, 2] 每张图像的宽、高
        "detection_success_mask": np.zeros(count, dtype=np.bool_),  # [N] FaceMesh 是否成功执行
        "valid_mask": np.zeros(count, dtype=np.bool_),  # [N] 左右两眼是否都有效
        "left_eye_valid_mask": np.zeros(count, dtype=np.bool_),  # [N] 左眼是否通过质量检查
        "right_eye_valid_mask": np.zeros(count, dtype=np.bool_),  # [N] 右眼是否通过质量检查
        "left_eye_points_px": np.full(point_shape, np.nan, dtype=np.float32),  # [N, 15, 2] 左眼 Eye15 像素坐标
        "right_eye_points_px": np.full(point_shape, np.nan, dtype=np.float32),  # [N, 15, 2] 右眼 Eye15 像素坐标
        "left_eye_points_crop_norm": np.full(point_shape, np.nan, dtype=np.float32),  # [N, 15, 2] 左眼 Eye15 相对裁剪框的归一化坐标
        "right_eye_points_crop_norm": np.full(point_shape, np.nan, dtype=np.float32),  # [N, 15, 2] 右眼 Eye15 相对裁剪框的归一化坐标
        "left_eye_points_local": np.full(point_shape, np.nan, dtype=np.float32),  # [N, 15, 2] 左眼 Eye15 局部坐标
        "right_eye_points_local": np.full(point_shape, np.nan, dtype=np.float32),  # [N, 15, 2] 右眼 Eye15 局部坐标
        "left_eye_point_valid_mask": np.zeros(
            (count, POINT_COUNT_PER_EYE), dtype=np.bool_
        ),  # [N, 15] 每个左眼点是否有效
        "right_eye_point_valid_mask": np.zeros(
            (count, POINT_COUNT_PER_EYE), dtype=np.bool_
        ),  # [N, 15] 每个右眼点是否有效
        "left_eye_geometry": np.full(
            (count, GEOMETRY_FEATURE_COUNT), np.nan, dtype=np.float32
        ),  # [N, 10] 左眼的 10 个几何特征
        "right_eye_geometry": np.full(
            (count, GEOMETRY_FEATURE_COUNT), np.nan, dtype=np.float32
        ),  # [N, 10] 右眼的 10 个几何特征
        "sample_quality_score": np.zeros(count, dtype=np.float32),  # [N] 样本综合质量分数
    }


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    """统计有限数值的数量、极值和 5/50/95 分位数，供报告审计。"""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: None for key in ("count", "min", "p05", "median", "p95", "max")}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256，供 NPZ 产物完整性记录。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_npz_atomic(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    """先写临时压缩 NPZ 再原子替换目标，避免半成品归档。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        if temporary.stat().st_size <= 0:
            raise RuntimeError("Temporary NPZ is empty.")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """以 UTF-8 写入临时 JSON 后原子替换质量报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_metadata(args: argparse.Namespace, row_count: int) -> dict[str, Any]:
    """构建 Eye15 点序、左右语义、坐标约定和阈值等可复现实验元数据。"""

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "dataset_csv": str(args.csv),
        "row_count": row_count,
        "point_count_per_eye": POINT_COUNT_PER_EYE,
        "point_names": list(EYE_POINT_NAMES),
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
        "anatomical_side_semantics": {
            "left": "subject-left/image-right/MediaPipe iris group 473",
            "right": "subject-right/image-left/MediaPipe iris group 468",
        },
        "mediapipe_indices": {
            "left": list(LEFT_EYE_MEDIAPIPE_INDICES),
            "right": list(RIGHT_EYE_MEDIAPIPE_INDICES),
        },
        "coordinate_conventions": {
            "points_px": "source image pixels, origin top-left",
            "points_crop_norm": "(point - anatomical eye bbox origin) / bbox size",
            "points_local": (
                "origin at canthus midpoint; x is outer-to-inner; y points from "
                "upper to lower eyelid; both axes divided by canthus distance"
            ),
        },
        "quality_score_note": (
            "Label-free heuristic in [0,1], not a calibrated probability. It combines "
            "image/crop containment, iris circularity, centre consistency, radius, "
            "and eyelid opening."
        ),
        "parameters": {
            "min_detection_confidence": args.min_detection_confidence,
            "min_canthus_distance_px": args.min_canthus_distance_px,
            "min_crop_inside_ratio": args.min_crop_inside_ratio,
            "min_quality_score": args.min_quality_score,
            "limit": args.limit,
        },
    }


def main() -> int:
    """运行独立 MediaPipe Eye15 提取流程并写出 NPZ 与统计报告。"""

    args = parse_args()
    validate_args(args)
    try:
        import mediapipe as mp
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Extraction requires MediaPipe, NumPy, and Pillow.") from error

    rows = read_rows(args.csv, dataset_name=args.dataset_name, limit=args.limit)
    arrays = _empty_arrays(len(rows))
    detection_reasons = [""] * len(rows)
    left_invalid_reasons = [""] * len(rows)
    right_invalid_reasons = [""] * len(rows)
    resolved_source_paths = [""] * len(rows)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
    ) as detector:
        for index, row in enumerate(rows):
            try:
                image_path = resolve_image_asset_path(
                    row, "source_image_path", args.image_root
                )
                resolved_source_paths[index] = str(image_path.resolve())
                with Image.open(image_path) as image:
                    rgb = np.asarray(image.convert("RGB"))
                height, width = rgb.shape[:2]
                arrays["image_size_px"][index] = (width, height)
                result = detector.process(rgb)
                if not result.multi_face_landmarks:
                    raise RuntimeError("MediaPipe Face Mesh found no face")
                landmarks = result.multi_face_landmarks[0].landmark
                left_points = extract_selected_points(
                    landmarks,
                    LEFT_EYE_MEDIAPIPE_INDICES,
                    width=width,
                    height=height,
                )
                right_points = extract_selected_points(
                    landmarks,
                    RIGHT_EYE_MEDIAPIPE_INDICES,
                    width=width,
                    height=height,
                )
                left_fit = fit_eye_geometry(
                    left_points,
                    bbox=eye_bbox(row, "left"),
                    image_width=width,
                    image_height=height,
                    min_canthus_distance_px=args.min_canthus_distance_px,
                    min_crop_inside_ratio=args.min_crop_inside_ratio,
                    min_quality_score=args.min_quality_score,
                )
                right_fit = fit_eye_geometry(
                    right_points,
                    bbox=eye_bbox(row, "right"),
                    image_width=width,
                    image_height=height,
                    min_canthus_distance_px=args.min_canthus_distance_px,
                    min_crop_inside_ratio=args.min_crop_inside_ratio,
                    min_quality_score=args.min_quality_score,
                )
                arrays["detection_success_mask"][index] = True
                arrays["left_eye_points_px"][index] = left_points
                arrays["right_eye_points_px"][index] = right_points
                arrays["left_eye_points_crop_norm"][index] = left_fit.crop_normalized
                arrays["right_eye_points_crop_norm"][index] = right_fit.crop_normalized
                arrays["left_eye_points_local"][index] = left_fit.local
                arrays["right_eye_points_local"][index] = right_fit.local
                arrays["left_eye_point_valid_mask"][index] = left_fit.point_valid_mask
                arrays["right_eye_point_valid_mask"][index] = right_fit.point_valid_mask
                arrays["left_eye_geometry"][index] = left_fit.features
                arrays["right_eye_geometry"][index] = right_fit.features
                arrays["left_eye_valid_mask"][index] = left_fit.valid
                arrays["right_eye_valid_mask"][index] = right_fit.valid
                arrays["valid_mask"][index] = left_fit.valid and right_fit.valid
                arrays["sample_quality_score"][index] = min(
                    float(left_fit.features[-1]), float(right_fit.features[-1])
                )
                left_invalid_reasons[index] = left_fit.invalid_reason
                right_invalid_reasons[index] = right_fit.invalid_reason
            except Exception as error:
                detection_reasons[index] = f"{type(error).__name__}: {error}"
            processed = index + 1
            if processed % args.log_every == 0 or processed == len(rows):
                success = int(arrays["detection_success_mask"][:processed].sum())
                valid = int(arrays["valid_mask"][:processed].sum())
                print(
                    f"Processed {processed}/{len(rows)} "
                    f"(detected={success}, valid={valid})"
                )

    metadata = build_metadata(args, len(rows))
    payload: dict[str, np.ndarray] = {
        "sample_id": np.asarray([row["sample_id"] for row in rows]),
        "dataset": np.asarray([row["dataset"] for row in rows]),
        "image_name": np.asarray([row["image_name"] for row in rows]),
        "source_image_path": np.asarray(resolved_source_paths),
        "detection_reason": np.asarray(detection_reasons),
        "left_eye_invalid_reason": np.asarray(left_invalid_reasons),
        "right_eye_invalid_reason": np.asarray(right_invalid_reasons),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
        **arrays,
    }
    write_npz_atomic(args.output, payload)

    detected = arrays["detection_success_mask"]
    valid = arrays["valid_mask"]
    feature_index = {name: index for index, name in enumerate(GEOMETRY_FEATURE_NAMES)}
    report = {
        **metadata,
        "output_npz": str(args.output),
        "output_npz_sha256": sha256_file(args.output),
        "counts": {
            "total": len(rows),
            "detection_success": int(detected.sum()),
            "detection_failed": int((~detected).sum()),
            "left_eye_valid": int(arrays["left_eye_valid_mask"].sum()),
            "right_eye_valid": int(arrays["right_eye_valid_mask"].sum()),
            "both_eyes_valid": int(valid.sum()),
        },
        "detection_failure_types": dict(
            Counter(
                reason.split(":", 1)[0]
                for reason in detection_reasons
                if reason
            )
        ),
        "left_eye_invalid_reasons": dict(
            Counter(
                item
                for reason in left_invalid_reasons
                for item in reason.split(";")
                if item
            )
        ),
        "right_eye_invalid_reasons": dict(
            Counter(
                item
                for reason in right_invalid_reasons
                for item in reason.split(";")
                if item
            )
        ),
        "distributions": {
            "sample_quality_score": distribution(
                arrays["sample_quality_score"][detected]
            ),
            "left_iris_center_y_local": distribution(
                arrays["left_eye_geometry"][detected, feature_index["iris_center_y_local"]]
            ),
            "right_iris_center_y_local": distribution(
                arrays["right_eye_geometry"][detected, feature_index["iris_center_y_local"]]
            ),
            "left_eyelid_opening_local": distribution(
                arrays["left_eye_geometry"][detected, feature_index["eyelid_opening_local"]]
            ),
            "right_eyelid_opening_local": distribution(
                arrays["right_eye_geometry"][detected, feature_index["eyelid_opening_local"]]
            ),
        },
    }
    write_json_atomic(args.report, report)
    print(f"Wrote NPZ: {args.output}")
    print(f"Wrote report: {args.report}")
    print(
        f"Detected {int(detected.sum())}/{len(rows)}; "
        f"both eyes valid {int(valid.sum())}/{len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
