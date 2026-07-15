"""Build the unified ModelV1 gaze dataset.

The script reads the CrossGaze data-collection folders, joins Vicon labels with
InsightFace crop metadata by image name, and writes a flat CSV suitable for a
future PyTorch Dataset.

It intentionally depends only on the Python standard library. If NumPy is
installed, pass --write-npz to also emit compact array files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_SOURCE_ROOT = Path(r"D:\GithubCode\CrossGaze-main\baseline\data_collection")
DEFAULT_OUTPUT_DIR = Path("data/processed")

IMAGE_W = 1920.0
IMAGE_H = 1080.0
W_REF = 1920.0
H_REF = 1080.0

FACE_CROP_SIZE = (224.0, 224.0)
EYE_CROP_SIZE = (60.0, 36.0)

FX = 1367.8584
FY = 1369.0087
CX = 957.9159
CY = 543.3381

# The old recorder names this CAMERA_OPTICAL_TO_RIGID_MATRIX, but its own
# comments and usage treat it as T_rigid_optical.
HAND_EYE_RIGID_TO_OPTICAL = [
    [0.00146143, 0.50358286, 0.86394569, 0.06705835],
    [-0.99981876, -0.01566358, 0.01082137, 0.03084992],
    [0.01898194, -0.86380493, 0.50346870, -0.03494481],
    [0.0, 0.0, 0.0, 1.0],
]

CROP_CAM_COLUMNS = [f"crop_cam_{idx:02d}" for idx in range(36)]
SCENE_COLUMNS = [f"scene_{idx:02d}" for idx in range(25)]

BASE_COLUMNS = [
    "sample_id",
    "dataset",
    "image_name",
    "frame_idx",
    "source_dataset_dir",
    "source_image_path",
    "face_path",
    "left_eye_path",
    "right_eye_path",
    "has_source_image",
    "has_face_image",
    "has_left_eye_image",
    "has_right_eye_image",
    "pnp_status",
    "z_table_mm",
    "uv_gt_u_mm",
    "uv_gt_v_mm",
    "gaze_target_w_x_mm",
    "gaze_target_w_y_mm",
    "gaze_target_w_z_mm",
    "table_origin_w_x_mm",
    "table_origin_w_y_mm",
    "table_origin_w_z_mm",
    "table_target_z_delta_mm",
    "gaze_cam_recomputed_x_mm",
    "gaze_cam_recomputed_y_mm",
    "gaze_cam_recomputed_z_mm",
    "gaze_cam_csv_x_mm",
    "gaze_cam_csv_y_mm",
    "gaze_cam_csv_z_mm",
    "gaze_cam_error_mm",
    "t_wc_x_mm",
    "t_wc_y_mm",
    "t_wc_z_mm",
    "t_cw_x_mm",
    "t_cw_y_mm",
    "t_cw_z_mm",
    "left_eye_camera_x_mm",
    "left_eye_camera_y_mm",
    "left_eye_camera_z_mm",
    "right_eye_camera_x_mm",
    "right_eye_camera_y_mm",
    "right_eye_camera_z_mm",
    "face_bbox_x",
    "face_bbox_y",
    "face_bbox_w",
    "face_bbox_h",
    "left_eye_bbox_x",
    "left_eye_bbox_y",
    "left_eye_bbox_w",
    "left_eye_bbox_h",
    "right_eye_bbox_x",
    "right_eye_bbox_y",
    "right_eye_bbox_w",
    "right_eye_bbox_h",
]

OUTPUT_COLUMNS = BASE_COLUMNS + CROP_CAM_COLUMNS + SCENE_COLUMNS


class DatasetBuildError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="CrossGaze data_collection directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for modelv1_dataset.csv and modelv1_report.json.",
    )
    parser.add_argument(
        "--output-name",
        default="modelv1_dataset",
        help="Output file stem.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names or numbers, for example: 2 3 4.",
    )
    parser.add_argument(
        "--pnp-csv",
        default="pnp_face_depth_scale_1010.csv",
        help="Preferred PnP CSV name. Falls back to pnp_face_depth.csv.",
    )
    parser.add_argument(
        "--require-pnp",
        action="store_true",
        help="Keep only rows with a successful PnP record.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Keep rows even when face/eye crop images are missing.",
    )
    parser.add_argument(
        "--include-failed-detections",
        action="store_true",
        help="Keep InsightFace rows whose status is not success.",
    )
    parser.add_argument(
        "--handeye-translation-scale",
        type=float,
        default=1000.0,
        help="Scale applied to hand-eye translation. Use 1000 when stored in meters.",
    )
    parser.add_argument(
        "--table-z-scope",
        choices=["per-dataset", "global"],
        default="per-dataset",
        help="Use per-dataset or global mean gaze_target_tz as table height.",
    )
    parser.add_argument(
        "--write-npz",
        action="store_true",
        help="Also write modelv1_dataset.npz if numpy is installed.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key)) for key in OUTPUT_COLUMNS})


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.10g}"
    return str(value)


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def as_int_text(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return str(value).strip()


def normalize_dataset_names(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            result.add(f"dataset_dual_rigid_body_{text}")
        else:
            result.add(text)
    return result


def discover_dataset_dirs(source_root: Path, requested: set[str] | None) -> list[Path]:
    if not source_root.exists():
        raise DatasetBuildError(f"Source root does not exist: {source_root}")
    dirs = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and path.name.startswith("dataset_dual_rigid_body_")
    )
    if requested is not None:
        dirs = [path for path in dirs if path.name in requested]
    return dirs


def choose_one(paths: Iterable[Path]) -> Path | None:
    paths = sorted(paths)
    return paths[-1] if paths else None


def choose_pnp_csv(dataset_dir: Path, preferred_name: str) -> Path | None:
    preferred = dataset_dir / preferred_name
    if preferred.exists():
        return preferred
    fallback = dataset_dir / "pnp_face_depth.csv"
    if fallback.exists():
        return fallback
    scaled = sorted(dataset_dir.glob("pnp_face_depth_scale_*.csv"))
    return scaled[-1] if scaled else None


def require_columns(rows: list[dict[str, str]], columns: list[str], label: str) -> bool:
    if not rows:
        return False
    keys = set(rows[0].keys())
    missing = [column for column in columns if column not in keys]
    if missing:
        return False
    return True


def status_success(row: dict[str, str]) -> bool:
    return row.get("status", "success").strip().lower() == "success"


def index_by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        value = row.get(key, "").strip()
        if value:
            result[value] = row
    return result


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def matvec(a: list[list[float]], v: list[float]) -> list[float]:
    return [sum(a[i][k] * v[k] for k in range(len(v))) for i in range(len(a))]


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def vec_scale(a: list[float], scale: float) -> list[float]:
    return [x * scale for x in a]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def flatten(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def camera_pose_from_row(row: dict[str, str], handeye_translation_scale: float) -> dict[str, list[float] | list[list[float]]]:
    rotation_keys = [
        ["cam_r11", "cam_r12", "cam_r13"],
        ["cam_r21", "cam_r22", "cam_r23"],
        ["cam_r31", "cam_r32", "cam_r33"],
    ]
    r_wr = []
    for row_keys in rotation_keys:
        r_row = []
        for key in row_keys:
            value = as_float(row, key)
            if value is None:
                raise ValueError(f"Missing camera rotation column: {key}")
            r_row.append(value)
        r_wr.append(r_row)

    t_wr = []
    for key in ["cam_tx", "cam_ty", "cam_tz"]:
        value = as_float(row, key)
        if value is None:
            raise ValueError(f"Missing camera translation column: {key}")
        t_wr.append(value)

    r_rc = [matrix_row[:3] for matrix_row in HAND_EYE_RIGID_TO_OPTICAL[:3]]
    t_rc = [matrix_row[3] * handeye_translation_scale for matrix_row in HAND_EYE_RIGID_TO_OPTICAL[:3]]

    r_wc = matmul(r_wr, r_rc)
    t_wc = vec_add(matvec(r_wr, t_rc), t_wr)
    r_cw = transpose(r_wc)
    t_cw = vec_scale(matvec(r_cw, t_wc), -1.0)
    return {"r_wc": r_wc, "t_wc": t_wc, "r_cw": r_cw, "t_cw": t_cw}


def effective_bbox(row: dict[str, str], prefix: str) -> tuple[float, float, float, float] | None:
    x = as_float(row, f"{prefix}_x")
    y = as_float(row, f"{prefix}_y")
    w = as_float(row, f"{prefix}_width")
    h = as_float(row, f"{prefix}_height")
    if x is None or y is None or w is None or h is None:
        return None
    if w <= 0 or h <= 0:
        return None
    x = max(0.0, min(IMAGE_W - 1.0, x))
    y = max(0.0, min(IMAGE_H - 1.0, y))
    x2 = max(x + 1.0, min(IMAGE_W, x + w))
    y2 = max(y + 1.0, min(IMAGE_H, y + h))
    return x, y, x2 - x, y2 - y


def normalized_bbox(bbox: tuple[float, float, float, float]) -> list[float]:
    x, y, w, h = bbox
    return [
        (x + 0.5 * w) / IMAGE_W,
        (y + 0.5 * h) / IMAGE_H,
        w / IMAGE_W,
        h / IMAGE_H,
    ]


def crop_affine(bbox: tuple[float, float, float, float], target_size: tuple[float, float]) -> list[float]:
    x, y, w, h = bbox
    target_w, target_h = target_size
    sx = target_w / w
    sy = target_h / h
    return [sx, 0.0, -x * sx, 0.0, sy, -y * sy]


def build_crop_cam_vec(insight_row: dict[str, str]) -> tuple[list[float], dict[str, tuple[float, float, float, float]]] | None:
    face = effective_bbox(insight_row, "face")
    left_eye = effective_bbox(insight_row, "left_eye")
    right_eye = effective_bbox(insight_row, "right_eye")
    if face is None or left_eye is None or right_eye is None:
        return None

    vector = []
    vector.extend(normalized_bbox(face))
    vector.extend(normalized_bbox(left_eye))
    vector.extend(normalized_bbox(right_eye))
    vector.extend([IMAGE_W / W_REF, IMAGE_H / H_REF])
    vector.extend([FX / IMAGE_W, FY / IMAGE_H, CX / IMAGE_W, CY / IMAGE_H])
    vector.extend(crop_affine(face, FACE_CROP_SIZE))
    vector.extend(crop_affine(left_eye, EYE_CROP_SIZE))
    vector.extend(crop_affine(right_eye, EYE_CROP_SIZE))
    if len(vector) != 36:
        raise AssertionError(f"crop_cam_vec must be 36D, got {len(vector)}")
    return vector, {"face": face, "left_eye": left_eye, "right_eye": right_eye}


def build_scene_vec(
    r_cw: list[list[float]],
    t_cw: list[float],
    t_wc: list[float],
    z_table: float,
) -> tuple[list[float], dict[str, list[float] | float]]:
    n_w = [0.0, 0.0, 1.0]
    e1_w = [1.0, 0.0, 0.0]
    e2_w = [0.0, 1.0, 0.0]
    o_table_w = [t_wc[0], t_wc[1], z_table]

    n_c = matvec(r_cw, n_w)
    o_table_c = vec_add(matvec(r_cw, o_table_w), t_cw)
    e1_c = matvec(r_cw, e1_w)
    e2_c = matvec(r_cw, e2_w)
    d_c = dot(n_c, o_table_c)

    vector = []
    vector.extend(n_c)
    vector.append(d_c)
    vector.extend(o_table_c)
    vector.extend(e1_c)
    vector.extend(e2_c)
    vector.extend(flatten(r_cw))
    vector.extend(t_cw)
    if len(vector) != 25:
        raise AssertionError(f"scene_vec must be 25D, got {len(vector)}")
    return vector, {
        "n_c": n_c,
        "d_c": d_c,
        "o_table_w": o_table_w,
        "o_table_c": o_table_c,
        "e1_c": e1_c,
        "e2_c": e2_c,
    }


def required_gaze_target(row: dict[str, str]) -> list[float] | None:
    values = [as_float(row, key) for key in ["gaze_target_tx", "gaze_target_ty", "gaze_target_tz"]]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values]


def old_gaze_cam(row: dict[str, str]) -> list[float] | None:
    values = [as_float(row, key) for key in ["gaze_cam_tx", "gaze_cam_ty", "gaze_cam_tz"]]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values]


def existing_image_paths(dataset_dir: Path, image_name: str) -> dict[str, Path]:
    return {
        "source": dataset_dir / "insightface_img" / image_name,
        "face": dataset_dir / "insightface_face" / image_name,
        "left_eye": dataset_dir / "insightface_eyes" / "left_eye" / image_name,
        "right_eye": dataset_dir / "insightface_eyes" / "right_eye" / image_name,
    }


def bool_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def pnp_fields(row: dict[str, str] | None) -> dict[str, float | str | None]:
    if row is None:
        return {
            "pnp_status": "",
            "left_eye_camera_x_mm": None,
            "left_eye_camera_y_mm": None,
            "left_eye_camera_z_mm": None,
            "right_eye_camera_x_mm": None,
            "right_eye_camera_y_mm": None,
            "right_eye_camera_z_mm": None,
        }
    return {
        "pnp_status": row.get("status", ""),
        "left_eye_camera_x_mm": as_float(row, "left_eye_camera_x_mm"),
        "left_eye_camera_y_mm": as_float(row, "left_eye_camera_y_mm"),
        "left_eye_camera_z_mm": as_float(row, "left_eye_camera_z_mm"),
        "right_eye_camera_x_mm": as_float(row, "right_eye_camera_x_mm"),
        "right_eye_camera_y_mm": as_float(row, "right_eye_camera_y_mm"),
        "right_eye_camera_z_mm": as_float(row, "right_eye_camera_z_mm"),
    }


def vector_columns(prefix_columns: list[str], values: list[float]) -> dict[str, float]:
    return {column: values[idx] for idx, column in enumerate(prefix_columns)}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return ordered[int(idx)]
    return ordered[low] * (high - idx) + ordered[high] * (idx - low)


def stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "p95": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": mean(clean),
        "p95": percentile(clean, 0.95),
    }


def collect_table_z_values(dataset_dirs: list[Path]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for dataset_dir in dataset_dirs:
        data_log = choose_one(dataset_dir.glob("data_log_*.csv"))
        if data_log is None:
            result[dataset_dir.name] = []
            continue
        rows = read_csv(data_log)
        values = []
        for row in rows:
            gaze_target = required_gaze_target(row)
            if gaze_target is not None:
                values.append(gaze_target[2])
        result[dataset_dir.name] = values
    return result


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = normalize_dataset_names(args.datasets)
    dataset_dirs = discover_dataset_dirs(args.source_root, requested)
    z_values_by_dataset = collect_table_z_values(dataset_dirs)
    global_z_values = [value for values in z_values_by_dataset.values() for value in values]
    global_z_table = mean(global_z_values) if global_z_values else None

    output_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "source_root": str(args.source_root),
        "output_dir": str(args.output_dir),
        "table_z_scope": args.table_z_scope,
        "handeye_translation_scale": args.handeye_translation_scale,
        "require_pnp": args.require_pnp,
        "allow_missing_images": args.allow_missing_images,
        "include_failed_detections": args.include_failed_detections,
        "datasets": [],
        "overall": {},
    }
    all_errors: list[float] = []

    for dataset_dir in dataset_dirs:
        dataset_report: dict[str, Any] = {
            "dataset": dataset_dir.name,
            "dataset_dir": str(dataset_dir),
            "data_log_csv": None,
            "insightface_csv": None,
            "pnp_csv": None,
            "z_table_mm": None,
            "counts": {
                "data_log_rows": 0,
                "insightface_rows": 0,
                "pnp_rows": 0,
                "joined_rows": 0,
                "written_rows": 0,
            },
            "skipped": {
                "missing_data_log": 0,
                "missing_insightface_csv": 0,
                "missing_gaze_target": 0,
                "missing_insightface_row": 0,
                "failed_insightface": 0,
                "invalid_bbox": 0,
                "missing_required_images": 0,
                "missing_pnp": 0,
                "failed_pnp": 0,
                "camera_pose_error": 0,
            },
            "missing_image_counts": {
                "source": 0,
                "face": 0,
                "left_eye": 0,
                "right_eye": 0,
            },
            "gaze_cam_error_mm": {},
        }

        data_log = choose_one(dataset_dir.glob("data_log_*.csv"))
        if data_log is None:
            dataset_report["skipped"]["missing_data_log"] = 1
            report["datasets"].append(dataset_report)
            continue
        dataset_report["data_log_csv"] = str(data_log)

        insightface_csv = dataset_dir / "insightface_coordinates.csv"
        if not insightface_csv.exists():
            dataset_report["skipped"]["missing_insightface_csv"] = 1
            report["datasets"].append(dataset_report)
            continue
        dataset_report["insightface_csv"] = str(insightface_csv)

        pnp_csv = choose_pnp_csv(dataset_dir, args.pnp_csv)
        if pnp_csv is not None:
            dataset_report["pnp_csv"] = str(pnp_csv)

        data_rows = read_csv(data_log)
        insight_rows = read_csv(insightface_csv)
        pnp_rows = read_csv(pnp_csv) if pnp_csv is not None else []
        dataset_report["counts"]["data_log_rows"] = len(data_rows)
        dataset_report["counts"]["insightface_rows"] = len(insight_rows)
        dataset_report["counts"]["pnp_rows"] = len(pnp_rows)

        if not require_columns(data_rows, ["image_filename"], "data_log"):
            dataset_report["skipped"]["missing_data_log"] = 1
            report["datasets"].append(dataset_report)
            continue

        z_values = z_values_by_dataset.get(dataset_dir.name, [])
        if args.table_z_scope == "global":
            z_table = global_z_table
        else:
            z_table = mean(z_values) if z_values else None
        dataset_report["z_table_mm"] = z_table
        if z_table is None:
            dataset_report["skipped"]["missing_gaze_target"] = len(data_rows)
            report["datasets"].append(dataset_report)
            continue

        insight_by_name = index_by(insight_rows, "image_name")
        pnp_by_name = index_by(pnp_rows, "image_name")
        dataset_errors: list[float] = []

        for data_row in data_rows:
            image_name = data_row.get("image_filename", "").strip()
            if not image_name:
                dataset_report["skipped"]["missing_gaze_target"] += 1
                continue

            gaze_target_w = required_gaze_target(data_row)
            if gaze_target_w is None:
                dataset_report["skipped"]["missing_gaze_target"] += 1
                continue

            insight_row = insight_by_name.get(image_name)
            if insight_row is None:
                dataset_report["skipped"]["missing_insightface_row"] += 1
                continue

            if not args.include_failed_detections and not status_success(insight_row):
                dataset_report["skipped"]["failed_insightface"] += 1
                continue

            crop_result = build_crop_cam_vec(insight_row)
            if crop_result is None:
                dataset_report["skipped"]["invalid_bbox"] += 1
                continue
            crop_cam_vec, bboxes = crop_result
            dataset_report["counts"]["joined_rows"] += 1

            image_paths = existing_image_paths(dataset_dir, image_name)
            image_exists = {key: bool_exists(path) for key, path in image_paths.items()}
            for key, exists in image_exists.items():
                if not exists:
                    dataset_report["missing_image_counts"][key] += 1

            if not args.allow_missing_images and not (
                image_exists["face"] and image_exists["left_eye"] and image_exists["right_eye"]
            ):
                dataset_report["skipped"]["missing_required_images"] += 1
                continue

            pnp_row = pnp_by_name.get(image_name)
            if args.require_pnp:
                if pnp_row is None:
                    dataset_report["skipped"]["missing_pnp"] += 1
                    continue
                if not status_success(pnp_row):
                    dataset_report["skipped"]["failed_pnp"] += 1
                    continue

            try:
                pose = camera_pose_from_row(data_row, args.handeye_translation_scale)
            except ValueError:
                dataset_report["skipped"]["camera_pose_error"] += 1
                continue

            r_cw = pose["r_cw"]  # type: ignore[assignment]
            t_cw = pose["t_cw"]  # type: ignore[assignment]
            t_wc = pose["t_wc"]  # type: ignore[assignment]
            scene_vec, scene_meta = build_scene_vec(r_cw, t_cw, t_wc, z_table)  # type: ignore[arg-type]

            gaze_cam_recomputed = vec_add(matvec(r_cw, gaze_target_w), t_cw)  # type: ignore[arg-type]
            gaze_cam_csv = old_gaze_cam(data_row)
            gaze_cam_error = None
            if gaze_cam_csv is not None:
                gaze_cam_error = norm(vec_sub(gaze_cam_recomputed, gaze_cam_csv))
                dataset_errors.append(gaze_cam_error)
                all_errors.append(gaze_cam_error)

            o_table_w = scene_meta["o_table_w"]
            uv_gt = [
                gaze_target_w[0] - o_table_w[0],  # type: ignore[index]
                gaze_target_w[1] - o_table_w[1],  # type: ignore[index]
            ]
            z_delta = gaze_target_w[2] - z_table

            pnp_data = pnp_fields(pnp_row)
            row: dict[str, Any] = {
                "sample_id": f"{dataset_dir.name}/{Path(image_name).stem}",
                "dataset": dataset_dir.name,
                "image_name": image_name,
                "frame_idx": as_int_text(data_row, "frame_idx"),
                "source_dataset_dir": str(dataset_dir),
                "source_image_path": str(image_paths["source"]),
                "face_path": str(image_paths["face"]),
                "left_eye_path": str(image_paths["left_eye"]),
                "right_eye_path": str(image_paths["right_eye"]),
                "has_source_image": image_exists["source"],
                "has_face_image": image_exists["face"],
                "has_left_eye_image": image_exists["left_eye"],
                "has_right_eye_image": image_exists["right_eye"],
                "z_table_mm": z_table,
                "uv_gt_u_mm": uv_gt[0],
                "uv_gt_v_mm": uv_gt[1],
                "gaze_target_w_x_mm": gaze_target_w[0],
                "gaze_target_w_y_mm": gaze_target_w[1],
                "gaze_target_w_z_mm": gaze_target_w[2],
                "table_origin_w_x_mm": o_table_w[0],  # type: ignore[index]
                "table_origin_w_y_mm": o_table_w[1],  # type: ignore[index]
                "table_origin_w_z_mm": o_table_w[2],  # type: ignore[index]
                "table_target_z_delta_mm": z_delta,
                "gaze_cam_recomputed_x_mm": gaze_cam_recomputed[0],
                "gaze_cam_recomputed_y_mm": gaze_cam_recomputed[1],
                "gaze_cam_recomputed_z_mm": gaze_cam_recomputed[2],
                "gaze_cam_csv_x_mm": gaze_cam_csv[0] if gaze_cam_csv else None,
                "gaze_cam_csv_y_mm": gaze_cam_csv[1] if gaze_cam_csv else None,
                "gaze_cam_csv_z_mm": gaze_cam_csv[2] if gaze_cam_csv else None,
                "gaze_cam_error_mm": gaze_cam_error,
                "t_wc_x_mm": t_wc[0],  # type: ignore[index]
                "t_wc_y_mm": t_wc[1],  # type: ignore[index]
                "t_wc_z_mm": t_wc[2],  # type: ignore[index]
                "t_cw_x_mm": t_cw[0],  # type: ignore[index]
                "t_cw_y_mm": t_cw[1],  # type: ignore[index]
                "t_cw_z_mm": t_cw[2],  # type: ignore[index]
                "face_bbox_x": bboxes["face"][0],
                "face_bbox_y": bboxes["face"][1],
                "face_bbox_w": bboxes["face"][2],
                "face_bbox_h": bboxes["face"][3],
                "left_eye_bbox_x": bboxes["left_eye"][0],
                "left_eye_bbox_y": bboxes["left_eye"][1],
                "left_eye_bbox_w": bboxes["left_eye"][2],
                "left_eye_bbox_h": bboxes["left_eye"][3],
                "right_eye_bbox_x": bboxes["right_eye"][0],
                "right_eye_bbox_y": bboxes["right_eye"][1],
                "right_eye_bbox_w": bboxes["right_eye"][2],
                "right_eye_bbox_h": bboxes["right_eye"][3],
            }
            row.update(pnp_data)
            row.update(vector_columns(CROP_CAM_COLUMNS, crop_cam_vec))
            row.update(vector_columns(SCENE_COLUMNS, scene_vec))
            output_rows.append(row)
            dataset_report["counts"]["written_rows"] += 1

        dataset_report["gaze_cam_error_mm"] = stats(dataset_errors)
        report["datasets"].append(dataset_report)

    report["overall"] = {
        "dataset_count": len(dataset_dirs),
        "sample_count": len(output_rows),
        "global_z_table_mm": global_z_table,
        "gaze_cam_error_mm": stats(all_errors),
        "crop_cam_dim": 36,
        "scene_dim": 25,
    }
    return output_rows, report


def write_report(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_npz(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise DatasetBuildError("NumPy is required for --write-npz. Install numpy or omit --write-npz.") from exc

    crop = np.asarray([[row[column] for column in CROP_CAM_COLUMNS] for row in rows], dtype=np.float32)
    scene = np.asarray([[row[column] for column in SCENE_COLUMNS] for row in rows], dtype=np.float32)
    uv = np.asarray([[row["uv_gt_u_mm"], row["uv_gt_v_mm"]] for row in rows], dtype=np.float32)
    gaze_target_w = np.asarray(
        [[row["gaze_target_w_x_mm"], row["gaze_target_w_y_mm"], row["gaze_target_w_z_mm"]] for row in rows],
        dtype=np.float32,
    )
    image_names = np.asarray([row["image_name"] for row in rows], dtype=object)
    face_paths = np.asarray([row["face_path"] for row in rows], dtype=object)
    left_eye_paths = np.asarray([row["left_eye_path"] for row in rows], dtype=object)
    right_eye_paths = np.asarray([row["right_eye_path"] for row in rows], dtype=object)
    np.savez_compressed(
        path,
        crop_cam_vec=crop,
        scene_vec=scene,
        uv_gt=uv,
        gaze_target_w=gaze_target_w,
        image_name=image_names,
        face_path=face_paths,
        left_eye_path=left_eye_paths,
        right_eye_path=right_eye_paths,
    )


def main() -> int:
    args = parse_args()
    rows, report = build_dataset(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_name}.csv"
    report_path = args.output_dir / f"{args.output_name}_report.json"
    npz_path = args.output_dir / f"{args.output_name}.npz"

    write_csv(csv_path, rows)
    write_report(report_path, report)
    if args.write_npz:
        write_npz(npz_path, rows)

    print(f"Wrote {len(rows)} samples")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    if args.write_npz:
        print(f"NPZ: {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
