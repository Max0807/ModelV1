"""Generate shared virtual-camera face/eye images without touching old data.

Example (small smoke run)::

    python scripts/generate_virtual_camera_normalized_images.py \
        --dataset-csv data/processed/modelv1_dataset3.csv \
                      data/processed/modelv1_dataset4.csv \
        --depth-prior-csv data/processed/depth_priors_deca_crop_v1.csv \
        --landmark-csv data/processed/mediapipe_pnp_landmarks_3.csv \
                       data/processed/mediapipe_pnp_landmarks_4.csv \
        --limit-per-dataset 5

The default output root is
``data/processed/virtual_camera_v2_anatomical_eyes``. Existing files are never
overwritten unless ``--overwrite`` is explicitly supplied. Version 2 uses
anatomical subject-left/subject-right eye directory semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.data.virtual_camera import (  # noqa: E402
    CameraCalibration,
    EyeCropConfig,
    VirtualCameraConfig,
    build_virtual_camera_transform,
    render_shared_normalized_inputs,
)
from modelv1.depth_prior import (  # noqa: E402
    CROSSGAZE_CAMERA_MATRIX,
    CROSSGAZE_DIST_COEFFS,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "virtual_camera_v2_anatomical_eyes"
)
DEFAULT_DEPTH_PRIOR_CSV = (
    PROJECT_ROOT / "data" / "processed" / "depth_priors_deca_crop_v1.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate shared virtual-camera face and eye images."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more ModelV1 dataset CSVs; their row order defines output order.",
    )
    parser.add_argument(
        "--depth-prior-csv",
        type=Path,
        default=DEFAULT_DEPTH_PRIOR_CSV,
        help="CSV containing successful PnP rotation and tvec columns.",
    )
    parser.add_argument(
        "--landmark-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more MediaPipe PnP landmark CSVs containing eye canthi.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--face-width", type=int, default=224)
    parser.add_argument("--face-height", type=int, default=224)
    parser.add_argument("--virtual-focal-length-px", type=float, default=480.0)
    parser.add_argument("--virtual-distance-mm", type=float, default=600.0)
    parser.add_argument("--eye-width", type=int, default=60)
    parser.add_argument("--eye-height", type=int, default=36)
    parser.add_argument("--eye-span-scale", type=float, default=1.8)
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional development limit; omitted means all rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files only inside --output-root.",
    )
    args = parser.parse_args()
    if args.limit_per_dataset is not None and args.limit_per_dataset <= 0:
        parser.error("--limit-per-dataset must be positive.")
    return args


def read_csv_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def index_unique(rows: Iterable[dict[str, str]], source_name: str) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            raise ValueError(f"{source_name} contains a row without sample_id.")
        if sample_id in index:
            raise ValueError(f"Duplicate sample_id in {source_name}: {sample_id}")
        index[sample_id] = row
    return index


def finite_float(row: dict[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"missing_or_invalid_{column}") from exc
    if not np.isfinite(value):
        raise ValueError(f"non_finite_{column}")
    return value


def rotation_from_depth_row(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            [finite_float(row, f"rotation_{r}{c}") for c in range(3)]
            for r in range(3)
        ],
        dtype=np.float64,
    )


def center_from_depth_row(row: dict[str, str]) -> np.ndarray:
    # Initial infrastructure intentionally uses the face-origin tvec from PnP.
    return np.array(
        [finite_float(row, f"tvec_{axis}_mm") for axis in "xyz"],
        dtype=np.float64,
    )


def eye_corners_from_landmark_row(
    row: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return anatomical-left and anatomical-right canthi, in that order.

    The legacy MediaPipe CSV names image-left landmarks 33/133 as ``left`` and
    image-right landmarks 362/263 as ``right``.  For the non-mirrored source
    images those are anatomical right and anatomical left, respectively.  The
    output directories and ModelV1 eye paths use anatomical side names, so the
    legacy columns must be swapped here.
    """

    # Anatomical left eye = image-right = legacy ``right_eye_*`` columns.
    left = np.array(
        [
            [finite_float(row, "right_eye_outer_x"), finite_float(row, "right_eye_outer_y")],
            [finite_float(row, "right_eye_inner_x"), finite_float(row, "right_eye_inner_y")],
        ],
        dtype=np.float64,
    )
    # Anatomical right eye = image-left = legacy ``left_eye_*`` columns.
    right = np.array(
        [
            [finite_float(row, "left_eye_outer_x"), finite_float(row, "left_eye_outer_y")],
            [finite_float(row, "left_eye_inner_x"), finite_float(row, "left_eye_inner_y")],
        ],
        dtype=np.float64,
    )
    return left, right


def safe_component(value: str, fallback: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    component = component.strip("._")
    return component or fallback


def choose_source_image(
    dataset_row: dict[str, str],
    landmark_row: dict[str, str],
    depth_row: dict[str, str],
) -> Path:
    candidates = []
    for row in (dataset_row, landmark_row, depth_row):
        raw_path = row.get("source_image_path", "").strip()
        if raw_path:
            candidates.append(Path(raw_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if candidates:
        raise FileNotFoundError(f"source_image_not_found:{candidates[0]}")
    raise FileNotFoundError("source_image_path_missing")


def read_image(path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for offline image generation.") from exc
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"image_decode_failed:{path}")
    return image


def write_png(path: Path, image: np.ndarray, overwrite: bool) -> None:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for offline image generation.") from exc
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"image_encode_failed:{path}")
    path.write_bytes(encoded.tobytes())


def output_paths(
    output_root: Path,
    dataset_row: dict[str, str],
) -> tuple[Path, Path, Path]:
    dataset_name = safe_component(dataset_row.get("dataset", ""), "unknown_dataset")
    image_name = dataset_row.get("image_name", "").strip()
    if image_name:
        image_stem = Path(image_name).stem
    else:
        image_stem = dataset_row["sample_id"].replace("\\", "/").rsplit("/", 1)[-1]
    image_stem = safe_component(image_stem, "sample")
    return (
        output_root / "face" / dataset_name / f"{image_stem}.png",
        output_root / "left_eye" / dataset_name / f"{image_stem}.png",
        output_root / "right_eye" / dataset_name / f"{image_stem}.png",
    )


def point_record(prefix: str, points: np.ndarray) -> dict[str, float]:
    record: dict[str, float] = {}
    for point_index, point in enumerate(points):
        record[f"{prefix}_{point_index}_x"] = float(point[0])
        record[f"{prefix}_{point_index}_y"] = float(point[1])
    return record


def crop_record(prefix: str, center: np.ndarray, size: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_crop_center_x": float(center[0]),
        f"{prefix}_crop_center_y": float(center[1]),
        f"{prefix}_crop_source_width": float(size[0]),
        f"{prefix}_crop_source_height": float(size[1]),
    }


def write_manifest(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifest.csv"
    report_path = output_root / "report.json"
    for protected_path in (manifest_path, report_path):
        if protected_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {protected_path}; choose a new --output-root "
                "or explicitly pass --overwrite."
            )

    dataset_rows = read_csv_rows(args.dataset_csv)
    # Dataset inputs may overlap; duplicates are dangerous because they map to
    # identical output paths, so reject them instead of silently choosing one.
    index_unique(dataset_rows, "dataset CSVs")
    depth_index = index_unique(read_csv_rows([args.depth_prior_csv]), "depth-prior CSV")
    landmark_index = index_unique(read_csv_rows(args.landmark_csv), "landmark CSVs")

    virtual_config = VirtualCameraConfig(
        output_width=args.face_width,
        output_height=args.face_height,
        focal_length_px=args.virtual_focal_length_px,
        distance_mm=args.virtual_distance_mm,
    )
    eye_config = EyeCropConfig(
        output_width=args.eye_width,
        output_height=args.eye_height,
        horizontal_span_scale=args.eye_span_scale,
    )
    calibration = CameraCalibration(
        camera_matrix=np.asarray(CROSSGAZE_CAMERA_MATRIX, dtype=np.float64),
        distortion_coefficients=np.asarray(CROSSGAZE_DIST_COEFFS, dtype=np.float64),
    )

    selected_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    manifest_rows: list[dict[str, Any]] = []
    for dataset_row in dataset_rows:
        sample_id = dataset_row["sample_id"].strip()
        dataset_name = dataset_row.get("dataset", "unknown_dataset").strip()
        if (
            args.limit_per_dataset is not None
            and selected_counts[dataset_name] >= args.limit_per_dataset
        ):
            continue
        selected_counts[dataset_name] += 1
        base_record: dict[str, Any] = {
            "sample_id": sample_id,
            "dataset": dataset_name,
            "image_name": dataset_row.get("image_name", ""),
            "normalization_status": "failed",
            "normalization_reason": "",
            "center_source": "pnp_tvec",
            "eye_side_semantics": "anatomical_subject_left_right",
            "landmark_eye_side_conversion": "legacy_image_side_to_anatomical",
        }
        try:
            depth_row = depth_index.get(sample_id)
            if depth_row is None:
                raise ValueError("depth_prior_row_missing")
            landmark_row = landmark_index.get(sample_id)
            if landmark_row is None:
                raise ValueError("landmark_row_missing")
            depth_status = depth_row.get("depth_prior_status", depth_row.get("status", ""))
            if depth_status.strip().lower() != "success":
                raise ValueError(f"invalid_depth_prior_status:{depth_status or 'missing'}")
            if landmark_row.get("status", "").strip().lower() != "success":
                raise ValueError(
                    "invalid_landmark_status:"
                    f"{landmark_row.get('status', '').strip() or 'missing'}"
                )

            source_path = choose_source_image(dataset_row, landmark_row, depth_row)
            source_image = read_image(source_path)
            transform = build_virtual_camera_transform(
                calibration=calibration,
                head_rotation_camera_from_head=rotation_from_depth_row(depth_row),
                face_center_camera_mm=center_from_depth_row(depth_row),
                config=virtual_config,
            )
            left_source, right_source = eye_corners_from_landmark_row(landmark_row)
            normalized = render_shared_normalized_inputs(
                source_image=source_image,
                calibration=calibration,
                transform=transform,
                left_eye_corners_source_xy=left_source,
                right_eye_corners_source_xy=right_source,
                eye_crop_config=eye_config,
            )
            face_path, left_path, right_path = output_paths(output_root, dataset_row)
            write_png(face_path, normalized.face_image, args.overwrite)
            write_png(left_path, normalized.left_eye.image, args.overwrite)
            write_png(right_path, normalized.right_eye.image, args.overwrite)

            base_record.update(
                {
                    "source_image_path": str(source_path.resolve()),
                    "normalized_face_path": str(face_path.resolve()),
                    "normalized_left_eye_path": str(left_path.resolve()),
                    "normalized_right_eye_path": str(right_path.resolve()),
                    "normalization_status": "success",
                    "normalization_reason": "",
                    "eye_width": eye_config.output_width,
                    "eye_height": eye_config.output_height,
                    "eye_span_scale": eye_config.horizontal_span_scale,
                    "pnp_reprojection_error_mean_px": depth_row.get(
                        "reprojection_error_mean_px", ""
                    ),
                    "pnp_reprojection_error_max_px": depth_row.get(
                        "reprojection_error_max_px", ""
                    ),
                    "pnp_confidence": depth_row.get("pnp_confidence", ""),
                    **transform.as_record(),
                    **point_record(
                        "left_eye_normalized",
                        normalized.left_eye_corners_normalized,
                    ),
                    **point_record(
                        "right_eye_normalized",
                        normalized.right_eye_corners_normalized,
                    ),
                    **crop_record(
                        "left_eye",
                        normalized.left_eye.source_center_xy,
                        normalized.left_eye.source_size_wh,
                    ),
                    **crop_record(
                        "right_eye",
                        normalized.right_eye.source_center_xy,
                        normalized.right_eye.source_size_wh,
                    ),
                }
            )
            status_counts["success"] += 1
        except Exception as exc:
            base_record["normalization_reason"] = f"{type(exc).__name__}:{exc}"
            status_counts["failed"] += 1
        manifest_rows.append(base_record)

    write_manifest(manifest_path, manifest_rows, args.overwrite)
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_csvs": [str(path.resolve()) for path in args.dataset_csv],
        "depth_prior_csv": str(args.depth_prior_csv.resolve()),
        "landmark_csvs": [str(path.resolve()) for path in args.landmark_csv],
        "output_root": str(output_root),
        "center_source": "pnp_tvec",
        "eye_side_semantics": "anatomical_subject_left_right",
        "landmark_eye_side_conversion": "legacy_image_side_to_anatomical",
        "shared_virtual_camera": True,
        "undistort_before_homography": True,
        "virtual_camera": {
            "width": virtual_config.output_width,
            "height": virtual_config.output_height,
            "focal_length_px": virtual_config.focal_length_px,
            "distance_mm": virtual_config.distance_mm,
        },
        "eye_crop": {
            "width": eye_config.output_width,
            "height": eye_config.output_height,
            "horizontal_span_scale": eye_config.horizontal_span_scale,
        },
        "camera_matrix": calibration.camera_matrix.tolist(),
        "distortion_coefficients": calibration.distortion_coefficients.tolist(),
        "selected_per_dataset": dict(selected_counts),
        "status_counts": dict(status_counts),
        "manifest_path": str(manifest_path),
    }
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    report = generate(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
