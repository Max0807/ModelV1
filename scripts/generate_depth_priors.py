"""Generate per-sample metric face-depth priors from DECA-FLAME and solvePnP.

The script reads ModelV1's dataset CSV and the baseline's saved MediaPipe PnP
landmark CSVs. It creates a separate depth-prior table so the existing training
CSV is never overwritten during preprocessing.
python scripts/generate_depth_priors.py --device auto --batch-size 8
python scripts/generate_depth_priors.py --face-preprocess legacy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.depth_prior import (
    CROSSGAZE_CAMERA_MATRIX,
    CROSSGAZE_DIST_COEFFS,
    DEFAULT_DECA_CROP_SCALE,
    DecaFlameConfig,
    DecaFlameExtractor,
    FACE_PREPROCESS_CHOICES,
    FACE_PREPROCESS_DECA,
    FACE_PREPROCESS_LEGACY,
    PnpCamera,
    PnpConfig,
    prepare_deca_face_image,
    solve_pnp_face_depth,
)


DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "modelv1_dataset.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "depth_prior_v1.csv"
DEFAULT_FIXED_SCALE_MM_PER_FLAME_UNIT = 1010.0

PNP_LABELS = (
    "left_eye_outer",
    "left_eye_inner",
    "right_eye_inner",
    "right_eye_outer",
    "nose_tip",
    "mouth_left",
    "mouth_right",
    "chin",
)

OUTPUT_FIELDS = (
    "sample_id",
    "dataset",
    "image_name",
    "source_image_path",
    "face_path",
    "deca_face_preprocess",
    "deca_crop_left_px",
    "deca_crop_top_px",
    "deca_crop_width_px",
    "deca_crop_height_px",
    "pnp_landmark_csv",
    "depth_prior_status",
    "reason",
    "left_eye_camera_x_mm",
    "left_eye_camera_y_mm",
    "left_eye_camera_z_mm",
    "right_eye_camera_x_mm",
    "right_eye_camera_y_mm",
    "right_eye_camera_z_mm",
    "face_depth_z_mm",
    "rvec_x_rad",
    "rvec_y_rad",
    "rvec_z_rad",
    "tvec_x_mm",
    "tvec_y_mm",
    "tvec_z_mm",
    "rotation_00",
    "rotation_01",
    "rotation_02",
    "rotation_10",
    "rotation_11",
    "rotation_12",
    "rotation_20",
    "rotation_21",
    "rotation_22",
    "reprojection_error_mean_px",
    "reprojection_error_max_px",
    "pnp_num_points",
    "pnp_inlier_count",
    "pnp_confidence",
    "depth_is_plausible",
    "scale_mm_per_flame_unit",
    "outer_scale_mm_per_flame_unit",
    "inner_scale_mm_per_flame_unit",
    "outer_flame_distance",
    "inner_flame_distance",
    "scale_disagreement_ratio",
)

FIELD_TYPES = {
    "sample_id": "string",
    "dataset": "string",
    "image_name": "string",
    "source_image_path": "string",
    "face_path": "string",
    "deca_face_preprocess": "string: deca or legacy",
    "deca_crop_left_px": "float: source-image pixels",
    "deca_crop_top_px": "float: source-image pixels",
    "deca_crop_width_px": "float: source-image pixels",
    "deca_crop_height_px": "float: source-image pixels",
    "pnp_landmark_csv": "string",
    "depth_prior_status": "string: success or failed",
    "reason": "string",
    "left_eye_camera_x_mm": "float: image-left eye-canthus midpoint camera x, mm",
    "left_eye_camera_y_mm": "float: image-left eye-canthus midpoint camera y, mm",
    "left_eye_camera_z_mm": "float: image-left eye-canthus midpoint camera z, mm",
    "right_eye_camera_x_mm": "float: image-right eye-canthus midpoint camera x, mm",
    "right_eye_camera_y_mm": "float: image-right eye-canthus midpoint camera y, mm",
    "right_eye_camera_z_mm": "float: image-right eye-canthus midpoint camera z, mm",
    "face_depth_z_mm": "float: mm",
    "rvec_x_rad": "float: Rodrigues rotation vector component, radians",
    "rvec_y_rad": "float: Rodrigues rotation vector component, radians",
    "rvec_z_rad": "float: Rodrigues rotation vector component, radians",
    "tvec_x_mm": "float: mm",
    "tvec_y_mm": "float: mm",
    "tvec_z_mm": "float: mm",
    "rotation_00": "float: rotation matrix element",
    "rotation_01": "float: rotation matrix element",
    "rotation_02": "float: rotation matrix element",
    "rotation_10": "float: rotation matrix element",
    "rotation_11": "float: rotation matrix element",
    "rotation_12": "float: rotation matrix element",
    "rotation_20": "float: rotation matrix element",
    "rotation_21": "float: rotation matrix element",
    "rotation_22": "float: rotation matrix element",
    "reprojection_error_mean_px": "float: pixels",
    "reprojection_error_max_px": "float: pixels",
    "pnp_num_points": "integer",
    "pnp_inlier_count": "integer",
    "pnp_confidence": "float in [0, 1], heuristic quality score",
    "depth_is_plausible": "boolean",
    "scale_mm_per_flame_unit": "float: mm per FLAME unit",
    "outer_scale_mm_per_flame_unit": "float: diagnostic mm per FLAME unit",
    "inner_scale_mm_per_flame_unit": "float: diagnostic mm per FLAME unit",
    "outer_flame_distance": "float: FLAME unit",
    "inner_flame_distance": "float: FLAME unit",
    "scale_disagreement_ratio": "float: relative disagreement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an offline metric depth-prior CSV for ModelV1."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Defaults to <output>.metadata.json.",
    )
    parser.add_argument(
        "--deca-root",
        type=Path,
        default=PROJECT_ROOT / "DECA-master",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to the checkpoint configured by official DECA.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:0. auto uses CUDA when available.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--face-preprocess",
        choices=FACE_PREPROCESS_CHOICES,
        default=FACE_PREPROCESS_DECA,
        help=(
            "DECA input crop. deca uses the official-style square similarity crop "
            "from the original image; legacy uses the existing face_path resize."
        ),
    )
    parser.add_argument(
        "--deca-crop-scale",
        type=float,
        default=DEFAULT_DECA_CROP_SCALE,
        help="Square DECA crop scale relative to official bbox old_size (default: 1.25).",
    )
    parser.add_argument("--outer-eye-distance-mm", type=float, default=105.0)
    parser.add_argument("--inner-eye-distance-mm", type=float, default=38.0)
    parser.add_argument(
        "--fixed-scale-mm-per-flame-unit",
        type=float,
        default=DEFAULT_FIXED_SCALE_MM_PER_FLAME_UNIT,
        help=(
            "Use a dataset-level fixed metric scale. The default 1010.0 matches "
            "the selected CrossGaze baseline scale."
        ),
    )
    parser.add_argument(
        "--use-measured-scale-per-frame",
        action="store_true",
        help="Do not use a fixed scale; average the measured inner/outer scales per frame.",
    )
    parser.add_argument("--use-ransac", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output CSV and metadata JSON.",
    )
    return parser.parse_args()


def read_dataset_rows(csv_path: Path, limit: int | None) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Dataset CSV has no rows: {csv_path}")

    required = {
        "sample_id",
        "dataset",
        "image_name",
        "source_dataset_dir",
        "source_image_path",
        "face_path",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Dataset CSV is missing columns: {sorted(missing)}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Dataset CSV contains duplicate sample_id values")
    return rows


def read_landmark_rows(landmark_csv: Path) -> dict[str, dict[str, str]]:
    if not landmark_csv.is_file():
        raise FileNotFoundError(f"Saved MediaPipe landmark CSV does not exist: {landmark_csv}")
    with landmark_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_name", "status", "reason"}
    for label in PNP_LABELS:
        required.add(f"{label}_x")
        required.add(f"{label}_y")
    if not rows:
        raise ValueError(f"Saved MediaPipe landmark CSV has no rows: {landmark_csv}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"Saved MediaPipe landmark CSV is missing columns {sorted(missing)}: {landmark_csv}"
        )

    by_image_name = {row["image_name"]: row for row in rows}
    if len(by_image_name) != len(rows):
        raise ValueError(f"Duplicate image_name values in {landmark_csv}")
    return by_image_name


def get_image_points(
    row: dict[str, str],
    landmark_cache: dict[Path, dict[str, dict[str, str]]],
) -> tuple[dict[str, tuple[float, float]], Path]:
    landmark_csv = Path(row["source_dataset_dir"]) / "mediapipe_pnp_landmarks.csv"
    if landmark_csv not in landmark_cache:
        landmark_cache[landmark_csv] = read_landmark_rows(landmark_csv)
    try:
        landmark_row = landmark_cache[landmark_csv][row["image_name"]]
    except KeyError as error:
        raise KeyError(
            f"No saved MediaPipe PnP landmarks for image {row['image_name']!r} in {landmark_csv}"
        ) from error
    if landmark_row["status"] != "success":
        raise RuntimeError(
            "MediaPipe PnP detection failed: " + landmark_row.get("reason", "unknown")
        )

    points: dict[str, tuple[float, float]] = {}
    for label in PNP_LABELS:
        try:
            points[label] = (
                float(landmark_row[f"{label}_x"]),
                float(landmark_row[f"{label}_y"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid saved 2D point for {label!r}") from error
    return points, landmark_csv


def require_runtime_modules() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Depth-prior generation requires numpy, torch, and Pillow."
        ) from error
    return np, torch, Image


def face_bbox_from_row(row: dict[str, str]) -> tuple[float, float, float, float]:
    try:
        return tuple(
            float(row[f"face_bbox_{axis}"]) for axis in ("x", "y", "w", "h")
        )  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Dataset row has invalid face_bbox_x/y/w/h values") from error


def load_deca_face_tensor(
    row: dict[str, str],
    *,
    face_preprocess: str,
    deca_crop_scale: float,
    image_size: int,
    np: Any,
    torch: Any,
    image_cls: Any,
) -> tuple[Any, Any]:
    source_path = Path(row["source_image_path"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Source image does not exist: {source_path}")
    with image_cls.open(source_path) as image:
        source_image = np.asarray(image.convert("RGB"))

    legacy_face_image = None
    if face_preprocess == FACE_PREPROCESS_LEGACY:
        face_path = Path(row["face_path"])
        if not face_path.is_file():
            raise FileNotFoundError(f"Face image does not exist: {face_path}")
        with image_cls.open(face_path) as image:
            legacy_face_image = np.asarray(image.convert("RGB"))

    input_image, crop_transform = prepare_deca_face_image(
        source_image,
        face_bbox_from_row(row),
        mode=face_preprocess,
        input_size=image_size,
        deca_crop_scale=deca_crop_scale,
        legacy_face_image_rgb=legacy_face_image,
    )
    array = np.asarray(input_image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1), crop_transform


def empty_record(
    row: dict[str, str],
    status: str,
    reason: str,
    *,
    face_preprocess: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {field: "" for field in OUTPUT_FIELDS}
    record.update(
        {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "image_name": row["image_name"],
            "source_image_path": row["source_image_path"],
            "face_path": row["face_path"],
            "deca_face_preprocess": face_preprocess,
            "depth_prior_status": status,
            "reason": reason,
        }
    )
    return record


def result_record(
    row: dict[str, str], landmark_csv: Path, result: Any, crop_transform: Any
) -> dict[str, Any]:
    record = empty_record(row, "success", "", face_preprocess=crop_transform.mode)
    record["pnp_landmark_csv"] = str(landmark_csv)
    record.update(crop_transform.as_record())
    record.update(result.as_record())
    record["outer_flame_distance"] = float(result.scale.outer_flame_distance)
    record["inner_flame_distance"] = float(result.scale.inner_flame_distance)

    rvec = result.rvec.reshape(3)
    tvec = result.tvec_mm.reshape(3)
    rotation = result.rotation_matrix.reshape(3, 3)
    for axis, value in zip(("x", "y", "z"), rvec):
        record[f"rvec_{axis}_rad"] = float(value)
    for axis, value in zip(("x", "y", "z"), tvec):
        record[f"tvec_{axis}_mm"] = float(value)
    for row_index in range(3):
        for column_index in range(3):
            record[f"rotation_{row_index}{column_index}"] = float(
                rotation[row_index, column_index]
            )
    return record


def write_csv_atomic(output_path: Path, records: Iterable[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(records)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_json_atomic(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def resolve_device(requested: str) -> str | None:
    if requested == "auto":
        return None
    return requested


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.fixed_scale_mm_per_flame_unit is not None and args.fixed_scale_mm_per_flame_unit <= 0:
        raise ValueError("--fixed-scale-mm-per-flame-unit must be positive")
    if args.outer_eye_distance_mm <= 0 or args.inner_eye_distance_mm <= 0:
        raise ValueError("Measured eye distances must be positive")
    if args.deca_crop_scale <= 0:
        raise ValueError("--deca-crop-scale must be positive")

    metadata_path = args.metadata_output or args.output.with_suffix(
        args.output.suffix + ".metadata.json"
    )
    for path in (args.output, metadata_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Use --overwrite to replace it."
            )

    rows = read_dataset_rows(args.csv, args.limit)
    np, torch, image_cls = require_runtime_modules()
    landmark_cache: dict[Path, dict[str, dict[str, str]]] = {}
    records_by_id: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, str], dict[str, tuple[float, float]], Path]] = []

    for row in rows:
        try:
            image_points, landmark_csv = get_image_points(row, landmark_cache)
            required_image_path = (
                Path(row["source_image_path"])
                if args.face_preprocess == FACE_PREPROCESS_DECA
                else Path(row["face_path"])
            )
            if not required_image_path.is_file():
                raise FileNotFoundError(f"Image does not exist: {required_image_path}")
            face_bbox_from_row(row)
            pending.append((row, image_points, landmark_csv))
        except Exception as error:
            records_by_id[row["sample_id"]] = empty_record(
                row, "failed", str(error), face_preprocess=args.face_preprocess
            )

    scale_for_pnp = (
        None if args.use_measured_scale_per_frame else args.fixed_scale_mm_per_flame_unit
    )
    extractor = DecaFlameExtractor(
        DecaFlameConfig(
            deca_root=args.deca_root,
            pretrained_model_path=args.checkpoint,
            device=resolve_device(args.device),
        )
    )
    camera = PnpCamera(CROSSGAZE_CAMERA_MATRIX, CROSSGAZE_DIST_COEFFS)
    pnp_config = PnpConfig(use_ransac=args.use_ransac)

    print(
        f"Samples={len(rows)}, preflight_failed={len(records_by_id)}, "
        f"to_process={len(pending)}, device={extractor.device}, "
        f"face_preprocess={args.face_preprocess}"
    )
    for start in range(0, len(pending), args.batch_size):
        requested_batch = pending[start : start + args.batch_size]
        batch_items: list[tuple[dict[str, str], dict[str, tuple[float, float]], Path, Any]] = []
        tensors: list[Any] = []
        for row, image_points, landmark_csv in requested_batch:
            try:
                tensor, crop_transform = load_deca_face_tensor(
                    row,
                    face_preprocess=args.face_preprocess,
                    deca_crop_scale=args.deca_crop_scale,
                    image_size=extractor.config.image_size,
                    np=np,
                    torch=torch,
                    image_cls=image_cls,
                )
                tensors.append(tensor)
                batch_items.append((row, image_points, landmark_csv, crop_transform))
            except Exception as error:
                records_by_id[row["sample_id"]] = empty_record(
                    row, "failed", str(error), face_preprocess=args.face_preprocess
                )
        if not batch_items:
            continue

        face_batch = torch.stack(tensors, dim=0)
        output = extractor.extract(face_batch)
        for index, (row, image_points, landmark_csv, crop_transform) in enumerate(batch_items):
            try:
                result = solve_pnp_face_depth(
                    image_points_by_label=image_points,
                    landmarks3d=output.landmarks3d[index].numpy(),
                    camera=camera,
                    scale_mm_per_flame_unit=scale_for_pnp,
                    outer_eye_distance_mm=args.outer_eye_distance_mm,
                    inner_eye_distance_mm=args.inner_eye_distance_mm,
                    config=pnp_config,
                )
                records_by_id[row["sample_id"]] = result_record(
                    row, landmark_csv, result, crop_transform
                )
            except Exception as error:
                records_by_id[row["sample_id"]] = empty_record(
                    row, "failed", str(error), face_preprocess=args.face_preprocess
                )
        print(f"Processed {min(start + len(requested_batch), len(pending))}/{len(pending)}")

    records = [records_by_id[row["sample_id"]] for row in rows]
    status_counts = Counter(record["depth_prior_status"] for record in records)
    metadata: dict[str, Any] = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.csv.resolve()),
        "output_csv": str(args.output.resolve()),
        "sample_count": len(records),
        "status_counts": dict(status_counts),
        "field_types": FIELD_TYPES,
        "coordinate_convention": {
            "input_2d": "Original-image pixels from saved MediaPipe Face Mesh PnP landmarks.",
            "flame_3d": "DECA-FLAME local coordinates scaled into millimetres before PnP.",
            "output_3d": "OpenCV camera coordinates in millimetres.",
        },
        "camera": {
            "camera_matrix": CROSSGAZE_CAMERA_MATRIX,
            "dist_coeffs": CROSSGAZE_DIST_COEFFS,
        },
        "scale_policy": {
            "mode": "per_frame_measured" if scale_for_pnp is None else "fixed",
            "fixed_scale_mm_per_flame_unit": scale_for_pnp,
            "outer_eye_distance_mm": args.outer_eye_distance_mm,
            "inner_eye_distance_mm": args.inner_eye_distance_mm,
        },
        "pnp": {
            "use_ransac": args.use_ransac,
            "model": "OpenCV solvePnP iterative",
        },
        "deca": {
            "root": str(args.deca_root.resolve()),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else "official DECA config default",
            "face_preprocess": args.face_preprocess,
            "deca_crop_scale": args.deca_crop_scale,
            "input_preprocess": (
                "RGB source-image crop, official DECA-style square similarity crop "
                "when face_preprocess=deca; float32 pixels in [0, 1]"
            ),
        },
        "uncertainty_note": (
            "pnp_confidence is a heuristic quality score. This preprocessing step "
            "does not generate calibrated z_sigma_mm or eye_center_sigma_mm."
        ),
    }
    write_csv_atomic(args.output, records)
    write_json_atomic(metadata_path, metadata)
    print(f"Wrote depth-prior CSV: {args.output}")
    print(f"Wrote metadata JSON: {metadata_path}")
    print(f"Status counts: {dict(status_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
