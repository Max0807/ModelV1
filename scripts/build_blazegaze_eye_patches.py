"""Generate released-style BlazeGaze homography eye patches from Stage 3 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.blazegaze.eye_patch import (
    DEFAULT_FACE_PADDING_COEFS,
    EYE_PATCH_HEIGHT,
    EYE_PATCH_WIDTH,
    EXACT_EYE_PATCH_SOURCE,
    REQUIRED_LANDMARK_INDICES,
    build_blazegaze_eye_patch,
    landmarks_to_pixel_array,
)


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT.parent
    / "WebEyeTrack-main"
    / "python"
    / "webeyetrack"
    / "model_weights"
    / "face_landmarker_v2_with_blendshapes.task"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "preprocessed_blazegaze" / "table7"
DEFAULT_PATCH_DIR = DEFAULT_OUTPUT_ROOT / "eye_patches"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_ROOT / "blazegaze_eye_patches.csv"
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "blazegaze_eye_patches.report.json"
REQUIRED_STAGE3_COLUMNS = (
    "sample_id",
    "dataset",
    "image_name",
    "source_image_path",
)
MANIFEST_FIELDS = (
    "sample_id",
    "dataset",
    "image_name",
    "source_image_path",
    "eye_patch_path",
    "eye_strip_source",
    "status",
    "reason",
    "landmark_count",
    "source_width_px",
    "source_height_px",
    "crop_top_y",
    "crop_bottom_y",
    "source_quad_area_px2",
    "homography_condition",
    "black_pixel_fraction",
    *(f"homography_{row}{column}" for row in range(3) for column in range(3)),
    *(
        f"landmark_{index}_{axis}_px"
        for index in REQUIRED_LANDMARK_INDICES
        for axis in "xy"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage3-csv", action="append", required=True,
        help="Stage 3 CSV; repeat for multiple participants.",
    )
    parser.add_argument(
        "--face-landmarker-model", type=Path, default=DEFAULT_MODEL_PATH,
        help="MediaPipe Tasks face-landmarker .task file.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PATCH_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(paths: list[str], limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = [name for name in REQUIRED_STAGE3_COLUMNS if name not in fields]
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            for raw_row in reader:
                row = dict(raw_row)
                sample_id = str(row["sample_id"]).strip()
                if not sample_id:
                    raise ValueError(f"{path} contains an empty sample_id.")
                if sample_id in seen:
                    raise ValueError(f"Duplicate Stage 3 sample {sample_id!r}.")
                seen.add(sample_id)
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_png_atomic(image_rgb: np.ndarray, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Eye patch already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    try:
        Image.fromarray(image_rgb, mode="RGB").save(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _empty_manifest_row(row: dict[str, str]) -> dict[str, object]:
    output: dict[str, object] = {field: "" for field in MANIFEST_FIELDS}
    output.update(
        {
            "sample_id": str(row["sample_id"]).strip(),
            "dataset": str(row["dataset"]).strip(),
            "image_name": str(row["image_name"]).strip(),
            "source_image_path": str(row["source_image_path"]).strip(),
            "eye_strip_source": EXACT_EYE_PATCH_SOURCE,
            "status": "failed",
        }
    )
    return output


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if not args.face_landmarker_model.is_file():
        raise FileNotFoundError(
            f"FaceLandmarker model is missing: {args.face_landmarker_model}"
        )
    if not args.overwrite:
        existing = [path for path in (args.manifest, args.report) if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing}")

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise RuntimeError(
            "BlazeGaze eye-patch generation requires MediaPipe Tasks."
        ) from error

    rows = _read_rows(args.stage3_csv, args.limit)
    if not rows:
        raise ValueError("No Stage 3 rows were selected.")

    base_options = python.BaseOptions(model_asset_path=str(args.face_landmarker_model))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE,
    )

    manifest_rows: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    landmark_counts: Counter[int] = Counter()
    black_fractions: list[float] = []
    homography_conditions: list[float] = []

    with vision.FaceLandmarker.create_from_options(options) as detector:
        for index, row in enumerate(rows, start=1):
            output = _empty_manifest_row(row)
            try:
                source_path = Path(str(row["source_image_path"]).strip())
                if not source_path.is_file():
                    raise FileNotFoundError(f"Source image is missing: {source_path}")
                with Image.open(source_path) as image:
                    rgb = np.ascontiguousarray(image.convert("RGB"), dtype=np.uint8)
                height, width = rgb.shape[:2]
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection = detector.detect(mp_image)
                if not detection.face_landmarks:
                    raise RuntimeError("MediaPipe FaceLandmarker found no face")
                landmarks = detection.face_landmarks[0]
                landmark_count = len(landmarks)
                landmark_counts[landmark_count] += 1
                if landmark_count != 478:
                    raise ValueError(
                        f"Expected the released 478-point topology, got {landmark_count}."
                    )
                landmarks_px = landmarks_to_pixel_array(landmarks, width, height)
                geometry = build_blazegaze_eye_patch(rgb, landmarks_px)

                dataset_name = str(row["dataset"]).strip()
                output_path = (
                    args.output_dir
                    / dataset_name
                    / f"{Path(str(row['image_name'])).stem}.png"
                )
                _save_png_atomic(geometry.patch_rgb, output_path, args.overwrite)

                output.update(
                    {
                        "eye_patch_path": str(output_path.resolve()),
                        "status": "success",
                        "reason": "",
                        "landmark_count": landmark_count,
                        "source_width_px": width,
                        "source_height_px": height,
                        "crop_top_y": geometry.crop_top_y,
                        "crop_bottom_y": geometry.crop_bottom_y,
                        "source_quad_area_px2": geometry.source_quad_area_px2,
                        "homography_condition": geometry.homography_condition,
                        "black_pixel_fraction": geometry.black_pixel_fraction,
                    }
                )
                for matrix_row in range(3):
                    for matrix_column in range(3):
                        output[f"homography_{matrix_row}{matrix_column}"] = float(
                            geometry.homography[matrix_row, matrix_column]
                        )
                for landmark_index in REQUIRED_LANDMARK_INDICES:
                    output[f"landmark_{landmark_index}_x_px"] = float(
                        landmarks_px[landmark_index, 0]
                    )
                    output[f"landmark_{landmark_index}_y_px"] = float(
                        landmarks_px[landmark_index, 1]
                    )
                black_fractions.append(geometry.black_pixel_fraction)
                homography_conditions.append(geometry.homography_condition)
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                output["reason"] = reason
                reason_counts[reason] += 1
            manifest_rows.append(output)
            if index % args.log_every == 0 or index == len(rows):
                success_count = sum(
                    item["status"] == "success" for item in manifest_rows
                )
                print(
                    f"BlazeGaze eye patches: {index}/{len(rows)} "
                    f"(success={success_count}, failed={index - success_count})"
                )

    _write_csv_atomic(args.manifest, manifest_rows)
    success_count = sum(row["status"] == "success" for row in manifest_rows)
    report: dict[str, object] = {
        "schema_version": "blazegaze_mediapipe478_homography_v1",
        "stage3_csvs": [str(Path(path)) for path in args.stage3_csv],
        "face_landmarker_model": str(args.face_landmarker_model.resolve()),
        "face_landmarker_model_sha256": _sha256(args.face_landmarker_model),
        "output_dir": str(args.output_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "input_rows": len(rows),
        "success": success_count,
        "failed": len(rows) - success_count,
        "reason_counts": dict(reason_counts),
        "landmark_count_distribution": {
            str(key): value for key, value in sorted(landmark_counts.items())
        },
        "required_landmark_indices": list(REQUIRED_LANDMARK_INDICES),
        "homography_corner_order": [103, 150, 379, 332],
        "face_padding_coefs_xy": list(DEFAULT_FACE_PADDING_COEFS),
        "output_shape_hwc": [EYE_PATCH_HEIGHT, EYE_PATCH_WIDTH, 3],
        "color_space": "RGB",
        "mean_black_pixel_fraction": (
            mean(black_fractions) if black_fractions else None
        ),
        "max_black_pixel_fraction": max(black_fractions) if black_fractions else None,
        "max_homography_condition": (
            max(homography_conditions) if homography_conditions else None
        ),
        "implementation_reference": (
            "WebEyeTrack python/webeyetrack/model_based.py::obtain_eyepatch"
        ),
    }
    _write_json_atomic(args.report, report)
    print(
        f"Wrote {success_count}/{len(rows)} exact BlazeGaze eye patches; "
        f"manifest: {args.manifest}"
    )
    return 0 if success_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
