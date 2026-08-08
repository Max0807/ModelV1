"""Generate metric iris/pupil-proxy depth priors from known IPD and PnP pose.

The script deliberately does not treat a FLAME mesh vertex as an eyeball
centre.  It uses FLAME only for face orientation and the left-to-right eye-line
direction, then reconstructs the two observed iris centres from calibrated
camera rays constrained to the measured interpupillary distance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.depth_prior import (
    CROSSGAZE_CAMERA_MATRIX,
    CROSSGAZE_DIST_COEFFS,
    DEFAULT_PNP_MAPPING,
    DEFAULT_DECA_CROP_SCALE,
    FACE_PREPROCESS_CHOICES,
    FACE_PREPROCESS_DECA,
    DecaFlameConfig,
    DecaFlameExtractor,
    PnpCamera,
    compute_eye_canthus_midpoints,
    prepare_deca_face_image,
    solve_pnp_face_depth,
)
from modelv1.depth_prior.eye_proxy_validation import (
    detect_iris_centres_rgb,
    load_iris_centres_csv,
    project_flame_local_points,
)
from modelv1.depth_prior.iris_ipd import (
    assign_iris_groups_to_anatomical_sides,
    reconstruct_iris_centres_from_ipd,
)
from modelv1.depth_prior.dataset_assets import resolve_image_asset_path
from modelv1.processed_artifacts import processed_dataset_artifacts


DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "modelv1_dataset11.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "depth_priors_iris_ipd_65mm_v1_11.csv"
FIELDS = (
    "sample_id", "dataset", "image_name", "source_image_path", "depth_prior_status", "reason",
    "left_eye_camera_x_mm", "left_eye_camera_y_mm", "left_eye_camera_z_mm",
    "right_eye_camera_x_mm", "right_eye_camera_y_mm", "right_eye_camera_z_mm",
    "left_iris_camera_x_mm", "left_iris_camera_y_mm", "left_iris_camera_z_mm",
    "right_iris_camera_x_mm", "right_iris_camera_y_mm", "right_iris_camera_z_mm",
    "iris_midpoint_camera_x_mm", "iris_midpoint_camera_y_mm", "iris_midpoint_depth_z_mm",
    "measured_ipd_mm", "iris_left_group", "iris_right_group", "iris_assignment_cost_px",
    "iris_baseline_residual_mm", "iris_ray_system_condition",
    "pnp_reprojection_error_mean_px", "pnp_reprojection_error_max_px",
    "pnp_rvec_x_rad", "pnp_rvec_y_rad", "pnp_rvec_z_rad",
    "pnp_scale_mm_per_flame_unit", "fixed_shape_calibration_count",
    "depth_uncertainty_status", "depth_uncertainty_reason",
    "depth_log_scale_std", "depth_scale_p05", "depth_scale_p50",
    "depth_scale_p95", "depth_uncertainty_sample_count",
    "depth_bootstrap_success_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-id", default=None, help="Derive dataset, iris, and output paths from one dataset ID.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--ipd-mm", type=float, default=65.0)
    parser.add_argument("--max-ipd-residual-mm", type=float, default=3.0)
    parser.add_argument("--max-ray-condition", type=float, default=250.0)
    parser.add_argument("--pnp-scale-mm-per-flame-unit", type=float, default=1010.0)
    parser.add_argument("--calibration-dataset", default="dataset_dual_rigid_body_3")
    parser.add_argument("--calibration-start", type=int, default=129)
    parser.add_argument("--calibration-stop", type=int, default=231)
    parser.add_argument(
        "--calibration-index-mode",
        choices=("frame_idx", "position"),
        default="frame_idx",
        help=(
            "Interpret --calibration-start/--calibration-stop as source frame_idx "
            "(default; missing/deleted frames are allowed) or as zero-based positions "
            "within the retained rows of --calibration-dataset."
        ),
    )
    parser.add_argument("--face-preprocess", choices=FACE_PREPROCESS_CHOICES, default=FACE_PREPROCESS_DECA)
    parser.add_argument("--deca-crop-scale", type=float, default=DEFAULT_DECA_CROP_SCALE)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Optional root that contains dataset directories, used when image paths "
            "saved in the CSV no longer exist. For example: "
            "D:/GithubCode/CrossGaze-main/CrossGaze-main/data_collection"
        ),
    )
    parser.add_argument("--iris-csv", type=Path, default=None)
    parser.add_argument(
        "--pnp-landmark-csv",
        type=Path,
        default=None,
        help="Optional generated MediaPipe PnP landmark table for this dataset.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--process-limit", type=int, default=None, help="Optional limit applied only after fixed-shape calibration; useful for diagnostics.")
    parser.add_argument(
        "--estimate-depth-uncertainty",
        action="store_true",
        help=(
            "Estimate a label-free per-frame log-depth scale uncertainty by "
            "perturbing the saved PnP and iris observations. This is required "
            "by fixed_prior/learned_reweight depth-distribution training."
        ),
    )
    parser.add_argument(
        "--uncertainty-bootstrap-samples",
        type=int,
        default=64,
        help="Observation-perturbation PnP/Iris solves per successful frame.",
    )
    parser.add_argument(
        "--uncertainty-iris-std-px",
        type=float,
        default=0.5,
        help="Independent Gaussian standard deviation applied to each iris-centre pixel coordinate.",
    )
    parser.add_argument(
        "--uncertainty-ipd-std-mm",
        type=float,
        default=0.0,
        help="Optional measured-IPD uncertainty; zero keeps the IPD fixed.",
    )
    parser.add_argument(
        "--uncertainty-min-success-count",
        type=int,
        default=32,
        help="Minimum successful perturbation solves required to mark uncertainty success.",
    )
    parser.add_argument(
        "--uncertainty-seed",
        type=int,
        default=42,
        help="Deterministic global seed mixed with sample_id for uncertainty sampling.",
    )
    parser.add_argument("--trace-stages", action="store_true", help="Print the processing stage for each generated record.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dataset_id is not None:
        artifacts = processed_dataset_artifacts(args.dataset_id, args.processed_dir)
        args.csv = artifacts.dataset_csv
        args.output = artifacts.iris_ipd_depth_prior
        args.iris_csv = artifacts.mediapipe_iris_centres
        args.pnp_landmark_csv = artifacts.mediapipe_pnp_landmarks
        args.calibration_dataset = artifacts.dataset_name
    return args


def read_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "dataset", "image_name", "source_dataset_dir", "source_image_path", "face_path", "face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Dataset CSV is empty or lacks required image/face columns.")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive.")
        rows = rows[:limit]
    return rows


def read_pnp_points(
    row: dict[str, str],
    cache: dict[Path, dict[str, dict[str, str]]],
    pnp_landmark_csv: Path | None,
) -> dict[str, tuple[float, float]]:
    path = pnp_landmark_csv or (Path(row["source_dataset_dir"]) / "mediapipe_pnp_landmarks.csv")
    if path not in cache:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            cache[path] = {item["image_name"]: item for item in csv.DictReader(handle)}
    item = cache[path].get(row["image_name"])
    if item is None or item.get("status") != "success":
        raise RuntimeError("No successful saved MediaPipe PnP observation.")
    labels = ("left_eye_outer", "left_eye_inner", "right_eye_inner", "right_eye_outer", "nose_tip", "mouth_left", "mouth_right", "chin")
    return {label: (float(item[f"{label}_x"]), float(item[f"{label}_y"])) for label in labels}


def load_deca_input(
    row: dict[str, str],
    *,
    face_preprocess: str,
    crop_scale: float,
    image_size: int,
    image_root: Path | None,
    np: Any,
    torch: Any,
    image_cls: Any,
) -> tuple[Any, Any, Any]:
    source_image_path = resolve_image_asset_path(row, "source_image_path", image_root)
    with image_cls.open(source_image_path) as image:
        source = np.asarray(image.convert("RGB"))
    legacy = None
    if face_preprocess != FACE_PREPROCESS_DECA:
        face_path = resolve_image_asset_path(row, "face_path", image_root)
        with image_cls.open(face_path) as image:
            legacy = np.asarray(image.convert("RGB"))
    bbox = tuple(float(row[f"face_bbox_{name}"]) for name in ("x", "y", "w", "h"))
    deca_image, _ = prepare_deca_face_image(source, bbox, mode=face_preprocess, input_size=image_size, deca_crop_scale=crop_scale, legacy_face_image_rgb=legacy)
    tensor = torch.from_numpy(deca_image.copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return source, tensor, bbox


def fixed_shape_from_calibration(
    rows: list[dict[str, str]],
    extractor: Any,
    args: argparse.Namespace,
    np: Any,
    torch: Any,
    image_cls: Any,
) -> tuple[Any, list[dict[str, str]], list[int]]:
    candidates = [row for row in rows if row["dataset"] == args.calibration_dataset]
    if not candidates:
        raise ValueError(f"No rows found for calibration dataset {args.calibration_dataset!r}.")

    missing_indices: list[int] = []
    if args.calibration_index_mode == "position":
        selected = candidates[args.calibration_start : args.calibration_stop + 1]
    else:
        if "frame_idx" not in candidates[0]:
            raise ValueError("Dataset CSV lacks frame_idx; use --calibration-index-mode position or add frame_idx.")
        selected = []
        present_indices: set[int] = set()
        for row in candidates:
            raw_frame_idx = row.get("frame_idx", "")
            try:
                frame_idx = int(raw_frame_idx)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid frame_idx {raw_frame_idx!r} in sample {row.get('sample_id', '<unknown>')!r}.") from error
            if args.calibration_start <= frame_idx <= args.calibration_stop:
                selected.append(row)
                present_indices.add(frame_idx)
        missing_indices = [
            frame_idx
            for frame_idx in range(args.calibration_start, args.calibration_stop + 1)
            if frame_idx not in present_indices
        ]

    if not selected:
        raise ValueError(
            "Calibration range selected no rows. Check --calibration-index-mode and "
            "the dataset/frame_idx convention."
        )
    shapes = []
    for row in selected:
        _, tensor, _ = load_deca_input(
            row,
            face_preprocess=args.face_preprocess,
            crop_scale=args.deca_crop_scale,
            image_size=extractor.config.image_size,
            image_root=args.image_root,
            np=np,
            torch=torch,
            image_cls=image_cls,
        )
        output = extractor.extract(tensor)
        shapes.append(output.parameters[0, :100].numpy())
    return np.median(np.stack(shapes, axis=0), axis=0), selected, missing_indices


def empty_record(row: dict[str, str], reason: str) -> dict[str, Any]:
    record = {field: "" for field in FIELDS}
    record.update({"sample_id": row["sample_id"], "dataset": row["dataset"], "image_name": row["image_name"], "source_image_path": row["source_image_path"], "depth_prior_status": "failed", "reason": reason})
    return record


def write_csv_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _pnp_projected_points_and_centered_residuals(
    pnp_points: dict[str, tuple[float, float]],
    landmarks: Any,
    pnp: Any,
    camera: PnpCamera,
    np: Any,
) -> tuple[tuple[str, ...], Any, Any]:
    """Return baseline PnP projections and a centered residual pool.

    The residual bootstrap deliberately perturbs only the detected image
    observations; the fixed FLAME shape and metric scale remain unchanged.
    """

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required for depth uncertainty estimation.") from error
    labels = tuple(item.label for item in DEFAULT_PNP_MAPPING)
    observed = np.asarray([pnp_points[label] for label in labels], dtype=np.float64)
    object_points_mm = np.asarray(
        [landmarks[item.flame_landmark_index] for item in DEFAULT_PNP_MAPPING],
        dtype=np.float64,
    ) * float(pnp.scale.scale_mm_per_flame_unit)
    projected, _ = cv2.projectPoints(
        object_points_mm,
        np.asarray(pnp.rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(pnp.tvec_mm, dtype=np.float64).reshape(3, 1),
        camera.camera_matrix,
        camera.dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    residuals = observed - projected
    return labels, projected, residuals - residuals.mean(axis=0, keepdims=True)


def estimate_depth_uncertainty(
    *,
    row: dict[str, str],
    pnp_points: dict[str, tuple[float, float]],
    landmarks: Any,
    pnp: Any,
    assigned_iris: dict[str, Any],
    direction: Any,
    baseline_depth_z_mm: float,
    camera: PnpCamera,
    args: argparse.Namespace,
    np: Any,
) -> dict[str, Any]:
    """Estimate per-frame common log-depth-scale uncertainty without labels."""

    if baseline_depth_z_mm <= 0 or not np.isfinite(baseline_depth_z_mm):
        return {
            "depth_uncertainty_status": "failed",
            "depth_uncertainty_reason": "baseline midpoint depth is not positive and finite",
        }
    labels, projected, residuals = _pnp_projected_points_and_centered_residuals(
        pnp_points, landmarks, pnp, camera, np
    )
    seed_material = f"{args.uncertainty_seed}:{row['sample_id']}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    log_scales: list[float] = []
    left_base = np.asarray(assigned_iris["left"], dtype=np.float64)
    right_base = np.asarray(assigned_iris["right"], dtype=np.float64)
    for _ in range(args.uncertainty_bootstrap_samples):
        sampled_residual_indices = rng.integers(0, len(labels), size=len(labels))
        perturbed_pnp = {
            label: tuple(projected[index] + residuals[sampled_residual_indices[index]])
            for index, label in enumerate(labels)
        }
        try:
            perturbed_pose = solve_pnp_face_depth(
                perturbed_pnp,
                landmarks,
                camera,
                scale_mm_per_flame_unit=args.pnp_scale_mm_per_flame_unit,
            )
            left_iris = left_base + rng.normal(
                0.0, args.uncertainty_iris_std_px, size=2
            )
            right_iris = right_base + rng.normal(
                0.0, args.uncertainty_iris_std_px, size=2
            )
            ipd_mm = args.ipd_mm + rng.normal(0.0, args.uncertainty_ipd_std_mm)
            if ipd_mm <= 0:
                continue
            reconstruction = reconstruct_iris_centres_from_ipd(
                left_iris,
                right_iris,
                head_left_to_right_direction_flame=direction,
                flame_to_camera_rotation=perturbed_pose.rotation_matrix,
                camera_matrix=camera.camera_matrix,
                dist_coeffs=camera.dist_coeffs,
                ipd_mm=float(ipd_mm),
                max_condition=args.max_ray_condition,
                max_baseline_residual_mm=args.max_ipd_residual_mm,
            )
            scale = float(
                reconstruction.midpoint_camera_xyz_mm[2] / baseline_depth_z_mm
            )
            if scale > 0 and np.isfinite(scale):
                log_scales.append(float(np.log(scale)))
        except Exception:
            # Failed perturbation solves are expected observations of an
            # ill-conditioned frame; their count is retained in the record.
            continue
    success_count = len(log_scales)
    record: dict[str, Any] = {
        "depth_uncertainty_sample_count": success_count,
        "depth_bootstrap_success_count": success_count,
    }
    if success_count < args.uncertainty_min_success_count:
        record.update(
            {
                "depth_uncertainty_status": "failed",
                "depth_uncertainty_reason": (
                    "insufficient successful perturbation solves: "
                    f"{success_count}/{args.uncertainty_bootstrap_samples}"
                ),
            }
        )
        return record
    scales = np.exp(np.asarray(log_scales, dtype=np.float64))
    log_scale_std = float(np.std(log_scales, ddof=1))
    if not np.isfinite(log_scale_std) or log_scale_std < 0:
        record.update(
            {
                "depth_uncertainty_status": "failed",
                "depth_uncertainty_reason": "non-finite log-depth uncertainty",
            }
        )
        return record
    record.update(
        {
            "depth_uncertainty_status": "success",
            "depth_uncertainty_reason": "",
            "depth_log_scale_std": log_scale_std,
            "depth_scale_p05": float(np.percentile(scales, 5.0)),
            "depth_scale_p50": float(np.percentile(scales, 50.0)),
            "depth_scale_p95": float(np.percentile(scales, 95.0)),
        }
    )
    return record


def main() -> int:
    args = parse_args()
    if args.ipd_mm <= 0 or args.pnp_scale_mm_per_flame_unit <= 0:
        raise ValueError("--ipd-mm and --pnp-scale-mm-per-flame-unit must be positive.")
    if args.max_ipd_residual_mm <= 0 or args.max_ray_condition <= 1:
        raise ValueError("IPD residual limit must be positive and ray condition must exceed 1.")
    if args.calibration_start < 0 or args.calibration_stop < args.calibration_start:
        raise ValueError("Calibration indices must be non-negative and inclusive.")
    if args.process_limit is not None and args.process_limit <= 0:
        raise ValueError("--process-limit must be positive.")
    if args.uncertainty_bootstrap_samples < 2:
        raise ValueError("--uncertainty-bootstrap-samples must be at least 2.")
    if not 0 < args.uncertainty_min_success_count <= args.uncertainty_bootstrap_samples:
        raise ValueError(
            "--uncertainty-min-success-count must lie in "
            "[1, --uncertainty-bootstrap-samples]."
        )
    if args.uncertainty_iris_std_px < 0 or args.uncertainty_ipd_std_mm < 0:
        raise ValueError("Uncertainty standard deviations must be non-negative.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --overwrite to replace it.")
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Generation requires NumPy, PyTorch, and Pillow.") from error

    rows = read_rows(args.csv, args.limit)
    extractor = DecaFlameExtractor(DecaFlameConfig(device=None if args.device == "auto" else args.device))
    fixed_shape, calibration_rows, missing_calibration_indices = fixed_shape_from_calibration(rows, extractor, args, np, torch, Image)
    print(
        f"Fixed shape: median of {len(calibration_rows)} rows from {args.calibration_dataset} "
        f"using {args.calibration_index_mode} [{args.calibration_start}:{args.calibration_stop}] inclusive"
    )
    if missing_calibration_indices:
        print(
            f"Calibration range has {len(missing_calibration_indices)} missing/deleted frame_idx values; "
            "continuing with the available rows."
        )
    camera = PnpCamera(CROSSGAZE_CAMERA_MATRIX, CROSSGAZE_DIST_COEFFS)
    pnp_cache: dict[Path, dict[str, dict[str, str]]] = {}
    records: list[dict[str, Any]] = []
    processing_rows = rows if args.process_limit is None else rows[: args.process_limit]
    for index, row in enumerate(processing_rows, start=1):
        try:
            if args.trace_stages:
                print(f"[{index}/{len(processing_rows)}] load image: {row['sample_id']}", flush=True)
            source, tensor, _ = load_deca_input(
                row,
                face_preprocess=args.face_preprocess,
                crop_scale=args.deca_crop_scale,
                image_size=extractor.config.image_size,
                image_root=args.image_root,
                np=np,
                torch=torch,
                image_cls=Image,
            )
            if args.trace_stages:
                print(f"[{index}/{len(processing_rows)}] DECA inference", flush=True)
            output = extractor.extract(tensor, fixed_shape_params=fixed_shape)
            landmarks = output.landmarks3d[0].numpy()
            if args.trace_stages:
                print(f"[{index}/{len(processing_rows)}] saved PnP points + solvePnP", flush=True)
            pnp_points = read_pnp_points(row, pnp_cache, args.pnp_landmark_csv)
            pnp = solve_pnp_face_depth(
                pnp_points,
                landmarks,
                camera,
                scale_mm_per_flame_unit=args.pnp_scale_mm_per_flame_unit,
            )
            if args.trace_stages:
                print(f"[{index}/{len(processing_rows)}] load iris centres", flush=True)
            iris = load_iris_centres_csv(args.iris_csv, row["image_name"], sample_id=row["sample_id"]) if args.iris_csv else detect_iris_centres_rgb(source)
            left_canthus, right_canthus = compute_eye_canthus_midpoints(landmarks)
            if args.trace_stages:
                print(f"[{index}/{len(processing_rows)}] project canthus", flush=True)
            canthus_pixels = project_flame_local_points(np.stack((left_canthus, right_canthus)), rotation_matrix=pnp.rotation_matrix, tvec_mm=pnp.tvec_mm, scale_mm_per_flame_unit=pnp.scale.scale_mm_per_flame_unit, camera_matrix=camera.camera_matrix, dist_coeffs=camera.dist_coeffs)
            assigned = assign_iris_groups_to_anatomical_sides(iris.as_mapping(), left_canthus_projection_xy=canthus_pixels[0], right_canthus_projection_xy=canthus_pixels[1])
            direction = right_canthus - left_canthus
            if args.trace_stages:
                print(f"[{index}/{len(processing_rows)}] reconstruct iris/IPD", flush=True)
            result = reconstruct_iris_centres_from_ipd(assigned["left"], assigned["right"], head_left_to_right_direction_flame=direction, flame_to_camera_rotation=pnp.rotation_matrix, camera_matrix=camera.camera_matrix, dist_coeffs=camera.dist_coeffs, ipd_mm=args.ipd_mm, max_condition=args.max_ray_condition, max_baseline_residual_mm=args.max_ipd_residual_mm)
            left, right, midpoint = result.left_iris_camera_xyz_mm, result.right_iris_camera_xyz_mm, result.midpoint_camera_xyz_mm
            record = {field: "" for field in FIELDS}
            record.update({
                "sample_id": row["sample_id"], "dataset": row["dataset"], "image_name": row["image_name"], "source_image_path": row["source_image_path"], "depth_prior_status": "success", "reason": "",
                # Compatibility fields deliberately contain iris/pupil proxies; explicit fields below retain provenance.
                "left_eye_camera_x_mm": float(left[0]), "left_eye_camera_y_mm": float(left[1]), "left_eye_camera_z_mm": float(left[2]),
                "right_eye_camera_x_mm": float(right[0]), "right_eye_camera_y_mm": float(right[1]), "right_eye_camera_z_mm": float(right[2]),
                "left_iris_camera_x_mm": float(left[0]), "left_iris_camera_y_mm": float(left[1]), "left_iris_camera_z_mm": float(left[2]),
                "right_iris_camera_x_mm": float(right[0]), "right_iris_camera_y_mm": float(right[1]), "right_iris_camera_z_mm": float(right[2]),
                "iris_midpoint_camera_x_mm": float(midpoint[0]), "iris_midpoint_camera_y_mm": float(midpoint[1]), "iris_midpoint_depth_z_mm": float(midpoint[2]),
                "measured_ipd_mm": args.ipd_mm, "iris_left_group": assigned["left_group"], "iris_right_group": assigned["right_group"], "iris_assignment_cost_px": assigned["assignment_cost_px"],
                "iris_baseline_residual_mm": result.baseline_residual_mm, "iris_ray_system_condition": result.ray_system_condition,
                "pnp_reprojection_error_mean_px": pnp.reprojection_error_mean_px, "pnp_reprojection_error_max_px": pnp.reprojection_error_max_px,
                "pnp_rvec_x_rad": float(pnp.rvec[0]), "pnp_rvec_y_rad": float(pnp.rvec[1]), "pnp_rvec_z_rad": float(pnp.rvec[2]),
                "pnp_scale_mm_per_flame_unit": args.pnp_scale_mm_per_flame_unit, "fixed_shape_calibration_count": len(calibration_rows),
            })
            if args.estimate_depth_uncertainty:
                record.update(
                    estimate_depth_uncertainty(
                        row=row,
                        pnp_points=pnp_points,
                        landmarks=landmarks,
                        pnp=pnp,
                        assigned_iris=assigned,
                        direction=direction,
                        baseline_depth_z_mm=float(midpoint[2]),
                        camera=camera,
                        args=args,
                        np=np,
                    )
                )
            records.append(record)
        except Exception as error:
            records.append(empty_record(row, f"{type(error).__name__}: {error}"))
        if index % 20 == 0 or index == len(processing_rows):
            print(f"Processed {index}/{len(processing_rows)}")
    write_csv_atomic(args.output, records)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "method": "metric iris/pupil-proxy ray reconstruction constrained by measured IPD", "ipd_mm": args.ipd_mm,
        "quality_thresholds": {"max_ipd_residual_mm": args.max_ipd_residual_mm, "max_ray_condition": args.max_ray_condition},
        "fixed_shape": {
            "dataset": args.calibration_dataset,
            "index_mode": args.calibration_index_mode,
            "start_inclusive": args.calibration_start,
            "stop_inclusive": args.calibration_stop,
            "count": len(calibration_rows),
            "missing_frame_indices": missing_calibration_indices,
        },
        "pnp": {"role": "head rotation and canthus direction only", "scale_mm_per_flame_unit": args.pnp_scale_mm_per_flame_unit},
        "depth_uncertainty": {
            "enabled": args.estimate_depth_uncertainty,
            "method": "centered PnP reprojection-residual bootstrap plus Gaussian iris-centre perturbation",
            "bootstrap_samples": args.uncertainty_bootstrap_samples,
            "iris_std_px": args.uncertainty_iris_std_px,
            "ipd_std_mm": args.uncertainty_ipd_std_mm,
            "min_success_count": args.uncertainty_min_success_count,
            "seed": args.uncertainty_seed,
            "quantity": "sample standard deviation of log(midpoint_depth / baseline_midpoint_depth)",
            "uses_depth_gt": False,
            "uses_uv_gt": False,
        },
        "image_root_override": str(args.image_root) if args.image_root is not None else None,
        "warning": "Iris-centre proxies are not anatomical eyeball centres.",
        "status_counts": {status: sum(record["depth_prior_status"] == status for record in records) for status in ("success", "failed")},
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
