"""Analyze per-sample vertical gaze error for a trained ModelV1 checkpoint.

The script runs a deterministic train/validation split, ranks samples by
absolute table-local v error, and writes:

* ``vertical_gaze_samples.csv``: every sample, worst first;
* ``worst_vertical_gaze_samples.csv``: the requested top-N subset;
* ``vertical_gaze_bins.csv``: metrics grouped by ground-truth v coordinate;
* ``vertical_gaze_summary.json``: machine-readable metrics and artifact paths;
* ``vertical_gaze_bins.png``: a four-panel binned diagnostic figure.

The signed error convention is always ``prediction - ground truth``.  A sample
is marked bad when its absolute v error is at or above ``--bad-percentile`` or
when the prediction is non-finite.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1 import ModelV1
from modelv1.data import (
    DIRECT_UV_TARGET_FRAME_TABLE_LOCAL,
    DIRECT_UV_TARGET_FRAME_VIRTUAL_CAMERA,
    IMAGE_SOURCE_LEGACY,
    IMAGE_SOURCE_VIRTUAL_CAMERA,
    VirtualCameraManifest,
    build_modelv1_dataloaders,
    canonical_direct_uv_target_frame,
    canonical_image_source,
)
from modelv1.data.normalization import (
    EyeGeometryNormalizer,
    EyeGeometryQualityNormalizer,
    UVTargetNormalizer,
)
from modelv1.depth_distribution import (
    DEPTH_DISTRIBUTION_LEARNED_REWEIGHT,
    DEPTH_DISTRIBUTION_POINT,
)
from modelv1.geometry import virtual_camera_xy_to_table_uv
from modelv1.model import PREDICTION_MODE_GAZE_GEOMETRY
from scripts.train_modelv1 import (
    checkpoint_model_state,
    load_config,
    make_model_config,
    move_batch_to_device,
    resolve_device,
    resolve_numbered_dataset_artifacts,
    resolve_preprocessed_v2_artifacts,
    resolve_project_path,
    resolve_sample_id_filter_manifest_path,
    resolve_table7_stage3_artifacts,
    resolve_virtual_camera_manifest_path,
    seed_everything,
    validate_config,
)


SAMPLE_COLUMNS = (
    "bad_rank",
    "vertical_error_percentile",
    "is_bad",
    "bad_reason",
    "prediction_valid",
    "sample_id",
    "dataset",
    "image_name",
    "frame_idx",
    "gt_u_mm",
    "gt_v_mm",
    "pred_u_mm",
    "pred_v_mm",
    "error_u_mm",
    "error_v_mm",
    "abs_error_u_mm",
    "abs_error_v_mm",
    "epe_mm",
    "v_minus_u_abs_error_mm",
    "v_dominates_u",
    "gt_v_bin",
    "gt_v_bin_label",
    "pred_sigma_u_mm",
    "pred_sigma_v_mm",
    "pred_uv_cov_mm2",
    "geometry_valid",
    "ray_valid_rate",
    "mean_abs_normal_dot_gaze",
    "mean_lambda_mm",
    "virtual_distance_scale",
    "z_table_mm",
    "head_pitch_deg",
    "head_yaw_deg",
    "head_roll_deg",
    "pnp_status",
    "source_dataset_dir",
    "source_image_path",
    "face_path",
    "left_eye_path",
    "right_eye_path",
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
)

BIN_COLUMNS = (
    "bin_index",
    "bin_label",
    "gt_v_lower_mm",
    "gt_v_upper_mm",
    "sample_count",
    "valid_prediction_count",
    "invalid_prediction_count",
    "gt_v_mean_mm",
    "mae_u_mm",
    "mae_v_mm",
    "median_abs_v_error_mm",
    "p90_abs_v_error_mm",
    "signed_u_bias_mm",
    "signed_v_bias_mm",
    "mean_epe_mm",
    "bad_sample_rate",
    "v_dominance_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best.pt or last.pt.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Training YAML. By default, config.yaml beside the checkpoint run "
            "is used, then the config embedded in the checkpoint as fallback."
        ),
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run>/analysis/vertical_gaze_error_<split>.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete device such as cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override data.batch_size for analysis only.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override data.num_workers for analysis only.",
    )
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument(
        "--binning",
        choices=("quantile", "uniform"),
        default="quantile",
        help="How ground-truth v coordinates are divided into bins.",
    )
    parser.add_argument(
        "--bad-percentile",
        type=float,
        default=90.0,
        help="Mark samples at or above this absolute-v-error percentile as bad.",
    )
    parser.add_argument(
        "--worst-count",
        type=int,
        default=50,
        help="Number of worst samples copied to the dedicated CSV/JSON list.",
    )
    parser.add_argument(
        "--print-worst",
        type=int,
        default=15,
        help="Number of worst samples printed after analysis; use 0 to disable.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.num_bins < 2:
        raise ValueError("--num-bins must be at least 2.")
    if not 0.0 < args.bad_percentile < 100.0:
        raise ValueError("--bad-percentile must lie strictly between 0 and 100.")
    if args.worst_count <= 0:
        raise ValueError("--worst-count must be positive.")
    if args.print_worst < 0:
        raise ValueError("--print-worst must be non-negative.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch before weights_only was added.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), Mapping
    ):
        raise ValueError(f"Checkpoint has no model state mapping: {path}")
    return checkpoint


def _load_analysis_config(
    *,
    explicit_path: Path | None,
    run_dir: Path,
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if explicit_path is not None:
        path = resolve_project_path(explicit_path).resolve()
        config = load_config(path)
        source = str(path)
    else:
        run_config_path = run_dir / "config.yaml"
        if run_config_path.is_file():
            config = load_config(run_config_path)
            source = str(run_config_path.resolve())
        else:
            embedded = checkpoint.get("config")
            if not isinstance(embedded, Mapping):
                raise ValueError(
                    "No run config.yaml was found and the checkpoint has no "
                    "embedded config. Pass --config explicitly."
                )
            config = copy.deepcopy(dict(embedded))
            validate_config(config)
            source = "checkpoint:config"

    embedded = checkpoint.get("config")
    if isinstance(embedded, Mapping) and isinstance(embedded.get("model"), Mapping):
        configured_model = make_model_config(config["model"])
        checkpoint_model = make_model_config(embedded["model"])
        if configured_model != checkpoint_model:
            raise ValueError(
                "The selected config has a different model architecture from the "
                "checkpoint. Use the checkpoint run's config.yaml."
            )
    return config, source


def _normalizer_from_checkpoint(
    checkpoint: Mapping[str, Any],
    key: str,
    normalizer_type: Any,
) -> Any | None:
    state = checkpoint.get(key)
    if state is None:
        return None
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint field {key!r} must be a mapping or null.")
    return normalizer_type.from_state_dict(state)


def _resolve_data_inputs(
    config: Mapping[str, Any],
    model_config: Any,
    artifact_dir: Path,
) -> tuple[Path, Path | None, Path | None, frozenset[str] | None]:
    data_config = config["data"]
    uses_gaze_geometry = (
        model_config.prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY
    )
    require_depth_prior = model_config.use_eye_geometry or uses_gaze_geometry
    uses_table7_stage3 = (
        data_config.get("table7_stage3_csv_paths") is not None
    )
    preprocessed_virtual_manifests: list[Path] = []

    if uses_table7_stage3:
        dataset_csv = resolve_table7_stage3_artifacts(
            data_config,
            artifact_dir,
        )
        depth_prior = None
    elif data_config.get("preprocessed_v2_root") is not None:
        (
            dataset_csv,
            depth_prior,
            preprocessed_virtual_manifests,
        ) = resolve_preprocessed_v2_artifacts(
            data_config,
            artifact_dir,
            require_depth_prior=require_depth_prior,
        )
    else:
        dataset_csv, depth_prior = resolve_numbered_dataset_artifacts(
            data_config,
            artifact_dir,
            require_depth_prior=require_depth_prior,
        )

    image_source = canonical_image_source(
        data_config.get("image_source", IMAGE_SOURCE_LEGACY)
    )
    virtual_manifest = resolve_virtual_camera_manifest_path(
        data_config,
        artifact_dir,
        fallback_paths=(
            preprocessed_virtual_manifests
            if image_source == IMAGE_SOURCE_VIRTUAL_CAMERA
            else None
        ),
    )
    filter_manifest = resolve_sample_id_filter_manifest_path(
        data_config,
        artifact_dir,
        fallback_paths=(
            preprocessed_virtual_manifests
            if image_source == IMAGE_SOURCE_LEGACY
            else None
        ),
    )
    allowed_sample_ids = (
        frozenset(VirtualCameraManifest.load(filter_manifest).records)
        if filter_manifest is not None
        else None
    )
    return dataset_csv, depth_prior, virtual_manifest, allowed_sample_ids


def _build_analysis_loader(
    *,
    config: Mapping[str, Any],
    model_config: Any,
    checkpoint: Mapping[str, Any],
    dataset_csv: Path,
    depth_prior: Path | None,
    virtual_manifest: Path | None,
    allowed_sample_ids: frozenset[str] | None,
    split: str,
    batch_size_override: int | None,
    num_workers_override: int | None,
) -> tuple[DataLoader, UVTargetNormalizer | None]:
    data_config = config["data"]
    uses_gaze_geometry = (
        model_config.prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY
    )
    require_depth_uncertainty = uses_gaze_geometry and (
        model_config.depth_distribution_mode != DEPTH_DISTRIBUTION_POINT
        or model_config.use_depth_correction
    )
    require_pnp_quality = uses_gaze_geometry and (
        model_config.depth_distribution_mode
        == DEPTH_DISTRIBUTION_LEARNED_REWEIGHT
        or model_config.use_depth_correction
    )

    uv_normalizer = _normalizer_from_checkpoint(
        checkpoint, "normalizer", UVTargetNormalizer
    )
    eye_normalizer = _normalizer_from_checkpoint(
        checkpoint, "eye_geometry_normalizer", EyeGeometryNormalizer
    )
    quality_normalizer = _normalizer_from_checkpoint(
        checkpoint,
        "eye_geometry_quality_normalizer",
        EyeGeometryQualityNormalizer,
    )
    correction_normalizer = _normalizer_from_checkpoint(
        checkpoint,
        "depth_correction_geometry_normalizer",
        EyeGeometryNormalizer,
    )
    if not uses_gaze_geometry and uv_normalizer is None:
        raise ValueError("Direct-UV checkpoint has no saved UV normalizer.")

    eye_geometry_pseudo_label_paths = [
        resolve_project_path(path)
        for path in data_config.get("eye_geometry_pseudo_label_paths", ())
    ] or None

    batch_size = int(
        batch_size_override
        if batch_size_override is not None
        else data_config["batch_size"]
    )
    num_workers = int(
        num_workers_override
        if num_workers_override is not None
        else data_config["num_workers"]
    )
    train_loader, val_loader = build_modelv1_dataloaders(
        csv_path=dataset_csv,
        train_datasets=data_config.get("train_datasets", ("3", "4")),
        val_datasets=data_config.get("val_datasets", ("5",)),
        split_mode=data_config["split_mode"],
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(data_config["pin_memory"]),
        normalize_images=bool(data_config["normalize_images"]),
        # Analysis must be deterministic, including when --split=train.
        train_paired_eye_transform=None,
        normalize_uv_targets=not uses_gaze_geometry,
        target_normalizer=uv_normalizer,
        use_eye_geometry=model_config.use_eye_geometry,
        use_gaze_geometry=uses_gaze_geometry,
        require_depth_uncertainty=require_depth_uncertainty,
        require_pnp_quality=require_pnp_quality,
        use_depth_correction=model_config.use_depth_correction,
        filter_invalid_depth_prior_samples=bool(
            data_config.get("skip_invalid_depth_prior_samples", True)
        ),
        depth_prior_csv_path=depth_prior,
        eye_geometry_normalizer=eye_normalizer,
        eye_geometry_gate_mode=model_config.eye_geometry_gate_mode,
        eye_geometry_quality_normalizer=quality_normalizer,
        depth_correction_geometry_normalizer=correction_normalizer,
        eye_geometry_representation=model_config.eye_geometry_representation,
        scene_representation=model_config.scene_representation,
        image_source=str(data_config.get("image_source", IMAGE_SOURCE_LEGACY)),
        virtual_camera_manifest_path=virtual_manifest,
        filter_invalid_virtual_camera_samples=bool(
            data_config.get("skip_invalid_virtual_camera_samples", False)
        ),
        allowed_sample_ids=allowed_sample_ids,
        direct_uv_target_frame=str(
            data_config.get(
                "direct_uv_target_frame",
                DIRECT_UV_TARGET_FRAME_TABLE_LOCAL,
            )
        ),
        use_table_frame_film=model_config.use_table_frame_film,
        precomputed_table_frame7=(
            data_config.get("table7_stage3_csv_paths") is not None
        ),
        eye_geometry_pseudo_label_paths=eye_geometry_pseudo_label_paths,
        require_eye_geometry_pseudo_labels=bool(
            data_config.get("require_eye_geometry_pseudo_labels", False)
        ),
    )
    selected = val_loader if split == "val" else train_loader
    if split == "val":
        return selected, uv_normalizer
    # The training loader shuffles by design. Re-wrap the same deterministic
    # dataset so ranks and output order do not depend on a sampler RNG state.
    return (
        DataLoader(
            selected.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(data_config["pin_memory"]),
            drop_last=False,
        ),
        uv_normalizer,
    )


def _read_metadata(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise ValueError(f"Dataset CSV has no sample_id column: {csv_path}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ValueError(f"Duplicate sample_id in dataset CSV: {sample_id}")
            rows[sample_id] = row
    return rows


def _batch_strings(batch: Mapping[str, object], key: str) -> list[str]:
    values = batch.get(key)
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"batch[{key!r}] must be a string sequence.")
    return [str(value) for value in values]


def _path_at(batch: Mapping[str, object], key: str, index: int) -> str:
    paths = batch.get("paths")
    if not isinstance(paths, Mapping):
        return ""
    values = paths.get(key)
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return ""
    return str(values[index])


def _optional_tensor(output: Mapping[str, object], key: str) -> torch.Tensor | None:
    value = output.get(key)
    return value if torch.is_tensor(value) else None


def _predict_table_uv(
    *,
    model: ModelV1,
    batch: Mapping[str, object],
    uv_normalizer: UVTargetNormalizer | None,
    direct_uv_target_frame: str,
) -> tuple[torch.Tensor, Mapping[str, object]]:
    output = model(batch)
    output_mapping: Mapping[str, object] = output if isinstance(output, Mapping) else {}
    if model.config.prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY:
        prediction = output_mapping.get("uv_mean_mm")
        if not torch.is_tensor(prediction):
            raise TypeError("Geometry model output has no tensor uv_mean_mm.")
        return prediction.float(), output_mapping

    normalized = output_mapping.get("uv") if output_mapping else output
    if not torch.is_tensor(normalized):
        raise TypeError("Direct-UV model output must contain a tensor prediction.")
    if uv_normalizer is None:
        raise RuntimeError("Direct-UV inference requires the checkpoint normalizer.")
    point_xy_or_uv_mm = uv_normalizer.denormalize(normalized.float())
    if direct_uv_target_frame == DIRECT_UV_TARGET_FRAME_TABLE_LOCAL:
        return point_xy_or_uv_mm, output_mapping
    if direct_uv_target_frame != DIRECT_UV_TARGET_FRAME_VIRTUAL_CAMERA:
        raise AssertionError(f"Unhandled direct UV target frame: {direct_uv_target_frame}")
    frame = batch.get("table_frame7_n")
    if not torch.is_tensor(frame):
        raise TypeError("Virtual-camera UV inference requires batch['table_frame7_n'].")
    table_uv = virtual_camera_xy_to_table_uv(
            point_xy_or_uv_mm,
            frame.float(),
            distance_scale_mm=model.config.table_distance_scale_mm,
        )
    return table_uv, output_mapping


def _rounded(value: float | int | None, digits: int = 6) -> float | int | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def _tensor_item(tensor: torch.Tensor | None, index: int) -> float | None:
    if tensor is None:
        return None
    value = tensor[index]
    if value.numel() != 1:
        raise ValueError("Expected a per-sample scalar diagnostic tensor.")
    return _rounded(float(value.detach().cpu().item()))


def collect_sample_records(
    *,
    model: ModelV1,
    loader: DataLoader,
    device: torch.device,
    uv_normalizer: UVTargetNormalizer | None,
    direct_uv_target_frame: str,
    metadata: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            device_batch = move_batch_to_device(batch, device)
            prediction, output = _predict_table_uv(
                model=model,
                batch=device_batch,
                uv_normalizer=uv_normalizer,
                direct_uv_target_frame=direct_uv_target_frame,
            )
            target = device_batch.get("uv_gt_table_mm")
            if not torch.is_tensor(target):
                raise TypeError("batch['uv_gt_table_mm'] must be a tensor.")
            prediction = prediction.detach().float().cpu()
            target = target.detach().float().cpu()
            sample_ids = _batch_strings(batch, "sample_id")
            datasets = _batch_strings(batch, "dataset")
            image_names = _batch_strings(batch, "image_name")
            if prediction.shape != target.shape or prediction.shape != (len(sample_ids), 2):
                raise ValueError(
                    "Prediction/target batch shape mismatch: "
                    f"prediction={tuple(prediction.shape)}, "
                    f"target={tuple(target.shape)}, ids={len(sample_ids)}."
                )

            covariance = _optional_tensor(output, "uv_covariance_mm2")
            covariance = covariance.detach().float().cpu() if covariance is not None else None
            geometry_valid = _optional_tensor(output, "geometry_any_physically_valid")
            geometry_valid = (
                geometry_valid.detach().float().cpu() if geometry_valid is not None else None
            )
            ray_valid = _optional_tensor(output, "ray_valid_mask")
            ray_valid_rate = (
                ray_valid.detach().float().cpu().mean(dim=-1)
                if ray_valid is not None
                else None
            )
            normal_dot = _optional_tensor(output, "normal_dot_gaze")
            mean_abs_normal_dot = (
                normal_dot.detach().float().cpu().abs().reshape(len(sample_ids), -1).mean(dim=-1)
                if normal_dot is not None
                else None
            )
            lambdas = _optional_tensor(output, "lambda_hypotheses_mm")
            mean_lambda = (
                lambdas.detach().float().cpu().reshape(len(sample_ids), -1).mean(dim=-1)
                if lambdas is not None
                else None
            )
            virtual_scale = batch.get("virtual_distance_scale")
            virtual_scale = (
                virtual_scale.detach().float().cpu().reshape(len(sample_ids), -1).mean(dim=-1)
                if torch.is_tensor(virtual_scale)
                else None
            )

            for index, sample_id in enumerate(sample_ids):
                row = metadata.get(sample_id, {})
                gt_u, gt_v = (float(value) for value in target[index].tolist())
                pred_u, pred_v = (float(value) for value in prediction[index].tolist())
                prediction_valid = math.isfinite(pred_u) and math.isfinite(pred_v)
                if prediction_valid:
                    error_u = pred_u - gt_u
                    error_v = pred_v - gt_v
                    abs_u = abs(error_u)
                    abs_v = abs(error_v)
                    epe = math.hypot(error_u, error_v)
                else:
                    error_u = error_v = abs_u = abs_v = epe = None

                sigma_u = sigma_v = uv_cov = None
                if covariance is not None:
                    matrix = covariance[index]
                    if matrix.shape == (2, 2):
                        sigma_u = math.sqrt(max(float(matrix[0, 0]), 0.0))
                        sigma_v = math.sqrt(max(float(matrix[1, 1]), 0.0))
                        uv_cov = float(matrix[0, 1])

                record = {
                    "bad_rank": None,
                    "vertical_error_percentile": None,
                    "is_bad": False,
                    "bad_reason": "",
                    "prediction_valid": prediction_valid,
                    "sample_id": sample_id,
                    "dataset": datasets[index],
                    "image_name": image_names[index],
                    "frame_idx": row.get("frame_idx", ""),
                    "gt_u_mm": _rounded(gt_u),
                    "gt_v_mm": _rounded(gt_v),
                    "pred_u_mm": _rounded(pred_u),
                    "pred_v_mm": _rounded(pred_v),
                    "error_u_mm": _rounded(error_u),
                    "error_v_mm": _rounded(error_v),
                    "abs_error_u_mm": _rounded(abs_u),
                    "abs_error_v_mm": _rounded(abs_v),
                    "epe_mm": _rounded(epe),
                    "v_minus_u_abs_error_mm": (
                        _rounded(abs_v - abs_u)
                        if abs_v is not None and abs_u is not None
                        else None
                    ),
                    "v_dominates_u": (
                        bool(abs_v > abs_u)
                        if abs_v is not None and abs_u is not None
                        else None
                    ),
                    "gt_v_bin": None,
                    "gt_v_bin_label": "",
                    "pred_sigma_u_mm": _rounded(sigma_u),
                    "pred_sigma_v_mm": _rounded(sigma_v),
                    "pred_uv_cov_mm2": _rounded(uv_cov),
                    "geometry_valid": (
                        bool(_tensor_item(geometry_valid, index))
                        if geometry_valid is not None
                        else None
                    ),
                    "ray_valid_rate": _tensor_item(ray_valid_rate, index),
                    "mean_abs_normal_dot_gaze": _tensor_item(
                        mean_abs_normal_dot, index
                    ),
                    "mean_lambda_mm": _tensor_item(mean_lambda, index),
                    "virtual_distance_scale": _tensor_item(virtual_scale, index),
                    "z_table_mm": row.get("z_table_mm", ""),
                    "head_pitch_deg": row.get("head_pitch_deg", ""),
                    "head_yaw_deg": row.get("head_yaw_deg", ""),
                    "head_roll_deg": row.get("head_roll_deg", ""),
                    "pnp_status": row.get("pnp_status", ""),
                    "source_dataset_dir": row.get("source_dataset_dir", ""),
                    "source_image_path": _path_at(batch, "source", index)
                    or row.get("source_image_path", ""),
                    "face_path": _path_at(batch, "face", index),
                    "left_eye_path": _path_at(batch, "left_eye", index),
                    "right_eye_path": _path_at(batch, "right_eye", index),
                }
                for key in SAMPLE_COLUMNS:
                    if "_bbox_" in key:
                        record[key] = row.get(key, "")
                records.append(record)
    if not records:
        raise ValueError("The selected analysis split contains no samples.")
    return records


def rank_and_mark_bad_samples(
    records: list[dict[str, Any]], bad_percentile: float
) -> float:
    finite_errors = np.asarray(
        [
            float(record["abs_error_v_mm"])
            for record in records
            if record["abs_error_v_mm"] is not None
        ],
        dtype=np.float64,
    )
    if finite_errors.size == 0:
        raise ValueError("All predictions are non-finite; no error percentile is defined.")
    threshold = float(np.percentile(finite_errors, bad_percentile))
    ascending = np.sort(finite_errors)

    records.sort(
        key=lambda record: (
            record["prediction_valid"],
            -float(record["abs_error_v_mm"] or 0.0),
        )
    )
    for rank, record in enumerate(records, start=1):
        record["bad_rank"] = rank
        error = record["abs_error_v_mm"]
        if error is None:
            record["vertical_error_percentile"] = 100.0
            record["is_bad"] = True
            record["bad_reason"] = "non_finite_prediction"
            continue
        percentile = 100.0 * float(np.searchsorted(ascending, error, side="right")) / len(
            ascending
        )
        record["vertical_error_percentile"] = round(percentile, 3)
        record["is_bad"] = bool(float(error) >= threshold)
        record["bad_reason"] = (
            f"abs_v_error_at_or_above_p{bad_percentile:g}"
            if record["is_bad"]
            else ""
        )
    return threshold


def _make_bin_edges(values: np.ndarray, num_bins: int, strategy: str) -> np.ndarray:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Ground-truth v values must be a non-empty finite array.")
    if np.all(values == values[0]):
        delta = max(abs(float(values[0])) * 1e-6, 0.5)
        return np.asarray([values[0] - delta, values[0] + delta], dtype=np.float64)
    if strategy == "quantile":
        edges = np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1))
    elif strategy == "uniform":
        edges = np.linspace(float(values.min()), float(values.max()), num_bins + 1)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
    edges = np.unique(edges.astype(np.float64))
    if edges.size < 2:
        raise ValueError("Could not form at least one ground-truth v bin.")
    return edges


def build_binned_metrics(
    records: list[dict[str, Any]], num_bins: int, strategy: str
) -> tuple[list[dict[str, Any]], list[float]]:
    gt_v = np.asarray([float(record["gt_v_mm"]) for record in records], dtype=np.float64)
    edges = _make_bin_edges(gt_v, min(num_bins, len(records)), strategy)
    indices = np.searchsorted(edges[1:-1], gt_v, side="right")
    bins: list[dict[str, Any]] = []

    for bin_index in range(len(edges) - 1):
        member_indices = np.flatnonzero(indices == bin_index)
        lower = float(edges[bin_index])
        upper = float(edges[bin_index + 1])
        close = "]" if bin_index == len(edges) - 2 else ")"
        label = f"[{lower:.1f}, {upper:.1f}{close}"
        members = [records[int(index)] for index in member_indices]
        for member in members:
            member["gt_v_bin"] = bin_index
            member["gt_v_bin_label"] = label
        valid = [member for member in members if member["prediction_valid"]]

        def values(key: str) -> np.ndarray:
            return np.asarray([float(member[key]) for member in valid], dtype=np.float64)

        if valid:
            abs_u = values("abs_error_u_mm")
            abs_v = values("abs_error_v_mm")
            error_u = values("error_u_mm")
            error_v = values("error_v_mm")
            epe = values("epe_mm")
            mae_u = float(abs_u.mean())
            mae_v = float(abs_v.mean())
            median_v = float(np.median(abs_v))
            p90_v = float(np.percentile(abs_v, 90.0))
            bias_u = float(error_u.mean())
            bias_v = float(error_v.mean())
            mean_epe = float(epe.mean())
            dominance_rate = float(
                np.mean([bool(member["v_dominates_u"]) for member in valid])
            )
        else:
            mae_u = mae_v = median_v = p90_v = None
            bias_u = bias_v = mean_epe = dominance_rate = None

        bins.append(
            {
                "bin_index": bin_index,
                "bin_label": label,
                "gt_v_lower_mm": _rounded(lower),
                "gt_v_upper_mm": _rounded(upper),
                "sample_count": len(members),
                "valid_prediction_count": len(valid),
                "invalid_prediction_count": len(members) - len(valid),
                "gt_v_mean_mm": _rounded(
                    np.mean([float(member["gt_v_mm"]) for member in members])
                ),
                "mae_u_mm": _rounded(mae_u),
                "mae_v_mm": _rounded(mae_v),
                "median_abs_v_error_mm": _rounded(median_v),
                "p90_abs_v_error_mm": _rounded(p90_v),
                "signed_u_bias_mm": _rounded(bias_u),
                "signed_v_bias_mm": _rounded(bias_v),
                "mean_epe_mm": _rounded(mean_epe),
                "bad_sample_rate": _rounded(
                    np.mean([bool(member["is_bad"]) for member in members])
                ),
                "v_dominance_rate": _rounded(dominance_rate),
            }
        )
    return bins, [float(value) for value in edges]


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if bool(record["prediction_valid"])]
    if not valid:
        raise ValueError("No valid predictions are available for summary metrics.")

    def array(key: str) -> np.ndarray:
        return np.asarray([float(record[key]) for record in valid], dtype=np.float64)

    abs_u = array("abs_error_u_mm")
    abs_v = array("abs_error_v_mm")
    error_u = array("error_u_mm")
    error_v = array("error_v_mm")
    epe = array("epe_mm")
    percentile_levels = (50, 75, 90, 95, 99, 100)
    return {
        "sample_count": len(records),
        "valid_prediction_count": len(valid),
        "invalid_prediction_count": len(records) - len(valid),
        "bad_sample_count": sum(bool(record["is_bad"]) for record in records),
        "mae_u_mm": _rounded(abs_u.mean()),
        "mae_v_mm": _rounded(abs_v.mean()),
        "mae_v_minus_mae_u_mm": _rounded(abs_v.mean() - abs_u.mean()),
        "mae_v_over_mae_u": _rounded(
            abs_v.mean() / abs_u.mean() if abs_u.mean() > 0 else None
        ),
        "median_abs_u_error_mm": _rounded(np.median(abs_u)),
        "median_abs_v_error_mm": _rounded(np.median(abs_v)),
        "signed_u_bias_mm": _rounded(error_u.mean()),
        "signed_v_bias_mm": _rounded(error_v.mean()),
        "rmse_u_mm": _rounded(math.sqrt(float(np.mean(error_u**2)))),
        "rmse_v_mm": _rounded(math.sqrt(float(np.mean(error_v**2)))),
        "mean_epe_mm": _rounded(epe.mean()),
        "median_epe_mm": _rounded(np.median(epe)),
        "v_dominance_rate": _rounded(
            np.mean([bool(record["v_dominates_u"]) for record in valid])
        ),
        "abs_v_error_percentiles_mm": {
            f"p{level}": _rounded(np.percentile(abs_v, level))
            for level in percentile_levels
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_value(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [float(row[key]) if row[key] is not None else np.nan for row in rows],
        dtype=np.float64,
    )


def save_binned_plot(
    path: Path,
    bins: Sequence[Mapping[str, Any]],
    *,
    split: str,
    checkpoint_name: str,
    dpi: int,
) -> None:
    labels = [str(row["bin_label"]) for row in bins]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    mae_u = _plot_value(bins, "mae_u_mm")
    mae_v = _plot_value(bins, "mae_v_mm")
    axes[0, 0].bar(x - width / 2, mae_u, width, label="MAE u")
    axes[0, 0].bar(x + width / 2, mae_v, width, label="MAE v")
    axes[0, 0].set_ylabel("Absolute error (mm)")
    axes[0, 0].set_title("Axis MAE by ground-truth v bin")
    axes[0, 0].legend()

    axes[0, 1].plot(
        x,
        _plot_value(bins, "median_abs_v_error_mm"),
        marker="o",
        label="Median |v error|",
    )
    axes[0, 1].plot(
        x,
        _plot_value(bins, "p90_abs_v_error_mm"),
        marker="o",
        label="P90 |v error|",
    )
    axes[0, 1].set_ylabel("Vertical absolute error (mm)")
    axes[0, 1].set_title("Vertical error tail by ground-truth v bin")
    axes[0, 1].legend()

    axes[1, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[1, 0].bar(x, _plot_value(bins, "signed_v_bias_mm"), color="#d95f02")
    axes[1, 0].set_ylabel("Mean signed v error (mm)")
    axes[1, 0].set_title("Vertical bias (prediction - ground truth)")

    counts = _plot_value(bins, "sample_count")
    bad_rate = 100.0 * _plot_value(bins, "bad_sample_rate")
    axes[1, 1].bar(x, counts, color="#7570b3", alpha=0.75, label="Samples")
    axes[1, 1].set_ylabel("Sample count")
    bad_axis = axes[1, 1].twinx()
    bad_axis.plot(x, bad_rate, color="#e7298a", marker="o", label="Bad rate")
    bad_axis.set_ylabel("Bad-sample rate (%)")
    bad_axis.set_ylim(bottom=0.0)
    axes[1, 1].set_title("Coverage and bad-sample concentration")
    handles_a, labels_a = axes[1, 1].get_legend_handles_labels()
    handles_b, labels_b = bad_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles_a + handles_b, labels_a + labels_b, loc="upper left")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        axis.set_xlabel("Ground-truth v range (mm)")
    fig.suptitle(
        f"Vertical gaze error analysis | split={split} | {checkpoint_name}",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _worst_json_record(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "bad_rank",
        "sample_id",
        "dataset",
        "image_name",
        "frame_idx",
        "gt_v_mm",
        "pred_v_mm",
        "error_v_mm",
        "abs_error_v_mm",
        "abs_error_u_mm",
        "epe_mm",
        "vertical_error_percentile",
        "bad_reason",
        "source_image_path",
        "face_path",
        "left_eye_path",
        "right_eye_path",
    )
    return {key: record.get(key) for key in keys}


def _print_worst(records: Sequence[Mapping[str, Any]], count: int) -> None:
    if count == 0:
        return
    print("\nWorst samples by absolute table-local v error:")
    print("rank | abs_v_mm | abs_u_mm | gt_v_mm | pred_v_mm | sample_id")
    for record in records[:count]:
        def display(key: str) -> str:
            value = record.get(key)
            return "invalid" if value is None else f"{float(value):.3f}"

        print(
            f"{int(record['bad_rank']):4d} | "
            f"{display('abs_error_v_mm'):>8} | "
            f"{display('abs_error_u_mm'):>8} | "
            f"{display('gt_v_mm'):>8} | "
            f"{display('pred_v_mm'):>9} | "
            f"{record['sample_id']}"
        )


def main() -> int:
    args = parse_args()
    _validate_args(args)
    checkpoint_path = resolve_project_path(args.checkpoint).resolve()
    run_dir = checkpoint_path.parent.parent
    output_dir = (
        resolve_project_path(args.output_dir).resolve()
        if args.output_dir is not None
        else run_dir / "analysis" / f"vertical_gaze_error_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = _load_checkpoint(checkpoint_path)
    config, config_source = _load_analysis_config(
        explicit_path=args.config,
        run_dir=run_dir,
        checkpoint=checkpoint,
    )
    model_config = make_model_config(config["model"])
    seed_everything(
        int(config["experiment"]["seed"]),
        bool(config["experiment"]["deterministic"]),
    )
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset_csv, depth_prior, virtual_manifest, allowed_ids = (
        _resolve_data_inputs(config, model_config, output_dir / "resolved_inputs")
    )
    loader, uv_normalizer = _build_analysis_loader(
        config=config,
        model_config=model_config,
        checkpoint=checkpoint,
        dataset_csv=dataset_csv,
        depth_prior=depth_prior,
        virtual_manifest=virtual_manifest,
        allowed_sample_ids=allowed_ids,
        split=args.split,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
    )

    model = ModelV1(model_config).to(device)
    model_state, model_weight_source = checkpoint_model_state(
        checkpoint,
        prefer_ema=True,
    )
    model.load_state_dict(model_state, strict=True)
    model.eval()
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_best_epe = checkpoint.get("best_val_epe_mm")
    # Release optimizer/scheduler/scaler state before image inference.
    del checkpoint

    metadata = _read_metadata(dataset_csv)
    direct_uv_target_frame = canonical_direct_uv_target_frame(
        config["data"].get(
            "direct_uv_target_frame", DIRECT_UV_TARGET_FRAME_TABLE_LOCAL
        )
    )
    print(
        f"Analyzing {len(loader.dataset)} samples from split={args.split!r} "
        f"with {model_weight_source} weights ..."
    )
    records = collect_sample_records(
        model=model,
        loader=loader,
        device=device,
        uv_normalizer=uv_normalizer,
        direct_uv_target_frame=direct_uv_target_frame,
        metadata=metadata,
    )
    bad_threshold = rank_and_mark_bad_samples(records, args.bad_percentile)
    bins, bin_edges = build_binned_metrics(records, args.num_bins, args.binning)
    overall = summarize_records(records)
    worst = records[: min(args.worst_count, len(records))]

    sample_csv = output_dir / "vertical_gaze_samples.csv"
    worst_csv = output_dir / "worst_vertical_gaze_samples.csv"
    bins_csv = output_dir / "vertical_gaze_bins.csv"
    summary_json = output_dir / "vertical_gaze_summary.json"
    bins_png = output_dir / "vertical_gaze_bins.png"
    _write_csv(sample_csv, records, SAMPLE_COLUMNS)
    _write_csv(worst_csv, worst, SAMPLE_COLUMNS)
    _write_csv(bins_csv, bins, BIN_COLUMNS)
    save_binned_plot(
        bins_png,
        bins,
        split=args.split,
        checkpoint_name=checkpoint_path.name,
        dpi=args.dpi,
    )

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_best_val_epe_mm": _rounded(checkpoint_best_epe),
        "model_weight_source": model_weight_source,
        "config_source": config_source,
        "dataset_csv": str(dataset_csv.resolve()),
        "split": args.split,
        "device": str(device),
        "prediction_mode": model_config.prediction_mode,
        "direct_uv_target_frame": direct_uv_target_frame,
        "signed_error_convention": "prediction_minus_ground_truth",
        "bad_sample_definition": {
            "metric": "abs_error_v_mm",
            "percentile": args.bad_percentile,
            "threshold_mm": _rounded(bad_threshold),
            "comparison": "greater_than_or_equal",
            "non_finite_predictions_are_bad": True,
        },
        "overall": overall,
        "binning": {
            "axis": "gt_v_mm",
            "strategy": args.binning,
            "requested_bin_count": args.num_bins,
            "actual_bin_count": len(bins),
            "edges_mm": [_rounded(value) for value in bin_edges],
            "bins": bins,
        },
        "worst_samples": [_worst_json_record(record) for record in worst],
        "artifacts": {
            "all_samples_csv": str(sample_csv.resolve()),
            "worst_samples_csv": str(worst_csv.resolve()),
            "bins_csv": str(bins_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
            "bins_png": str(bins_png.resolve()),
        },
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    _print_worst(records, min(args.print_worst, len(records)))
    print(f"\nBad threshold: |v error| >= {bad_threshold:.3f} mm")
    print(f"All samples CSV: {sample_csv}")
    print(f"Worst samples CSV: {worst_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Binned plot: {bins_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
