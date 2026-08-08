"""Train legacy direct-UV ModelV1 or V4 gaze-plus-geometry models.

The script owns the complete training lifecycle: deterministic data split,
model/optimizer/scheduler setup, AMP, validation, W&B and local logging,
checkpointing, and resume validation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1 import (
    GazeGeometryLoss,
    GazeGeometryLossConfig,
    ModelV1,
    ModelV1Config,
    UVLossConfig,
    UVRegressionLoss,
)
from modelv1.losses import compute_uv_metrics_mm
from modelv1.deca_cache import DecaFeatureCache
from modelv1.processed_artifacts import processed_dataset_artifacts
from modelv1.data import (
    EyeAppearanceAugmentationConfig,
    IMAGE_SOURCE_LEGACY,
    IMAGE_SOURCE_VIRTUAL_CAMERA,
    PairedEyeAppearanceAugmentation,
    build_modelv1_dataloaders,
    get_depth_correction_geometry_normalizer,
    get_eye_geometry_normalizer,
    get_eye_geometry_quality_normalizer,
    get_uv_target_normalizer,
    canonical_image_source,
    merge_virtual_camera_manifests,
)
from modelv1.data.normalization import (
    EyeGeometryNormalizer,
    EyeGeometryQualityNormalizer,
    UVTargetNormalizer,
)
from modelv1.depth_prior.face_preprocess import (
    DEFAULT_DECA_CROP_SCALE,
    FACE_PREPROCESS_CHOICES,
    FACE_PREPROCESS_DECA,
    FACE_PREPROCESS_LEGACY,
)
from modelv1.experiment_diary import append_completed_experiment
from modelv1.geometry_gate import (
    EYE_GEOMETRY_GATE_LEARNED_RESIDUAL,
    EYE_GEOMETRY_GATE_NONE,
    canonical_eye_geometry_gate_mode,
)
from modelv1.deca_cache import (
    DECA_FEATURE_REPRESENTATION_FULL236,
    DECA_FEATURE_REPRESENTATION_NONE,
    canonical_deca_feature_representation,
    deca_feature_representation_dim,
)
from modelv1.model import (
    DECA_BRANCH_MODE_FACTORIZED_GEOMETRY,
    FACE_IMAGE_BACKBONE,
    FACE_IMAGE_PRETRAINED_DATASET,
    GAZE_PREDICTION_FRAME_CAMERA,
    GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA,
    PREDICTION_MODE_GAZE_GEOMETRY,
    canonical_deca_branch_mode,
    canonical_gaze_prediction_frame,
)
from modelv1.depth_distribution import (
    DEPTH_DISTRIBUTION_LEARNED_REWEIGHT,
    DEPTH_DISTRIBUTION_POINT,
)
from modelv1.data.depth_prior import (
    EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
    canonical_eye_geometry_representation,
    eye_geometry_representation_dim,
)
from modelv1.scene import (
    SCENE_REPRESENTATION_FULL25,
    canonical_scene_representation,
    scene_representation_dim,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "modelv1" / "train_random_80_20_100.yaml"
BASE_METRIC_NAMES = ("epe_mm", "median_epe_mm", "mae_u_mm", "mae_v_mm")
V4_GEOMETRY_METRIC_NAMES = (
    "physical_valid_rate",
    "mean_lambda_mm",
    "behind_ray_rate",
    "parallel_ray_rate",
    "gaze_axis_error_deg",
    "directed_gaze_error_deg",
)
V4_LOSS_METRIC_NAMES = (
    "pseudo_gaze_cosine_loss",
    "weighted_pseudo_gaze_loss",
    "uv_gaussian_nll",
    "weighted_uv_gaussian_nll",
)
DEPTH_CORRECTION_METRIC_NAMES = (
    "depth_shared_scale_mean",
    "depth_log_correction_abs_mean",
)
TUPLE_CONFIG_KEYS = (
    "deca_factor_hidden_dims",
    "face_hidden_dims",
    "crop_cam_hidden_dims",
    "scene_hidden_dims",
    "fusion_hidden_dims",
    "depth_reweighter_hidden_dims",
    "depth_correction_hidden_dims",
)
EYE_BACKBONE_TRAIN_FROM_STAGES = (
    "stem",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
)


def metric_names_for_model(
    config: ModelV1Config,
    loss_config: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    if config.prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY:
        names = (*BASE_METRIC_NAMES, *V4_GEOMETRY_METRIC_NAMES)
        if config.use_depth_correction:
            names = (*names, *DEPTH_CORRECTION_METRIC_NAMES)
        if loss_config is not None and (
            float(loss_config.get("gaze_angular_weight", 0.0)) > 0
            or float(loss_config.get("uv_gaussian_nll_weight", 0.0)) > 0
        ):
            names = (*names, *V4_LOSS_METRIC_NAMES)
        return names
    return BASE_METRIC_NAMES


def _split_identity(values: Any, fallback: str) -> str:
    if not isinstance(values, (list, tuple)):
        return fallback
    normalized = [str(value).strip() for value in values if str(value).strip()]
    return "_".join(normalized) if normalized else fallback


def apply_numbered_experiment_identity(config: dict[str, Any]) -> None:
    """Derive a unique run name and W&B metadata from numbered data splits."""

    data_config = config["data"]
    experiment_config = config["experiment"]
    if not data_config.get("dataset_ids") or not bool(
        experiment_config.get("auto_experiment_identity", True)
    ):
        return
    train_id = _split_identity(data_config.get("train_datasets"), "random_train")
    val_id = _split_identity(data_config.get("val_datasets"), "random_val")
    prediction_mode = str(config["model"].get("prediction_mode", "direct_uv"))
    model_stage = "v4" if prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY else prediction_mode
    prior_kind = str(data_config.get("depth_prior_kind", "none")).lower()
    depth_mode = str(config["model"].get("depth_distribution_mode", "point")).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_stem = (
        f"numbered_train{train_id}_val{val_id}_{model_stage}_{prior_kind}_{depth_mode}"
    )
    experiment_config["run_name"] = f"{run_stem}_{timestamp}"
    experiment_config["description"] = (
        f"Auto-generated numbered-artifact split: train={train_id}, val={val_id}, "
        f"prior={prior_kind}, depth_mode={depth_mode}."
    )
    wandb_config = config["logging"]["wandb"]
    wandb_config["project"] = str(
        experiment_config.get("numbered_wandb_project", "ModelV1_numbered_V4")
    )
    wandb_config["tags"] = [
        "ModelV1",
        model_stage,
        "numbered_artifacts",
        prediction_mode,
        prior_kind,
        f"depth_{depth_mode}",
        f"train_dataset_{train_id}",
        f"val_dataset_{val_id}",
        f"datasets_{_split_identity(data_config.get('dataset_ids'), 'unknown')}",
        "imagenet_resnet18",
        "eye_augmentation",
    ]


def metric_fields(metric_names: tuple[str, ...]) -> list[str]:
    return [
        "epoch",
        "global_step",
        "lr",
        "train_loss",
        *(f"train_{name}" for name in metric_names),
        "train_seconds",
        "train_samples_per_second",
        "val_loss",
        *(f"val_{name}" for name in metric_names),
        "val_seconds",
        "val_samples_per_second",
        "epoch_seconds",
        "elapsed_seconds",
        "best_val_epe_mm",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--resume", type=Path, default=None, help="Path to last.pt or best.pt.")
    parser.add_argument("--device", default=None, help="Override training.device.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one epoch with W&B disabled to verify the complete training path.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Top-level training config must be a mapping.")
    data_section = config.get("data")
    if isinstance(data_section, dict):
        data_section["deca_face_preprocess"] = str(
            data_section.get("deca_face_preprocess", FACE_PREPROCESS_DECA)
        ).strip().lower()
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    for key in ("experiment", "data", "model", "loss", "training", "logging"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Config requires a {key!r} mapping.")
    if int(config["training"].get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive.")
    if int(config["data"].get("batch_size", 0)) <= 0:
        raise ValueError("data.batch_size must be positive.")
    if float(config["data"].get("val_ratio", 0.0)) <= 0:
        raise ValueError("data.val_ratio must be positive.")
    if float(config["training"]["optimizer"].get("lr", 0.0)) <= 0:
        raise ValueError("training.optimizer.lr must be positive.")
    experiment_config = config["experiment"]
    if not isinstance(experiment_config.get("description"), str):
        raise ValueError("experiment.description must be a string.")
    if not isinstance(experiment_config.get("diary_enabled"), bool):
        raise ValueError("experiment.diary_enabled must be true or false.")
    if (
        experiment_config["diary_enabled"]
        and not str(experiment_config.get("diary_path", "")).strip()
    ):
        raise ValueError(
            "experiment.diary_path is required when diary_enabled=true."
        )
    wandb_config = config["logging"].get("wandb")
    if not isinstance(wandb_config, dict):
        raise ValueError("logging.wandb must be a mapping.")
    for key in ("enabled", "project", "entity", "mode", "tags"):
        if key not in wandb_config:
            raise ValueError(f"logging.wandb requires {key!r} in the config file.")
    if not isinstance(wandb_config["enabled"], bool):
        raise ValueError("logging.wandb.enabled must be true or false.")
    if not str(wandb_config["project"]).strip():
        raise ValueError("logging.wandb.project must be a non-empty name.")
    if not str(wandb_config["mode"]).strip():
        raise ValueError("logging.wandb.mode must be non-empty.")
    if not isinstance(wandb_config["tags"], list):
        raise ValueError("logging.wandb.tags must be a list.")
    face_preprocess = str(
        config["data"].get("deca_face_preprocess", FACE_PREPROCESS_DECA)
    ).strip().lower()
    if face_preprocess not in FACE_PREPROCESS_CHOICES:
        raise ValueError(
            "data.deca_face_preprocess must be one of "
            f"{FACE_PREPROCESS_CHOICES}, got {face_preprocess!r}."
        )
    if float(config["data"].get("deca_crop_scale", DEFAULT_DECA_CROP_SCALE)) <= 0:
        raise ValueError("data.deca_crop_scale must be positive.")
    use_eye_geometry = config["model"].get("use_eye_geometry", False)
    if not isinstance(use_eye_geometry, bool):
        raise ValueError("model.use_eye_geometry must be true or false.")
    use_crop_cam = config["model"].get("use_crop_cam", True)
    if not isinstance(use_crop_cam, bool):
        raise ValueError("model.use_crop_cam must be true or false.")
    use_face_image = config["model"].get("use_face_image", False)
    if not isinstance(use_face_image, bool):
        raise ValueError("model.use_face_image must be true or false.")
    load_face_image = config["data"].get("load_face_image", False)
    if not isinstance(load_face_image, bool):
        raise ValueError("data.load_face_image must be true or false.")
    if use_face_image and not load_face_image:
        raise ValueError(
            "data.load_face_image must be true when model.use_face_image=true."
        )
    deca_feature_representation = canonical_deca_feature_representation(
        config["model"].get(
            "deca_feature_representation",
            DECA_FEATURE_REPRESENTATION_FULL236,
        )
    )
    expected_deca_feature_dim = deca_feature_representation_dim(
        deca_feature_representation
    )
    if int(config["model"].get("deca_feature_dim", 236)) != expected_deca_feature_dim:
        raise ValueError(
            f"model.deca_feature_representation={deca_feature_representation!r} "
            f"requires model.deca_feature_dim={expected_deca_feature_dim}."
        )
    deca_branch_mode = canonical_deca_branch_mode(
        config["model"].get("deca_branch_mode", "flat")
    )
    if (
        deca_branch_mode == DECA_BRANCH_MODE_FACTORIZED_GEOMETRY
        and deca_feature_representation != "geometry156"
    ):
        raise ValueError(
            "model.deca_branch_mode='factorized_geometry' requires "
            "model.deca_feature_representation='geometry156'."
        )
    scene_representation = canonical_scene_representation(
        config["model"].get(
            "scene_representation",
            SCENE_REPRESENTATION_FULL25,
        )
    )
    expected_scene_dim = scene_representation_dim(scene_representation)
    if int(config["model"].get("scene_dim", 25)) != expected_scene_dim:
        raise ValueError(
            f"model.scene_representation={scene_representation!r} requires "
            f"model.scene_dim={expected_scene_dim}."
        )
    eye_geometry_representation = canonical_eye_geometry_representation(
        config["model"].get(
            "eye_geometry_representation",
            EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
        )
    )
    expected_eye_geometry_dim = eye_geometry_representation_dim(
        eye_geometry_representation
    )
    if int(config["model"].get("eye_geometry_dim", 6)) != expected_eye_geometry_dim:
        raise ValueError(
            f"model.eye_geometry_representation={eye_geometry_representation!r} "
            f"requires model.eye_geometry_dim={expected_eye_geometry_dim}."
        )
    gate_mode = canonical_eye_geometry_gate_mode(
        config["model"].get("eye_geometry_gate_mode", EYE_GEOMETRY_GATE_NONE)
    )
    if not use_eye_geometry and gate_mode != EYE_GEOMETRY_GATE_NONE:
        raise ValueError(
            "model.eye_geometry_gate_mode must be 'none' when "
            "model.use_eye_geometry=false."
        )
    if int(config["model"].get("eye_geometry_quality_dim", 4)) != 4:
        raise ValueError("model.eye_geometry_quality_dim must be 4 for V2.2.")
    has_numbered_dataset_ids = bool(config["data"].get("dataset_ids"))
    if (
        use_eye_geometry
        and not has_numbered_dataset_ids
        and not config["data"].get("depth_prior_csv_path")
    ):
        raise ValueError(
            "data.depth_prior_csv_path is required when model.use_eye_geometry=true."
        )
    gate_regularization_weight = float(
        config["loss"].get("gate_regularization_weight", 0.0)
    )
    if gate_regularization_weight < 0:
        raise ValueError("loss.gate_regularization_weight must be non-negative.")
    prediction_mode = str(
        config["model"].get("prediction_mode", "direct_uv")
    ).strip().lower()
    image_source = canonical_image_source(
        config["data"].get("image_source", IMAGE_SOURCE_LEGACY)
    )
    gaze_prediction_frame = canonical_gaze_prediction_frame(
        config["model"].get(
            "gaze_prediction_frame",
            GAZE_PREDICTION_FRAME_CAMERA,
        )
    )
    virtual_manifest_path = config["data"].get("virtual_camera_manifest_path")
    virtual_manifest_paths = config["data"].get("virtual_camera_manifest_paths")
    if virtual_manifest_path and virtual_manifest_paths:
        raise ValueError(
            "Use either data.virtual_camera_manifest_path or "
            "data.virtual_camera_manifest_paths, not both."
        )
    if virtual_manifest_paths is not None and (
        not isinstance(virtual_manifest_paths, (list, tuple))
        or not virtual_manifest_paths
        or not all(str(path).strip() for path in virtual_manifest_paths)
    ):
        raise ValueError(
            "data.virtual_camera_manifest_paths must be a non-empty YAML list."
        )
    if (
        image_source == IMAGE_SOURCE_VIRTUAL_CAMERA
        and not virtual_manifest_path
        and not virtual_manifest_paths
    ):
        raise ValueError(
            "data.virtual_camera_manifest_path or "
            "data.virtual_camera_manifest_paths is required when "
            "data.image_source='virtual_camera'."
        )
    if image_source == IMAGE_SOURCE_LEGACY and (
        virtual_manifest_path or virtual_manifest_paths
    ):
        raise ValueError(
            "Virtual-camera manifest options require "
            "data.image_source='virtual_camera'."
        )
    if prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY:
        if (
            image_source == IMAGE_SOURCE_VIRTUAL_CAMERA
            and gaze_prediction_frame
            != GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA
        ):
            raise ValueError(
                "Virtual-camera images in gaze_geometry mode require "
                "model.gaze_prediction_frame='virtual_camera'."
            )
        if (
            gaze_prediction_frame == GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA
            and image_source != IMAGE_SOURCE_VIRTUAL_CAMERA
        ):
            raise ValueError(
                "model.gaze_prediction_frame='virtual_camera' requires "
                "data.image_source='virtual_camera'."
            )
    if (
        prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY
        and not has_numbered_dataset_ids
        and not config["data"].get("depth_prior_csv_path")
    ):
        raise ValueError(
            "data.depth_prior_csv_path is required for gaze_geometry mode."
        )
    trainable_components = str(
        config["training"].get("trainable_components", "all")
    ).strip().lower()
    if trainable_components not in {
        "all",
        "gaze_model",
        "depth_reweighter",
        "depth_correction",
    }:
        raise ValueError(
            "training.trainable_components must be all, gaze_model, "
            "depth_reweighter, or depth_correction."
        )
    use_depth_correction = bool(
        config["model"].get("use_depth_correction", False)
    )
    if trainable_components == "depth_correction" and not use_depth_correction:
        raise ValueError(
            "training.trainable_components='depth_correction' requires "
            "model.use_depth_correction=true."
        )
    depth_correction_prior_weight = float(
        config["loss"].get("depth_correction_prior_weight", 0.0)
    )
    if depth_correction_prior_weight < 0:
        raise ValueError(
            "loss.depth_correction_prior_weight must be non-negative."
        )
    if depth_correction_prior_weight > 0 and not use_depth_correction:
        raise ValueError(
            "loss.depth_correction_prior_weight requires "
            "model.use_depth_correction=true."
        )
    make_eye_appearance_augmentation(config["data"])
    validate_eye_backbone_schedule(config["training"], config["model"])


def make_eye_appearance_augmentation(
    data_config: Mapping[str, Any],
) -> PairedEyeAppearanceAugmentation | None:
    section = data_config.get("eye_augmentation")
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValueError("data.eye_augmentation must be a mapping.")
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("data.eye_augmentation.enabled must be true or false.")
    if not enabled:
        return None
    required = (
        "photometric_probability",
        "brightness_min",
        "brightness_max",
        "contrast_min",
        "contrast_max",
        "gamma_min",
        "gamma_max",
        "blur_probability",
        "blur_kernel_size",
        "blur_sigma_min",
        "blur_sigma_max",
        "noise_probability",
        "noise_std_max",
        "occlusion_probability",
        "occlusion_area_min",
        "occlusion_area_max",
        "occlusion_aspect_min",
        "occlusion_aspect_max",
    )
    missing = [key for key in required if key not in section]
    if missing:
        raise ValueError(
            "Enabled data.eye_augmentation is missing keys: "
            f"{missing}."
        )
    values = {key: section[key] for key in required}
    values["blur_kernel_size"] = int(values["blur_kernel_size"])
    augmentation_config = EyeAppearanceAugmentationConfig(**values)
    return PairedEyeAppearanceAugmentation(augmentation_config)


def validate_eye_backbone_schedule(
    training_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> None:
    section = training_config.get("eye_backbone_schedule")
    if section is None:
        return
    if not isinstance(section, Mapping):
        raise ValueError("training.eye_backbone_schedule must be a mapping.")
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            "training.eye_backbone_schedule.enabled must be true or false."
        )
    if not enabled:
        return
    required = ("frozen_epochs", "unfreeze_from", "lr_multiplier")
    missing = [key for key in required if key not in section]
    if missing:
        raise ValueError(
            "Enabled training.eye_backbone_schedule is missing keys: "
            f"{missing}."
        )
    frozen_epochs = int(section["frozen_epochs"])
    if frozen_epochs < 0:
        raise ValueError("eye_backbone_schedule.frozen_epochs must be non-negative.")
    unfreeze_from = str(section["unfreeze_from"]).strip().lower()
    if unfreeze_from not in EYE_BACKBONE_TRAIN_FROM_STAGES:
        raise ValueError(
            "eye_backbone_schedule.unfreeze_from must be one of "
            f"{EYE_BACKBONE_TRAIN_FROM_STAGES}, got {unfreeze_from!r}."
        )
    lr_multiplier = float(section["lr_multiplier"])
    if not 0 < lr_multiplier <= 1:
        raise ValueError(
            "eye_backbone_schedule.lr_multiplier must lie in (0, 1]."
        )
    if str(training_config.get("trainable_components", "all")).lower() in {
        "depth_reweighter",
        "depth_correction",
    }:
        raise ValueError(
            "Eye-backbone scheduling must be disabled while training only the "
            "depth reweighter or depth correction head."
        )
    backbone = str(model_config.get("eye_backbone", "resnet18")).lower()
    if not backbone.startswith("resnet"):
        raise ValueError(
            "Eye-backbone scheduling requires a ResNet eye backbone, got "
            f"{backbone!r}."
        )


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv_records(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def _merge_csv_files(paths: list[Path], output: Path, *, label: str) -> Path:
    """Merge records and reject duplicate sample IDs early.

    Dataset CSVs must retain an identical schema.  Numbered depth-prior CSVs
    may legitimately span preprocessing revisions, for example after adding
    per-sample depth uncertainty.  Those are merged using the ordered union of
    columns; absent fields are written as empty and are later treated as
    unavailable by :class:`DepthPriorTable`.
    """

    expected_fields: list[str] | None = None
    seen_ids: set[str] = set()
    merged: list[dict[str, str]] = []
    for path in paths:
        rows, fields = _read_csv_records(path)
        if "sample_id" not in fields:
            raise ValueError(f"{label} CSV has no sample_id column: {path}")
        if expected_fields is None:
            expected_fields = fields
        elif fields != expected_fields:
            if label != "depth-prior":
                raise ValueError(
                    f"Cannot merge {label} CSV files with different columns: "
                    f"{paths[0]} and {path}"
                )
            expected_fields.extend(
                field for field in fields if field not in expected_fields
            )
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate sample_id={sample_id!r} while merging {label} files."
                )
            seen_ids.add(sample_id)
            merged.append(row)
    assert expected_fields is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=expected_fields)
        writer.writeheader()
        writer.writerows(merged)
    return output


def _merge_deca_caches(paths: list[Path], output: Path) -> Path:
    """Create a run-local, auditable union of independently cached datasets."""

    caches = [DecaFeatureCache.load(path) for path in paths]
    base_metadata = dict(caches[0].metadata)
    for cache in caches[1:]:
        for key in ("face_preprocess", "deca_crop_scale"):
            if cache.metadata.get(key) != base_metadata.get(key):
                raise ValueError(
                    f"Cannot merge DECA caches with different {key}: "
                    f"{paths[0]} and {cache.path}"
                )
    sample_ids = [sample_id for cache in caches for sample_id in cache.sample_ids]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id values while merging DECA caches.")
    metadata = dict(base_metadata)
    metadata["merged_from"] = [str(path) for path in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        deca_feat=np.concatenate([cache.features for cache in caches], axis=0),
        sample_id=np.asarray(sample_ids),
        image_sha256=np.asarray(
            [digest for cache in caches for digest in cache.image_sha256]
        ),
        metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )
    return output


def resolve_numbered_dataset_artifacts(
    data_config: Mapping[str, Any],
    run_dir: Path,
    *,
    require_deca_features: bool,
    require_depth_prior: bool,
) -> tuple[Path, Path | None, Path | None]:
    """Resolve ``data.dataset_ids`` into one or several numbered artifacts.

    A single ID is used directly.  Multiple IDs are merged under the run
    directory, so the exact data union is retained with the experiment instead
    of creating or mutating a global combined CSV/cache.
    """

    raw_ids = data_config.get("dataset_ids")
    if raw_ids is None:
        dataset_csv = resolve_project_path(data_config["csv_path"])
        deca_cache = (
            resolve_project_path(data_config["deca_cache_path"])
            if require_deca_features
            else None
        )
        depth_prior = (
            resolve_project_path(data_config["depth_prior_csv_path"])
            if require_depth_prior
            else None
        )
        return dataset_csv, deca_cache, depth_prior
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise ValueError("data.dataset_ids must be a non-empty YAML list when provided.")
    processed_dir = resolve_project_path(
        data_config.get("processed_data_dir", "data/processed")
    )
    artifacts = [processed_dataset_artifacts(value, processed_dir) for value in raw_ids]
    # Iris/PnP geometry training must not load rows whose depth reconstruction
    # failed.  The end-to-end preparation workflow writes this filtered CSV.
    use_trainable_depth_subset = bool(
        data_config.get("use_trainable_depth_subset", require_depth_prior)
    )
    dataset_paths = [
        item.common_depth_prior_dataset_csv
        if use_trainable_depth_subset
        else item.dataset_csv
        for item in artifacts
    ]
    deca_paths = [item.deca_cache for item in artifacts]
    prior_kind = str(data_config.get("depth_prior_kind", "iris_ipd_65mm")).strip().lower()
    if prior_kind not in {"iris_ipd_65mm", "pnp1010"}:
        raise ValueError(
            "data.depth_prior_kind must be 'iris_ipd_65mm' or 'pnp1010'."
        )
    prior_paths = [
        item.iris_ipd_depth_prior if prior_kind == "iris_ipd_65mm" else item.pnp_depth_prior
        for item in artifacts
    ]
    required_paths = [
        *dataset_paths,
        *(deca_paths if require_deca_features else []),
        *(prior_paths if require_depth_prior else []),
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        preview = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing numbered processed artifact(s):\n  {preview}")
    if len(artifacts) == 1:
        return (
            dataset_paths[0],
            deca_paths[0] if require_deca_features else None,
            prior_paths[0] if require_depth_prior else None,
        )

    merged_dir = run_dir / "resolved_numbered_data"
    dataset_csv = _merge_csv_files(dataset_paths, merged_dir / "dataset.csv", label="dataset")
    deca_cache = (
        _merge_deca_caches(deca_paths, merged_dir / "deca_features.npz")
        if require_deca_features
        else None
    )
    depth_prior = (
        _merge_csv_files(prior_paths, merged_dir / "depth_prior.csv", label="depth-prior")
        if require_depth_prior
        else None
    )
    (merged_dir / "sources.json").write_text(
        json.dumps(
            {
                "dataset_ids": [item.dataset_id for item in artifacts],
                "use_trainable_depth_subset": use_trainable_depth_subset,
                "dataset_csvs": [str(path) for path in dataset_paths],
                "deca_caches": (
                    [str(path) for path in deca_paths]
                    if require_deca_features
                    else []
                ),
                "depth_prior_kind": prior_kind if require_depth_prior else None,
                "depth_prior_csvs": [str(path) for path in prior_paths]
                if require_depth_prior
                else [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_csv, deca_cache, depth_prior


def resolve_virtual_camera_manifest_path(
    data_config: Mapping[str, Any],
    run_dir: Path,
) -> Path | None:
    """Resolve one manifest or merge configured manifests inside ``run_dir``."""

    singular = data_config.get("virtual_camera_manifest_path")
    plural = data_config.get("virtual_camera_manifest_paths")
    if singular and plural:
        raise ValueError(
            "Use either virtual_camera_manifest_path or "
            "virtual_camera_manifest_paths, not both."
        )
    if singular is not None:
        return resolve_project_path(singular)
    if plural is None:
        return None
    if not isinstance(plural, (list, tuple)) or not plural:
        raise ValueError("virtual_camera_manifest_paths must be a non-empty list.")
    source_paths = [resolve_project_path(path) for path in plural]
    if len(source_paths) == 1:
        return source_paths[0]
    merged_path = run_dir / "resolved_virtual_camera" / "manifest.csv"
    return merge_virtual_camera_manifests(source_paths, merged_path)


def validate_deca_cache_preprocess(
    cache_path: Path,
    data_config: Mapping[str, Any],
) -> None:
    """Reject a feature cache rendered with different DECA image preprocessing."""

    expected_mode = str(
        data_config.get("deca_face_preprocess", FACE_PREPROCESS_DECA)
    ).strip().lower()
    expected_scale = float(
        data_config.get("deca_crop_scale", DEFAULT_DECA_CROP_SCALE)
    )
    cache = DecaFeatureCache.load(cache_path)
    # Caches produced before this field existed were necessarily legacy crops.
    cache_mode = str(
        cache.metadata.get("face_preprocess", FACE_PREPROCESS_LEGACY)
    ).strip().lower()
    if cache_mode != expected_mode:
        raise ValueError(
            "DECA cache preprocessing mismatch: "
            f"config requests {expected_mode!r}, but {cache_path} was built with "
            f"{cache_mode!r}. Rebuild the cache with scripts/cache_deca_features.py."
        )
    if expected_mode == FACE_PREPROCESS_DECA:
        cache_scale = cache.metadata.get("deca_crop_scale")
        if cache_scale is None or abs(float(cache_scale) - expected_scale) > 1e-8:
            raise ValueError(
                "DECA cache crop-scale mismatch: "
                f"config requests {expected_scale}, but {cache_path} stores "
                f"{cache_scale!r}. Rebuild the cache with the same scale."
            )


def make_run_dir(config: dict[str, Any], resume: Path | None) -> Path:
    if resume is not None:
        checkpoint = resolve_project_path(resume)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("Resume checkpoint must live in a run's checkpoints directory.")
        run_dir = checkpoint.parent.parent
        config["experiment"]["run_name"] = run_dir.name
        return run_dir

    project = str(config["logging"]["wandb"]["project"])
    run_name = config["experiment"].get("run_name")
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["experiment"]["run_name"] = str(run_name)
    output_dir = resolve_project_path(config["experiment"]["output_dir"])
    run_dir = output_dir / project / str(run_name)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    return run_dir


def setup_file_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("modelv1.train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def save_config(config: Mapping[str, Any], run_dir: Path) -> None:
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, allow_unicode=False, sort_keys=False)


class MetricsCsvWriter:
    def __init__(self, path: Path, metric_names: tuple[str, ...]) -> None:
        self.fields = metric_fields(metric_names)
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open("r", encoding="utf-8", newline="") as existing_handle:
                existing_fields = next(csv.reader(existing_handle), [])
            if existing_fields != self.fields:
                raise ValueError(
                    f"Existing metrics file uses a different schema: {path}. "
                    "Start a new run instead of resuming this run."
                )
        self.handle = path.open("a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields)
        if not exists:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, metrics: Mapping[str, float | int]) -> None:
        self.writer.writerow({key: metrics[key] for key in self.fields})
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class NoOpTracker:
    def log(self, metrics: Mapping[str, float | int], step: int) -> None:
        del metrics, step

    def finish(self) -> None:
        pass


def start_tracker(
    config: Mapping[str, Any],
    run_dir: Path,
    metric_names: tuple[str, ...],
    static_metadata: Mapping[str, Any] | None = None,
) -> Any:
    wandb_config = config["logging"]["wandb"]
    if not bool(wandb_config["enabled"]):
        return NoOpTracker()
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "W&B logging is enabled, but wandb is not installed. Run pip install -r requirements.txt."
        ) from exc

    run_config = to_jsonable(config)
    if static_metadata is not None:
        run_config["model_statistics"] = to_jsonable(static_metadata)

    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        mode=wandb_config["mode"],
        name=config["experiment"]["run_name"],
        tags=wandb_config["tags"],
        dir=str(run_dir),
        config=run_config,
    )
    run.define_metric("epoch")
    for split in ("train", "val"):
        run.define_metric(f"{split}/loss", step_metric="epoch", summary="min")
        for name in metric_names:
            summary = (
                "max"
                if name in {"physical_valid_rate", "mean_lambda_mm"}
                else "min"
            )
            run.define_metric(
                f"{split}/{name}",
                step_metric="epoch",
                summary=summary,
            )
    run.define_metric("checkpoint/best_val_epe_mm", step_metric="epoch", summary="min")
    return run


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is unavailable.")
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def make_model_config(section: Mapping[str, Any]) -> ModelV1Config:
    values = dict(section)
    for key in TUPLE_CONFIG_KEYS:
        if key in values:
            values[key] = tuple(values[key])
    return ModelV1Config(**values)


def configure_trainable_components(model: ModelV1, mode: str) -> None:
    """Apply the explicit V4 stage policy before optimizer construction."""

    mode = str(mode).strip().lower()
    model._trainable_components = mode
    if mode == "all":
        return
    if mode == "gaze_model":
        if model.depth_reweighter is not None:
            for parameter in model.depth_reweighter.parameters():
                parameter.requires_grad_(False)
        if model.depth_correction_head is not None:
            for parameter in model.depth_correction_head.parameters():
                parameter.requires_grad_(False)
        return
    if mode == "depth_reweighter":
        if model.depth_reweighter is None:
            raise ValueError(
                "depth_reweighter stage requires depth_distribution_mode="
                "'learned_reweight'."
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.depth_reweighter.parameters():
            parameter.requires_grad_(True)
        return
    if mode == "depth_correction":
        if model.depth_correction_head is None:
            raise ValueError(
                "depth_correction stage requires model.use_depth_correction=true."
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.depth_correction_head.parameters():
            parameter.requires_grad_(True)
        return
    raise ValueError(f"Unknown trainable_components={mode!r}.")


def enabled_eye_backbone_schedule(
    training_config: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    section = training_config.get("eye_backbone_schedule")
    if isinstance(section, Mapping) and bool(section.get("enabled", False)):
        return section
    return None


def apply_eye_backbone_schedule(
    model: ModelV1,
    training_config: Mapping[str, Any],
    epoch: int,
) -> str | None:
    """Apply the configured freeze boundary and describe a state transition."""

    section = enabled_eye_backbone_schedule(training_config)
    if section is None:
        return None
    frozen_epochs = int(section["frozen_epochs"])
    target = (
        None
        if epoch <= frozen_epochs
        else str(section["unfreeze_from"]).strip().lower()
    )
    encoders = model.eye_resnet_encoders()
    previous = tuple(encoder.backbone_train_from for encoder in encoders)
    if previous and all(value == target for value in previous):
        return None
    model.set_eye_backbone_train_from(target)
    state = "fully frozen" if target is None else f"trainable from {target}"
    return f"epoch={epoch} eye ResNet backbone is now {state}"


def optimizer_parameter_groups(
    model: ModelV1,
    training_config: Mapping[str, Any],
) -> list[dict[str, object]] | list[nn.Parameter]:
    """Build stable parameter groups, including later-unfrozen backbone tensors."""

    section = enabled_eye_backbone_schedule(training_config)
    if section is None:
        return [parameter for parameter in model.parameters() if parameter.requires_grad]
    backbone_parameters = model.eye_backbone_parameters()
    if not backbone_parameters:
        raise ValueError("Enabled eye-backbone schedule found no ResNet parameters.")
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    non_backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not non_backbone_parameters:
        raise ValueError("Eye-backbone schedule found no trainable non-backbone parameters.")
    base_lr = float(training_config["optimizer"]["lr"])
    return [
        {
            "params": non_backbone_parameters,
            "lr": base_lr,
            "group_name": "gaze_and_fusion",
        },
        {
            "params": list(backbone_parameters),
            "lr": base_lr * float(section["lr_multiplier"]),
            "group_name": "eye_resnet_backbone",
        },
    ]


def build_training_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_config = training_config["scheduler"]
    if scheduler_config["name"].lower() != "cosine":
        raise ValueError("Only cosine scheduler is supported by the V1 training config.")
    epochs = int(training_config["epochs"])
    eta_min = float(scheduler_config["eta_min"])
    if enabled_eye_backbone_schedule(training_config) is None:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )

    # A shared multiplicative cosine preserves the confirmed 0.1 backbone LR
    # ratio all the way to eta_min; CosineAnnealingLR accepts only one absolute
    # eta_min and would otherwise destroy that ratio near the end of training.
    base_lr = float(training_config["optimizer"]["lr"])
    eta_ratio = eta_min / base_lr

    def cosine_multiplier(step: int) -> float:
        progress = min(max(step, 0), epochs) / epochs
        return eta_ratio + (1.0 - eta_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[cosine_multiplier for _ in optimizer.param_groups],
    )


def initialize_compatible_model_weights(
    model: ModelV1,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[int, int]:
    """Load shape-compatible weights for point -> prior -> reweight stages."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError(
            f"Initialization checkpoint has no model state: {checkpoint_path}"
        )
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current
        and torch.is_tensor(value)
        and current[key].shape == value.shape
    }
    if not compatible:
        raise ValueError(
            f"No shape-compatible weights found in {checkpoint_path}."
        )
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(current) - len(compatible)


def move_batch_to_device(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def run_epoch(
    *,
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    grad_clip_norm: float | None,
    gate_regularization_weight: float,
) -> tuple[dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    if (
        is_train
        and getattr(model, "_trainable_components", "all")
        in {"depth_reweighter", "depth_correction"}
    ):
        # Frozen gaze features must be deterministic: keep their BatchNorm
        # running statistics and dropout state fixed while training the scorer.
        model.eval()
        component = getattr(model, "_trainable_components")
        trainable_module = (
            getattr(model, "depth_reweighter", None)
            if component == "depth_reweighter"
            else getattr(model, "depth_correction_head", None)
        )
        if trainable_module is None:
            raise RuntimeError(
                f"Frozen-gaze stage has no module for {component!r}."
            )
        trainable_module.train()
    start = time.perf_counter()
    total_loss = 0.0
    sample_count = 0
    predictions: list[Tensor] = []
    targets_mm: list[Tensor] = []
    geometry_metric_sums = {
        name: 0.0 for name in V4_GEOMETRY_METRIC_NAMES
    }
    if getattr(getattr(model, "config", None), "use_depth_correction", False):
        geometry_metric_sums.update(
            {name: 0.0 for name in DEPTH_CORRECTION_METRIC_NAMES}
        )
    v4_loss_metric_sums = {
        name: 0.0 for name in V4_LOSS_METRIC_NAMES
    }
    optimizer_steps = 0
    optimizer_skipped_steps = 0

    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        batch_size = int(device_batch["uv_gt"].shape[0])
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            with autocast_context(amp_enabled):
                use_gate_regularization = (
                    is_train and gate_regularization_weight > 0
                )
                model_output = model(
                    device_batch,
                    return_features=use_gate_regularization,
                )
                if isinstance(criterion, GazeGeometryLoss):
                    if not isinstance(model_output, Mapping):
                        raise TypeError("V4 model must return a geometry output mapping.")
                    loss_outputs = criterion(model_output, device_batch)
                    loss = loss_outputs["loss"]
                    uv_pred_mm = model_output["uv_mean_mm"]
                    pseudo_gaze_cosine_loss = loss_outputs["gaze_angular"]
                    v4_loss_metric_sums["pseudo_gaze_cosine_loss"] += (
                        float(pseudo_gaze_cosine_loss) * batch_size
                    )
                    v4_loss_metric_sums["weighted_pseudo_gaze_loss"] += (
                        float(pseudo_gaze_cosine_loss)
                        * criterion.config.gaze_angular_weight
                        * batch_size
                    )
                    uv_gaussian_nll = loss_outputs["uv_gaussian_nll"]
                    v4_loss_metric_sums["uv_gaussian_nll"] += (
                        float(uv_gaussian_nll) * batch_size
                    )
                    v4_loss_metric_sums["weighted_uv_gaussian_nll"] += (
                        float(uv_gaussian_nll)
                        * criterion.config.uv_gaussian_nll_weight
                        * batch_size
                    )
                    with torch.no_grad():
                        ray_valid = model_output["ray_valid_mask"].float()
                        lambdas = model_output["lambda_hypotheses_mm"].float()
                        normal_dot = model_output["normal_dot_gaze"].abs().float()
                        raw_eyes = device_batch["raw_eye_geometry_mm"]
                        gaze_target = device_batch["gaze_target_camera_mm"]
                        if not torch.is_tensor(raw_eyes) or not torch.is_tensor(
                            gaze_target
                        ):
                            raise TypeError(
                                "V4 physical metrics require tensor eye and gaze target data."
                            )
                        eye_midpoint = 0.5 * (
                            raw_eyes[:, :3] + raw_eyes[:, 3:]
                        )
                        gaze_gt = torch.nn.functional.normalize(
                            gaze_target - eye_midpoint,
                            dim=-1,
                            eps=1e-8,
                        )
                        gaze_pred = torch.nn.functional.normalize(
                            model_output["gaze_direction_c"],
                            dim=-1,
                            eps=1e-8,
                        )
                        direction_cosine = (
                            gaze_pred * gaze_gt
                        ).sum(dim=-1).clamp(-1.0, 1.0)
                        directed_error = torch.rad2deg(
                            torch.acos(direction_cosine)
                        )
                        axis_error = torch.rad2deg(
                            torch.acos(direction_cosine.abs())
                        )
                        geometry_metric_sums["physical_valid_rate"] += float(
                            ray_valid.mean(dim=-1).sum()
                        )
                        geometry_metric_sums["mean_lambda_mm"] += float(
                            lambdas.mean(dim=-1).sum()
                        )
                        geometry_metric_sums["behind_ray_rate"] += float(
                            (
                                lambdas
                                <= model.config.geometry_min_lambda_mm
                            ).float().mean(dim=-1).sum()
                        )
                        geometry_metric_sums["parallel_ray_rate"] += float(
                            (
                                normal_dot.squeeze(-1)
                                < model.config.geometry_min_abs_normal_dot_gaze
                            ).float().mean(dim=-1).sum()
                        )
                        geometry_metric_sums["gaze_axis_error_deg"] += float(
                            axis_error.sum()
                        )
                        geometry_metric_sums[
                            "directed_gaze_error_deg"
                        ] += float(directed_error.sum())
                        if model.config.use_depth_correction:
                            shared_scale = model_output["depth_shared_scale"].float()
                            log_correction = model_output[
                                "depth_log_scale_correction"
                            ].float()
                            geometry_metric_sums[
                                "depth_shared_scale_mean"
                            ] += float(shared_scale.sum())
                            geometry_metric_sums[
                                "depth_log_correction_abs_mean"
                            ] += float(log_correction.abs().sum())
                else:
                    if isinstance(model_output, dict):
                        uv_pred = model_output["uv"]
                    else:
                        uv_pred = model_output
                    if not isinstance(criterion, UVRegressionLoss):
                        raise TypeError("Legacy training requires UVRegressionLoss.")
                    loss = criterion(uv_pred, device_batch["uv_target"])
                    uv_pred_mm = criterion.normalizer.denormalize(uv_pred)
                if use_gate_regularization:
                    gate_delta = model_output.get("eye_geometry_gate_delta")
                    valid_mask = device_batch.get("eye_geometry_valid_mask")
                    if gate_delta is None or not torch.is_tensor(valid_mask):
                        raise RuntimeError(
                            "Gate regularization requires learned gate delta "
                            "and eye_geometry_valid_mask."
                        )
                    valid_mask = valid_mask.to(
                        device=gate_delta.device,
                        dtype=gate_delta.dtype,
                    )
                    gate_regularization = (
                        gate_delta.square() * valid_mask
                    ).sum() / valid_mask.sum().clamp_min(1.0)
                    loss = (
                        loss
                        + gate_regularization_weight * gate_regularization
                    )
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scale_before_step = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                if amp_enabled and float(scaler.get_scale()) < scale_before_step:
                    optimizer_skipped_steps += 1
                else:
                    optimizer_steps += 1

        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
        predictions.append(uv_pred_mm.detach().cpu())
        targets_mm.append(device_batch["uv_gt"].detach().cpu())

    if sample_count == 0:
        raise RuntimeError("DataLoader yielded no samples.")
    metric_tensors = compute_uv_metrics_mm(
        torch.cat(predictions),
        torch.cat(targets_mm),
    )
    elapsed = time.perf_counter() - start
    metrics = {"loss": total_loss / sample_count}
    if is_train:
        metrics["optimizer_steps"] = float(optimizer_steps)
        metrics["optimizer_skipped_steps"] = float(optimizer_skipped_steps)
    metrics.update({name: float(value) for name, value in metric_tensors.items()})
    if isinstance(criterion, GazeGeometryLoss):
        metrics.update(
            {
                name: total / sample_count
                for name, total in geometry_metric_sums.items()
            }
        )
        metrics.update(
            {
                name: total / sample_count
                for name, total in v4_loss_metric_sums.items()
            }
        )
    metrics["seconds"] = elapsed
    metrics["samples_per_second"] = sample_count / elapsed if elapsed > 0 else 0.0
    return metrics, sample_count


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch_state = state["torch"]
    if not torch.is_tensor(torch_state):
        raise TypeError("Checkpoint torch RNG state must be a tensor.")
    torch.set_rng_state(torch_state.cpu())
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = state["cuda"]
        if not isinstance(cuda_states, (list, tuple)) or not all(
            torch.is_tensor(value) for value in cuda_states
        ):
            raise TypeError("Checkpoint CUDA RNG state must be a tensor sequence.")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    best_val_epe_mm: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    normalizer: UVTargetNormalizer | None,
    eye_geometry_normalizer: EyeGeometryNormalizer | None,
    eye_geometry_quality_normalizer: EyeGeometryQualityNormalizer | None,
    depth_correction_geometry_normalizer: EyeGeometryNormalizer | None,
    config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_epe_mm": best_val_epe_mm,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "normalizer": normalizer.state_dict() if normalizer is not None else None,
        "eye_geometry_normalizer": (
            eye_geometry_normalizer.state_dict()
            if eye_geometry_normalizer is not None
            else None
        ),
        "eye_geometry_quality_normalizer": (
            eye_geometry_quality_normalizer.state_dict()
            if eye_geometry_quality_normalizer is not None
            else None
        ),
        "depth_correction_geometry_normalizer": (
            depth_correction_geometry_normalizer.state_dict()
            if depth_correction_geometry_normalizer is not None
            else None
        ),
        "config": to_jsonable(config),
        "rng_state": capture_rng_state(),
    }
    try:
        torch.save(payload, temporary_path)
        if temporary_path.stat().st_size <= 0:
            raise RuntimeError("temporary checkpoint is empty")
        os.replace(temporary_path, path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        try:
            free_mib = shutil.disk_usage(path.parent).free / (1024**2)
            free_text = f"{free_mib:.1f} MiB free"
        except OSError:
            free_text = "free space unavailable"
        raise RuntimeError(
            f"Failed to save checkpoint atomically to {path} ({free_text}). "
            "Any previously completed checkpoint at that path was preserved."
        ) from exc


def update_best_checkpoint_from_last(last_path: Path, best_path: Path) -> None:
    """Atomically point ``best.pt`` at the just-written ``last.pt``.

    NTFS hard links avoid storing the identical checkpoint twice when the
    current epoch is both latest and best. A copy fallback keeps the training
    script portable to filesystems without hard-link support.
    """

    if not last_path.is_file():
        raise FileNotFoundError(f"Last checkpoint does not exist: {last_path}")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = best_path.with_name(
        f".{best_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        try:
            os.link(last_path, temporary_path)
        except OSError:
            shutil.copy2(last_path, temporary_path)
        os.replace(temporary_path, best_path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        try:
            free_mib = shutil.disk_usage(best_path.parent).free / (1024**2)
            free_text = f"{free_mib:.1f} MiB free"
        except OSError:
            free_text = "free space unavailable"
        raise RuntimeError(
            f"Failed to update best checkpoint at {best_path} ({free_text})."
        ) from exc


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    normalizer: UVTargetNormalizer | None,
    eye_geometry_normalizer: EyeGeometryNormalizer | None,
    eye_geometry_quality_normalizer: EyeGeometryQualityNormalizer | None,
    depth_correction_geometry_normalizer: EyeGeometryNormalizer | None,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device)
    required = {"model", "optimizer", "scheduler", "scaler", "normalizer", "epoch", "global_step"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Resume checkpoint is missing keys: {sorted(missing)}")
    saved_normalizer_state = checkpoint["normalizer"]
    if normalizer is None:
        if saved_normalizer_state is not None:
            raise ValueError(
                "Resume checkpoint uses legacy UV normalization, but current "
                "V4 config predicts millimetres through geometry."
            )
    else:
        if saved_normalizer_state is None:
            raise ValueError("Resume checkpoint has no UV normalization state.")
        saved_normalizer = UVTargetNormalizer.from_state_dict(saved_normalizer_state)
        if not torch.allclose(
            saved_normalizer.mean_mm,
            normalizer.mean_mm,
        ) or not torch.allclose(saved_normalizer.std_mm, normalizer.std_mm):
            raise ValueError(
                "Resume checkpoint uses different UV normalization statistics. "
                "Use the original split/config for this checkpoint."
            )
    saved_eye_geometry_state = checkpoint.get("eye_geometry_normalizer")
    if eye_geometry_normalizer is None:
        if saved_eye_geometry_state is not None:
            raise ValueError(
                "Resume checkpoint uses eye geometry, but the current config disables it."
            )
    else:
        if saved_eye_geometry_state is None:
            raise ValueError(
                "Resume checkpoint has no eye geometry normalization statistics."
            )
        saved_eye_geometry_normalizer = EyeGeometryNormalizer.from_state_dict(
            saved_eye_geometry_state
        )
        if not torch.allclose(
            saved_eye_geometry_normalizer.mean,
            eye_geometry_normalizer.mean,
        ) or not torch.allclose(
            saved_eye_geometry_normalizer.std,
            eye_geometry_normalizer.std,
        ):
            raise ValueError(
                "Resume checkpoint uses different eye geometry normalization "
                "statistics. Use the original split and depth-prior CSV."
            )
    saved_quality_state = checkpoint.get("eye_geometry_quality_normalizer")
    if eye_geometry_quality_normalizer is None:
        if saved_quality_state is not None:
            raise ValueError(
                "Resume checkpoint uses eye geometry quality normalization, "
                "but the current config does not."
            )
    else:
        if saved_quality_state is None:
            raise ValueError(
                "Resume checkpoint has no eye geometry quality normalization "
                "statistics."
            )
        saved_quality_normalizer = EyeGeometryQualityNormalizer.from_state_dict(
            saved_quality_state
        )
        if not torch.allclose(
            saved_quality_normalizer.mean,
            eye_geometry_quality_normalizer.mean,
        ) or not torch.allclose(
            saved_quality_normalizer.std,
            eye_geometry_quality_normalizer.std,
        ):
            raise ValueError(
                "Resume checkpoint uses different eye geometry quality "
                "normalization statistics."
            )
    saved_correction_state = checkpoint.get(
        "depth_correction_geometry_normalizer"
    )
    if depth_correction_geometry_normalizer is None:
        if saved_correction_state is not None:
            raise ValueError(
                "Resume checkpoint uses depth correction geometry "
                "normalization, but the current config does not."
            )
    else:
        if saved_correction_state is None:
            raise ValueError(
                "Resume checkpoint has no depth correction geometry "
                "normalization statistics."
            )
        saved_correction_normalizer = EyeGeometryNormalizer.from_state_dict(
            saved_correction_state
        )
        if not torch.allclose(
            saved_correction_normalizer.mean,
            depth_correction_geometry_normalizer.mean,
        ) or not torch.allclose(
            saved_correction_normalizer.std,
            depth_correction_geometry_normalizer.std,
        ):
            raise ValueError(
                "Resume checkpoint uses different depth correction geometry "
                "normalization statistics."
            )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        float(checkpoint.get("best_val_epe_mm", float("inf"))),
    )


def build_epoch_record(
    *,
    epoch: int,
    global_step: int,
    lr: float,
    train: Mapping[str, float],
    val: Mapping[str, float],
    epoch_seconds: float,
    elapsed_seconds: float,
    best_val_epe_mm: float,
    metric_names: tuple[str, ...],
) -> dict[str, float | int]:
    record: dict[str, float | int] = {
        "epoch": epoch,
        "global_step": global_step,
        "lr": lr,
        "epoch_seconds": epoch_seconds,
        "elapsed_seconds": elapsed_seconds,
        "best_val_epe_mm": best_val_epe_mm,
    }
    for prefix, values in (("train", train), ("val", val)):
        record[f"{prefix}_loss"] = values["loss"]
        for name in metric_names:
            record[f"{prefix}_{name}"] = values[name]
        record[f"{prefix}_seconds"] = values["seconds"]
        record[f"{prefix}_samples_per_second"] = values["samples_per_second"]
    return record


def wandb_metrics(
    record: Mapping[str, float | int],
    metric_names: tuple[str, ...],
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "epoch": record["epoch"],
        "global_step": record["global_step"],
        "learning_rate": record["lr"],
        "time/epoch_seconds": record["epoch_seconds"],
        "time/elapsed_seconds": record["elapsed_seconds"],
        "checkpoint/best_val_epe_mm": record["best_val_epe_mm"],
    }
    for prefix in ("train", "val"):
        metrics[f"{prefix}/loss"] = record[f"{prefix}_loss"]
        for name in metric_names:
            metrics[f"{prefix}/{name}"] = record[f"{prefix}_{name}"]
        metrics[f"time/{prefix}_seconds"] = record[f"{prefix}_seconds"]
        metrics[f"{prefix}/samples_per_second"] = record[f"{prefix}_samples_per_second"]
    return metrics


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    config = load_config(resolve_project_path(args.config))
    if args.resume is None and not args.dry_run:
        apply_numbered_experiment_identity(config)
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.dry_run:
        config["training"]["epochs"] = 1
        config["training"]["checkpoint"]["enabled"] = False
        config["logging"]["wandb"]["enabled"] = False
        config["experiment"]["run_name"] = "dry_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    resume_path = resolve_project_path(args.resume) if args.resume is not None else None
    initialization_value = config["training"].get("initialization_checkpoint")
    initialization_path = (
        resolve_project_path(initialization_value)
        if initialization_value
        else None
    )
    if resume_path is not None and initialization_path is not None:
        raise ValueError(
            "Use either --resume or training.initialization_checkpoint, not both."
        )
    if initialization_path is not None and not initialization_path.is_file():
        raise FileNotFoundError(
            f"Initialization checkpoint does not exist: {initialization_path}"
        )
    trainable_components = (
        str(config["training"].get("trainable_components", "all"))
        .strip()
        .lower()
    )
    if (
        trainable_components in {"depth_reweighter", "depth_correction"}
        and resume_path is None
        and initialization_path is None
    ):
        raise ValueError(
            f"{trainable_components} stage freezes gaze and therefore requires "
            "either --resume or an explicit training.initialization_checkpoint "
            "from a trained gaze stage."
        )
    run_dir = make_run_dir(config, resume_path)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_file_logger(run_dir / "train.log")
    if resume_path is None:
        save_config(config, run_dir)

    seed_everything(int(config["experiment"]["seed"]), bool(config["experiment"]["deterministic"]))
    device = resolve_device(str(config["training"]["device"]))
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    data_config = config["data"]
    model_config = make_model_config(config["model"])
    uses_deca_features = (
        model_config.deca_feature_representation
        != DECA_FEATURE_REPRESENTATION_NONE
    )
    eye_augmentation = make_eye_appearance_augmentation(data_config)
    active_metric_names = metric_names_for_model(
        model_config,
        config["loss"],
    )
    uses_gaze_geometry = (
        model_config.prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY
    )
    dataset_csv_path, deca_cache_path, depth_prior_csv_path = (
        resolve_numbered_dataset_artifacts(
            data_config,
            run_dir,
            require_deca_features=uses_deca_features,
            require_depth_prior=model_config.use_eye_geometry or uses_gaze_geometry,
        )
    )
    if deca_cache_path is not None:
        validate_deca_cache_preprocess(deca_cache_path, data_config)
    require_depth_uncertainty = (
        uses_gaze_geometry
        and (
            model_config.depth_distribution_mode != DEPTH_DISTRIBUTION_POINT
            or model_config.use_depth_correction
        )
    )
    require_pnp_quality = (
        uses_gaze_geometry
        and (
            model_config.depth_distribution_mode
            == DEPTH_DISTRIBUTION_LEARNED_REWEIGHT
            or model_config.use_depth_correction
        )
    )
    virtual_camera_manifest_path = resolve_virtual_camera_manifest_path(
        data_config,
        run_dir,
    )
    train_loader, val_loader = build_modelv1_dataloaders(
        csv_path=dataset_csv_path,
        train_datasets=data_config.get("train_datasets", ("3", "4")),
        val_datasets=data_config.get("val_datasets", ("5",)),
        split_mode=data_config["split_mode"],
        all_datasets=data_config["all_datasets"],
        val_ratio=float(data_config["val_ratio"]),
        split_seed=int(data_config["split_seed"]),
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"]),
        normalize_images=bool(data_config["normalize_images"]),
        load_face_image=bool(data_config["load_face_image"]),
        train_paired_eye_transform=eye_augmentation,
        deca_cache_path=deca_cache_path,
        require_deca_features=uses_deca_features,
        normalize_uv_targets=not uses_gaze_geometry,
        use_eye_geometry=model_config.use_eye_geometry,
        use_gaze_geometry=uses_gaze_geometry,
        require_depth_uncertainty=require_depth_uncertainty,
        require_pnp_quality=require_pnp_quality,
        use_depth_correction=model_config.use_depth_correction,
        filter_invalid_depth_prior_samples=bool(
            data_config.get("skip_invalid_depth_prior_samples", True)
        ),
        depth_prior_csv_path=depth_prior_csv_path,
        eye_geometry_gate_mode=model_config.eye_geometry_gate_mode,
        eye_geometry_representation=model_config.eye_geometry_representation,
        scene_representation=model_config.scene_representation,
        deca_feature_representation=model_config.deca_feature_representation,
        image_source=str(
            data_config.get("image_source", IMAGE_SOURCE_LEGACY)
        ),
        virtual_camera_manifest_path=virtual_camera_manifest_path,
        filter_invalid_virtual_camera_samples=bool(
            data_config.get("skip_invalid_virtual_camera_samples", False)
        ),
    )
    normalizer = get_uv_target_normalizer(train_loader.dataset)
    if not uses_gaze_geometry and normalizer is None:
        raise RuntimeError("Training requires normalized UV targets.")
    eye_geometry_normalizer = get_eye_geometry_normalizer(train_loader.dataset)
    if model_config.use_eye_geometry and eye_geometry_normalizer is None:
        raise RuntimeError(
            "Eye geometry is enabled but its training normalizer is unavailable."
        )
    eye_geometry_quality_normalizer = get_eye_geometry_quality_normalizer(
        train_loader.dataset
    )
    depth_correction_geometry_normalizer = (
        get_depth_correction_geometry_normalizer(train_loader.dataset)
    )
    if (
        model_config.use_depth_correction
        and depth_correction_geometry_normalizer is None
    ):
        raise RuntimeError(
            "Depth correction is enabled but its training geometry normalizer "
            "is unavailable."
        )
    if (
        (
            model_config.eye_geometry_gate_mode
            == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
            or require_pnp_quality
        )
        and eye_geometry_quality_normalizer is None
    ):
        raise RuntimeError(
            "Learned eye geometry gate requires a training quality normalizer."
        )

    model = ModelV1(model_config).to(device)
    if initialization_path is not None:
        loaded_count, skipped_count = initialize_compatible_model_weights(
            model,
            initialization_path,
            device,
        )
        initialization_text = (
            f"initialized {loaded_count} tensors from {initialization_path}; "
            f"left {skipped_count} tensors at configured initialization"
        )
        logger.info(initialization_text)
        print(initialization_text)
    configure_trainable_components(
        model,
        str(config["training"].get("trainable_components", "all")),
    )
    initial_schedule_message = apply_eye_backbone_schedule(
        model,
        config["training"],
        epoch=1,
    )
    loss_config = dict(config["loss"])
    gate_regularization_weight = float(
        loss_config.pop("gate_regularization_weight", 0.0)
    )
    if (
        model_config.eye_geometry_gate_mode
        != EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
    ):
        gate_regularization_weight = 0.0
    if uses_gaze_geometry:
        criterion: nn.Module = GazeGeometryLoss(
            GazeGeometryLossConfig(**loss_config)
        )
        gate_regularization_weight = 0.0
    else:
        assert normalizer is not None
        criterion = UVRegressionLoss(normalizer, UVLossConfig(**loss_config))
    optimizer_config = config["training"]["optimizer"]
    if optimizer_config["name"].lower() != "adamw":
        raise ValueError("Only AdamW is supported by the V1 training config.")
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, config["training"]),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = build_training_scheduler(optimizer, config["training"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    global_step = 0
    best_val_epe_mm = float("inf")
    if resume_path is not None:
        start_epoch, global_step, best_val_epe_mm = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            normalizer=normalizer,
            eye_geometry_normalizer=eye_geometry_normalizer,
            eye_geometry_quality_normalizer=eye_geometry_quality_normalizer,
            depth_correction_geometry_normalizer=(
                depth_correction_geometry_normalizer
            ),
            device=device,
        )
    resumed_schedule_message = apply_eye_backbone_schedule(
        model,
        config["training"],
        epoch=start_epoch,
    )

    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    setup_text = (
        f"run_dir={run_dir} device={device} amp={amp_enabled} "
        f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"deca_feature_representation={model_config.deca_feature_representation} "
        f"use_face_image={model_config.use_face_image} "
        f"face_image_backbone={FACE_IMAGE_BACKBONE} "
        f"face_image_pretrained={FACE_IMAGE_PRETRAINED_DATASET} "
        f"freeze_face_image_backbone={model_config.freeze_face_image_backbone} "
        f"visual_attention={model_config.visual_attention_heads}x"
        f"{model_config.visual_attention_dim} "
        f"eye_backbone={model_config.eye_backbone} "
        f"eye_backbone_weights={model_config.eye_backbone_weights} "
        f"eye_augmentation={eye_augmentation is not None} "
        f"image_source={data_config.get('image_source', IMAGE_SOURCE_LEGACY)} "
        f"gaze_prediction_frame={model_config.gaze_prediction_frame} "
        f"use_crop_cam={model_config.use_crop_cam} "
        f"scene_representation={model_config.scene_representation} "
        f"eye_geometry_gate_mode={model_config.eye_geometry_gate_mode} "
        f"parameters_total={total_parameter_count:,} "
        f"parameters_trainable={trainable_parameter_count:,} start_epoch={start_epoch}"
    )
    logger.info(setup_text)
    print(setup_text)
    schedule_message = (
        resumed_schedule_message
        if resume_path is not None
        else initial_schedule_message
    )
    if schedule_message is not None:
        logger.info(schedule_message)
        print(schedule_message)
    optimizer_group_text = ", ".join(
        f"{group.get('group_name', f'group_{index}')}={float(group['lr']):.3e}"
        for index, group in enumerate(optimizer.param_groups)
    )
    logger.info("optimizer learning rates: %s", optimizer_group_text)
    print(f"optimizer learning rates: {optimizer_group_text}")

    tracker = start_tracker(
        config,
        run_dir,
        active_metric_names,
        static_metadata={
            "use_face_image": model_config.use_face_image,
            "face_image_backbone": FACE_IMAGE_BACKBONE,
            "face_image_pretrained": FACE_IMAGE_PRETRAINED_DATASET,
            "freeze_face_image_backbone": (
                model_config.freeze_face_image_backbone
            ),
            "visual_attention_dim": model_config.visual_attention_dim,
            "visual_attention_heads": model_config.visual_attention_heads,
            "eye_backbone": model_config.eye_backbone,
            "eye_backbone_weights": model_config.eye_backbone_weights,
            "eye_augmentation": eye_augmentation is not None,
            "image_source": data_config.get(
                "image_source", IMAGE_SOURCE_LEGACY
            ),
            "gaze_prediction_frame": model_config.gaze_prediction_frame,
            "parameters_total": total_parameter_count,
            "parameters_trainable": trainable_parameter_count,
        },
    )
    metrics_writer = MetricsCsvWriter(
        run_dir / "metrics.csv",
        active_metric_names,
    )
    started_at = time.perf_counter()
    epochs = int(config["training"]["epochs"])
    terminal_interval = int(config["logging"]["terminal_every_epochs"])
    grad_clip_value = config["training"].get("grad_clip_norm")
    grad_clip_norm = float(grad_clip_value) if grad_clip_value is not None else None

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started_at = time.perf_counter()
            schedule_transition = apply_eye_backbone_schedule(
                model,
                config["training"],
                epoch=epoch,
            )
            if schedule_transition is not None:
                logger.info(schedule_transition)
                print(schedule_transition)
            lr = float(optimizer.param_groups[0]["lr"])
            train_metrics, train_samples = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip_norm=grad_clip_norm,
                gate_regularization_weight=gate_regularization_weight,
            )
            optimizer_steps = int(train_metrics.pop("optimizer_steps"))
            optimizer_skipped_steps = int(
                train_metrics.pop("optimizer_skipped_steps")
            )
            if optimizer_steps == 0:
                raise RuntimeError(
                    "Every optimizer step in this epoch was skipped because "
                    "gradients were non-finite. Disable FP16 AMP for the "
                    "gaze-geometry experiment or reduce the geometry loss "
                    "gradient range."
                )
            if optimizer_skipped_steps > 0:
                skipped_text = (
                    f"epoch={epoch} skipped {optimizer_skipped_steps}/"
                    f"{optimizer_steps + optimizer_skipped_steps} optimizer "
                    "steps because AMP detected non-finite gradients"
                )
                logger.warning(skipped_text)
                print(skipped_text)
            global_step += optimizer_steps
            val_metrics, _ = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip_norm=None,
                gate_regularization_weight=0.0,
            )
            scheduler.step()

            improved = val_metrics["epe_mm"] < best_val_epe_mm
            if improved:
                best_val_epe_mm = val_metrics["epe_mm"]
            record = build_epoch_record(
                epoch=epoch,
                global_step=global_step,
                lr=lr,
                train=train_metrics,
                val=val_metrics,
                epoch_seconds=time.perf_counter() - epoch_started_at,
                elapsed_seconds=time.perf_counter() - started_at,
                best_val_epe_mm=best_val_epe_mm,
                metric_names=active_metric_names,
            )
            metrics_writer.write(record)
            tracker.log(
                wandb_metrics(record, active_metric_names),
                step=epoch,
            )
            logger.info(json.dumps(record, ensure_ascii=True))

            checkpoint_enabled = bool(
                config["training"]["checkpoint"].get("enabled", True)
            )
            save_last = checkpoint_enabled and bool(
                config["training"]["checkpoint"]["save_last"]
            )
            last_checkpoint_path = checkpoints_dir / "last.pt"
            if save_last:
                save_checkpoint(
                    last_checkpoint_path,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_epe_mm=best_val_epe_mm,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    normalizer=normalizer,
                    eye_geometry_normalizer=eye_geometry_normalizer,
                    eye_geometry_quality_normalizer=eye_geometry_quality_normalizer,
                    depth_correction_geometry_normalizer=(
                        depth_correction_geometry_normalizer
                    ),
                    config=config,
                )
            if improved and checkpoint_enabled:
                best_checkpoint_path = checkpoints_dir / "best.pt"
                if save_last:
                    update_best_checkpoint_from_last(
                        last_checkpoint_path,
                        best_checkpoint_path,
                    )
                else:
                    save_checkpoint(
                        best_checkpoint_path,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_epe_mm=best_val_epe_mm,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        normalizer=normalizer,
                        eye_geometry_normalizer=eye_geometry_normalizer,
                        eye_geometry_quality_normalizer=(
                            eye_geometry_quality_normalizer
                        ),
                        depth_correction_geometry_normalizer=(
                            depth_correction_geometry_normalizer
                        ),
                        config=config,
                    )

            if epoch % terminal_interval == 0 or epoch == epochs or epoch == start_epoch:
                print(
                    f"epoch {epoch:03d}/{epochs} | lr={lr:.3e} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_epe={val_metrics['epe_mm']:.2f} mm | "
                    + (
                        f"val_valid={val_metrics['physical_valid_rate']:.3f} | "
                        f"val_lambda={val_metrics['mean_lambda_mm']:.1f} mm | "
                        f"val_axis={val_metrics['gaze_axis_error_deg']:.2f} deg | "
                        if uses_gaze_geometry
                        else ""
                    )
                    +
                    f"best={best_val_epe_mm:.2f} mm | "
                    f"epoch_time={record['epoch_seconds']:.1f}s"
                )
    finally:
        metrics_writer.close()
        tracker.finish()

    if bool(config["experiment"]["diary_enabled"]) and not args.dry_run:
        diary_path = resolve_project_path(config["experiment"]["diary_path"])
        diary_updated = append_completed_experiment(
            config=config,
            run_dir=run_dir,
            diary_path=diary_path,
            parameter_count=total_parameter_count,
            train_sample_count=len(train_loader.dataset),
            val_sample_count=len(val_loader.dataset),
        )
        diary_text = (
            f"experiment diary updated: {diary_path}"
            if diary_updated
            else f"experiment diary already contains this run: {diary_path}"
        )
        logger.info(diary_text)
        print(diary_text)

    completed_text = (
        f"completed epochs={epochs} best_val_epe_mm={best_val_epe_mm:.4f} "
        f"run_dir={run_dir}"
    )
    logger.info(completed_text)
    print(completed_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
