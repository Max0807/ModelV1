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
    DirectTableUVLoss,
    DirectVirtualTableUVLoss,
    GazeGeometryLoss,
    GazeGeometryLossConfig,
    ModelV1,
    ModelV1Config,
    UVLossConfig,
    UVRegressionLoss,
)
from modelv1.losses import compute_uv_metrics_mm, probabilistic_eye_keypoint_nll
from modelv1.vertical_eye_geometry import (
    eye_landmark_validation_metrics,
    equivariant_landmark_consistency_loss,
    mirror_shape_consistency_loss,
    probabilistic_landmark_supervision_losses,
    sample_eye_affine_matrices,
    stack_binocular_outputs,
    warp_eye_images,
)
from modelv1.processed_artifacts import processed_dataset_artifacts
from modelv1.preprocessed_v2 import (
    DEFAULT_TRAINING_DEPTH_METHOD,
    load_preprocessed_v2_training_inputs,
)
from modelv1.data import (
    DIRECT_UV_TARGET_FRAME_TABLE_LOCAL,
    DIRECT_UV_TARGET_FRAME_VIRTUAL_CAMERA,
    FaceAppearanceAugmentation,
    IMAGE_SOURCE_LEGACY,
    IMAGE_SOURCE_VIRTUAL_CAMERA,
    PairedEyeAppearanceAugmentation,
    VirtualCameraManifest,
    build_modelv1_dataloaders,
    canonical_direct_uv_target_frame,
    get_depth_correction_geometry_normalizer,
    get_eye_geometry_normalizer,
    get_eye_geometry_quality_normalizer,
    get_uv_target_normalizer,
    canonical_image_source,
    eye_augmentation_config_for_policy,
    face_augmentation_config_for_policy,
    merge_virtual_camera_manifests,
)
from modelv1.data.normalization import (
    EyeGeometryNormalizer,
    EyeGeometryQualityNormalizer,
    UVTargetNormalizer,
)
from modelv1.experiment_diary import append_completed_experiment
from modelv1.geometry_gate import (
    EYE_GEOMETRY_GATE_LEARNED_RESIDUAL,
    EYE_GEOMETRY_GATE_NONE,
    canonical_eye_geometry_gate_mode,
)
from modelv1.model import (
    FACE_IMAGE_BACKBONE,
    FACE_IMAGE_PRETRAINED_DATASET,
    GAZE_PREDICTION_FRAME_CAMERA,
    GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA,
    PREDICTION_MODE_DIRECT_UV,
    PREDICTION_MODE_GAZE_GEOMETRY,
    canonical_gaze_prediction_frame,
)
from modelv1.model_factory import (
    MODEL_ARCHITECTURE_BLAZE_STYLE,
    build_experiment_model,
    build_experiment_model_config,
    canonical_model_architecture,
)
from modelv1.blaze_style_model import blaze_embedding_consistency_loss
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
    SCENE_REPRESENTATION_TABLE_FRAME7,
    canonical_scene_representation,
    scene_representation_dim,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "modelv1"
    / "train_dataset11_val_dataset3_direct_uv_virtual_camera_film_v2.yaml"
)
FINALIZED_EYE_IMAGE_SIZE = (90, 56)
FINALIZED_EYE_IMAGE_SOURCE = IMAGE_SOURCE_LEGACY
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
    "crop_cam_hidden_dims",
    "scene_hidden_dims",
    "fusion_hidden_dims",
    "depth_reweighter_hidden_dims",
    "depth_correction_hidden_dims",
    "eye_keypoint_heatmap_size",
)
EYE_BACKBONE_TRAIN_FROM_STAGES = (
    "stem",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
)
AMP_DTYPE_FLOAT16 = "float16"
AMP_DTYPE_BFLOAT16 = "bfloat16"
AMP_DTYPES = (AMP_DTYPE_FLOAT16, AMP_DTYPE_BFLOAT16)


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
    else:
        names = BASE_METRIC_NAMES
    if config.use_eye_keypoint_auxiliary:
        names = (
            *names,
            "eye_keypoint_nll",
            "weighted_eye_keypoint_nll",
            "eye_keypoint_mean_std_norm",
            "eye_keypoint_effective_weight_per_sample",
            "eye_landmark_coordinate_loss",
            "weighted_eye_landmark_coordinate_loss",
            "eye_landmark_heatmap_loss",
            "weighted_eye_landmark_heatmap_loss",
            "eye_landmark_visibility_loss",
            "weighted_eye_landmark_visibility_loss",
            "eye_landmark_equivariance_loss",
            "weighted_eye_landmark_equivariance_loss",
            "eye_landmark_mirror_shape_loss",
            "weighted_eye_landmark_mirror_shape_loss",
            "eye_landmark_nme",
            "eye_landmark_iris_center_error_px",
            "eye_landmark_vertical_error_px",
            "eye_landmark_ellipse_reprojection_error",
            "eye_landmark_equivariance_error_norm",
            "eye_landmark_teacher_quality_error_corr",
        )
    if config.use_landmark_guided_eye_fusion:
        names = (
            *names,
            "landmark_guided_global_gate_mean",
            "landmark_guided_applied_gate_abs_mean",
            "landmark_guided_quality_mean",
            "landmark_guided_delta_norm_mean",
        )
    if loss_config is not None:
        if float(loss_config.get("blaze_reconstruction_weight", 0.0)) > 0:
            names = (
                *names,
                "blaze_reconstruction_mse",
                "weighted_blaze_reconstruction_mse",
            )
        if float(loss_config.get("blaze_consistency_weight", 0.0)) > 0:
            names = (
                *names,
                "blaze_embedding_consistency",
                "weighted_blaze_embedding_consistency",
            )
    return names


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
    prior_kind = (
        DEFAULT_TRAINING_DEPTH_METHOD
        if data_config.get("preprocessed_v2_root") is not None
        else str(data_config.get("depth_prior_kind", "none")).lower()
    )
    depth_mode = str(config["model"].get("depth_distribution_mode", "point")).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_stem = (
        f"numbered_train{train_id}_val{val_id}_{model_stage}_{prior_kind}_{depth_mode}"
    )
    experiment_config["run_name"] = f"{run_stem}_{timestamp}"
    wandb_config = config["logging"]["wandb"]
    wandb_config["project"] = str(
        experiment_config.get("numbered_wandb_project", "ModelV1_numbered_V4")
    )
    tags = [
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
    if str(data_config.get("face_augmentation", "none")).lower() != "none":
        tags.append("face_augmentation")
    wandb_config["tags"] = tags


GRADIENT_DIAGNOSTIC_FIELDS = (
    "train_grad_norm_pre_clip_mean",
    "train_grad_norm_pre_clip_max",
    "train_grad_clip_coefficient_mean",
    "train_grad_clipped_step_fraction",
)
EMA_DIAGNOSTIC_FIELDS = (
    "ema_updates",
    "val_uses_ema",
)


def metric_fields(
    metric_names: tuple[str, ...],
    optimizer_group_names: tuple[str, ...] = (),
) -> list[str]:
    return [
        "epoch",
        "global_step",
        "lr",
        *(f"lr_{name}" for name in optimizer_group_names),
        "train_loss",
        *(f"train_{name}" for name in metric_names),
        *GRADIENT_DIAGNOSTIC_FIELDS,
        *EMA_DIAGNOSTIC_FIELDS,
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


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Top-level training config must be a mapping: {path}")
    return config


def _deep_merge_config(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(defaults)
    for key, value in overrides.items():
        inherited = merged.get(key)
        if isinstance(inherited, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_config(inherited, value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    config = _load_yaml_mapping(path)
    defaults_value = config.pop("defaults", None)
    if defaults_value is not None:
        if not isinstance(defaults_value, str) or not defaults_value.strip():
            raise ValueError("defaults must be a non-empty YAML file path.")
        defaults_path = Path(defaults_value)
        if not defaults_path.is_absolute():
            defaults_path = path.parent / defaults_path
        defaults = _load_yaml_mapping(defaults_path.resolve())
        if "defaults" in defaults:
            raise ValueError("Nested defaults files are not supported.")
        config = _deep_merge_config(defaults, config)
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
    num_workers = config["data"].get("num_workers", 0)
    if isinstance(num_workers, bool) or int(num_workers) < 0:
        raise ValueError("data.num_workers must be a non-negative integer.")
    persistent_workers = config["data"].get("persistent_workers", False)
    if not isinstance(persistent_workers, bool):
        raise ValueError("data.persistent_workers must be true or false.")
    amp = config["training"].get("amp", False)
    if not isinstance(amp, bool):
        raise ValueError("training.amp must be true or false.")
    canonical_amp_dtype_name(
        config["training"].get("amp_dtype", AMP_DTYPE_FLOAT16)
    )
    fused = config["training"]["optimizer"].get("fused", False)
    if not isinstance(fused, bool):
        raise ValueError("training.optimizer.fused must be true or false.")
    if float(config["training"]["optimizer"].get("lr", 0.0)) <= 0:
        raise ValueError("training.optimizer.lr must be positive.")
    ema_config = config["training"].get("ema", {})
    if not isinstance(ema_config, Mapping):
        raise ValueError("training.ema must be a mapping.")
    if not isinstance(ema_config.get("enabled", False), bool):
        raise ValueError("training.ema.enabled must be true or false.")
    ema_decay = float(ema_config.get("decay", 0.99))
    if not 0.0 < ema_decay < 1.0:
        raise ValueError("training.ema.decay must lie in (0, 1).")
    if not isinstance(ema_config.get("start_after_warmup", True), bool):
        raise ValueError("training.ema.start_after_warmup must be true or false.")
    experiment_config = config["experiment"]
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
    model_architecture = canonical_model_architecture(
        config["model"].get("architecture", "modelv1")
    )
    eye_width, eye_height = FINALIZED_EYE_IMAGE_SIZE
    use_eye_geometry = config["model"].get("use_eye_geometry", False)
    if not isinstance(use_eye_geometry, bool):
        raise ValueError("model.use_eye_geometry must be true or false.")
    use_crop_cam = config["model"].get("use_crop_cam", True)
    if not isinstance(use_crop_cam, bool):
        raise ValueError("model.use_crop_cam must be true or false.")
    use_face_image = config["model"].get("use_face_image", False)
    if not isinstance(use_face_image, bool):
        raise ValueError("model.use_face_image must be true or false.")
    if not use_face_image:
        raise ValueError(
            "The finalized training path requires model.use_face_image=true."
        )
    use_eye_keypoint_auxiliary = config["model"].get(
        "use_eye_keypoint_auxiliary", False
    )
    if not isinstance(use_eye_keypoint_auxiliary, bool):
        raise ValueError(
            "model.use_eye_keypoint_auxiliary must be true or false."
        )
    use_landmark_guided_eye_fusion = config["model"].get(
        "use_landmark_guided_eye_fusion", False
    )
    if not isinstance(use_landmark_guided_eye_fusion, bool):
        raise ValueError(
            "model.use_landmark_guided_eye_fusion must be true or false."
        )
    if use_landmark_guided_eye_fusion and not use_eye_keypoint_auxiliary:
        raise ValueError(
            "model.use_landmark_guided_eye_fusion=true requires "
            "model.use_eye_keypoint_auxiliary=true."
        )
    if model_architecture == MODEL_ARCHITECTURE_BLAZE_STYLE:
        if use_eye_keypoint_auxiliary:
            raise ValueError(
                "Blaze-style architecture requires "
                "model.use_eye_keypoint_auxiliary=false."
            )
        if use_landmark_guided_eye_fusion:
            raise ValueError(
                "Blaze-style architecture requires "
                "model.use_landmark_guided_eye_fusion=false."
            )
    use_reconstruction_auxiliary = config["model"].get(
        "use_reconstruction_auxiliary",
        False,
    )
    if not isinstance(use_reconstruction_auxiliary, bool):
        raise ValueError(
            "model.use_reconstruction_auxiliary must be true or false."
        )
    reconstruction_sample_limit = config["model"].get(
        "reconstruction_sample_limit",
        8,
    )
    if (
        isinstance(reconstruction_sample_limit, bool)
        or int(reconstruction_sample_limit) <= 0
    ):
        raise ValueError("model.reconstruction_sample_limit must be positive.")
    blaze_reconstruction_weight = float(
        config["loss"].get("blaze_reconstruction_weight", 0.0)
    )
    blaze_consistency_weight = float(
        config["loss"].get("blaze_consistency_weight", 0.0)
    )
    if blaze_reconstruction_weight < 0 or blaze_consistency_weight < 0:
        raise ValueError("Blaze auxiliary loss weights must be non-negative.")
    if (
        blaze_reconstruction_weight > 0 or blaze_consistency_weight > 0
    ) and model_architecture != MODEL_ARCHITECTURE_BLAZE_STYLE:
        raise ValueError(
            "Blaze reconstruction/consistency losses require "
            "model.architecture='blaze_style'."
        )
    if blaze_reconstruction_weight > 0 and not use_reconstruction_auxiliary:
        raise ValueError(
            "blaze_reconstruction_weight > 0 requires "
            "model.use_reconstruction_auxiliary=true."
        )
    if use_reconstruction_auxiliary and blaze_reconstruction_weight <= 0:
        raise ValueError(
            "model.use_reconstruction_auxiliary=true requires a positive "
            "loss.blaze_reconstruction_weight."
        )
    if blaze_reconstruction_weight > 0 and not bool(
        config["data"].get("normalize_images", True)
    ):
        raise ValueError(
            "Blaze reconstruction requires data.normalize_images=true so the "
            "decoder target can invert the configured ImageNet normalization."
        )
    eye_geometry_pseudo_label_paths = config["data"].get(
        "eye_geometry_pseudo_label_paths"
    )
    if eye_geometry_pseudo_label_paths is not None and (
        not isinstance(eye_geometry_pseudo_label_paths, (list, tuple))
        or not eye_geometry_pseudo_label_paths
        or not all(str(path).strip() for path in eye_geometry_pseudo_label_paths)
    ):
        raise ValueError(
            "data.eye_geometry_pseudo_label_paths must be a non-empty YAML list."
        )
    require_eye_geometry_pseudo_labels = config["data"].get(
        "require_eye_geometry_pseudo_labels", False
    )
    if not isinstance(require_eye_geometry_pseudo_labels, bool):
        raise ValueError(
            "data.require_eye_geometry_pseudo_labels must be true or false."
        )
    if require_eye_geometry_pseudo_labels and not eye_geometry_pseudo_label_paths:
        raise ValueError(
            "data.require_eye_geometry_pseudo_labels=true requires "
            "data.eye_geometry_pseudo_label_paths."
        )
    if use_eye_keypoint_auxiliary and not eye_geometry_pseudo_label_paths:
        raise ValueError(
            "model.use_eye_keypoint_auxiliary=true requires "
            "data.eye_geometry_pseudo_label_paths."
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
    preprocessed_v2_root = config["data"].get("preprocessed_v2_root")
    uses_preprocessed_v2 = preprocessed_v2_root is not None
    table7_stage3_csv_paths = config["data"].get("table7_stage3_csv_paths")
    uses_table7_stage3 = table7_stage3_csv_paths is not None
    if uses_table7_stage3 and (
        not isinstance(table7_stage3_csv_paths, (list, tuple))
        or not table7_stage3_csv_paths
        or not all(str(path).strip() for path in table7_stage3_csv_paths)
    ):
        raise ValueError(
            "data.table7_stage3_csv_paths must be a non-empty YAML list."
        )
    if uses_preprocessed_v2 and uses_table7_stage3:
        raise ValueError(
            "Use either data.preprocessed_v2_root or "
            "data.table7_stage3_csv_paths, not both."
        )
    if uses_table7_stage3 and not has_numbered_dataset_ids:
        raise ValueError(
            "data.dataset_ids is required with data.table7_stage3_csv_paths."
        )
    if uses_preprocessed_v2:
        if not str(preprocessed_v2_root).strip():
            raise ValueError("data.preprocessed_v2_root must be a non-empty path.")
        if not has_numbered_dataset_ids:
            raise ValueError(
                "data.dataset_ids is required with data.preprocessed_v2_root."
            )
        incompatible = [
            key
            for key in (
                "csv_path",
                "depth_prior_csv_path",
                "processed_data_dir",
                "depth_prior_kind",
                "use_trainable_depth_subset",
                "virtual_camera_manifest_path",
                "virtual_camera_manifest_paths",
                "sample_id_filter_virtual_camera_manifest_paths",
            )
            if config["data"].get(key) is not None
        ]
        if incompatible:
            raise ValueError(
                "preprocessed-v2 training inputs are authoritative; remove "
                f"legacy/explicit data keys: {incompatible}."
            )
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
    eye_keypoint_nll_weight = float(
        config["loss"].get("eye_keypoint_nll_weight", 0.0)
    )
    eye_keypoint_min_std_norm = float(
        config["loss"].get("eye_keypoint_min_std_norm", 0.01)
    )
    eye_landmark_coordinate_weight = float(
        config["loss"].get("eye_landmark_coordinate_weight", 0.0)
    )
    eye_landmark_heatmap_weight = float(
        config["loss"].get("eye_landmark_heatmap_weight", 0.0)
    )
    eye_landmark_visibility_weight = float(
        config["loss"].get("eye_landmark_visibility_weight", 0.0)
    )
    eye_landmark_equivariance_weight = float(
        config["loss"].get("eye_landmark_equivariance_weight", 0.0)
    )
    eye_landmark_mirror_shape_weight = float(
        config["loss"].get("eye_landmark_mirror_shape_weight", 0.0)
    )
    eye_landmark_huber_delta_norm = float(
        config["loss"].get("eye_landmark_huber_delta_norm", 0.02)
    )
    eye_landmark_heatmap_sigma_min_px = float(
        config["loss"].get("eye_landmark_heatmap_sigma_min_px", 1.0)
    )
    eye_landmark_heatmap_sigma_max_px = float(
        config["loss"].get("eye_landmark_heatmap_sigma_max_px", 3.0)
    )
    eye_landmark_equivariance_rotation_deg = float(
        config["loss"].get("eye_landmark_equivariance_rotation_deg", 4.0)
    )
    eye_landmark_equivariance_translation_norm = float(
        config["loss"].get("eye_landmark_equivariance_translation_norm", 0.03)
    )
    eye_landmark_equivariance_scale_min = float(
        config["loss"].get("eye_landmark_equivariance_scale_min", 0.97)
    )
    eye_landmark_equivariance_scale_max = float(
        config["loss"].get("eye_landmark_equivariance_scale_max", 1.03)
    )
    prediction_mode = str(
        config["model"].get("prediction_mode", "direct_uv")
    ).strip().lower()
    image_source = canonical_image_source(
        config["data"].get("image_source", IMAGE_SOURCE_LEGACY)
    )
    direct_uv_target_frame = canonical_direct_uv_target_frame(
        config["data"].get(
            "direct_uv_target_frame",
            DIRECT_UV_TARGET_FRAME_TABLE_LOCAL,
        )
    )
    gaze_prediction_frame = canonical_gaze_prediction_frame(
        config["model"].get(
            "gaze_prediction_frame",
            GAZE_PREDICTION_FRAME_CAMERA,
        )
    )
    use_table_frame_film = bool(
        config["model"].get("use_table_frame_film", False)
    )
    virtual_manifest_path = config["data"].get("virtual_camera_manifest_path")
    virtual_manifest_paths = config["data"].get("virtual_camera_manifest_paths")
    sample_filter_manifest_paths = config["data"].get(
        "sample_id_filter_virtual_camera_manifest_paths"
    )
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
    if sample_filter_manifest_paths is not None and (
        not isinstance(sample_filter_manifest_paths, (list, tuple))
        or not sample_filter_manifest_paths
        or not all(str(path).strip() for path in sample_filter_manifest_paths)
    ):
        raise ValueError(
            "data.sample_id_filter_virtual_camera_manifest_paths must be "
            "a non-empty YAML list."
        )
    if (
        image_source == IMAGE_SOURCE_VIRTUAL_CAMERA
        and not virtual_manifest_path
        and not virtual_manifest_paths
        and not uses_preprocessed_v2
        and not uses_table7_stage3
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
    if uses_table7_stage3:
        if prediction_mode != PREDICTION_MODE_DIRECT_UV:
            raise ValueError(
                "data.table7_stage3_csv_paths currently supports only "
                "model.prediction_mode='direct_uv'."
            )
        if use_eye_geometry:
            raise ValueError(
                "data.table7_stage3_csv_paths does not load a depth-prior "
                "eye-geometry branch."
            )
        if use_crop_cam:
            raise ValueError(
                "data.table7_stage3_csv_paths requires model.use_crop_cam=false."
            )
        # if image_source != IMAGE_SOURCE_VIRTUAL_CAMERA:
        #     raise ValueError(
        #         "data.table7_stage3_csv_paths requires "
        #         "data.image_source='virtual_camera'."
        #     )
        if scene_representation != SCENE_REPRESENTATION_TABLE_FRAME7:
            raise ValueError(
                "data.table7_stage3_csv_paths requires "
                "model.scene_representation='table_frame7'."
            )
        if direct_uv_target_frame != DIRECT_UV_TARGET_FRAME_TABLE_LOCAL:
            raise ValueError(
                "data.table7_stage3_csv_paths supports only "
                "data.direct_uv_target_frame='table_local'."
            )
        if virtual_manifest_path or virtual_manifest_paths:
            raise ValueError(
                "Stage3 rows already contain virtual_face_path and "
                "table_frame7_n_0..6; remove legacy virtual-camera manifests."
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
        # if (
        #     gaze_prediction_frame == GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA
        #     and image_source != IMAGE_SOURCE_VIRTUAL_CAMERA
        # ):
        #     raise ValueError(
        #         "model.gaze_prediction_frame='virtual_camera' requires "
        #         "data.image_source='virtual_camera'."
        #     )
    if direct_uv_target_frame == DIRECT_UV_TARGET_FRAME_VIRTUAL_CAMERA:
        if prediction_mode != PREDICTION_MODE_DIRECT_UV:
            raise ValueError(
                "data.direct_uv_target_frame='virtual_camera' requires "
                "model.prediction_mode='direct_uv'."
            )
        # if image_source != IMAGE_SOURCE_VIRTUAL_CAMERA:
        #     raise ValueError(
        #         "data.direct_uv_target_frame='virtual_camera' requires "
        #         "data.image_source='virtual_camera'."
        #     )
        if (
            canonical_scene_representation(
                config["model"].get(
                    "scene_representation",
                    SCENE_REPRESENTATION_FULL25,
                )
            )
            != SCENE_REPRESENTATION_TABLE_FRAME7
        ):
            raise ValueError(
                "data.direct_uv_target_frame='virtual_camera' requires "
                "model.scene_representation='table_frame7'."
            )
    if use_table_frame_film:
        if prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY:
            raise ValueError(
                "model.use_table_frame_film requires prediction_mode='direct_uv'."
            )
        # if image_source != IMAGE_SOURCE_VIRTUAL_CAMERA:
        #     raise ValueError(
        #         "model.use_table_frame_film requires "
        #         "data.image_source='virtual_camera'."
        #     )
        if (
            canonical_scene_representation(
                config["model"].get(
                    "scene_representation",
                    SCENE_REPRESENTATION_FULL25,
                )
            )
            != SCENE_REPRESENTATION_TABLE_FRAME7
        ):
            raise ValueError(
                "model.use_table_frame_film requires "
                "scene_representation='table_frame7'."
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
    make_eye_appearance_augmentation(config["data"])
    make_face_appearance_augmentation(config["data"])
    validate_eye_backbone_schedule(config["training"], config["model"])
    validate_scheduler_config(config["training"])


def make_eye_appearance_augmentation(
    data_config: Mapping[str, Any],
) -> PairedEyeAppearanceAugmentation | None:
    policy = data_config.get("eye_augmentation", "none")
    if policy is None:
        return None
    if not isinstance(policy, str):
        raise ValueError("data.eye_augmentation must be a named policy string.")
    augmentation_config = eye_augmentation_config_for_policy(policy)
    if augmentation_config is None:
        return None
    return PairedEyeAppearanceAugmentation(augmentation_config)


def make_face_appearance_augmentation(
    data_config: Mapping[str, Any],
) -> FaceAppearanceAugmentation | None:
    policy = data_config.get("face_augmentation", "none")
    if policy is None:
        return None
    if not isinstance(policy, str):
        raise ValueError("data.face_augmentation must be a named policy string.")
    augmentation_config = face_augmentation_config_for_policy(policy)
    if augmentation_config is None:
        return None
    return FaceAppearanceAugmentation(augmentation_config)


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
    architecture = canonical_model_architecture(
        model_config.get("architecture", "modelv1")
    )
    if architecture == MODEL_ARCHITECTURE_BLAZE_STYLE:
        raise ValueError(
            "training.eye_backbone_schedule must be disabled for the "
            "Blaze-style architecture."
        )
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


def validate_scheduler_config(training_config: Mapping[str, Any]) -> None:
    scheduler_config = training_config["scheduler"]
    if scheduler_config["name"].lower() != "cosine":
        raise ValueError("Only cosine scheduler is supported by the V1 config.")
    warmup_epochs = int(scheduler_config.get("warmup_epochs", 0))
    epochs = int(training_config["epochs"])
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError(
            "training.scheduler.warmup_epochs must lie in [0, training.epochs)."
        )
    start_factor = float(scheduler_config.get("warmup_start_factor", 1.0))
    if not 0 < start_factor <= 1:
        raise ValueError(
            "training.scheduler.warmup_start_factor must lie in (0, 1]."
        )
    eta_min = float(scheduler_config["eta_min"])
    base_lr = float(training_config["optimizer"]["lr"])
    if not 0 <= eta_min <= base_lr:
        raise ValueError(
            "training.scheduler.eta_min must lie in [0, training.optimizer.lr]."
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


def resolve_table7_stage3_artifacts(
    data_config: Mapping[str, Any],
    run_dir: Path,
) -> Path:
    """Resolve and audit one or more final Iris65 Stage3 training CSVs."""

    raw_paths = data_config.get("table7_stage3_csv_paths")
    if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
        raise ValueError(
            "data.table7_stage3_csv_paths must be a non-empty YAML list."
        )
    paths = [resolve_project_path(path) for path in raw_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Stage3 TableFrame7 CSV(s):\n  {preview}")

    resolved_dir = run_dir / "resolved_table7_stage3"
    dataset_csv = (
        paths[0]
        if len(paths) == 1
        else _merge_csv_files(
            paths,
            resolved_dir / "dataset.csv",
            label="table7-stage3",
        )
    )
    resolved_dir.mkdir(parents=True, exist_ok=True)
    rows, _ = _read_csv_records(dataset_csv)
    (resolved_dir / "sources.json").write_text(
        json.dumps(
            {
                "artifact_source": "iris65_metric_head_origin_stage3_table7",
                "dataset_csvs": [str(path) for path in paths],
                "sample_count": len(rows),
                "virtual_camera_center_source": (
                    "Iris65-derived metric head origin"
                ),
                "table_frame_source": "precomputed table_frame7_n_0..6",
                "eye_image_source": FINALIZED_EYE_IMAGE_SOURCE,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_csv


def resolve_numbered_dataset_artifacts(
    data_config: Mapping[str, Any],
    run_dir: Path,
    *,
    require_depth_prior: bool,
) -> tuple[Path, Path | None]:
    """Resolve ``data.dataset_ids`` into one or several numbered artifacts.

    A single ID is used directly.  Multiple IDs are merged under the run
    directory, so the exact data union is retained with the experiment instead
    of creating or mutating a global combined CSV.
    """

    raw_ids = data_config.get("dataset_ids")
    if raw_ids is None:
        dataset_csv = resolve_project_path(data_config["csv_path"])
        depth_prior = (
            resolve_project_path(data_config["depth_prior_csv_path"])
            if require_depth_prior
            else None
        )
        return dataset_csv, depth_prior
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
        *(prior_paths if require_depth_prior else []),
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        preview = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing numbered processed artifact(s):\n  {preview}")
    if len(artifacts) == 1:
        return (
            dataset_paths[0],
            prior_paths[0] if require_depth_prior else None,
        )

    merged_dir = run_dir / "resolved_numbered_data"
    dataset_csv = _merge_csv_files(dataset_paths, merged_dir / "dataset.csv", label="dataset")
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
    return dataset_csv, depth_prior


def resolve_preprocessed_v2_artifacts(
    data_config: Mapping[str, Any],
    run_dir: Path,
    *,
    require_depth_prior: bool,
) -> tuple[Path, Path | None, list[Path]]:
    """Resolve and merge versioned preprocessing training-input contracts."""

    raw_ids = data_config.get("dataset_ids")
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids:
        raise ValueError(
            "data.dataset_ids must be a non-empty YAML list for preprocessed-v2."
        )
    root = resolve_project_path(data_config["preprocessed_v2_root"])
    depth_method = DEFAULT_TRAINING_DEPTH_METHOD
    inputs = load_preprocessed_v2_training_inputs(root, raw_ids, depth_method)

    dataset_paths = [item.dataset_csv for item in inputs]
    prior_paths = [item.depth_prior_csv for item in inputs]
    virtual_manifest_paths = [item.virtual_camera_manifest for item in inputs]
    for item in inputs:
        rows, _ = _read_csv_records(item.dataset_csv)
        if len(rows) != item.sample_count:
            raise ValueError(
                f"Preprocessed-v2 sample_count mismatch for Dataset {item.dataset_id}: "
                f"manifest={item.sample_count}, dataset_csv={len(rows)}."
            )
        wrong_dataset = [
            row.get("sample_id", "")
            for row in rows
            if row.get("dataset") != item.dataset_name
        ]
        if wrong_dataset:
            raise ValueError(
                f"Preprocessed-v2 Dataset {item.dataset_id} CSV contains "
                f"{len(wrong_dataset)} rows from another dataset."
            )

    resolved_dir = run_dir / "resolved_preprocessed_v2"
    if len(inputs) == 1:
        dataset_csv = dataset_paths[0]
        depth_prior = prior_paths[0] if require_depth_prior else None
    else:
        dataset_csv = _merge_csv_files(
            dataset_paths,
            resolved_dir / "dataset.csv",
            label="dataset",
        )
        depth_prior = (
            _merge_csv_files(
                prior_paths,
                resolved_dir / "depth_prior.csv",
                label="depth-prior",
            )
            if require_depth_prior
            else None
        )

    resolved_dir.mkdir(parents=True, exist_ok=True)
    (resolved_dir / "sources.json").write_text(
        json.dumps(
            {
                "artifact_source": "preprocessed_v2",
                "preprocessed_v2_root": str(root),
                "depth_method": depth_method,
                "dataset_ids": [item.dataset_id for item in inputs],
                "training_input_manifests": [
                    str(item.manifest_path) for item in inputs
                ],
                "dataset_csvs": [str(path) for path in dataset_paths],
                "depth_prior_csvs": (
                    [str(path) for path in prior_paths]
                    if require_depth_prior
                    else []
                ),
                "virtual_camera_manifests": [
                    str(path) for path in virtual_manifest_paths
                ],
                "sample_counts": {
                    item.dataset_id: item.sample_count for item in inputs
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_csv, depth_prior, virtual_manifest_paths


def resolve_virtual_camera_manifest_path(
    data_config: Mapping[str, Any],
    run_dir: Path,
    fallback_paths: list[Path] | None = None,
) -> Path | None:
    """Resolve one manifest or merge configured manifests inside ``run_dir``."""

    singular = data_config.get("virtual_camera_manifest_path")
    plural = data_config.get("virtual_camera_manifest_paths")
    if fallback_paths and (singular is not None or plural is not None):
        raise ValueError(
            "Preprocessed-v2 virtual-camera manifests cannot be combined with "
            "explicit virtual_camera_manifest_path(s)."
        )
    if singular and plural:
        raise ValueError(
            "Use either virtual_camera_manifest_path or "
            "virtual_camera_manifest_paths, not both."
        )
    if singular is not None:
        return resolve_project_path(singular)
    if plural is None and not fallback_paths:
        return None
    if fallback_paths:
        source_paths = fallback_paths
    else:
        if not isinstance(plural, (list, tuple)) or not plural:
            raise ValueError("virtual_camera_manifest_paths must be a non-empty list.")
        source_paths = [resolve_project_path(path) for path in plural]
    if len(source_paths) == 1:
        return source_paths[0]
    merged_path = run_dir / "resolved_virtual_camera" / "manifest.csv"
    return merge_virtual_camera_manifests(source_paths, merged_path)


def resolve_sample_id_filter_manifest_path(
    data_config: Mapping[str, Any],
    run_dir: Path,
    fallback_paths: list[Path] | None = None,
) -> Path | None:
    """Resolve manifests used only to lock a paired experiment's samples."""

    values = data_config.get("sample_id_filter_virtual_camera_manifest_paths")
    if fallback_paths and values is not None:
        raise ValueError(
            "Preprocessed-v2 sample filtering cannot be combined with explicit "
            "sample_id_filter_virtual_camera_manifest_paths."
        )
    if values is None and not fallback_paths:
        return None
    if fallback_paths:
        source_paths = fallback_paths
    else:
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(
                "sample_id_filter_virtual_camera_manifest_paths must be a "
                "non-empty list."
            )
        source_paths = [resolve_project_path(path) for path in values]
    if len(source_paths) == 1:
        return source_paths[0]
    merged_path = run_dir / "resolved_sample_id_filter" / "manifest.csv"
    return merge_virtual_camera_manifests(source_paths, merged_path)


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
    def __init__(
        self,
        path: Path,
        metric_names: tuple[str, ...],
        optimizer_group_names: tuple[str, ...] = (),
    ) -> None:
        expected_fields = metric_fields(metric_names, optimizer_group_names)
        self.fields = expected_fields
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open("r", encoding="utf-8", newline="") as existing_handle:
                existing_fields = next(csv.reader(existing_handle), [])
            compatible_existing_fields = [
                field for field in expected_fields if field in set(existing_fields)
            ]
            if compatible_existing_fields != existing_fields:
                raise ValueError(
                    f"Existing metrics file uses a different schema: {path}. "
                    "Start a new run instead of resuming this run."
                )
            # Older runs do not have the new diagnostic columns. Preserve their
            # existing CSV schema so checkpoint resume remains backward compatible.
            self.fields = existing_fields
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
    architecture = canonical_model_architecture(
        values.get("architecture", "modelv1")
    )
    values.pop("architecture", None)
    for key in TUPLE_CONFIG_KEYS:
        if key in values:
            values[key] = tuple(values[key])
    return build_experiment_model_config(architecture, values)


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
        return [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    backbone_parameters = model.eye_backbone_parameters()
    face_backbone_parameters = [
        parameter
        for parameter in model.visual_encoder.face_encoder.backbone.parameters()
        if parameter.requires_grad
    ]
    face_backbone_ids = {
        id(parameter) for parameter in face_backbone_parameters
    }
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    non_backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if (
            parameter.requires_grad
            and id(parameter) not in backbone_ids
            and id(parameter) not in face_backbone_ids
        )
    ]
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
        {
            "params": face_backbone_parameters,
            "lr": 3e-5,
            "group_name": "face_inception_resnet_backbone",
        },
    ]


def build_training_optimizer(
    model: ModelV1,
    training_config: Mapping[str, Any],
    device: torch.device,
) -> torch.optim.AdamW:
    optimizer_config = training_config["optimizer"]
    if str(optimizer_config["name"]).lower() != "adamw":
        raise ValueError("Only AdamW is supported by the V1 training config.")
    fused = bool(optimizer_config.get("fused", False)) and device.type == "cuda"
    return torch.optim.AdamW(
        optimizer_parameter_groups(model, training_config),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        fused=fused,
    )


def optimizer_group_name(group: Mapping[str, Any], index: int) -> str:
    return str(group.get("group_name", f"group_{index}"))


def optimizer_group_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = optimizer_group_name(group, index)
        if name in rates:
            raise ValueError(f"Duplicate optimizer parameter group name: {name!r}.")
        rates[name] = float(group["lr"])
    return rates


def optimizer_group_parameter_counts(
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = optimizer_group_name(group, index)
        counts[name] = sum(
            parameter.numel()
            for parameter in group["params"]
            if parameter.requires_grad
        )
    return counts


class ExponentialMovingAverage:
    """Maintain evaluation-only exponential averages of model state."""

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float,
        update_after_step: int,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie in (0, 1).")
        if update_after_step < 0:
            raise ValueError("EMA update_after_step must be non-negative.")
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.optimization_steps = 0
        self.num_updates = 0
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self._ema_parameters = tuple(self.module.parameters())
        self._ema_buffers = tuple(self.module.buffers())

    @property
    def is_active(self) -> bool:
        return self.num_updates > 0

    @torch.no_grad()
    def _copy_from(self, model: nn.Module) -> None:
        model_parameters = tuple(model.parameters())
        model_buffers = tuple(model.buffers())
        if len(model_parameters) != len(self._ema_parameters):
            raise RuntimeError("EMA model parameter structure changed during training.")
        if len(model_buffers) != len(self._ema_buffers):
            raise RuntimeError("EMA model buffer structure changed during training.")
        for ema_parameter, model_parameter in zip(
            self._ema_parameters,
            model_parameters,
        ):
            ema_parameter.copy_(model_parameter)
        for ema_buffer, model_buffer in zip(self._ema_buffers, model_buffers):
            ema_buffer.copy_(model_buffer)

    @torch.no_grad()
    def update(self, model: nn.Module) -> bool:
        self.optimization_steps += 1
        if self.optimization_steps <= self.update_after_step:
            return False

        model_parameters = tuple(model.parameters())
        model_buffers = tuple(model.buffers())
        if self.num_updates == 0:
            self._copy_from(model)
        else:
            torch._foreach_lerp_(
                self._ema_parameters,
                model_parameters,
                1.0 - self.decay,
            )
            for ema_buffer, model_buffer in zip(
                self._ema_buffers,
                model_buffers,
            ):
                if torch.is_floating_point(ema_buffer):
                    ema_buffer.lerp_(model_buffer, 1.0 - self.decay)
                else:
                    ema_buffer.copy_(model_buffer)
        self.num_updates += 1
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "model": self.module.state_dict(),
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "optimization_steps": self.optimization_steps,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        saved_decay = float(state["decay"])
        saved_update_after_step = int(state["update_after_step"])
        if not math.isclose(saved_decay, self.decay):
            raise ValueError(
                f"EMA decay mismatch: checkpoint={saved_decay}, config={self.decay}."
            )
        if saved_update_after_step != self.update_after_step:
            raise ValueError(
                "EMA update_after_step mismatch: "
                f"checkpoint={saved_update_after_step}, "
                f"config={self.update_after_step}."
            )
        model_state = state["model"]
        if not isinstance(model_state, Mapping):
            raise TypeError("EMA checkpoint model state must be a mapping.")
        self.module.load_state_dict(model_state)
        self.optimization_steps = int(state["optimization_steps"])
        self.num_updates = int(state["num_updates"])

    def reset_from_model(
        self,
        model: nn.Module,
        *,
        optimization_steps: int,
    ) -> None:
        self._copy_from(model)
        self.optimization_steps = int(optimization_steps)
        self.num_updates = 0


def checkpoint_model_state(
    checkpoint: Mapping[str, Any],
    *,
    prefer_ema: bool,
) -> tuple[Mapping[str, Any], str]:
    if prefer_ema:
        ema_state = checkpoint.get("ema")
        if (
            isinstance(ema_state, Mapping)
            and int(ema_state.get("num_updates", 0)) > 0
        ):
            ema_model_state = ema_state.get("model")
            if isinstance(ema_model_state, Mapping):
                return ema_model_state, "ema"
    model_state = checkpoint.get("model", checkpoint)
    if not isinstance(model_state, Mapping):
        raise ValueError("Checkpoint has no model state mapping.")
    return model_state, "raw"


def scheduler_steps_per_optimizer_step(
    training_config: Mapping[str, Any],
) -> bool:
    return int(training_config["scheduler"].get("warmup_epochs", 0)) > 0


def build_training_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: Mapping[str, Any],
    *,
    steps_per_epoch: int = 1,
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_config = training_config["scheduler"]
    if scheduler_config["name"].lower() != "cosine":
        raise ValueError("Only cosine scheduler is supported by the V1 training config.")
    epochs = int(training_config["epochs"])
    eta_min = float(scheduler_config["eta_min"])
    warmup_epochs = int(scheduler_config.get("warmup_epochs", 0))
    if warmup_epochs == 0:
        if enabled_eye_backbone_schedule(training_config) is None:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=eta_min,
            )
        base_lr = float(training_config["optimizer"]["lr"])
        eta_ratio = eta_min / base_lr

        def epoch_cosine_multiplier(step: int) -> float:
            progress = min(max(step, 0), epochs) / epochs
            return eta_ratio + (1.0 - eta_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                epoch_cosine_multiplier for _ in optimizer.param_groups
            ],
        )

    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive.")
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = total_steps - warmup_steps
    start_factor = float(scheduler_config.get("warmup_start_factor", 0.1))
    eta_ratio = eta_min / float(training_config["optimizer"]["lr"])

    def warmup_cosine_multiplier(step: int) -> float:
        bounded_step = min(max(step, 0), total_steps)
        if bounded_step < warmup_steps:
            progress = bounded_step / warmup_steps
            return start_factor + (1.0 - start_factor) * progress
        progress = (bounded_step - warmup_steps) / cosine_steps
        return eta_ratio + (1.0 - eta_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            warmup_cosine_multiplier for _ in optimizer.param_groups
        ],
    )


def initialize_compatible_model_weights(
    model: ModelV1,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[int, int]:
    """Load shape-compatible weights for point -> prior -> reweight stages."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state, _ = checkpoint_model_state(checkpoint, prefer_ema=True)
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


def canonical_amp_dtype_name(value: object) -> str:
    name = str(value).strip().lower()
    aliases = {
        "fp16": AMP_DTYPE_FLOAT16,
        "float16": AMP_DTYPE_FLOAT16,
        "bf16": AMP_DTYPE_BFLOAT16,
        "bfloat16": AMP_DTYPE_BFLOAT16,
    }
    if name not in aliases:
        raise ValueError(
            f"training.amp_dtype must be one of {AMP_DTYPES}, got {value!r}."
        )
    return aliases[name]


def resolve_amp_settings(
    training_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[bool, str, torch.dtype]:
    dtype_name = canonical_amp_dtype_name(
        training_config.get("amp_dtype", AMP_DTYPE_FLOAT16)
    )
    dtype = (
        torch.bfloat16
        if dtype_name == AMP_DTYPE_BFLOAT16
        else torch.float16
    )
    enabled = bool(training_config.get("amp", False)) and device.type == "cuda"
    if (
        enabled
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "BF16 AMP was requested, but the selected CUDA device does not "
            "report BF16 support."
        )
    return enabled, dtype_name, dtype


def autocast_context(enabled: bool, dtype: torch.dtype):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def run_epoch(
    *,
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype = torch.float16,
    grad_clip_norm: float | None,
    gate_regularization_weight: float,
    blaze_reconstruction_weight: float = 0.0,
    blaze_consistency_weight: float = 0.0,
    eye_keypoint_nll_weight: float = 0.0,
    eye_keypoint_min_std_norm: float = 0.01,
    eye_landmark_coordinate_weight: float = 0.0,
    eye_landmark_heatmap_weight: float = 0.0,
    eye_landmark_visibility_weight: float = 0.0,
    eye_landmark_equivariance_weight: float = 0.0,
    eye_landmark_mirror_shape_weight: float = 0.0,
    eye_landmark_huber_delta_norm: float = 0.02,
    eye_landmark_heatmap_sigma_min_px: float = 1.0,
    eye_landmark_heatmap_sigma_max_px: float = 3.0,
    eye_landmark_equivariance_rotation_deg: float = 4.0,
    eye_landmark_equivariance_translation_norm: float = 0.03,
    eye_landmark_equivariance_scale_min: float = 0.97,
    eye_landmark_equivariance_scale_max: float = 1.03,
    ema: ExponentialMovingAverage | None = None,
) -> tuple[dict[str, float], int]:
    is_train = optimizer is not None
    runtime_model_config = getattr(model, "config", None)
    use_eye_keypoint_auxiliary = bool(
        getattr(runtime_model_config, "use_eye_keypoint_auxiliary", False)
    )
    use_landmark_guided_eye_fusion = bool(
        getattr(runtime_model_config, "use_landmark_guided_eye_fusion", False)
    )
    use_blaze_auxiliary = (
        blaze_reconstruction_weight > 0 or blaze_consistency_weight > 0
    )
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
    eye_keypoint_metric_sums = {
        "eye_keypoint_nll": 0.0,
        "eye_keypoint_mean_std_norm": 0.0,
        "eye_keypoint_effective_weight_per_sample": 0.0,
        "eye_landmark_coordinate_loss": 0.0,
        "eye_landmark_heatmap_loss": 0.0,
        "eye_landmark_visibility_loss": 0.0,
        "eye_landmark_equivariance_loss": 0.0,
        "eye_landmark_mirror_shape_loss": 0.0,
        "eye_landmark_equivariance_error_norm": 0.0,
    }
    blaze_auxiliary_metric_sums = {
        "blaze_reconstruction_mse": 0.0,
        "blaze_embedding_consistency": 0.0,
    }
    blaze_reconstruction_sample_count = 0
    eye_landmark_predictions: list[Tensor] = []
    eye_landmark_targets: list[Tensor] = []
    eye_landmark_valid_masks: list[Tensor] = []
    eye_landmark_teacher_qualities: list[Tensor] = []
    eye_landmark_image_size: tuple[int, int] | None = None
    landmark_guided_global_gate_sum = 0.0
    landmark_guided_applied_gate_abs_sum = 0.0
    landmark_guided_quality_sum = 0.0
    landmark_guided_delta_norm_sum = 0.0
    optimizer_steps = 0
    optimizer_skipped_steps = 0
    grad_norm_pre_clip_sum = 0.0
    grad_norm_pre_clip_max = 0.0
    grad_clip_coefficient_sum = 0.0
    grad_clipped_steps = 0
    gradient_diagnostic_steps = 0

    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        batch_size = int(device_batch["uv_gt"].shape[0])
        metric_target_mm = device_batch["uv_gt"]
        if not torch.is_tensor(metric_target_mm):
            raise TypeError("batch['uv_gt'] must be a tensor.")
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            with autocast_context(amp_enabled, amp_dtype):
                use_gate_regularization = (
                    is_train and gate_regularization_weight > 0
                )
                model_kwargs = {
                    "return_features": (
                        use_gate_regularization or use_blaze_auxiliary
                    )
                }
                if use_eye_keypoint_auxiliary or use_blaze_auxiliary:
                    model_kwargs["return_auxiliary"] = True
                model_output = model(device_batch, **model_kwargs)
                if use_landmark_guided_eye_fusion:
                    if not isinstance(model_output, Mapping):
                        raise RuntimeError(
                            "Landmark-guided fusion requires model mapping outputs."
                        )
                    landmark_guided_global_gate_sum += float(
                        model_output["landmark_guided_global_gate"]
                        .detach()
                        .sum()
                    )
                    landmark_guided_applied_gate_abs_sum += float(
                        model_output["landmark_guided_applied_gate"]
                        .detach()
                        .abs()
                        .sum()
                    )
                    landmark_guided_quality_sum += float(
                        model_output["landmark_guided_sample_quality"]
                        .detach()
                        .sum()
                    )
                    landmark_guided_delta_norm_sum += float(
                        model_output["landmark_guided_delta_norm"]
                        .detach()
                        .sum()
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
                elif isinstance(
                    criterion,
                    (DirectTableUVLoss, DirectVirtualTableUVLoss),
                ):
                    direct_uv_pred = model_output
                    loss_outputs = criterion(direct_uv_pred, device_batch)
                    loss = loss_outputs["loss"]
                    uv_pred_mm = loss_outputs["uv_pred_table_mm"]
                    metric_target_mm = device_batch["uv_gt_table_mm"]
                    if not torch.is_tensor(metric_target_mm):
                        raise TypeError("batch['uv_gt_table_mm'] must be a tensor.")
                else:
                    if isinstance(model_output, dict):
                        uv_pred = model_output["uv"]
                    else:
                        uv_pred = model_output
                    if not isinstance(criterion, UVRegressionLoss):
                        raise TypeError("Legacy training requires UVRegressionLoss.")
                    loss = criterion(uv_pred, device_batch["uv_target"])
                    uv_pred_mm = criterion.normalizer.denormalize(uv_pred)
                if use_blaze_auxiliary:
                    if not isinstance(model_output, Mapping):
                        raise TypeError(
                            "Blaze auxiliary losses require model mapping outputs."
                        )
                    if blaze_reconstruction_weight > 0:
                        reconstruction = model_output.get(
                            "reconstructed_eye_strip"
                        )
                        reconstruction_target = model_output.get(
                            "reconstruction_target_eye_strip"
                        )
                        if not torch.is_tensor(reconstruction) or not torch.is_tensor(
                            reconstruction_target
                        ):
                            raise RuntimeError(
                                "Blaze reconstruction requires decoder output "
                                "and a canonical eye-strip target."
                            )
                        reconstruction_loss = torch.nn.functional.mse_loss(
                            reconstruction.float(),
                            reconstruction_target.float(),
                        )
                        loss = (
                            loss
                            + blaze_reconstruction_weight
                            * reconstruction_loss
                        )
                        blaze_auxiliary_metric_sums[
                            "blaze_reconstruction_mse"
                        ] += float(reconstruction_loss.detach()) * int(
                            reconstruction.shape[0]
                        )
                        blaze_reconstruction_sample_count += int(
                            reconstruction.shape[0]
                        )
                    if blaze_consistency_weight > 0:
                        latent = model_output.get("blaze_eye_latent")
                        uv_targets = device_batch.get("uv_gt")
                        if not torch.is_tensor(latent) or not torch.is_tensor(
                            uv_targets
                        ):
                            raise RuntimeError(
                                "Blaze consistency requires blaze_eye_latent "
                                "and batch UV targets."
                            )
                        consistency_loss = blaze_embedding_consistency_loss(
                            latent,
                            uv_targets,
                        )
                        loss = (
                            loss
                            + blaze_consistency_weight * consistency_loss
                        )
                        blaze_auxiliary_metric_sums[
                            "blaze_embedding_consistency"
                        ] += float(consistency_loss.detach()) * batch_size
                if use_eye_keypoint_auxiliary:
                    if not isinstance(model_output, Mapping):
                        raise TypeError(
                            "Probabilistic eye-keypoint supervision requires "
                            "model mapping outputs."
                        )
                    target_xy = device_batch.get("eye_pseudo_landmarks_xy")
                    valid_mask = device_batch.get(
                        "eye_pseudo_landmark_valid_mask"
                    )
                    eye_quality = device_batch.get(
                        "eye_pseudo_effective_quality"
                    )
                    teacher_quality = device_batch.get(
                        "eye_pseudo_eye_quality"
                    )
                    eye_valid_mask = device_batch.get(
                        "eye_pseudo_eye_valid_mask"
                    )
                    if not all(
                        torch.is_tensor(value)
                        for value in (
                            target_xy,
                            valid_mask,
                            eye_quality,
                            teacher_quality,
                            eye_valid_mask,
                        )
                    ):
                        raise RuntimeError(
                            "Probabilistic eye-keypoint supervision requires "
                            "landmarks, post-augmentation masks, and quality."
                        )
                    predicted_mean_xy = stack_binocular_outputs(
                        model_output, "mean_xy"
                    )
                    heatmap_logits = stack_binocular_outputs(
                        model_output, "logits"
                    )
                    visibility_logits = stack_binocular_outputs(
                        model_output, "visibility_logits"
                    )
                    supervision_outputs = (
                        probabilistic_landmark_supervision_losses(
                            predicted_mean_xy=predicted_mean_xy,
                            heatmap_logits=heatmap_logits,
                            visibility_logits=visibility_logits,
                            target_xy=target_xy,
                            valid_mask=valid_mask,
                            teacher_quality=eye_quality,
                            visibility_supervision_weight=(
                                eye_valid_mask.float()
                                * teacher_quality.float()
                            ),
                            huber_delta_norm=eye_landmark_huber_delta_norm,
                            heatmap_sigma_min_px=(
                                eye_landmark_heatmap_sigma_min_px
                            ),
                            heatmap_sigma_max_px=(
                                eye_landmark_heatmap_sigma_max_px
                            ),
                        )
                    )
                    keypoint_outputs = probabilistic_eye_keypoint_nll(
                        model_output["left_eye_keypoint_mean_xy"],
                        model_output["right_eye_keypoint_mean_xy"],
                        model_output["left_eye_keypoint_covariance"],
                        model_output["right_eye_keypoint_covariance"],
                        target_xy,
                        valid_mask,
                        eye_quality,
                        min_std_norm=eye_keypoint_min_std_norm,
                    )
                    keypoint_loss = keypoint_outputs["loss"]
                    mirror_loss = mirror_shape_consistency_loss(
                        model_output["left_eye_keypoint_shape_embedding"],
                        model_output["right_eye_keypoint_shape_embedding"],
                        eye_valid_mask.float() * teacher_quality.float(),
                    )
                    left_eye_image = device_batch.get("left_eye")
                    right_eye_image = device_batch.get("right_eye")
                    if not torch.is_tensor(left_eye_image) or not torch.is_tensor(
                        right_eye_image
                    ):
                        raise RuntimeError(
                            "Eye-landmark equivariance requires left/right eye images."
                        )
                    equivariance_enabled = (
                        eye_landmark_equivariance_weight > 0
                    )
                    transform1 = sample_eye_affine_matrices(
                        batch_size,
                        device=device,
                        max_rotation_deg=(
                            eye_landmark_equivariance_rotation_deg
                            if equivariance_enabled
                            else 0.0
                        ),
                        max_translation_norm=(
                            eye_landmark_equivariance_translation_norm
                            if equivariance_enabled
                            else 0.0
                        ),
                        scale_min=(
                            eye_landmark_equivariance_scale_min
                            if equivariance_enabled
                            else 1.0
                        ),
                        scale_max=(
                            eye_landmark_equivariance_scale_max
                            if equivariance_enabled
                            else 1.0
                        ),
                        deterministic_sign=None if is_train else -1,
                    )
                    transform2 = sample_eye_affine_matrices(
                        batch_size,
                        device=device,
                        max_rotation_deg=(
                            eye_landmark_equivariance_rotation_deg
                            if equivariance_enabled
                            else 0.0
                        ),
                        max_translation_norm=(
                            eye_landmark_equivariance_translation_norm
                            if equivariance_enabled
                            else 0.0
                        ),
                        scale_min=(
                            eye_landmark_equivariance_scale_min
                            if equivariance_enabled
                            else 1.0
                        ),
                        scale_max=(
                            eye_landmark_equivariance_scale_max
                            if equivariance_enabled
                            else 1.0
                        ),
                        deterministic_sign=None if is_train else 1,
                    )
                    predict_eye_landmarks = getattr(
                        model, "predict_eye_landmarks", None
                    )
                    if not callable(predict_eye_landmarks) and equivariance_enabled:
                        raise RuntimeError(
                            "The model must expose predict_eye_landmarks for "
                            "equivariant supervision."
                        )
                    if equivariance_enabled and callable(predict_eye_landmarks):
                        augmented_output = predict_eye_landmarks(
                            torch.cat(
                                (
                                    warp_eye_images(left_eye_image, transform1),
                                    warp_eye_images(left_eye_image, transform2),
                                ),
                                dim=0,
                            ),
                            torch.cat(
                                (
                                    warp_eye_images(right_eye_image, transform1),
                                    warp_eye_images(right_eye_image, transform2),
                                ),
                                dim=0,
                            ),
                        )
                        view1_output = {
                            name: value[:batch_size]
                            for name, value in augmented_output.items()
                        }
                        view2_output = {
                            name: value[batch_size:]
                            for name, value in augmented_output.items()
                        }
                    else:
                        view1_output = {
                            "left_eye_keypoint_mean_xy": predicted_mean_xy[:, 0],
                            "right_eye_keypoint_mean_xy": predicted_mean_xy[:, 1],
                        }
                        view2_output = view1_output
                    equivariance_outputs = (
                        equivariant_landmark_consistency_loss(
                            stack_binocular_outputs(view1_output, "mean_xy"),
                            stack_binocular_outputs(view2_output, "mean_xy"),
                            transform1,
                            transform2,
                            eye_valid_mask.float() * teacher_quality.float(),
                            huber_delta_norm=eye_landmark_huber_delta_norm,
                        )
                    )
                    equivariance_loss = equivariance_outputs["loss"]
                    loss = (
                        loss
                        + eye_keypoint_nll_weight * keypoint_loss
                        + eye_landmark_coordinate_weight
                        * supervision_outputs["coordinate_loss"]
                        + eye_landmark_heatmap_weight
                        * supervision_outputs["heatmap_loss"]
                        + eye_landmark_visibility_weight
                        * supervision_outputs["visibility_loss"]
                        + eye_landmark_equivariance_weight * equivariance_loss
                        + eye_landmark_mirror_shape_weight * mirror_loss
                    )
                    eye_keypoint_metric_sums["eye_keypoint_nll"] += (
                        float(keypoint_loss.detach()) * batch_size
                    )
                    eye_keypoint_metric_sums[
                        "eye_keypoint_mean_std_norm"
                    ] += (
                        float(keypoint_outputs["mean_std_norm"].detach())
                        * batch_size
                    )
                    eye_keypoint_metric_sums[
                        "eye_keypoint_effective_weight_per_sample"
                    ] += float(keypoint_outputs["effective_weight"].detach())
                    for metric_name, output_name in (
                        ("eye_landmark_coordinate_loss", "coordinate_loss"),
                        ("eye_landmark_heatmap_loss", "heatmap_loss"),
                        ("eye_landmark_visibility_loss", "visibility_loss"),
                    ):
                        eye_keypoint_metric_sums[metric_name] += (
                            float(supervision_outputs[output_name].detach())
                            * batch_size
                        )
                    eye_keypoint_metric_sums[
                        "eye_landmark_equivariance_loss"
                    ] += float(equivariance_loss.detach()) * batch_size
                    eye_keypoint_metric_sums[
                        "eye_landmark_mirror_shape_loss"
                    ] += float(mirror_loss.detach()) * batch_size
                    eye_keypoint_metric_sums[
                        "eye_landmark_equivariance_error_norm"
                    ] += (
                        float(
                            equivariance_outputs[
                                "mean_error_norm"
                            ].detach()
                        )
                        * batch_size
                    )
                    eye_landmark_predictions.append(
                        predicted_mean_xy.detach().float().cpu()
                    )
                    eye_landmark_targets.append(target_xy.detach().float().cpu())
                    eye_landmark_valid_masks.append(
                        valid_mask.detach().float().cpu()
                    )
                    eye_landmark_teacher_qualities.append(
                        teacher_quality.detach().float().cpu()
                    )
                    eye_landmark_image_size = (
                        int(left_eye_image.shape[-1]),
                        int(left_eye_image.shape[-2]),
                    )
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
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip_norm,
                    )
                    grad_norm_value = float(grad_norm)
                    clip_coefficient = min(
                        1.0,
                        grad_clip_norm / (grad_norm_value + 1e-6),
                    )
                    grad_norm_pre_clip_sum += grad_norm_value
                    grad_norm_pre_clip_max = max(
                        grad_norm_pre_clip_max,
                        grad_norm_value,
                    )
                    grad_clip_coefficient_sum += clip_coefficient
                    grad_clipped_steps += int(clip_coefficient < 1.0)
                    gradient_diagnostic_steps += 1
                scale_before_step = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                if scaler.is_enabled() and float(scaler.get_scale()) < scale_before_step:
                    optimizer_skipped_steps += 1
                else:
                    optimizer_steps += 1
                    if ema is not None:
                        ema.update(model)
                    if scheduler is not None:
                        scheduler.step()

        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
        predictions.append(uv_pred_mm.detach().cpu())
        targets_mm.append(metric_target_mm.detach().cpu())

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
        if gradient_diagnostic_steps > 0:
            metrics["grad_norm_pre_clip_mean"] = (
                grad_norm_pre_clip_sum / gradient_diagnostic_steps
            )
            metrics["grad_norm_pre_clip_max"] = grad_norm_pre_clip_max
            metrics["grad_clip_coefficient_mean"] = (
                grad_clip_coefficient_sum / gradient_diagnostic_steps
            )
            metrics["grad_clipped_step_fraction"] = (
                grad_clipped_steps / gradient_diagnostic_steps
            )
        else:
            metrics["grad_norm_pre_clip_mean"] = float("nan")
            metrics["grad_norm_pre_clip_max"] = float("nan")
            metrics["grad_clip_coefficient_mean"] = 1.0
            metrics["grad_clipped_step_fraction"] = 0.0
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
    if use_eye_keypoint_auxiliary:
        metrics.update(
            {
                name: total / sample_count
                for name, total in eye_keypoint_metric_sums.items()
            }
        )
        metrics["weighted_eye_keypoint_nll"] = (
            eye_keypoint_nll_weight * metrics["eye_keypoint_nll"]
        )
        weight_by_metric = {
            "eye_landmark_coordinate_loss": eye_landmark_coordinate_weight,
            "eye_landmark_heatmap_loss": eye_landmark_heatmap_weight,
            "eye_landmark_visibility_loss": eye_landmark_visibility_weight,
            "eye_landmark_equivariance_loss": (
                eye_landmark_equivariance_weight
            ),
            "eye_landmark_mirror_shape_loss": (
                eye_landmark_mirror_shape_weight
            ),
        }
        for metric_name, metric_weight in weight_by_metric.items():
            metrics[f"weighted_{metric_name}"] = (
                metric_weight * metrics[metric_name]
            )
        if eye_landmark_image_size is None:
            raise RuntimeError("Eye-landmark validation received no image size.")
        validation_metrics = eye_landmark_validation_metrics(
            torch.cat(eye_landmark_predictions),
            torch.cat(eye_landmark_targets),
            torch.cat(eye_landmark_valid_masks),
            torch.cat(eye_landmark_teacher_qualities),
            eye_image_size=eye_landmark_image_size,
        )
        metrics.update(
            {name: float(value) for name, value in validation_metrics.items()}
        )
    if blaze_reconstruction_weight > 0:
        if blaze_reconstruction_sample_count <= 0:
            raise RuntimeError(
                "Blaze reconstruction was enabled but decoded no samples."
            )
        reconstruction_metric = (
            blaze_auxiliary_metric_sums["blaze_reconstruction_mse"]
            / blaze_reconstruction_sample_count
        )
        metrics["blaze_reconstruction_mse"] = reconstruction_metric
        metrics["weighted_blaze_reconstruction_mse"] = (
            blaze_reconstruction_weight * reconstruction_metric
        )
    if blaze_consistency_weight > 0:
        consistency_metric = (
            blaze_auxiliary_metric_sums["blaze_embedding_consistency"]
            / sample_count
        )
        metrics["blaze_embedding_consistency"] = consistency_metric
        metrics["weighted_blaze_embedding_consistency"] = (
            blaze_consistency_weight * consistency_metric
        )
    if use_landmark_guided_eye_fusion:
        metrics.update(
            {
                "landmark_guided_global_gate_mean": (
                    landmark_guided_global_gate_sum / sample_count
                ),
                "landmark_guided_applied_gate_abs_mean": (
                    landmark_guided_applied_gate_abs_sum / sample_count
                ),
                "landmark_guided_quality_mean": (
                    landmark_guided_quality_sum / sample_count
                ),
                "landmark_guided_delta_norm_mean": (
                    landmark_guided_delta_norm_sum / sample_count
                ),
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
    ema: ExponentialMovingAverage | None,
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
        "ema": ema.state_dict() if ema is not None else None,
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
    ema: ExponentialMovingAverage | None,
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
    if ema is not None:
        saved_ema_state = checkpoint.get("ema")
        if isinstance(saved_ema_state, Mapping):
            ema.load_state_dict(saved_ema_state)
        else:
            ema.reset_from_model(
                model,
                optimization_steps=int(checkpoint["global_step"]),
            )
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
    optimizer_group_lrs: Mapping[str, float],
    ema_updates: int,
    val_uses_ema: bool,
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
        "ema_updates": ema_updates,
        "val_uses_ema": int(val_uses_ema),
    }
    record.update(
        {f"lr_{name}": value for name, value in optimizer_group_lrs.items()}
    )
    for prefix, values in (("train", train), ("val", val)):
        record[f"{prefix}_loss"] = values["loss"]
        for name in metric_names:
            record[f"{prefix}_{name}"] = values[name]
        record[f"{prefix}_seconds"] = values["seconds"]
        record[f"{prefix}_samples_per_second"] = values["samples_per_second"]
    for name in GRADIENT_DIAGNOSTIC_FIELDS:
        record[name] = train[name[len("train_"):]]
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
    for name, value in record.items():
        if name.startswith("lr_"):
            metrics[f"learning_rate/{name[len('lr_'):]}"] = value
    for name in GRADIENT_DIAGNOSTIC_FIELDS:
        metrics[f"optimization/{name[len('train_'):]}"] = record[name]
    metrics["ema/updates"] = record["ema_updates"]
    metrics["ema/validation_uses_ema"] = record["val_uses_ema"]
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
    amp_enabled, amp_dtype_name, amp_dtype = resolve_amp_settings(
        config["training"],
        device,
    )
    data_config = config["data"]
    model_architecture = canonical_model_architecture(
        config["model"].get("architecture", "modelv1")
    )
    model_config = make_model_config(config["model"])
    eye_augmentation = make_eye_appearance_augmentation(data_config)
    face_augmentation = make_face_appearance_augmentation(data_config)
    active_metric_names = metric_names_for_model(
        model_config,
        config["loss"],
    )
    uses_gaze_geometry = (
        model_config.prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY
    )
    direct_uv_target_frame = canonical_direct_uv_target_frame(
        data_config.get(
            "direct_uv_target_frame",
            DIRECT_UV_TARGET_FRAME_TABLE_LOCAL,
        )
    )
    uses_table7_stage3 = data_config.get("table7_stage3_csv_paths") is not None
    preprocessed_v2_virtual_manifests: list[Path] = []
    if uses_table7_stage3:
        dataset_csv_path = resolve_table7_stage3_artifacts(
            data_config,
            run_dir,
        )
        depth_prior_csv_path = None
    elif data_config.get("preprocessed_v2_root") is not None:
        (
            dataset_csv_path,
            depth_prior_csv_path,
            preprocessed_v2_virtual_manifests,
        ) = resolve_preprocessed_v2_artifacts(
            data_config,
            run_dir,
            require_depth_prior=model_config.use_eye_geometry or uses_gaze_geometry,
        )
    else:
        dataset_csv_path, depth_prior_csv_path = (
            resolve_numbered_dataset_artifacts(
                data_config,
                run_dir,
                require_depth_prior=model_config.use_eye_geometry or uses_gaze_geometry,
            )
        )
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
        fallback_paths=(
            preprocessed_v2_virtual_manifests
            if canonical_image_source(
                data_config.get("image_source", IMAGE_SOURCE_LEGACY)
            )
            == IMAGE_SOURCE_VIRTUAL_CAMERA
            else None
        ),
    )
    sample_id_filter_manifest_path = resolve_sample_id_filter_manifest_path(
        data_config,
        run_dir,
        fallback_paths=(
            preprocessed_v2_virtual_manifests
            if canonical_image_source(
                data_config.get("image_source", IMAGE_SOURCE_LEGACY)
            )
            == IMAGE_SOURCE_LEGACY
            else None
        ),
    )
    allowed_sample_ids = (
        frozenset(
            VirtualCameraManifest.load(sample_id_filter_manifest_path).records
        )
        if sample_id_filter_manifest_path is not None
        else None
    )
    eye_geometry_pseudo_label_paths = None
    raw_eye_geometry_paths = data_config.get("eye_geometry_pseudo_label_paths")
    if raw_eye_geometry_paths:
        eye_geometry_pseudo_label_paths = [
            resolve_project_path(path) for path in raw_eye_geometry_paths
        ]
        missing_eye_geometry_paths = [
            path for path in eye_geometry_pseudo_label_paths if not path.is_file()
        ]
        if missing_eye_geometry_paths:
            raise FileNotFoundError(
                "Eye-geometry pseudo-label NPZ files do not exist: "
                f"{missing_eye_geometry_paths}"
            )
    train_loader, val_loader = build_modelv1_dataloaders(
        csv_path=dataset_csv_path,
        train_datasets=data_config.get("train_datasets", ("3", "4")),
        val_datasets=data_config.get("val_datasets", ("5",)),
        split_mode=str(data_config.get("split_mode", "explicit_datasets")),
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"]),
        persistent_workers=bool(data_config.get("persistent_workers", False)),
        normalize_images=bool(data_config["normalize_images"]),
        train_paired_eye_transform=eye_augmentation,
        train_face_transform=face_augmentation,
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
        image_source=str(
            data_config.get("image_source", IMAGE_SOURCE_LEGACY)
        ),
        virtual_camera_manifest_path=virtual_camera_manifest_path,
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
        precomputed_table_frame7=uses_table7_stage3,
        eye_geometry_pseudo_label_paths=eye_geometry_pseudo_label_paths,
        require_eye_geometry_pseudo_labels=bool(
            data_config.get("require_eye_geometry_pseudo_labels", False)
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

    model = build_experiment_model(model_architecture, model_config).to(device)
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
    blaze_auxiliary_loss_kwargs = {
        "blaze_reconstruction_weight": float(
            loss_config.pop("blaze_reconstruction_weight", 0.0)
        ),
        "blaze_consistency_weight": float(
            loss_config.pop("blaze_consistency_weight", 0.0)
        ),
    }
    gate_regularization_weight = float(
        loss_config.pop("gate_regularization_weight", 0.0)
    )
    eye_keypoint_nll_weight = float(
        loss_config.pop("eye_keypoint_nll_weight", 0.0)
    )
    eye_keypoint_min_std_norm = float(
        loss_config.pop("eye_keypoint_min_std_norm", 0.01)
    )
    eye_landmark_loss_kwargs = {
        "eye_landmark_coordinate_weight": float(
            loss_config.pop("eye_landmark_coordinate_weight", 0.0)
        ),
        "eye_landmark_heatmap_weight": float(
            loss_config.pop("eye_landmark_heatmap_weight", 0.0)
        ),
        "eye_landmark_visibility_weight": float(
            loss_config.pop("eye_landmark_visibility_weight", 0.0)
        ),
        "eye_landmark_equivariance_weight": float(
            loss_config.pop("eye_landmark_equivariance_weight", 0.0)
        ),
        "eye_landmark_mirror_shape_weight": float(
            loss_config.pop("eye_landmark_mirror_shape_weight", 0.0)
        ),
        "eye_landmark_huber_delta_norm": float(
            loss_config.pop("eye_landmark_huber_delta_norm", 0.02)
        ),
        "eye_landmark_heatmap_sigma_min_px": float(
            loss_config.pop("eye_landmark_heatmap_sigma_min_px", 1.0)
        ),
        "eye_landmark_heatmap_sigma_max_px": float(
            loss_config.pop("eye_landmark_heatmap_sigma_max_px", 3.0)
        ),
        "eye_landmark_equivariance_rotation_deg": float(
            loss_config.pop("eye_landmark_equivariance_rotation_deg", 4.0)
        ),
        "eye_landmark_equivariance_translation_norm": float(
            loss_config.pop(
                "eye_landmark_equivariance_translation_norm", 0.03
            )
        ),
        "eye_landmark_equivariance_scale_min": float(
            loss_config.pop("eye_landmark_equivariance_scale_min", 0.97)
        ),
        "eye_landmark_equivariance_scale_max": float(
            loss_config.pop("eye_landmark_equivariance_scale_max", 1.03)
        ),
    }
    if (
        model_config.eye_geometry_gate_mode
        != EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
    ):
        gate_regularization_weight = 0.0
    if not model_config.use_eye_keypoint_auxiliary:
        eye_keypoint_nll_weight = 0.0
        for weight_name in (
            "eye_landmark_coordinate_weight",
            "eye_landmark_heatmap_weight",
            "eye_landmark_visibility_weight",
            "eye_landmark_equivariance_weight",
            "eye_landmark_mirror_shape_weight",
        ):
            eye_landmark_loss_kwargs[weight_name] = 0.0
    if uses_gaze_geometry:
        criterion: nn.Module = GazeGeometryLoss(
            GazeGeometryLossConfig(**loss_config)
        )
        gate_regularization_weight = 0.0
    else:
        assert normalizer is not None
        uv_loss_config = UVLossConfig(**loss_config)
        if direct_uv_target_frame == DIRECT_UV_TARGET_FRAME_VIRTUAL_CAMERA:
            criterion = DirectVirtualTableUVLoss(
                normalizer,
                uv_loss_config,
                table_distance_scale_mm=model_config.table_distance_scale_mm,
            )
        else:
            criterion = DirectTableUVLoss(normalizer, uv_loss_config)
    optimizer = build_training_optimizer(
        model,
        config["training"],
        device,
    )
    scheduler = build_training_scheduler(
        optimizer,
        config["training"],
        steps_per_epoch=len(train_loader),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16
    )
    ema_config = config["training"].get("ema", {})
    ema_enabled = bool(ema_config.get("enabled", False))
    ema_decay = float(ema_config.get("decay", 0.99))
    ema_start_after_warmup = bool(
        ema_config.get("start_after_warmup", True)
    )
    ema_update_after_step = (
        int(config["training"]["scheduler"].get("warmup_epochs", 0))
        * len(train_loader)
        if ema_start_after_warmup
        else 0
    )
    ema = (
        ExponentialMovingAverage(
            model,
            decay=ema_decay,
            update_after_step=ema_update_after_step,
        )
        if ema_enabled
        else None
    )

    start_epoch = 1
    global_step = 0
    best_val_epe_mm = float("inf")
    if resume_path is not None:
        start_epoch, global_step, best_val_epe_mm = load_checkpoint(
            resume_path,
            model=model,
            ema=ema,
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
    if model_architecture == MODEL_ARCHITECTURE_BLAZE_STYLE:
        face_backbone_name = "blaze_style"
        face_pretrained_name = "none"
        eye_backbone_name = "blaze_style_shared"
        eye_backbone_weights_name = "none"
        visual_topology_name = "blaze_multibranch"
        eye_token_grid_name = "2x4_per_eye"
    else:
        face_backbone_name = FACE_IMAGE_BACKBONE
        face_pretrained_name = FACE_IMAGE_PRETRAINED_DATASET
        eye_backbone_name = model_config.eye_backbone
        eye_backbone_weights_name = model_config.eye_backbone_weights
        visual_topology_name = (
            f"cross_attention_{model_config.visual_attention_heads}x"
            f"{model_config.visual_attention_dim}"
        )
        eye_token_grid_name = "7x12"
    setup_text = (
        f"run_dir={run_dir} device={device} amp={amp_enabled} "
        f"architecture={model_architecture} "
        f"amp_dtype={amp_dtype_name if amp_enabled else 'disabled'} "
        f"grad_scaler={scaler.is_enabled()} "
        f"fused_adamw={bool(optimizer.defaults.get('fused', False))} "
        f"ema={ema is not None} "
        f"ema_decay={ema_decay if ema is not None else 'disabled'} "
        f"ema_update_after_step="
        f"{ema_update_after_step if ema is not None else 'disabled'} "
        f"num_workers={int(data_config['num_workers'])} "
        f"persistent_workers="
        f"{bool(train_loader.persistent_workers)} "
        f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"use_face_image={model_config.use_face_image} "
        f"face_image_backbone={face_backbone_name} "
        f"face_image_pretrained={face_pretrained_name} "
        f"freeze_face_image_backbone={model_config.freeze_face_image_backbone} "
        f"visual_topology={visual_topology_name} "
        f"use_eye_keypoint_auxiliary={model_config.use_eye_keypoint_auxiliary} "
        f"eye_keypoint_count={model_config.eye_keypoint_count} "
        f"eye_keypoint_features=layer2+layer3 "
        f"eye_keypoint_heatmap_size={model_config.eye_keypoint_heatmap_size} "
        f"eye_keypoint_nll_weight={eye_keypoint_nll_weight} "
        f"eye_keypoint_min_std_norm={eye_keypoint_min_std_norm} "
        f"eye_landmark_losses={eye_landmark_loss_kwargs} "
        f"use_landmark_guided_eye_fusion="
        f"{model_config.use_landmark_guided_eye_fusion} "
        f"use_reconstruction_auxiliary="
        f"{getattr(model_config, 'use_reconstruction_auxiliary', False)} "
        f"reconstruction_sample_limit="
        f"{getattr(model_config, 'reconstruction_sample_limit', None)} "
        f"blaze_auxiliary_losses={blaze_auxiliary_loss_kwargs} "
        f"landmark_guided_fusion_heads="
        f"{model_config.landmark_guided_fusion_heads} "
        f"landmark_guided_fusion_gate_init="
        f"{model_config.landmark_guided_fusion_gate_init} "
        f"eye_backbone={eye_backbone_name} "
        f"eye_token_grid={eye_token_grid_name} "
        f"eye_backbone_weights={eye_backbone_weights_name} "
        f"eye_image_size={FINALIZED_EYE_IMAGE_SIZE} "
        f"eye_augmentation={eye_augmentation is not None} "
        f"face_augmentation={face_augmentation is not None} "
        f"image_source={data_config.get('image_source', IMAGE_SOURCE_LEGACY)} "
        f"eye_image_source={FINALIZED_EYE_IMAGE_SOURCE} "
        f"direct_uv_target_frame="
        f"{data_config.get('direct_uv_target_frame', DIRECT_UV_TARGET_FRAME_TABLE_LOCAL)} "
        f"gaze_prediction_frame={model_config.gaze_prediction_frame} "
        f"use_table_frame_film={model_config.use_table_frame_film} "
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
    optimizer_group_names = tuple(
        optimizer_group_name(group, index)
        for index, group in enumerate(optimizer.param_groups)
    )
    optimizer_group_counts = optimizer_group_parameter_counts(optimizer)
    optimizer_group_text = ", ".join(
        f"{name}: lr={lr:.3e}, parameters={optimizer_group_counts[name]:,}"
        for name, lr in optimizer_group_learning_rates(optimizer).items()
    )
    logger.info("optimizer parameter groups: %s", optimizer_group_text)
    print(f"optimizer parameter groups: {optimizer_group_text}")

    tracker = start_tracker(
        config,
        run_dir,
        active_metric_names,
        static_metadata={
            "architecture": model_architecture,
            "use_face_image": model_config.use_face_image,
            "face_image_backbone": face_backbone_name,
            "face_image_pretrained": face_pretrained_name,
            "freeze_face_image_backbone": (
                model_config.freeze_face_image_backbone
            ),
            "visual_topology": visual_topology_name,
            "use_eye_keypoint_auxiliary": (
                model_config.use_eye_keypoint_auxiliary
            ),
            "use_reconstruction_auxiliary": getattr(
                model_config,
                "use_reconstruction_auxiliary",
                False,
            ),
            "reconstruction_sample_limit": getattr(
                model_config,
                "reconstruction_sample_limit",
                None,
            ),
            **blaze_auxiliary_loss_kwargs,
            "eye_keypoint_count": model_config.eye_keypoint_count,
            "eye_keypoint_auxiliary_feature_stage": (
                model_config.eye_keypoint_auxiliary_feature_stage
            ),
            "eye_keypoint_nll_weight": eye_keypoint_nll_weight,
            "eye_keypoint_min_std_norm": eye_keypoint_min_std_norm,
            "eye_backbone": eye_backbone_name,
            "eye_backbone_weights": eye_backbone_weights_name,
            "eye_augmentation": eye_augmentation is not None,
            "face_augmentation": face_augmentation is not None,
            "amp": amp_enabled,
            "amp_dtype": amp_dtype_name if amp_enabled else "disabled",
            "grad_scaler": scaler.is_enabled(),
            "fused_adamw": bool(optimizer.defaults.get("fused", False)),
            "ema": ema is not None,
            "ema_decay": ema_decay if ema is not None else None,
            "ema_update_after_step": (
                ema_update_after_step if ema is not None else None
            ),
            "num_workers": int(data_config["num_workers"]),
            "persistent_workers": bool(train_loader.persistent_workers),
            "image_source": data_config.get(
                "image_source", IMAGE_SOURCE_LEGACY
            ),
            "eye_image_source": FINALIZED_EYE_IMAGE_SOURCE,
            "eye_image_size": FINALIZED_EYE_IMAGE_SIZE,
            "direct_uv_target_frame": data_config.get(
                "direct_uv_target_frame",
                DIRECT_UV_TARGET_FRAME_TABLE_LOCAL,
            ),
            "gaze_prediction_frame": model_config.gaze_prediction_frame,
            "use_table_frame_film": model_config.use_table_frame_film,
            "parameters_total": total_parameter_count,
            "parameters_trainable": trainable_parameter_count,
        },
    )
    metrics_writer = MetricsCsvWriter(
        run_dir / "metrics.csv",
        active_metric_names,
        optimizer_group_names,
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
            optimizer_group_lrs = optimizer_group_learning_rates(optimizer)
            lr = next(iter(optimizer_group_lrs.values()))
            train_metrics, train_samples = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scheduler=(
                    scheduler
                    if scheduler_steps_per_optimizer_step(config["training"])
                    else None
                ),
                scaler=scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                grad_clip_norm=grad_clip_norm,
                gate_regularization_weight=gate_regularization_weight,
                eye_keypoint_nll_weight=eye_keypoint_nll_weight,
                eye_keypoint_min_std_norm=eye_keypoint_min_std_norm,
                ema=ema,
                **blaze_auxiliary_loss_kwargs,
                **eye_landmark_loss_kwargs,
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
            validation_uses_ema = ema is not None and ema.is_active
            validation_model = ema.module if validation_uses_ema else model
            val_metrics, _ = run_epoch(
                model=validation_model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scheduler=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                grad_clip_norm=None,
                gate_regularization_weight=0.0,
                eye_keypoint_nll_weight=eye_keypoint_nll_weight,
                eye_keypoint_min_std_norm=eye_keypoint_min_std_norm,
                **blaze_auxiliary_loss_kwargs,
                **eye_landmark_loss_kwargs,
            )
            if not scheduler_steps_per_optimizer_step(config["training"]):
                scheduler.step()

            improved = val_metrics["epe_mm"] < best_val_epe_mm
            if improved:
                best_val_epe_mm = val_metrics["epe_mm"]
            record = build_epoch_record(
                epoch=epoch,
                global_step=global_step,
                lr=lr,
                optimizer_group_lrs=optimizer_group_lrs,
                ema_updates=ema.num_updates if ema is not None else 0,
                val_uses_ema=validation_uses_ema,
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
                    ema=ema,
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
                        ema=ema,
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
                group_lr_text = ", ".join(
                    f"{name}={value:.3e}"
                    for name, value in optimizer_group_lrs.items()
                )
                print(
                    f"epoch {epoch:03d}/{epochs} | lr=[{group_lr_text}] | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_weights={'ema' if validation_uses_ema else 'raw'} "
                    f"(updates={ema.num_updates if ema is not None else 0}) | "
                    # f"grad_norm={train_metrics['grad_norm_pre_clip_mean']:.3f} "
                    # f"(max={train_metrics['grad_norm_pre_clip_max']:.3f}) | "
                    # f"clip_coef={train_metrics['grad_clip_coefficient_mean']:.3f} | "
                    # f"clipped={train_metrics['grad_clipped_step_fraction']:.1%} | "
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
