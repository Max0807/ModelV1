from __future__ import annotations

from pathlib import Path

import yaml

from modelv1.preprocessed_v2 import DEFAULT_TRAINING_DEPTH_METHOD
from scripts.train_modelv1 import load_config


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "modelv1"
    / "train_dataset11_val_dataset3_direct_uv_virtual_camera_film_v2.yaml"
)


def test_finalized_config_inherits_stable_direct_uv_defaults() -> None:
    source = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert source["defaults"] == "defaults_direct_uv.yaml"
    for key in (
        "require_eye_geometry_pseudo_labels",
        "direct_uv_target_frame",
        "skip_invalid_virtual_camera_samples",
        "split_mode",
    ):
        assert key not in source["data"]

    config = load_config(CONFIG_PATH)
    assert config["data"]["require_eye_geometry_pseudo_labels"] is True
    assert config["data"]["direct_uv_target_frame"] == "table_local"
    assert config["data"]["skip_invalid_virtual_camera_samples"] is True
    assert config["data"]["split_mode"] == "explicit_datasets"
    assert "preprocessed_v2_root" not in config["data"]
    assert config["data"]["table7_stage3_csv_paths"] == [
        (
            "data/preprocessed_table7/dataset13/stage3_virtual_table7/"
            "training_table7_dataset13.csv"
        ),
        (
            "data/preprocessed_table7/dataset11/stage3_virtual_table7/"
            "training_table7_dataset11.csv"
        ),
    ]
    assert config["training"]["eye_backbone_schedule"] == {
        "enabled": True,
        "frozen_epochs": 0,
        "unfreeze_from": "stem",
        "lr_multiplier": 0.05,
    }
    assert "face_backbone_schedule" not in config["training"]
    assert config["data"]["num_workers"] == 2
    assert config["data"]["persistent_workers"] is True
    assert config["training"]["amp"] is True
    assert config["training"]["amp_dtype"] == "bfloat16"
    assert config["training"]["optimizer"] == {
        "name": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "fused": True,
    }
    assert config["training"]["scheduler"]["warmup_epochs"] == 3
    assert config["training"]["scheduler"]["warmup_start_factor"] == 0.1
    assert config["training"]["ema"] == {
        "enabled": True,
        "decay": 0.99,
        "start_after_warmup": True,
    }


def test_retired_training_options_are_absent_from_finalized_config() -> None:
    config = load_config(CONFIG_PATH)
    assert DEFAULT_TRAINING_DEPTH_METHOD == "iris65"
    for key in (
        "depth_method",
        "eye_image_source",
        "eye_image_size",
        "iris_supervision_csv_paths",
        "all_datasets",
        "split_seed",
        "val_ratio",
        "load_face_image",
        "deca_face_preprocess",
        "deca_crop_scale",
    ):
        assert key not in config["data"]
    assert "description" not in config["experiment"]
    assert "use_eye_iris_auxiliary" not in config["model"]
    assert "iris_heatmap_weight" not in config["loss"]
    for key in (
        "use_virtual_distance_film",
        "use_low_dof_eye_template",
        "use_pitch_to_v_residual",
        "use_vertical_geometry_residual",
        "face_feature_mode",
        "visual_token_pooling",
        "use_binocular_self_attention",
    ):
        assert key not in config["model"]
    for key in (
        "eye_template_reprojection_weight",
        "eye_template_binocular_pitch_weight",
        "eye_template_pitch_huber_delta_deg",
        "vertical_residual_regularization_weight",
        "vertical_residual_supervision_weight",
        "vertical_residual_huber_beta_mm",
        "mixture_nll_weight",
        "mixture_kernel_sigma_mm",
        "gaze_direction_weight",
        "depth_prior_kl_weight",
        "depth_correction_prior_weight",
    ):
        assert key not in config["loss"]
    assert config["data"]["eye_augmentation"] == "appearance_mild_v1"
    assert config["data"]["face_augmentation"] == "appearance_mild_v1"
