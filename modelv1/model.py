"""Multi-branch ModelV1 gaze regressor.

DECA can be kept outside the training graph as an offline frozen feature branch,
or disabled completely with ``deca_feature_representation='none'``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import models

from modelv1.geometry_gate import (
    EYE_GEOMETRY_GATE_LEARNED_RESIDUAL,
    EYE_GEOMETRY_GATE_NONE,
    canonical_eye_geometry_gate_mode,
)
from modelv1.deca_cache import (
    DECA_FEATURE_REPRESENTATION_FULL236,
    DECA_FEATURE_REPRESENTATION_GEOMETRY156,
    DECA_FEATURE_REPRESENTATION_NONE,
    canonical_deca_feature_representation,
    deca_feature_representation_dim,
)
from modelv1.data.depth_prior import (
    EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
    canonical_eye_geometry_representation,
    eye_geometry_representation_dim,
)
from modelv1.data.eye_geometry_pseudo_labels import EYE_PSEUDO_POINT_COUNT
from modelv1.scene import (
    SCENE_REPRESENTATION_FULL25,
    SCENE_REPRESENTATION_TABLE_FRAME7,
    canonical_scene_representation,
    scene_representation_dim,
)
from modelv1.depth_distribution import (
    DEPTH_DISTRIBUTION_LEARNED_REWEIGHT,
    DepthHypothesisConfig,
    DepthHypothesisReweighter,
    PnpDepthHypothesisGenerator,
    canonical_depth_distribution_mode,
    mixture_uv_statistics,
)
from modelv1.gaze_distribution import tangent_cholesky_from_raw, tangent_sigma_points
from modelv1.geometry import RayTableGeometry, RayTableGeometryConfig
from modelv1.vertical_eye_geometry import (
    LandmarkGuidedEyeFusion,
    LowDOFDifferentiableEyeballTemplate,
    PitchToTableVResidual,
    ProbabilisticEyeLandmarkHead,
    VerticalEyeGeometryFeatureExtractor,
    VERTICAL_EYE_GEOMETRY_FEATURE_DIM,
    VerticalGeometryVResidual,
    stack_binocular_outputs,
)


DEFAULT_DECA_FEATURE_DIM = 236
PREDICTION_MODE_DIRECT_UV = "direct_uv"
PREDICTION_MODE_GAZE_GEOMETRY = "gaze_geometry"
PREDICTION_MODES = (PREDICTION_MODE_DIRECT_UV, PREDICTION_MODE_GAZE_GEOMETRY)
GAZE_PREDICTION_FRAME_CAMERA = "camera"
GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA = "virtual_camera"
GAZE_PREDICTION_FRAMES = (
    GAZE_PREDICTION_FRAME_CAMERA,
    GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA,
)
DECA_BATCH_KEYS = (
    "deca_feat",
    "deca_features",
    "face_deca_feat",
    "face_deca_features",
)
COMPACT_EYE_BACKBONE = "cnn"
DEFAULT_EYE_BACKBONE = "resnet18"
SMALL_IMAGE_RESNET18_BACKBONE = "resnet18_3x3"
FACE_IMAGE_PRETRAINED_DATASET = "vggface2"
FACE_IMAGE_BACKBONE = "inception_resnet_v1"
FACE_IMAGE_FEATURE_CHANNELS = 1792
EYE_IMAGE_FEATURE_CHANNELS = 512
FACE_IMAGE_TOKEN_COUNT = 9
FACE_FEATURE_MODE_NATIVE_3X3 = "native_3x3"
FACE_FEATURE_MODE_PADDED_MIXED7A_4X4 = "padded_mixed7a_4x4"
FACE_FEATURE_MODE_TOKEN_GRIDS = {
    FACE_FEATURE_MODE_NATIVE_3X3: (3, 3),
    FACE_FEATURE_MODE_PADDED_MIXED7A_4X4: (4, 4),
}
DEFAULT_EYE_IMAGE_TOKEN_GRID = (2, 2)
EYE_FEATURE_MODE_SINGLE_STAGE = "single_stage"
EYE_FEATURE_MODE_MULTISCALE_FPN24 = "multiscale_fpn24"
EYE_STEM_MODE_STANDARD = "standard"
EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL = "small_3x3_s2_no_maxpool"
VISUAL_TOKEN_POOLING_MEAN = "mean"
VISUAL_TOKEN_POOLING_LEARNED = "learned"
RESNET_EYE_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    SMALL_IMAGE_RESNET18_BACKBONE: (models.resnet18, models.ResNet18_Weights),
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "resnet101": (models.resnet101, models.ResNet101_Weights),
    "resnet152": (models.resnet152, models.ResNet152_Weights),
}
SUPPORTED_EYE_BACKBONES = (
    COMPACT_EYE_BACKBONE,
    "resnet18",
    SMALL_IMAGE_RESNET18_BACKBONE,
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
)
RESNET_STAGE_NAMES = (
    "none",
    "stem",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
)
NO_WEIGHT_VALUES = {"", "none", "null", "false", "random", "scratch"}
DECA_BRANCH_MODE_FLAT = "flat"
DECA_BRANCH_MODE_FACTORIZED_GEOMETRY = "factorized_geometry"
DECA_BRANCH_MODES = (
    DECA_BRANCH_MODE_FLAT,
    DECA_BRANCH_MODE_FACTORIZED_GEOMETRY,
)
DECA_GEOMETRY_FACTOR_DIMS = {
    "shape": 100,
    "exp": 50,
    "pose": 6,
}


def canonical_gaze_prediction_frame(value: str) -> str:
    frame = str(value).strip().lower()
    if frame not in GAZE_PREDICTION_FRAMES:
        raise ValueError(
            f"Unknown gaze_prediction_frame={value!r}; expected one of "
            f"{GAZE_PREDICTION_FRAMES}."
        )
    return frame


def canonical_deca_branch_mode(value: str) -> str:
    """Validate and normalize the DECA face-branch architecture choice."""

    mode = str(value).strip().lower()
    if mode not in DECA_BRANCH_MODES:
        supported = ", ".join(DECA_BRANCH_MODES)
        raise ValueError(
            f"Unknown deca_branch_mode={value!r}; expected one of: {supported}."
        )
    return mode


@dataclass(frozen=True)
class ModelV1Config:
    """Shape and width configuration for :class:`ModelV1`."""

    deca_feature_dim: int = DEFAULT_DECA_FEATURE_DIM
    deca_feature_representation: str = DECA_FEATURE_REPRESENTATION_FULL236
    deca_branch_mode: str = DECA_BRANCH_MODE_FLAT
    deca_factor_embedding_dim: int = 64
    deca_factor_hidden_dims: tuple[int, ...] = (64,)
    crop_cam_dim: int = 36
    use_crop_cam: bool = True
    scene_dim: int = 25
    scene_representation: str = SCENE_REPRESENTATION_FULL25
    eye_geometry_dim: int = 6
    eye_geometry_representation: str = EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D
    use_eye_geometry: bool = False
    eye_geometry_gate_mode: str = EYE_GEOMETRY_GATE_NONE
    eye_geometry_quality_dim: int = 4
    eye_geometry_gate_hidden_dim: int = 16
    eye_geometry_gate_min: float = 0.05
    eye_geometry_gate_max: float = 0.95
    eye_geometry_gate_delta_max: float = 2.0
    uv_dim: int = 2
    prediction_mode: str = PREDICTION_MODE_DIRECT_UV
    gaze_dim: int = 3
    gaze_prediction_frame: str = GAZE_PREDICTION_FRAME_CAMERA
    use_gaze_uncertainty: bool = False
    gaze_uncertainty_min_std_rad: float = 0.005
    gaze_uncertainty_max_std_rad: float = 0.50
    gaze_uncertainty_init_std_rad: float = 0.02
    depth_distribution_mode: str = "point"
    depth_num_bins: int = 1
    depth_range_sigma: float = 3.0
    depth_log_sigma_min: float = 0.001
    depth_log_sigma_max: float = 0.5
    depth_reweighter_hidden_dims: tuple[int, ...] = (64, 32)
    depth_reweighter_dropout: float = 0.1
    depth_reweighter_detach_context: bool = True
    use_depth_correction: bool = False
    depth_correction_hidden_dims: tuple[int, ...] = (64, 32)
    depth_correction_dropout: float = 0.1
    table_distance_scale_mm: float = 1000.0
    geometry_min_abs_normal_dot_gaze: float = 0.01
    geometry_min_lambda_mm: float = 0.0
    geometry_max_lambda_mm: float = 5000.0

    face_embedding_dim: int = 128
    face_hidden_dims: tuple[int, ...] = (256,)
    use_face_image: bool = False
    freeze_face_image_backbone: bool = False
    face_feature_mode: str = FACE_FEATURE_MODE_NATIVE_3X3
    eye_embedding_dim: int = 128
    per_eye_embedding_dim: int = 96
    eye_backbone: str = DEFAULT_EYE_BACKBONE
    eye_backbone_weights: str | None = None
    visual_embedding_dim: int = 128
    visual_attention_dim: int = 128
    visual_attention_heads: int = 8
    visual_attention_ffn_dim: int = 256
    visual_attention_dropout: float = 0.1
    use_binocular_self_attention: bool = False
    binocular_attention_heads: int = 4
    binocular_attention_dropout: float = 0.1
    use_eye_iris_auxiliary: bool = False
    eye_iris_auxiliary_feature_stage: str = "layer3"
    eye_iris_auxiliary_hidden_channels: int = 128
    use_eye_keypoint_auxiliary: bool = False
    eye_keypoint_count: int = EYE_PSEUDO_POINT_COUNT
    eye_keypoint_auxiliary_feature_stage: str = "layer3"
    eye_keypoint_auxiliary_hidden_channels: int = 128
    eye_keypoint_probability_temperature: float = 1.0
    eye_keypoint_heatmap_size: tuple[int, int] = (28, 45)
    eye_keypoint_predict_variance_correction: bool = True
    eye_keypoint_max_abs_log_std_correction: float = 1.5
    eye_keypoint_shape_embedding_dim: int = 32
    use_landmark_guided_eye_fusion: bool = False
    landmark_guided_fusion_heads: int = 4
    landmark_guided_fusion_dropout: float = 0.1
    landmark_guided_fusion_gate_init: float = 0.0
    landmark_guided_fusion_gate_max: float = 1.0
    landmark_guided_quality_logit_scale: float = 1.0
    landmark_guided_uncertainty_scale_norm: float = 0.25
    landmark_guided_min_quality: float = 0.1
    landmark_guided_detach_probability_for_sampling: bool = True
    landmark_guided_detach_landmark_statistics: bool = True
    landmark_guided_detach_quality: bool = True
    use_low_dof_eye_template: bool = False
    eye_template_image_size: tuple[int, int] = (90, 56)
    eye_template_pitch_range_deg: tuple[float, float] = (-35.0, 35.0)
    eye_template_yaw_range_deg: tuple[float, float] = (-45.0, 45.0)
    eye_template_pitch_bins: int = 29
    eye_template_yaw_bins: int = 31
    eye_template_posterior_temperature: float = 0.25
    eye_template_covariance_floor_px: float = 0.75
    eye_template_min_canthus_distance_px: float = 4.0
    eye_template_min_geometry_confidence: float = 0.1
    eye_template_center_energy_weight: float = 1.0
    eye_template_rim_energy_weight: float = 1.0
    eye_template_angle_prior_weight: float = 0.05
    eye_template_learnable_parameters: bool = True
    eye_template_use_reference_uncertainty: bool = True
    eye_template_use_aperture_visibility: bool = True
    use_pitch_to_v_residual: bool = False
    use_vertical_geometry_residual: bool = False
    vertical_geometry_residual_hidden_dims: tuple[int, ...] = (64, 32)
    vertical_geometry_residual_max_abs_delta_mm: float = 250.0
    vertical_geometry_residual_min_training_gate: float = 0.10
    vertical_geometry_residual_min_inference_confidence: float = 0.10
    pitch_to_v_target_frame: str = "table_local"
    pitch_to_v_hidden_dims: tuple[int, ...] = (32, 16)
    pitch_to_v_max_abs_delta_mm: float = 250.0
    pitch_to_v_pitch_scale_deg: float = 20.0
    pitch_to_v_uncertainty_scale_deg: float = 8.0
    pitch_to_v_disagreement_scale_deg: float = 6.0
    pitch_to_v_min_pitch_std_deg: float = 0.5
    pitch_to_v_min_active_gate: float = 0.05
    pitch_to_v_detach_uncertainty_gate: bool = True
    pitch_to_v_use_camera_table_sensitivity: bool = True
    pitch_to_v_min_abs_table_normal_z: float = 0.1
    visual_token_pooling: str = VISUAL_TOKEN_POOLING_MEAN
    visual_token_pooling_hidden_dim: int = 64
    visual_token_pooling_dropout: float = 0.1
    eye_stem_mode: str = EYE_STEM_MODE_STANDARD
    eye_feature_mode: str = EYE_FEATURE_MODE_SINGLE_STAGE
    eye_feature_stage: str = "layer4"
    eye_token_grid: tuple[int, int] = DEFAULT_EYE_IMAGE_TOKEN_GRID
    crop_cam_embedding_dim: int = 64
    crop_cam_hidden_dims: tuple[int, ...] = (128,)
    scene_embedding_dim: int = 64
    scene_hidden_dims: tuple[int, ...] = (128,)
    fusion_hidden_dims: tuple[int, ...] = (256, 128)
    use_virtual_distance_film: bool = False
    use_virtual_pose_film: bool = False
    virtual_pose_dim: int = 9
    virtual_distance_film_hidden_dims: tuple[int, ...] = (16,)
    virtual_distance_film_embedding_dim: int = 32
    virtual_distance_film_delta_max: float = 0.1

    branch_dropout: float = 0.1
    fusion_dropout: float = 0.2
    share_eye_encoder: bool = True
    detach_deca_features: bool = True

    def __post_init__(self) -> None:
        dims = {
            "deca_feature_dim": self.deca_feature_dim,
            "deca_factor_embedding_dim": self.deca_factor_embedding_dim,
            "crop_cam_dim": self.crop_cam_dim,
            "scene_dim": self.scene_dim,
            "eye_geometry_dim": self.eye_geometry_dim,
            "eye_geometry_quality_dim": self.eye_geometry_quality_dim,
            "eye_geometry_gate_hidden_dim": self.eye_geometry_gate_hidden_dim,
            "uv_dim": self.uv_dim,
            "gaze_dim": self.gaze_dim,
            "face_embedding_dim": self.face_embedding_dim,
            "eye_embedding_dim": self.eye_embedding_dim,
            "per_eye_embedding_dim": self.per_eye_embedding_dim,
            "visual_embedding_dim": self.visual_embedding_dim,
            "visual_attention_dim": self.visual_attention_dim,
            "visual_attention_heads": self.visual_attention_heads,
            "visual_attention_ffn_dim": self.visual_attention_ffn_dim,
            "binocular_attention_heads": self.binocular_attention_heads,
            "eye_iris_auxiliary_hidden_channels": (
                self.eye_iris_auxiliary_hidden_channels
            ),
            "eye_keypoint_count": self.eye_keypoint_count,
            "eye_keypoint_auxiliary_hidden_channels": (
                self.eye_keypoint_auxiliary_hidden_channels
            ),
            "eye_keypoint_shape_embedding_dim": (
                self.eye_keypoint_shape_embedding_dim
            ),
            "landmark_guided_fusion_heads": self.landmark_guided_fusion_heads,
            "eye_template_pitch_bins": self.eye_template_pitch_bins,
            "eye_template_yaw_bins": self.eye_template_yaw_bins,
            "visual_token_pooling_hidden_dim": self.visual_token_pooling_hidden_dim,
            "crop_cam_embedding_dim": self.crop_cam_embedding_dim,
            "scene_embedding_dim": self.scene_embedding_dim,
            "virtual_pose_dim": self.virtual_pose_dim,
            "virtual_distance_film_embedding_dim": (
                self.virtual_distance_film_embedding_dim
            ),
        }
        object.__setattr__(self, "eye_backbone", canonical_eye_backbone(self.eye_backbone))
        prediction_mode = str(self.prediction_mode).strip().lower()
        if prediction_mode not in PREDICTION_MODES:
            raise ValueError(
                f"prediction_mode must be one of {PREDICTION_MODES}, got "
                f"{self.prediction_mode!r}."
            )
        object.__setattr__(self, "prediction_mode", prediction_mode)
        object.__setattr__(
            self,
            "gaze_prediction_frame",
            canonical_gaze_prediction_frame(self.gaze_prediction_frame),
        )
        object.__setattr__(
            self,
            "depth_distribution_mode",
            canonical_depth_distribution_mode(self.depth_distribution_mode),
        )
        object.__setattr__(
            self,
            "deca_feature_representation",
            canonical_deca_feature_representation(self.deca_feature_representation),
        )
        object.__setattr__(
            self,
            "deca_branch_mode",
            canonical_deca_branch_mode(self.deca_branch_mode),
        )
        object.__setattr__(
            self,
            "eye_geometry_gate_mode",
            canonical_eye_geometry_gate_mode(self.eye_geometry_gate_mode),
        )
        object.__setattr__(
            self,
            "eye_geometry_representation",
            canonical_eye_geometry_representation(self.eye_geometry_representation),
        )
        object.__setattr__(
            self,
            "scene_representation",
            canonical_scene_representation(self.scene_representation),
        )
        if (
            self.eye_backbone not in RESNET_EYE_BACKBONES
            and self.eye_backbone_weights is not None
            and str(self.eye_backbone_weights).strip().lower() not in NO_WEIGHT_VALUES
        ):
            raise ValueError(
                "eye_backbone_weights is only supported for torchvision ResNet eye backbones."
            )
        non_positive = [
            name
            for name, value in dims.items()
            if value <= 0 and name != "deca_feature_dim"
        ]
        if non_positive:
            raise ValueError(f"ModelV1Config dimensions must be positive: {non_positive}")
        if not self.fusion_hidden_dims:
            raise ValueError("fusion_hidden_dims must contain at least one layer width.")
        if not isinstance(self.use_virtual_distance_film, bool):
            raise ValueError("use_virtual_distance_film must be a boolean.")
        if not isinstance(self.use_virtual_pose_film, bool):
            raise ValueError("use_virtual_pose_film must be a boolean.")
        if not isinstance(self.use_binocular_self_attention, bool):
            raise ValueError("use_binocular_self_attention must be a boolean.")
        if not isinstance(self.use_eye_iris_auxiliary, bool):
            raise ValueError("use_eye_iris_auxiliary must be a boolean.")
        if not isinstance(self.use_eye_keypoint_auxiliary, bool):
            raise ValueError("use_eye_keypoint_auxiliary must be a boolean.")
        if self.eye_keypoint_count != EYE_PSEUDO_POINT_COUNT:
            raise ValueError(
                "eye_keypoint_count must match the pseudo-label schema: "
                f"{EYE_PSEUDO_POINT_COUNT}."
            )
        if self.eye_keypoint_probability_temperature <= 0:
            raise ValueError(
                "eye_keypoint_probability_temperature must be positive."
            )
        if not isinstance(self.eye_keypoint_predict_variance_correction, bool):
            raise ValueError(
                "eye_keypoint_predict_variance_correction must be a boolean."
            )
        if self.eye_keypoint_max_abs_log_std_correction <= 0:
            raise ValueError(
                "eye_keypoint_max_abs_log_std_correction must be positive."
            )
        if not isinstance(self.use_landmark_guided_eye_fusion, bool):
            raise ValueError(
                "use_landmark_guided_eye_fusion must be a boolean."
            )
        for detach_name in (
            "landmark_guided_detach_probability_for_sampling",
            "landmark_guided_detach_landmark_statistics",
            "landmark_guided_detach_quality",
        ):
            if not isinstance(getattr(self, detach_name), bool):
                raise ValueError(f"{detach_name} must be a boolean.")
        if self.visual_attention_dim % self.landmark_guided_fusion_heads != 0:
            raise ValueError(
                "visual_attention_dim must be divisible by "
                "landmark_guided_fusion_heads."
            )
        if not 0 <= self.landmark_guided_fusion_dropout < 1:
            raise ValueError(
                "landmark_guided_fusion_dropout must lie in [0, 1)."
            )
        if self.landmark_guided_fusion_gate_max <= 0 or not (
            -self.landmark_guided_fusion_gate_max
            < self.landmark_guided_fusion_gate_init
            < self.landmark_guided_fusion_gate_max
        ):
            raise ValueError(
                "landmark_guided_fusion_gate_init must lie strictly inside "
                "(-landmark_guided_fusion_gate_max, "
                "landmark_guided_fusion_gate_max)."
            )
        if self.landmark_guided_quality_logit_scale < 0:
            raise ValueError(
                "landmark_guided_quality_logit_scale must be non-negative."
            )
        if self.landmark_guided_uncertainty_scale_norm <= 0:
            raise ValueError(
                "landmark_guided_uncertainty_scale_norm must be positive."
            )
        if not 0 <= self.landmark_guided_min_quality < 1:
            raise ValueError(
                "landmark_guided_min_quality must lie in [0, 1)."
            )
        if not isinstance(self.use_low_dof_eye_template, bool):
            raise ValueError("use_low_dof_eye_template must be a boolean.")
        if not isinstance(self.eye_template_learnable_parameters, bool):
            raise ValueError(
                "eye_template_learnable_parameters must be a boolean."
            )
        if not isinstance(self.eye_template_use_reference_uncertainty, bool):
            raise ValueError(
                "eye_template_use_reference_uncertainty must be a boolean."
            )
        if not isinstance(self.eye_template_use_aperture_visibility, bool):
            raise ValueError(
                "eye_template_use_aperture_visibility must be a boolean."
            )
        if not isinstance(self.use_pitch_to_v_residual, bool):
            raise ValueError("use_pitch_to_v_residual must be a boolean.")
        if not isinstance(self.use_vertical_geometry_residual, bool):
            raise ValueError(
                "use_vertical_geometry_residual must be a boolean."
            )
        if self.use_vertical_geometry_residual and self.use_pitch_to_v_residual:
            raise ValueError(
                "use_vertical_geometry_residual and use_pitch_to_v_residual "
                "are mutually exclusive."
            )
        if not isinstance(self.pitch_to_v_detach_uncertainty_gate, bool):
            raise ValueError(
                "pitch_to_v_detach_uncertainty_gate must be a boolean."
            )
        if not isinstance(self.pitch_to_v_use_camera_table_sensitivity, bool):
            raise ValueError(
                "pitch_to_v_use_camera_table_sensitivity must be a boolean."
            )
        if not 0 < self.pitch_to_v_min_abs_table_normal_z < 1:
            raise ValueError(
                "pitch_to_v_min_abs_table_normal_z must lie in (0, 1)."
            )
        pitch_to_v_target_frame = str(self.pitch_to_v_target_frame).strip().lower()
        if pitch_to_v_target_frame not in {"table_local", "virtual_camera"}:
            raise ValueError(
                "pitch_to_v_target_frame must be table_local or virtual_camera."
            )
        object.__setattr__(
            self, "pitch_to_v_target_frame", pitch_to_v_target_frame
        )
        if not self.pitch_to_v_hidden_dims or min(self.pitch_to_v_hidden_dims) <= 0:
            raise ValueError("pitch_to_v_hidden_dims must contain positive widths.")
        if min(
            self.pitch_to_v_max_abs_delta_mm,
            self.pitch_to_v_pitch_scale_deg,
            self.pitch_to_v_uncertainty_scale_deg,
            self.pitch_to_v_disagreement_scale_deg,
            self.pitch_to_v_min_pitch_std_deg,
        ) <= 0:
            raise ValueError("Pitch-to-v residual scales must be positive.")
        if not 0 <= self.pitch_to_v_min_active_gate <= 1:
            raise ValueError("pitch_to_v_min_active_gate must lie in [0, 1].")
        if self.eye_template_pitch_bins < 3 or self.eye_template_yaw_bins < 3:
            raise ValueError("Eye-template pitch/yaw bins must be at least three.")
        if self.eye_template_posterior_temperature <= 0:
            raise ValueError(
                "eye_template_posterior_temperature must be positive."
            )
        if self.eye_template_covariance_floor_px <= 0:
            raise ValueError("eye_template_covariance_floor_px must be positive.")
        if self.eye_template_min_canthus_distance_px <= 0:
            raise ValueError(
                "eye_template_min_canthus_distance_px must be positive."
            )
        if not 0 <= self.eye_template_min_geometry_confidence <= 1:
            raise ValueError(
                "eye_template_min_geometry_confidence must lie in [0, 1]."
            )
        if min(
            self.eye_template_center_energy_weight,
            self.eye_template_rim_energy_weight,
            self.eye_template_angle_prior_weight,
        ) < 0:
            raise ValueError("Eye-template energy weights must be non-negative.")
        if self.virtual_distance_film_delta_max <= 0:
            raise ValueError("virtual_distance_film_delta_max must be positive.")
        if self.branch_dropout < 0 or self.fusion_dropout < 0:
            raise ValueError("branch_dropout and fusion_dropout must be non-negative.")
        if not 0 <= self.visual_attention_dropout < 1:
            raise ValueError("visual_attention_dropout must be in [0, 1).")
        if not 0 <= self.binocular_attention_dropout < 1:
            raise ValueError("binocular_attention_dropout must be in [0, 1).")
        if not 0 <= self.visual_token_pooling_dropout < 1:
            raise ValueError("visual_token_pooling_dropout must be in [0, 1).")
        if self.visual_attention_dim % self.visual_attention_heads != 0:
            raise ValueError(
                "visual_attention_dim must be divisible by visual_attention_heads."
            )
        if self.visual_attention_dim % self.binocular_attention_heads != 0:
            raise ValueError(
                "visual_attention_dim must be divisible by binocular_attention_heads."
            )
        object.__setattr__(
            self,
            "eye_feature_stage",
            canonical_eye_feature_stage(self.eye_feature_stage),
        )
        object.__setattr__(
            self,
            "eye_feature_mode",
            canonical_eye_feature_mode(self.eye_feature_mode),
        )
        object.__setattr__(
            self,
            "eye_stem_mode",
            canonical_eye_stem_mode(self.eye_stem_mode),
        )
        object.__setattr__(
            self,
            "eye_iris_auxiliary_feature_stage",
            canonical_eye_iris_auxiliary_feature_stage(
                self.eye_iris_auxiliary_feature_stage
            ),
        )
        object.__setattr__(
            self,
            "eye_keypoint_auxiliary_feature_stage",
            canonical_eye_iris_auxiliary_feature_stage(
                self.eye_keypoint_auxiliary_feature_stage
            ),
        )
        object.__setattr__(
            self,
            "visual_token_pooling",
            canonical_visual_token_pooling(self.visual_token_pooling),
        )
        object.__setattr__(
            self,
            "face_feature_mode",
            canonical_face_feature_mode(self.face_feature_mode),
        )
        try:
            eye_token_grid = tuple(int(value) for value in self.eye_token_grid)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "eye_token_grid must be a two-item (height, width) sequence."
            ) from exc
        if len(eye_token_grid) != 2 or any(value <= 0 for value in eye_token_grid):
            raise ValueError(
                "eye_token_grid must contain two positive (height, width) values."
            )
        object.__setattr__(self, "eye_token_grid", eye_token_grid)
        try:
            eye_keypoint_heatmap_size = tuple(
                int(value) for value in self.eye_keypoint_heatmap_size
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "eye_keypoint_heatmap_size must be a two-item "
                "(height, width) sequence."
            ) from exc
        if len(eye_keypoint_heatmap_size) != 2 or any(
            value <= 1 for value in eye_keypoint_heatmap_size
        ):
            raise ValueError(
                "eye_keypoint_heatmap_size must contain two values greater than one."
            )
        object.__setattr__(
            self,
            "eye_keypoint_heatmap_size",
            eye_keypoint_heatmap_size,
        )
        for field_name in (
            "eye_template_image_size",
            "eye_template_pitch_range_deg",
            "eye_template_yaw_range_deg",
        ):
            try:
                pair = tuple(getattr(self, field_name))
            except TypeError as exc:
                raise ValueError(f"{field_name} must be a two-item sequence.") from exc
            if len(pair) != 2:
                raise ValueError(f"{field_name} must be a two-item sequence.")
            if field_name == "eye_template_image_size":
                pair = tuple(int(value) for value in pair)
                if min(pair) <= 1:
                    raise ValueError(
                        "eye_template_image_size values must be greater than one."
                    )
            else:
                pair = tuple(float(value) for value in pair)
                if pair[0] >= pair[1]:
                    raise ValueError(f"{field_name} must be strictly increasing.")
            object.__setattr__(self, field_name, pair)
        if self.use_face_image and self.eye_backbone not in RESNET_EYE_BACKBONES:
            raise ValueError(
                "Cross-attention visual fusion requires a torchvision ResNet "
                f"eye backbone, got {self.eye_backbone!r}."
            )
        if (
            self.use_face_image
            and self.eye_feature_mode == EYE_FEATURE_MODE_MULTISCALE_FPN24
            and self.eye_backbone not in {"resnet18", SMALL_IMAGE_RESNET18_BACKBONE}
        ):
            raise ValueError(
                "eye_feature_mode='multiscale_fpn24' is defined for the "
                "ResNet18 eye backbone."
            )
        if (
            self.use_face_image
            and self.eye_stem_mode == EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL
            and self.eye_backbone != "resnet18"
        ):
            raise ValueError(
                "eye_stem_mode='small_3x3_s2_no_maxpool' is defined for "
                "eye_backbone='resnet18'."
            )
        if self.use_eye_iris_auxiliary and not self.use_face_image:
            raise ValueError(
                "use_eye_iris_auxiliary=True requires use_face_image=True."
            )
        if self.use_eye_iris_auxiliary and self.eye_backbone != "resnet18":
            raise ValueError(
                "Iris auxiliary supervision is defined for eye_backbone='resnet18'."
            )
        if self.use_eye_keypoint_auxiliary and not self.use_face_image:
            raise ValueError(
                "use_eye_keypoint_auxiliary=True requires use_face_image=True."
            )
        if self.use_eye_keypoint_auxiliary and self.eye_backbone != "resnet18":
            raise ValueError(
                "Probabilistic eye-keypoint supervision is defined for "
                "eye_backbone='resnet18'."
            )
        if (
            self.use_landmark_guided_eye_fusion
            and not self.use_eye_keypoint_auxiliary
        ):
            raise ValueError(
                "use_landmark_guided_eye_fusion=True requires "
                "use_eye_keypoint_auxiliary=True."
            )
        if self.use_landmark_guided_eye_fusion and not self.use_face_image:
            raise ValueError(
                "use_landmark_guided_eye_fusion=True requires "
                "use_face_image=True."
            )
        if self.use_low_dof_eye_template and not self.use_eye_keypoint_auxiliary:
            raise ValueError(
                "use_low_dof_eye_template=True requires "
                "use_eye_keypoint_auxiliary=True."
            )
        if self.use_vertical_geometry_residual and not self.use_eye_keypoint_auxiliary:
            raise ValueError(
                "use_vertical_geometry_residual=True requires "
                "use_eye_keypoint_auxiliary=True."
            )
        if (
            self.use_vertical_geometry_residual
            and self.prediction_mode != PREDICTION_MODE_DIRECT_UV
        ):
            raise ValueError(
                "Vertical geometry residual correction requires direct_uv mode."
            )
        if self.use_vertical_geometry_residual and self.use_pitch_to_v_residual:
            raise ValueError(
                "Use either vertical geometry residual or pitch-to-v residual, "
                "not both."
            )
        if (
            not self.vertical_geometry_residual_hidden_dims
            or min(self.vertical_geometry_residual_hidden_dims) <= 0
        ):
            raise ValueError(
                "vertical_geometry_residual_hidden_dims must contain positive widths."
            )
        if self.vertical_geometry_residual_max_abs_delta_mm <= 0:
            raise ValueError(
                "vertical_geometry_residual_max_abs_delta_mm must be positive."
            )
        if not 0 <= self.vertical_geometry_residual_min_training_gate <= 1:
            raise ValueError(
                "vertical_geometry_residual_min_training_gate must lie in [0, 1]."
            )
        if not 0 <= self.vertical_geometry_residual_min_inference_confidence <= 1:
            raise ValueError(
                "vertical_geometry_residual_min_inference_confidence must lie in [0, 1]."
            )
        if self.use_pitch_to_v_residual and not self.use_low_dof_eye_template:
            raise ValueError(
                "use_pitch_to_v_residual=True requires "
                "use_low_dof_eye_template=True."
            )
        if (
            self.use_pitch_to_v_residual
            and self.prediction_mode != PREDICTION_MODE_DIRECT_UV
        ):
            raise ValueError(
                "Pitch-to-v residual correction currently requires direct_uv mode."
            )
        if (
            self.use_pitch_to_v_residual
            and self.scene_representation != SCENE_REPRESENTATION_TABLE_FRAME7
        ):
            raise ValueError(
                "Pitch-to-v residual correction requires scene_representation="
                "table_frame7."
            )
        if (
            self.use_eye_iris_auxiliary
            and self.prediction_mode != PREDICTION_MODE_DIRECT_UV
        ):
            raise ValueError(
                "Iris auxiliary supervision is currently supported only in "
                "prediction_mode='direct_uv'."
            )
        if self.use_face_image and not self.share_eye_encoder:
            raise ValueError(
                "Cross-attention visual fusion requires share_eye_encoder=True."
            )
        if not isinstance(self.use_eye_geometry, bool):
            raise ValueError("use_eye_geometry must be a boolean.")
        if not isinstance(self.use_crop_cam, bool):
            raise ValueError("use_crop_cam must be a boolean.")
        expected_deca_feature_dim = deca_feature_representation_dim(
            self.deca_feature_representation
        )
        if self.deca_feature_dim != expected_deca_feature_dim:
            raise ValueError(
                f"deca_feature_representation={self.deca_feature_representation!r} "
                f"requires deca_feature_dim={expected_deca_feature_dim}, got "
                f"{self.deca_feature_dim}."
            )
        if self.deca_feature_dim < 0:
            raise ValueError("deca_feature_dim must be non-negative.")
        if (
            self.deca_branch_mode == DECA_BRANCH_MODE_FACTORIZED_GEOMETRY
            and self.deca_feature_representation
            != DECA_FEATURE_REPRESENTATION_GEOMETRY156
        ):
            raise ValueError(
                "deca_branch_mode='factorized_geometry' requires "
                "deca_feature_representation='geometry156'."
            )
        if not isinstance(self.use_face_image, bool):
            raise ValueError("use_face_image must be a boolean.")
        if not isinstance(self.freeze_face_image_backbone, bool):
            raise ValueError("freeze_face_image_backbone must be a boolean.")
        expected_scene_dim = scene_representation_dim(
            self.scene_representation
        )
        if self.scene_dim != expected_scene_dim:
            raise ValueError(
                f"scene_representation={self.scene_representation!r} requires "
                f"scene_dim={expected_scene_dim}, got {self.scene_dim}."
            )
        expected_eye_geometry_dim = eye_geometry_representation_dim(
            self.eye_geometry_representation
        )
        if self.eye_geometry_dim != expected_eye_geometry_dim:
            raise ValueError(
                f"eye_geometry_representation={self.eye_geometry_representation!r} "
                f"requires eye_geometry_dim={expected_eye_geometry_dim}, got "
                f"{self.eye_geometry_dim}."
            )
        if (
            not self.use_eye_geometry
            and self.eye_geometry_gate_mode != EYE_GEOMETRY_GATE_NONE
        ):
            raise ValueError(
                "eye_geometry_gate_mode must be 'none' when "
                "use_eye_geometry=False."
            )
        if not 0.0 < self.eye_geometry_gate_min < self.eye_geometry_gate_max < 1.0:
            raise ValueError(
                "eye_geometry_gate_min/max must satisfy 0 < min < max < 1."
            )
        if self.eye_geometry_gate_delta_max <= 0:
            raise ValueError("eye_geometry_gate_delta_max must be positive.")
        if self.prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY:
            if self.use_virtual_distance_film:
                raise ValueError(
                    "Virtual-distance FiLM is only supported by direct_uv mode."
                )
            if self.use_virtual_pose_film:
                raise ValueError(
                    "Virtual-camera pose FiLM is only supported by direct_uv mode."
                )
            if self.gaze_dim != 3:
                raise ValueError("gaze_geometry prediction requires gaze_dim=3.")
            if not isinstance(self.use_gaze_uncertainty, bool):
                raise ValueError("use_gaze_uncertainty must be a boolean.")
            if not (
                0 < self.gaze_uncertainty_min_std_rad
                <= self.gaze_uncertainty_max_std_rad
            ):
                raise ValueError(
                    "gaze_uncertainty_min_std_rad/max_std_rad must satisfy "
                    "0 < min <= max."
                )
            if not (
                self.gaze_uncertainty_min_std_rad
                <= self.gaze_uncertainty_init_std_rad
                <= self.gaze_uncertainty_max_std_rad
            ):
                raise ValueError(
                    "gaze_uncertainty_init_std_rad must lie within the "
                    "configured min/max standard-deviation interval."
                )
            if self.use_crop_cam:
                raise ValueError(
                    "V4 gaze_geometry forbids use_crop_cam: camera metadata may "
                    "only enter the parameter-free geometry path."
                )
            if self.use_eye_geometry:
                raise ValueError(
                    "V4 gaze_geometry forbids learned eye-geometry fusion."
                )
            if self.scene_representation != SCENE_REPRESENTATION_TABLE_FRAME7:
                raise ValueError(
                    "V4 gaze_geometry requires scene_representation='table_frame7'."
                )
            DepthHypothesisConfig(
                mode=self.depth_distribution_mode,
                num_bins=self.depth_num_bins,
                range_sigma=self.depth_range_sigma,
                log_sigma_min=self.depth_log_sigma_min,
                log_sigma_max=self.depth_log_sigma_max,
            )
            RayTableGeometryConfig(
                table_distance_scale_mm=self.table_distance_scale_mm,
                min_abs_normal_dot_gaze=self.geometry_min_abs_normal_dot_gaze,
                min_lambda_mm=self.geometry_min_lambda_mm,
                max_lambda_mm=self.geometry_max_lambda_mm,
            )
            if (
                self.depth_distribution_mode
                == DEPTH_DISTRIBUTION_LEARNED_REWEIGHT
            ):
                if not self.depth_reweighter_hidden_dims:
                    raise ValueError(
                        "learned_reweight requires depth_reweighter_hidden_dims."
                    )
                if self.eye_geometry_quality_dim != 4:
                    raise ValueError("V4 PnP quality vector must be 4D.")
            if self.use_depth_correction:
                if not self.depth_correction_hidden_dims:
                    raise ValueError(
                        "use_depth_correction=True requires "
                        "depth_correction_hidden_dims."
                    )
                if self.eye_geometry_dim != 6 or self.eye_geometry_quality_dim != 4:
                    raise ValueError(
                        "DepthCorrectionHead requires 6D normalized PnP geometry "
                        "and 4D PnP quality (10D total)."
                    )
        else:
            if self.use_depth_correction:
                raise ValueError(
                    "use_depth_correction=True requires "
                    "prediction_mode='gaze_geometry'."
                )
            if (
                self.use_virtual_distance_film
                and self.scene_representation != SCENE_REPRESENTATION_TABLE_FRAME7
            ):
                raise ValueError(
                    "Virtual-distance FiLM requires "
                    "scene_representation='table_frame7'."
                )
            if (
                self.use_virtual_pose_film
                and self.scene_representation != SCENE_REPRESENTATION_TABLE_FRAME7
            ):
                raise ValueError(
                    "Virtual-camera pose FiLM requires "
                    "scene_representation='table_frame7'."
                )
            if (
                self.use_virtual_distance_film or self.use_virtual_pose_film
            ) and len(self.fusion_hidden_dims) < 2:
                raise ValueError(
                    "Direct-UV FiLM requires fusion_hidden_dims to contain "
                    "at least the FiLM width and output width."
                )
            if self.use_virtual_pose_film and self.virtual_pose_dim != 9:
                raise ValueError("Virtual-camera pose FiLM requires virtual_pose_dim=9.")
            if self.gaze_prediction_frame != GAZE_PREDICTION_FRAME_CAMERA:
                raise ValueError(
                    "gaze_prediction_frame='virtual_camera' requires "
                    "prediction_mode='gaze_geometry'."
                )
        if self.depth_reweighter_dropout < 0:
            raise ValueError("depth_reweighter_dropout must be non-negative.")
        if self.depth_correction_dropout < 0:
            raise ValueError("depth_correction_dropout must be non-negative.")


class ConvBlock(nn.Module):
    """Small convolution block for the eye encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CompactEyeImageEncoder(nn.Module):
    """Compact CNN for one eye crop."""

    def __init__(
        self,
        embedding_dim: int,
        dropout: float,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 96, stride=2),
            ConvBlock(96, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, image: Tensor) -> Tensor:
        image = ensure_image_batch(image, "eye")
        return self.net(image.float())


class ResNetEyeImageEncoder(nn.Module):
    """ResNet feature extractor for one eye crop."""

    def __init__(
        self,
        backbone: str,
        embedding_dim: int | None,
        dropout: float,
        weights: str | None = None,
        in_channels: int = 3,
        spatial_pool_size: tuple[int, int] | None = None,
        feature_map_stage: str = "layer4",
        feature_map_mode: str = EYE_FEATURE_MODE_SINGLE_STAGE,
        stem_mode: str = EYE_STEM_MODE_STANDARD,
    ) -> None:
        super().__init__()
        self.backbone_name = canonical_eye_backbone(backbone)
        self.spatial_pool_size = spatial_pool_size
        self.feature_map_stage = canonical_eye_feature_stage(feature_map_stage)
        self.feature_map_mode = canonical_eye_feature_mode(feature_map_mode)
        self.stem_mode = canonical_eye_stem_mode(stem_mode)
        if (
            self.feature_map_mode == EYE_FEATURE_MODE_MULTISCALE_FPN24
            and self.backbone_name not in {"resnet18", SMALL_IMAGE_RESNET18_BACKBONE}
        ):
            raise ValueError(
                "feature_map_mode='multiscale_fpn24' is defined for the "
                "ResNet18 eye backbone."
            )
        if (
            self.stem_mode == EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL
            and self.backbone_name != "resnet18"
        ):
            raise ValueError(
                "stem_mode='small_3x3_s2_no_maxpool' is defined for "
                "backbone='resnet18'."
            )
        resolved_weights = resolve_resnet_weights(self.backbone_name, weights)
        builder, _ = RESNET_EYE_BACKBONES[self.backbone_name]
        resnet = builder(weights=resolved_weights)
        if self.stem_mode == EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL:
            resnet.conv1 = replace_resnet_h5_small_stem(
                resnet.conv1,
                in_channels,
                bool(resolved_weights),
            )
            resnet.maxpool = nn.Identity()
            # The kernel support changes from 7x7 to 3x3, so inherited
            # ImageNet BatchNorm statistics are no longer representative.
            resnet.bn1.reset_running_stats()
        elif self.backbone_name == SMALL_IMAGE_RESNET18_BACKBONE:
            resnet.conv1 = replace_resnet_small_image_stem(
                resnet.conv1,
                in_channels,
                bool(resolved_weights),
            )
            resnet.maxpool = nn.Identity()
        elif in_channels != 3:
            resnet.conv1 = replace_first_conv(resnet.conv1, in_channels, bool(resolved_weights))
        feature_dim = (
            128
            if self.feature_map_mode == EYE_FEATURE_MODE_MULTISCALE_FPN24
            else resnet_stage_feature_channels(resnet, self.feature_map_stage)
        )
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.feature_map_channels = feature_dim
        if self.feature_map_mode == EYE_FEATURE_MODE_MULTISCALE_FPN24:
            if self.stem_mode == EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL:
                # H5 at 90x56: layer2/3/4 are 14x23, 7x12 and 4x6.
                # Keep the native layer3 grid for denser 7x12 eye tokens.
                self.fpn_layer2 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.SiLU(inplace=True),
                )
                self.fpn_layer3 = nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.SiLU(inplace=True),
                )
                self.fpn_layer4 = nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.SiLU(inplace=True),
                )
            else:
                # H3 at 90x56: layer2/3/4 are 7x12, 4x6 and 2x3.
                # Align them on the native layer3 grid.
                self.fpn_layer2 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.SiLU(inplace=True),
                )
                self.fpn_layer3 = nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.SiLU(inplace=True),
                )
                self.fpn_layer4 = nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.SiLU(inplace=True),
                )
            self.fpn_fusion = nn.Sequential(
                nn.Conv2d(384, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
            )
        else:
            self.fpn_layer2 = None
            self.fpn_layer3 = None
            self.fpn_layer4 = None
            self.fpn_fusion = None
        self.backbone_train_from: str | None = "stem"
        self._frozen_backbone_modules: tuple[nn.Module, ...] = ()
        self.proj = (
            nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            )
            if embedding_dim is not None
            else None
        )
        if resolved_weights is not None:
            mark_preserve_parameters(self.backbone)

    def forward(self, image: Tensor) -> Tensor:
        if self.proj is None:
            raise RuntimeError(
                "This eye encoder was created for spatial feature extraction. "
                "Call forward_feature_map instead."
            )
        feature_map = self.forward_feature_map(image)
        pooled = self.backbone.avgpool(feature_map).flatten(start_dim=1)
        return self.proj(pooled)

    def forward_feature_map(self, image: Tensor) -> Tensor:
        """Return a selected spatial ResNet stage before global pooling."""

        return self.forward_feature_maps(image)["feature_map"]

    def forward_feature_maps(self, image: Tensor) -> dict[str, Tensor]:
        """Return the fused map together with unpooled ResNet stage maps."""

        image = ensure_image_batch(image, "eye")
        x = image.float()
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(x)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        if self.feature_map_mode == EYE_FEATURE_MODE_MULTISCALE_FPN24:
            assert self.fpn_layer2 is not None
            assert self.fpn_layer3 is not None
            assert self.fpn_layer4 is not None
            assert self.fpn_fusion is not None
            layer2_aligned = self.fpn_layer2(layer2)
            layer3_aligned = self.fpn_layer3(layer3)
            target_size = layer3.shape[-2:]
            layer4_aligned = F.interpolate(
                self.fpn_layer4(layer4),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            feature_map = self.fpn_fusion(
                torch.cat((layer2_aligned, layer3_aligned, layer4_aligned), dim=1)
            )
        else:
            feature_map = (
                layer3
                if self.feature_map_stage == "layer3"
                else layer4
            )
        if (
            self.spatial_pool_size is not None
            and tuple(feature_map.shape[-2:]) != self.spatial_pool_size
        ):
            feature_map = F.adaptive_avg_pool2d(feature_map, self.spatial_pool_size)
        return {
            "feature_map": feature_map,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
        }

    def set_backbone_train_from(self, stage: str | None) -> None:
        """Freeze the backbone below ``stage``; ``None`` freezes it all."""

        train_from = (
            None
            if stage is None or str(stage).strip().lower() in NO_WEIGHT_VALUES
            else canonical_resnet_stage(stage)
        )
        trainable = False
        frozen_modules: list[nn.Module] = []
        for stage_name, modules in named_resnet_stages(self.backbone):
            if train_from is not None and stage_name == train_from:
                trainable = True
            for module in modules:
                module.requires_grad_(trainable)
                if not trainable:
                    frozen_modules.append(module)
        self.backbone_train_from = train_from
        self._frozen_backbone_modules = tuple(frozen_modules)
        if self.training:
            for module in self._frozen_backbone_modules:
                module.eval()

    def train(self, mode: bool = True) -> ResNetEyeImageEncoder:
        super().train(mode)
        if mode:
            # Frozen BatchNorm layers must not update running statistics.
            for module in self._frozen_backbone_modules:
                module.eval()
        return self


class EyeImageEncoder(nn.Module):
    """Configurable image encoder for one eye crop."""

    def __init__(
        self,
        embedding_dim: int,
        dropout: float,
        *,
        backbone: str,
        backbone_weights: str | None,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.backbone_name = canonical_eye_backbone(backbone)
        if self.backbone_name == COMPACT_EYE_BACKBONE:
            self.net = CompactEyeImageEncoder(embedding_dim, dropout, in_channels)
        else:
            self.net = ResNetEyeImageEncoder(
                self.backbone_name,
                embedding_dim,
                dropout,
                weights=backbone_weights,
                in_channels=in_channels,
            )

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)

    def resnet_encoder(self) -> ResNetEyeImageEncoder | None:
        return self.net if isinstance(self.net, ResNetEyeImageEncoder) else None


def _set_basic_conv_padding(module: nn.Module, padding: int) -> bool:
    conv = getattr(module, "conv", module)
    if not isinstance(conv, nn.Conv2d):
        return False
    conv.padding = (padding, padding)
    return True


def _module_at_path(module: nn.Module, path: tuple[str | int, ...]) -> nn.Module | None:
    current: nn.Module = module
    for part in path:
        try:
            if isinstance(part, int):
                current = current[part]  # type: ignore[index]
            else:
                current = getattr(current, part)
        except (AttributeError, IndexError, TypeError):
            return None
    return current


def adapt_mixed_7a_for_padded_4x4(mixed_7a: nn.Module) -> None:
    """Keep one extra spatial cell when Inception-ResNet downsamples 8x8 to 4x4."""

    patched_conv_paths = 0
    for path in (
        ("branch0",),
        ("branch0", 1),
        ("branch1", 1),
        ("branch2", 2),
    ):
        target = _module_at_path(mixed_7a, path)
        if target is not None and _set_basic_conv_padding(target, padding=1):
            patched_conv_paths += 1
    branch3 = getattr(mixed_7a, "branch3", None)
    patched_pool = False
    if isinstance(branch3, nn.MaxPool2d):
        mixed_7a.branch3 = nn.MaxPool2d(  # type: ignore[attr-defined]
            kernel_size=branch3.kernel_size,
            stride=branch3.stride,
            padding=1,
            dilation=branch3.dilation,
            return_indices=branch3.return_indices,
            ceil_mode=branch3.ceil_mode,
        )
        patched_pool = True
    if patched_conv_paths < 3 or not patched_pool:
        raise TypeError(
            "Could not adapt facenet-pytorch mixed_7a for 4x4 face tokens; "
            "the module layout is not recognized."
        )


def canonical_face_feature_mode(value: str) -> str:
    """Return the Inception-ResNet spatial-token topology for face images."""

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "native": FACE_FEATURE_MODE_NATIVE_3X3,
        "native_3x3": FACE_FEATURE_MODE_NATIVE_3X3,
        "3x3": FACE_FEATURE_MODE_NATIVE_3X3,
        "f0": FACE_FEATURE_MODE_NATIVE_3X3,
        "padded_mixed7a_4x4": FACE_FEATURE_MODE_PADDED_MIXED7A_4X4,
        "mixed7a_4x4": FACE_FEATURE_MODE_PADDED_MIXED7A_4X4,
        "4x4": FACE_FEATURE_MODE_PADDED_MIXED7A_4X4,
        "f1": FACE_FEATURE_MODE_PADDED_MIXED7A_4X4,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(FACE_FEATURE_MODE_TOKEN_GRIDS)
        raise ValueError(
            f"face_feature_mode must be one of: {supported}, got {value!r}."
        ) from exc


def face_feature_token_grid(feature_mode: str) -> tuple[int, int]:
    return FACE_FEATURE_MODE_TOKEN_GRIDS[canonical_face_feature_mode(feature_mode)]


class InceptionResnetFaceEncoder(nn.Module):
    """Return VGGFace2 Inception-ResNet spatial features for a 160x160 face."""

    def __init__(
        self,
        *,
        freeze_backbone: bool = False,
        feature_mode: str = FACE_FEATURE_MODE_NATIVE_3X3,
    ) -> None:
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The RGB face branch requires facenet-pytorch. Install the "
                "project requirements before setting use_face_image=True."
            ) from exc

        self.backbone_name = FACE_IMAGE_BACKBONE
        self.pretrained_dataset = FACE_IMAGE_PRETRAINED_DATASET
        self.freeze_backbone = freeze_backbone
        self.feature_mode = canonical_face_feature_mode(feature_mode)
        self.output_grid = face_feature_token_grid(self.feature_mode)
        self.backbone = InceptionResnetV1(
            pretrained=self.pretrained_dataset,
            classify=False,
        )
        if self.feature_mode == FACE_FEATURE_MODE_PADDED_MIXED7A_4X4:
            adapt_mixed_7a_for_padded_4x4(self.backbone.mixed_7a)
        # The pretrained identity-classification head is not used by the
        # embedding path and would otherwise remain as dead parameters.
        self.backbone.logits = nn.Identity()
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
        mark_preserve_parameters(self.backbone)

    def forward(self, image: Tensor) -> Tensor:
        image = ensure_image_batch(image, "face")
        x = image.float()
        for stage_name in (
            "conv2d_1a",
            "conv2d_2a",
            "conv2d_2b",
            "maxpool_3a",
            "conv2d_3b",
            "conv2d_4a",
            "conv2d_4b",
            "repeat_1",
            "mixed_6a",
            "repeat_2",
            "mixed_7a",
            "repeat_3",
            "block8",
        ):
            x = getattr(self.backbone, stage_name)(x)
        if tuple(x.shape[-2:]) != self.output_grid:
            raise ValueError(
                f"Expected a {self.output_grid[0]}x{self.output_grid[1]} "
                "Inception-ResNet face feature map for "
                f"feature_mode={self.feature_mode!r}, got {tuple(x.shape[-2:])}."
            )
        return x

    def train(self, mode: bool = True) -> InceptionResnetFaceEncoder:
        super().train(mode)
        if self.freeze_backbone:
            # A frozen pretrained backbone must also keep all BatchNorm
            # running statistics fixed.
            self.backbone.eval()
        return self


class ProbabilisticEyeKeypointHead(nn.Module):
    """Predict spatial distributions and their moments for eye landmarks.

    The head emits one categorical distribution over the feature-map grid for
    each keypoint. Its expectation is the normalized crop coordinate and its
    second central moment is a full 2D covariance. Consequently uncertainty is
    derived from the same distribution as the position instead of a separate,
    potentially inconsistent confidence regressor.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        hidden_channels: int,
        keypoint_count: int = EYE_PSEUDO_POINT_COUNT,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0 or keypoint_count <= 0:
            raise ValueError("Probabilistic keypoint-head dimensions must be positive.")
        if temperature <= 0:
            raise ValueError("Probabilistic keypoint temperature must be positive.")
        self.keypoint_count = int(keypoint_count)
        self.temperature = float(temperature)
        self.predictor = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, self.keypoint_count, kernel_size=1),
        )

    def forward(self, feature_map: Tensor) -> dict[str, Tensor]:
        if feature_map.ndim != 4:
            raise ValueError(
                "Eye keypoint feature_map must have shape [B, C, H, W], got "
                f"{tuple(feature_map.shape)}."
            )
        logits = self.predictor(feature_map)
        batch_size, keypoint_count, height, width = logits.shape
        if keypoint_count != self.keypoint_count or height <= 0 or width <= 0:
            raise RuntimeError("Probabilistic keypoint head produced an invalid shape.")

        # Moment computation remains FP32 under AMP so small covariances do not
        # underflow and uncertainty can safely drive later geometry fitting.
        flat_logits = logits.float().flatten(start_dim=2) / self.temperature
        probabilities = torch.softmax(flat_logits, dim=-1)
        y_coordinates, x_coordinates = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=logits.device),
            torch.linspace(0.0, 1.0, width, device=logits.device),
            indexing="ij",
        )
        coordinate_grid = torch.stack(
            (x_coordinates, y_coordinates), dim=-1
        ).reshape(height * width, 2)
        mean_xy = torch.einsum(
            "bkn,nc->bkc", probabilities, coordinate_grid
        )
        centered = coordinate_grid[None, None, :, :] - mean_xy[:, :, None, :]
        covariance = torch.einsum(
            "bkn,bknc,bknd->bkcd",
            probabilities,
            centered,
            centered,
        )
        variance_xy = torch.diagonal(covariance, dim1=-2, dim2=-1)
        log_probabilities = torch.log(probabilities.clamp_min(1e-12))
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        if height * width > 1:
            entropy = entropy / math.log(height * width)
        confidence = (1.0 - entropy).clamp(0.0, 1.0)
        return {
            "logits": logits,
            "probability_maps": probabilities.reshape(
                batch_size, keypoint_count, height, width
            ),
            "mean_xy": mean_xy,
            "covariance": covariance,
            "variance_xy": variance_xy,
            "entropy": entropy,
            "confidence": confidence,
        }


class CrossAttentionVisualEncoder(nn.Module):
    """Fuse binocular local tokens with global face tokens via cross-attention."""

    def __init__(
        self,
        *,
        attention_dim: int,
        output_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        eye_backbone: str,
        eye_backbone_weights: str | None,
        freeze_face_backbone: bool,
        face_feature_mode: str = FACE_FEATURE_MODE_NATIVE_3X3,
        eye_token_grid: tuple[int, int] = DEFAULT_EYE_IMAGE_TOKEN_GRID,
        eye_feature_stage: str = "layer4",
        eye_feature_mode: str = EYE_FEATURE_MODE_SINGLE_STAGE,
        eye_stem_mode: str = EYE_STEM_MODE_STANDARD,
        use_binocular_self_attention: bool = False,
        binocular_attention_heads: int = 4,
        binocular_attention_dropout: float = 0.1,
        use_iris_auxiliary: bool = False,
        iris_auxiliary_feature_stage: str = "layer3",
        iris_auxiliary_hidden_channels: int = 128,
        use_keypoint_auxiliary: bool = False,
        keypoint_count: int = EYE_PSEUDO_POINT_COUNT,
        keypoint_auxiliary_feature_stage: str = "layer3",
        keypoint_auxiliary_hidden_channels: int = 128,
        keypoint_probability_temperature: float = 1.0,
        keypoint_heatmap_size: tuple[int, int] = (28, 45),
        keypoint_predict_variance_correction: bool = True,
        keypoint_max_abs_log_std_correction: float = 1.5,
        keypoint_shape_embedding_dim: int = 32,
        use_landmark_guided_fusion: bool = False,
        landmark_guided_fusion_heads: int = 4,
        landmark_guided_fusion_dropout: float = 0.1,
        landmark_guided_fusion_gate_init: float = 0.0,
        landmark_guided_fusion_gate_max: float = 1.0,
        landmark_guided_quality_logit_scale: float = 1.0,
        landmark_guided_uncertainty_scale_norm: float = 0.25,
        landmark_guided_min_quality: float = 0.1,
        landmark_guided_detach_probability_for_sampling: bool = True,
        landmark_guided_detach_landmark_statistics: bool = True,
        landmark_guided_detach_quality: bool = True,
        use_low_dof_eye_template: bool = False,
        eye_template_image_size: tuple[int, int] = (90, 56),
        eye_template_pitch_range_deg: tuple[float, float] = (-35.0, 35.0),
        eye_template_yaw_range_deg: tuple[float, float] = (-45.0, 45.0),
        eye_template_pitch_bins: int = 29,
        eye_template_yaw_bins: int = 31,
        eye_template_posterior_temperature: float = 0.25,
        eye_template_covariance_floor_px: float = 0.75,
        eye_template_min_canthus_distance_px: float = 4.0,
        eye_template_min_geometry_confidence: float = 0.1,
        eye_template_center_energy_weight: float = 1.0,
        eye_template_rim_energy_weight: float = 1.0,
        eye_template_angle_prior_weight: float = 0.05,
        eye_template_learnable_parameters: bool = True,
        eye_template_use_reference_uncertainty: bool = True,
        eye_template_use_aperture_visibility: bool = True,
        token_pooling: str = VISUAL_TOKEN_POOLING_MEAN,
        token_pooling_hidden_dim: int = 64,
        token_pooling_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.eye_feature_stage = canonical_eye_feature_stage(eye_feature_stage)
        self.eye_feature_mode = canonical_eye_feature_mode(eye_feature_mode)
        self.eye_stem_mode = canonical_eye_stem_mode(eye_stem_mode)
        self.use_binocular_self_attention = bool(use_binocular_self_attention)
        self.use_iris_auxiliary = bool(use_iris_auxiliary)
        self.use_keypoint_auxiliary = bool(use_keypoint_auxiliary)
        self.use_landmark_guided_fusion = bool(use_landmark_guided_fusion)
        if self.use_landmark_guided_fusion and not self.use_keypoint_auxiliary:
            raise ValueError(
                "use_landmark_guided_fusion requires use_keypoint_auxiliary=True."
            )
        if use_low_dof_eye_template and not self.use_keypoint_auxiliary:
            raise ValueError(
                "use_low_dof_eye_template requires use_keypoint_auxiliary=True."
            )
        self.face_feature_mode = canonical_face_feature_mode(face_feature_mode)
        self.face_token_grid = face_feature_token_grid(self.face_feature_mode)
        self.face_token_count = math.prod(self.face_token_grid)
        self.iris_auxiliary_feature_stage = canonical_eye_iris_auxiliary_feature_stage(
            iris_auxiliary_feature_stage
        )
        self.keypoint_auxiliary_feature_stage = (
            canonical_eye_iris_auxiliary_feature_stage(
                keypoint_auxiliary_feature_stage
            )
        )
        self.token_pooling = canonical_visual_token_pooling(token_pooling)
        if binocular_attention_heads <= 0:
            raise ValueError("binocular_attention_heads must be positive.")
        if attention_dim % binocular_attention_heads != 0:
            raise ValueError(
                "attention_dim must be divisible by binocular_attention_heads."
            )
        if not 0 <= binocular_attention_dropout < 1:
            raise ValueError("binocular_attention_dropout must be in [0, 1).")
        if iris_auxiliary_hidden_channels <= 0:
            raise ValueError("iris_auxiliary_hidden_channels must be positive.")
        if token_pooling_hidden_dim <= 0:
            raise ValueError("token_pooling_hidden_dim must be positive.")
        if not 0 <= token_pooling_dropout < 1:
            raise ValueError("token_pooling_dropout must be in [0, 1).")
        self.eye_token_grid = tuple(int(value) for value in eye_token_grid)
        if len(self.eye_token_grid) != 2 or any(
            value <= 0 for value in self.eye_token_grid
        ):
            raise ValueError(
                "eye_token_grid must contain two positive (height, width) values."
            )
        self.binocular_eye_token_count = 2 * math.prod(self.eye_token_grid)
        self.face_encoder = InceptionResnetFaceEncoder(
            freeze_backbone=freeze_face_backbone,
            feature_mode=self.face_feature_mode,
        )
        self.eye_encoder = ResNetEyeImageEncoder(
            backbone=eye_backbone,
            embedding_dim=None,
            dropout=dropout,
            weights=eye_backbone_weights,
            spatial_pool_size=self.eye_token_grid,
            feature_map_stage=self.eye_feature_stage,
            feature_map_mode=self.eye_feature_mode,
            stem_mode=self.eye_stem_mode,
        )
        self.face_projection = nn.Sequential(
            nn.Linear(FACE_IMAGE_FEATURE_CHANNELS, attention_dim),
            nn.LayerNorm(attention_dim),
        )
        self.eye_projection = nn.Sequential(
            nn.Linear(self.eye_encoder.feature_map_channels, attention_dim),
            nn.LayerNorm(attention_dim),
        )
        iris_feature_channels = {
            "layer2": 128,
            "layer3": 256,
        }[self.iris_auxiliary_feature_stage]
        self.iris_heatmap_head = (
            nn.Sequential(
                nn.Conv2d(
                    iris_feature_channels,
                    iris_auxiliary_hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(iris_auxiliary_hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(iris_auxiliary_hidden_channels, 1, kernel_size=1),
            )
            if self.use_iris_auxiliary
            else None
        )
        self.eye_keypoint_head = (
            ProbabilisticEyeLandmarkHead(
                layer2_channels=128,
                layer3_channels=256,
                hidden_channels=keypoint_auxiliary_hidden_channels,
                keypoint_count=keypoint_count,
                output_size=keypoint_heatmap_size,
                temperature=keypoint_probability_temperature,
                predict_variance_correction=(
                    keypoint_predict_variance_correction
                ),
                max_abs_log_std_correction=(
                    keypoint_max_abs_log_std_correction
                ),
                shape_embedding_dim=keypoint_shape_embedding_dim,
            )
            if self.use_keypoint_auxiliary
            else None
        )
        self.eye_geometry_template = (
            LowDOFDifferentiableEyeballTemplate(
                eye_image_size=eye_template_image_size,
                pitch_range_deg=eye_template_pitch_range_deg,
                yaw_range_deg=eye_template_yaw_range_deg,
                pitch_bins=eye_template_pitch_bins,
                yaw_bins=eye_template_yaw_bins,
                posterior_temperature=eye_template_posterior_temperature,
                covariance_floor_px=eye_template_covariance_floor_px,
                min_canthus_distance_px=eye_template_min_canthus_distance_px,
                min_geometry_confidence=eye_template_min_geometry_confidence,
                center_energy_weight=eye_template_center_energy_weight,
                rim_energy_weight=eye_template_rim_energy_weight,
                angle_prior_weight=eye_template_angle_prior_weight,
                learnable_template=eye_template_learnable_parameters,
                use_reference_uncertainty=(
                    eye_template_use_reference_uncertainty
                ),
                use_aperture_visibility=(
                    eye_template_use_aperture_visibility
                ),
            )
            if use_low_dof_eye_template
            else None
        )
        self.landmark_guided_fusion = (
            LandmarkGuidedEyeFusion(
                token_dim=attention_dim,
                num_heads=landmark_guided_fusion_heads,
                dropout=landmark_guided_fusion_dropout,
                keypoint_count=keypoint_count,
                gate_init=landmark_guided_fusion_gate_init,
                gate_max=landmark_guided_fusion_gate_max,
                quality_logit_scale=landmark_guided_quality_logit_scale,
                uncertainty_scale_norm=(
                    landmark_guided_uncertainty_scale_norm
                ),
                min_quality=landmark_guided_min_quality,
                detach_probability_for_sampling=(
                    landmark_guided_detach_probability_for_sampling
                ),
                detach_landmark_statistics=(
                    landmark_guided_detach_landmark_statistics
                ),
                detach_quality=landmark_guided_detach_quality,
            )
            if self.use_landmark_guided_fusion
            else None
        )
        self.face_position = nn.Parameter(
            torch.zeros(1, self.face_token_count, attention_dim)
        )
        self.eye_position = nn.Parameter(
            torch.zeros(1, self.binocular_eye_token_count, attention_dim)
        )
        self.left_eye_type = nn.Parameter(torch.zeros(1, 1, attention_dim))
        self.right_eye_type = nn.Parameter(torch.zeros(1, 1, attention_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.binocular_self_attention = (
            nn.MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=binocular_attention_heads,
                dropout=binocular_attention_dropout,
                batch_first=True,
            )
            if self.use_binocular_self_attention
            else None
        )
        self.binocular_attention_norm = (
            nn.LayerNorm(attention_dim)
            if self.use_binocular_self_attention
            else None
        )
        self.attention_norm = nn.LayerNorm(attention_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, attention_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(attention_dim)
        self.token_score = (
            nn.Sequential(
                nn.LayerNorm(attention_dim),
                nn.Linear(attention_dim, token_pooling_hidden_dim),
                nn.GELU(),
                nn.Dropout(token_pooling_dropout),
                nn.Linear(token_pooling_hidden_dim, 1),
            )
            if self.token_pooling == VISUAL_TOKEN_POOLING_LEARNED
            else None
        )
        self.output_projection = (
            nn.Identity()
            if attention_dim == output_dim
            else nn.Sequential(
                nn.Linear(attention_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        )
        for embedding in (
            self.face_position,
            self.eye_position,
            self.left_eye_type,
            self.right_eye_type,
        ):
            nn.init.trunc_normal_(embedding, std=0.02)
        self.reset_token_pooling_to_mean()

    @staticmethod
    def _spatial_tokens(feature_map: Tensor) -> Tensor:
        return feature_map.flatten(start_dim=2).transpose(1, 2)

    def forward(
        self,
        face: Tensor,
        left_eye: Tensor,
        right_eye: Tensor,
        *,
        return_attention: bool = False,
        return_auxiliary: bool = False,
    ) -> dict[str, Tensor]:
        face_map = self.face_encoder(face)
        left_eye_maps = self.eye_encoder.forward_feature_maps(left_eye)
        right_eye_maps = self.eye_encoder.forward_feature_maps(right_eye)
        left_eye_map = left_eye_maps["feature_map"]
        right_eye_map = right_eye_maps["feature_map"]
        landmark_outputs: dict[str, Tensor] = {}
        if self.eye_keypoint_head is not None and (
            return_auxiliary or self.landmark_guided_fusion is not None
        ):
            landmark_outputs = self._eye_landmarks_from_feature_maps(
                left_eye_maps,
                right_eye_maps,
                include_template=return_auxiliary,
            )

        face_tokens = self.face_projection(self._spatial_tokens(face_map))
        left_tokens = self.eye_projection(self._spatial_tokens(left_eye_map))
        right_tokens = self.eye_projection(self._spatial_tokens(right_eye_map))
        if face_tokens.shape[1] != self.face_token_count:
            raise ValueError(
                f"Expected a {self.face_token_grid[0]}x{self.face_token_grid[1]} "
                "Inception-ResNet face feature map for "
                f"feature_mode={self.face_feature_mode!r}, "
                f"got {tuple(face_map.shape[-2:])}."
            )
        if (
            tuple(left_eye_map.shape[-2:]) != self.eye_token_grid
            or tuple(right_eye_map.shape[-2:]) != self.eye_token_grid
        ):
            raise ValueError(
                f"Expected two {self.eye_token_grid[0]}x{self.eye_token_grid[1]} "
                "ResNet eye feature maps, got "
                f"left={tuple(left_eye_map.shape[-2:])}, "
                f"right={tuple(right_eye_map.shape[-2:])}."
            )

        face_tokens = face_tokens + self.face_position
        left_count = left_tokens.shape[1]
        left_tokens = (
            left_tokens
            + self.eye_position[:, :left_count]
            + self.left_eye_type
        )
        right_tokens = (
            right_tokens
            + self.eye_position[:, left_count:]
            + self.right_eye_type
        )
        eye_tokens = torch.cat((left_tokens, right_tokens), dim=1)
        binocular_attention_weights: Tensor | None = None
        if self.binocular_self_attention is None:
            binocular_tokens = eye_tokens
        else:
            binocular_attended, binocular_attention_weights = (
                self.binocular_self_attention(
                    query=eye_tokens,
                    key=eye_tokens,
                    value=eye_tokens,
                    need_weights=return_attention,
                    average_attn_weights=False,
                )
            )
            assert self.binocular_attention_norm is not None
            binocular_tokens = self.binocular_attention_norm(
                eye_tokens + binocular_attended
            )
        landmark_fusion_outputs: dict[str, Tensor] = {}
        pre_landmark_eye_tokens: Tensor | None = None
        if self.landmark_guided_fusion is not None:
            pre_landmark_eye_tokens = binocular_tokens
            fusion_outputs = self.landmark_guided_fusion(
                binocular_tokens,
                left_layer2=left_eye_maps["layer2"],
                left_layer3=left_eye_maps["layer3"],
                right_layer2=right_eye_maps["layer2"],
                right_layer3=right_eye_maps["layer3"],
                predictions=landmark_outputs,
                return_attention=return_attention,
            )
            binocular_tokens = fusion_outputs["refined_eye_tokens"]
            landmark_fusion_outputs = {
                f"landmark_guided_{name}": value
                for name, value in fusion_outputs.items()
                if name not in {"refined_eye_tokens", "landmark_tokens"}
            }
        attended, attention_weights = self.cross_attention(
            query=binocular_tokens,
            key=face_tokens,
            value=face_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        attended = self.attention_norm(binocular_tokens + attended)
        fused_tokens = self.ffn_norm(attended + self.ffn(attended))
        token_pooling_weights: Tensor | None = None
        if self.token_score is None:
            pooled_tokens = fused_tokens.mean(dim=1)
        else:
            token_pooling_weights = torch.softmax(
                self.token_score(fused_tokens).squeeze(dim=-1),
                dim=1,
            )
            pooled_tokens = torch.sum(
                fused_tokens * token_pooling_weights.unsqueeze(dim=-1),
                dim=1,
            )
        visual_features = self.output_projection(pooled_tokens)

        output = {
            "visual_features": visual_features,
            "face_image_features": face_tokens.mean(dim=1),
            "eye_features": binocular_tokens.mean(dim=1),
        }
        if return_auxiliary and self.iris_heatmap_head is not None:
            output["left_iris_heatmap"] = self.iris_heatmap_head(
                left_eye_maps[self.iris_auxiliary_feature_stage]
            )
            output["right_iris_heatmap"] = self.iris_heatmap_head(
                right_eye_maps[self.iris_auxiliary_feature_stage]
            )
        if return_auxiliary and self.eye_keypoint_head is not None:
            output.update(landmark_outputs)
        output.update(landmark_fusion_outputs)
        if return_attention and attention_weights is not None:
            output.update(
                {
                    "face_tokens": face_tokens,
                    "eye_tokens": binocular_tokens,
                    "cross_attended_tokens": fused_tokens,
                    "cross_attention_weights": attention_weights,
                }
            )
            if binocular_attention_weights is not None:
                output["pre_binocular_eye_tokens"] = eye_tokens
                output["binocular_self_attention_weights"] = (
                    binocular_attention_weights
                )
            if pre_landmark_eye_tokens is not None:
                output["pre_landmark_eye_tokens"] = pre_landmark_eye_tokens
            if token_pooling_weights is not None:
                output["token_pooling_weights"] = token_pooling_weights
        return output

    def _eye_landmarks_from_feature_maps(
        self,
        left_eye_maps: Mapping[str, Tensor],
        right_eye_maps: Mapping[str, Tensor],
        *,
        include_template: bool = True,
    ) -> dict[str, Tensor]:
        if self.eye_keypoint_head is None:
            raise RuntimeError("The probabilistic eye-landmark head is disabled.")
        output: dict[str, Tensor] = {}
        for side, eye_maps in (
            ("left", left_eye_maps),
            ("right", right_eye_maps),
        ):
            prediction = self.eye_keypoint_head(
                eye_maps["layer2"],
                eye_maps["layer3"],
            )
            output.update(
                {
                    f"{side}_eye_keypoint_{name}": value
                    for name, value in prediction.items()
                }
            )
        if include_template and self.eye_geometry_template is not None:
            template_outputs = self.eye_geometry_template(
                stack_binocular_outputs(output, "mean_xy"),
                stack_binocular_outputs(output, "covariance"),
                stack_binocular_outputs(output, "visibility_probability"),
            )
            output.update(
                {
                    f"eye_template_{name}": value
                    for name, value in template_outputs.items()
                }
            )
        return output

    def predict_eye_landmarks(
        self,
        left_eye: Tensor,
        right_eye: Tensor,
        *,
        include_template: bool = True,
    ) -> dict[str, Tensor]:
        """Run only the shared eye encoder and probabilistic landmark head."""

        left_eye_maps = self.eye_encoder.forward_feature_maps(left_eye)
        right_eye_maps = self.eye_encoder.forward_feature_maps(right_eye)
        return self._eye_landmarks_from_feature_maps(
            left_eye_maps,
            right_eye_maps,
            include_template=include_template,
        )

    def reset_token_pooling_to_mean(self) -> None:
        """Start learned pooling as exact mean pooling before it specializes."""

        if self.token_score is None:
            return
        output = self.token_score[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)


class FaceBranch(nn.Module):
    """Embed frozen/offline DECA face features."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        detach_input: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.detach_input = detach_input
        self.net = make_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            dropout=dropout,
            input_layer_norm=True,
            activate_output=True,
        )

    def forward(self, deca_feat: Tensor) -> Tensor:
        deca_feat = ensure_vector_batch(deca_feat, "deca_feat")
        if deca_feat.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected deca_feat dim {self.input_dim}, got {deca_feat.shape[-1]}"
            )
        if self.detach_input:
            deca_feat = deca_feat.detach()
        return self.net(deca_feat.float())


class FactorizedGeometryFaceBranch(nn.Module):
    """Encode DECA shape, expression, and pose before cross-factor fusion.

    The input must be the ``geometry156`` cache representation ordered as
    ``[shape(100), exp(50), pose(6)]``.  Each semantic factor gets its own
    small MLP; their embeddings are concatenated and projected to the common
    face embedding consumed by the final multimodal fusion.
    """

    def __init__(
        self,
        embedding_dim: int,
        factor_embedding_dim: int,
        factor_hidden_dims: tuple[int, ...],
        fusion_hidden_dims: tuple[int, ...],
        dropout: float,
        detach_input: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = sum(DECA_GEOMETRY_FACTOR_DIMS.values())
        self.detach_input = detach_input
        self.shape_encoder = make_mlp(
            input_dim=DECA_GEOMETRY_FACTOR_DIMS["shape"],
            hidden_dims=factor_hidden_dims,
            output_dim=factor_embedding_dim,
            dropout=dropout,
            input_layer_norm=True,
            activate_output=True,
        )
        self.exp_encoder = make_mlp(
            input_dim=DECA_GEOMETRY_FACTOR_DIMS["exp"],
            hidden_dims=factor_hidden_dims,
            output_dim=factor_embedding_dim,
            dropout=dropout,
            input_layer_norm=True,
            activate_output=True,
        )
        self.pose_encoder = make_mlp(
            input_dim=DECA_GEOMETRY_FACTOR_DIMS["pose"],
            hidden_dims=factor_hidden_dims,
            output_dim=factor_embedding_dim,
            dropout=dropout,
            input_layer_norm=True,
            activate_output=True,
        )
        self.fusion = make_mlp(
            input_dim=3 * factor_embedding_dim,
            hidden_dims=fusion_hidden_dims,
            output_dim=embedding_dim,
            dropout=dropout,
            input_layer_norm=True,
            activate_output=True,
        )

    def forward(self, deca_feat: Tensor) -> Tensor:
        deca_feat = ensure_vector_batch(deca_feat, "deca_feat")
        if deca_feat.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected geometry156 deca_feat dim {self.input_dim}, got "
                f"{deca_feat.shape[-1]}"
            )
        if self.detach_input:
            deca_feat = deca_feat.detach()
        shape_end = DECA_GEOMETRY_FACTOR_DIMS["shape"]
        exp_end = shape_end + DECA_GEOMETRY_FACTOR_DIMS["exp"]
        shape_features = self.shape_encoder(deca_feat[:, :shape_end].float())
        exp_features = self.exp_encoder(deca_feat[:, shape_end:exp_end].float())
        pose_features = self.pose_encoder(deca_feat[:, exp_end:].float())
        return self.fusion(torch.cat((shape_features, exp_features, pose_features), dim=-1))


class EyeBranch(nn.Module):
    """Encode left/right eye crops and fuse them into one eye embedding."""

    def __init__(
        self,
        per_eye_dim: int,
        embedding_dim: int,
        dropout: float,
        share_encoder: bool = True,
        backbone: str = DEFAULT_EYE_BACKBONE,
        backbone_weights: str | None = None,
    ) -> None:
        super().__init__()
        self.share_encoder = share_encoder
        if share_encoder:
            self.eye_encoder = EyeImageEncoder(
                per_eye_dim,
                dropout,
                backbone=backbone,
                backbone_weights=backbone_weights,
            )
        else:
            self.left_eye_encoder = EyeImageEncoder(
                per_eye_dim,
                dropout,
                backbone=backbone,
                backbone_weights=backbone_weights,
            )
            self.right_eye_encoder = EyeImageEncoder(
                per_eye_dim,
                dropout,
                backbone=backbone,
                backbone_weights=backbone_weights,
            )
        self.project = make_mlp(
            input_dim=per_eye_dim * 2,
            hidden_dims=(embedding_dim,),
            output_dim=embedding_dim,
            dropout=dropout,
            input_layer_norm=False,
            activate_output=True,
        )

    def forward(self, left_eye: Tensor, right_eye: Tensor) -> Tensor:
        if self.share_encoder:
            left_features = self.eye_encoder(left_eye)
            right_features = self.eye_encoder(right_eye)
        else:
            left_features = self.left_eye_encoder(left_eye)
            right_features = self.right_eye_encoder(right_eye)
        return self.project(torch.cat([left_features, right_features], dim=-1))

    def resnet_encoders(self) -> tuple[ResNetEyeImageEncoder, ...]:
        encoders = (
            (self.eye_encoder,)
            if self.share_encoder
            else (self.left_eye_encoder, self.right_eye_encoder)
        )
        result: list[ResNetEyeImageEncoder] = []
        for encoder in encoders:
            resnet = encoder.resnet_encoder()
            if resnet is not None:
                result.append(resnet)
        return tuple(result)


class VectorBranch(nn.Module):
    """MLP branch for structured vector inputs."""

    def __init__(
        self,
        name: str,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.net = make_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            dropout=dropout,
            input_layer_norm=True,
            activate_output=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = ensure_vector_batch(x, self.name)
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.name} dim {self.input_dim}, got {x.shape[-1]}"
            )
        return self.net(x.float())


class DirectUVFiLMFusion(nn.Module):
    """Condition the first direct-UV fusion layer on virtual-camera metadata."""

    def __init__(
        self,
        *,
        input_dim: int,
        fusion_hidden_dims: tuple[int, ...],
        scale_hidden_dims: tuple[int, ...],
        scale_embedding_dim: int,
        dropout: float,
        delta_max: float,
        use_pose_condition: bool = False,
        pose_dim: int = 9,
    ) -> None:
        super().__init__()
        if len(fusion_hidden_dims) < 2:
            raise ValueError(
                "fusion_hidden_dims must contain a FiLM width and output width."
            )
        self.film_dim = fusion_hidden_dims[0]
        self.output_dim = fusion_hidden_dims[-1]
        self.delta_max = float(delta_max)
        self.use_pose_condition = bool(use_pose_condition)
        self.pose_dim = int(pose_dim)
        self.condition_dim = self.pose_dim if self.use_pose_condition else 1
        self.pre_film = make_mlp(
            input_dim=input_dim,
            hidden_dims=(),
            output_dim=self.film_dim,
            dropout=dropout,
            input_layer_norm=False,
            activate_output=True,
        )
        # LayerNorm on a scalar input would erase the scale-only condition.
        self.condition_encoder = make_mlp(
            input_dim=self.condition_dim,
            hidden_dims=scale_hidden_dims,
            output_dim=scale_embedding_dim,
            dropout=0.0,
            input_layer_norm=self.use_pose_condition,
            activate_output=True,
        )
        self.film_parameters = nn.Linear(scale_embedding_dim, 2 * self.film_dim)
        self.post_film = make_mlp(
            input_dim=self.film_dim,
            hidden_dims=fusion_hidden_dims[1:-1],
            output_dim=self.output_dim,
            dropout=dropout,
            input_layer_norm=False,
            activate_output=True,
        )

    def forward(
        self,
        fusion_input: Tensor,
        virtual_log_scale_normalized: Tensor | None = None,
        virtual_camera_pose_table: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.use_pose_condition:
            if virtual_camera_pose_table is None:
                raise ValueError(
                    "Missing virtual_camera_pose_table for virtual pose FiLM."
                )
            pose = ensure_vector_batch(
                virtual_camera_pose_table,
                "virtual_camera_pose_table",
            ).float()
            if pose.shape[-1] != self.pose_dim:
                raise ValueError(
                    f"virtual_camera_pose_table must have shape [B, {self.pose_dim}]."
                )
            if pose.shape[0] != fusion_input.shape[0]:
                raise ValueError(
                    "Virtual FiLM condition tensors must share the batch dimension."
                )
            if not torch.isfinite(pose).all():
                raise ValueError(
                    "Virtual FiLM condition tensors must contain only finite values."
                )
            condition = pose.to(device=fusion_input.device, dtype=fusion_input.dtype)
        else:
            if virtual_log_scale_normalized is None:
                raise ValueError(
                    "Missing virtual_log_scale_normalized for direct-UV FiLM."
                )
            scale = ensure_vector_batch(
                virtual_log_scale_normalized,
                "virtual_log_scale_normalized",
            ).float()
            if scale.shape[-1] != 1:
                raise ValueError("virtual_log_scale_normalized must have shape [B, 1].")
            if scale.shape[0] != fusion_input.shape[0]:
                raise ValueError(
                    "Fusion features and virtual scale must share the batch dimension."
                )
            if not torch.isfinite(scale).all():
                raise ValueError(
                    "virtual_log_scale_normalized must contain only finite values."
                )
            condition = scale.to(device=fusion_input.device, dtype=fusion_input.dtype)

        hidden = self.pre_film(fusion_input)
        condition_embedding = self.condition_encoder(condition)
        gamma_raw, beta_raw = self.film_parameters(condition_embedding).chunk(2, dim=-1)
        gamma = self.delta_max * torch.tanh(gamma_raw)
        beta = self.delta_max * torch.tanh(beta_raw)
        modulated = (1.0 + gamma) * hidden + beta
        output = self.post_film(modulated)
        features = {
            "virtual_film_condition": condition,
            "virtual_film_condition_embedding": condition_embedding,
            "virtual_film_gamma": gamma,
            "virtual_film_beta": beta,
            "virtual_film_features": modulated,
        }
        if self.use_pose_condition:
            features["virtual_pose_embedding"] = condition_embedding
        else:
            features["virtual_scale_embedding"] = condition_embedding
        return output, features

    def reset_to_identity(self) -> None:
        nn.init.zeros_(self.film_parameters.weight)
        nn.init.zeros_(self.film_parameters.bias)


class EyeGeometryQualityGate(nn.Module):
    """Predict a bounded logit-space correction to offline PnP reliability."""

    def __init__(self, quality_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.quality_dim = quality_dim
        self.hidden = nn.Linear(quality_dim, hidden_dim)
        self.activation = nn.SiLU(inplace=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, quality: Tensor) -> Tensor:
        quality = ensure_vector_batch(quality, "eye_geometry_quality_vec")
        if quality.shape[-1] != self.quality_dim:
            raise ValueError(
                "Expected eye_geometry_quality_vec dim "
                f"{self.quality_dim}, got {quality.shape[-1]}"
            )
        if not torch.isfinite(quality).all():
            raise ValueError(
                "eye_geometry_quality_vec must contain only finite values."
            )
        return self.output(self.activation(self.hidden(quality.float())))

    def reset_output_to_zero(self) -> None:
        """Start learned-residual mode at the fixed analytical gate."""

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)


class DepthCorrectionHead(nn.Module):
    """Predict one bounded shared log-scale correction for both PnP eyes.

    The 10D input contains only train-normalized, label-free PnP diagnostics:
    six perspective-aware binocular geometry values and four solver-quality
    values.  The output scales both camera-space eyes about the camera origin,
    preserving their image rays and relative binocular structure.
    """

    def __init__(
        self,
        geometry_dim: int,
        quality_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        range_sigma: float,
        log_sigma_min: float,
        log_sigma_max: float,
    ) -> None:
        super().__init__()
        if geometry_dim != 6 or quality_dim != 4:
            raise ValueError("DepthCorrectionHead requires a 6D+4D input.")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("DepthCorrectionHead hidden_dims must be positive.")
        if dropout < 0:
            raise ValueError("DepthCorrectionHead dropout must be non-negative.")
        if range_sigma <= 0 or log_sigma_min <= 0:
            raise ValueError("Depth correction sigma limits must be positive.")
        if log_sigma_max < log_sigma_min:
            raise ValueError("log_sigma_max must be >= log_sigma_min.")

        layers: list[nn.Module] = []
        input_dim = geometry_dim + quality_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(input_dim, width), nn.SiLU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)
        self.geometry_dim = geometry_dim
        self.quality_dim = quality_dim
        self.range_sigma = float(range_sigma)
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)

    def reset_output_to_identity(self) -> None:
        """Make the initial correction exactly ``scale=1``."""

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        pnp_geometry: Tensor,
        pnp_quality: Tensor,
        raw_eye_geometry_mm: Tensor,
        depth_log_scale_sigma: Tensor,
    ) -> dict[str, Tensor]:
        geometry = ensure_vector_batch(pnp_geometry, "pnp_geometry_vec").float()
        quality = ensure_vector_batch(pnp_quality, "pnp_quality_vec").float()
        raw_eyes = ensure_vector_batch(
            raw_eye_geometry_mm,
            "raw_eye_geometry_mm",
        ).float()
        sigma = ensure_vector_batch(
            depth_log_scale_sigma,
            "depth_log_scale_sigma",
        ).float()
        if geometry.shape[-1] != self.geometry_dim:
            raise ValueError(
                f"Expected pnp_geometry_vec dim {self.geometry_dim}, "
                f"got {geometry.shape[-1]}."
            )
        if quality.shape[-1] != self.quality_dim:
            raise ValueError(
                f"Expected pnp_quality_vec dim {self.quality_dim}, "
                f"got {quality.shape[-1]}."
            )
        if raw_eyes.shape[-1] != 6:
            raise ValueError("raw_eye_geometry_mm must have shape [B, 6].")
        if sigma.shape[-1] != 1:
            raise ValueError("depth_log_scale_sigma must have shape [B, 1].")
        batch_sizes = {
            geometry.shape[0],
            quality.shape[0],
            raw_eyes.shape[0],
            sigma.shape[0],
        }
        if len(batch_sizes) != 1:
            raise ValueError("Depth correction inputs must share one batch size.")
        if not all(
            torch.isfinite(value).all()
            for value in (geometry, quality, raw_eyes, sigma)
        ):
            raise ValueError("Depth correction inputs must contain only finite values.")

        raw_logit = self.output(self.hidden(torch.cat((geometry, quality), dim=-1)))
        effective_sigma = sigma.clamp(
            min=self.log_sigma_min,
            max=self.log_sigma_max,
        )
        max_abs_log_correction = self.range_sigma * effective_sigma
        log_scale_correction = max_abs_log_correction * torch.tanh(raw_logit)
        shared_scale = torch.exp(log_scale_correction)
        corrected_eyes = raw_eyes * shared_scale
        standardized_correction = log_scale_correction / effective_sigma
        return {
            "depth_correction_raw_logit": raw_logit,
            "depth_log_scale_correction": log_scale_correction,
            "depth_correction_effective_sigma": effective_sigma,
            "depth_correction_standardized": standardized_correction,
            "depth_shared_scale": shared_scale,
            "corrected_eye_geometry_mm": corrected_eyes,
        }

class ModelV1(nn.Module):
    """Multi-branch gaze model with optional crop/camera and eye geometry."""

    def __init__(self, config: ModelV1Config | None = None) -> None:
        super().__init__()
        self.config = config or ModelV1Config()
        self.uses_gaze_geometry = (
            self.config.prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY
        )
        self.uses_deca_features = (
            self.config.deca_feature_representation
            != DECA_FEATURE_REPRESENTATION_NONE
        )

        self.face_branch = (
            (
                FactorizedGeometryFaceBranch(
                    embedding_dim=self.config.face_embedding_dim,
                    factor_embedding_dim=self.config.deca_factor_embedding_dim,
                    factor_hidden_dims=self.config.deca_factor_hidden_dims,
                    fusion_hidden_dims=self.config.face_hidden_dims,
                    dropout=self.config.branch_dropout,
                    detach_input=self.config.detach_deca_features,
                )
                if self.config.deca_branch_mode
                == DECA_BRANCH_MODE_FACTORIZED_GEOMETRY
                else FaceBranch(
                    input_dim=self.config.deca_feature_dim,
                    embedding_dim=self.config.face_embedding_dim,
                    hidden_dims=self.config.face_hidden_dims,
                    dropout=self.config.branch_dropout,
                    detach_input=self.config.detach_deca_features,
                )
            )
            if self.uses_deca_features
            else None
        )
        self.visual_encoder = (
            CrossAttentionVisualEncoder(
                attention_dim=self.config.visual_attention_dim,
                output_dim=self.config.visual_embedding_dim,
                num_heads=self.config.visual_attention_heads,
                ffn_dim=self.config.visual_attention_ffn_dim,
                dropout=self.config.visual_attention_dropout,
                eye_backbone=self.config.eye_backbone,
                eye_backbone_weights=self.config.eye_backbone_weights,
                freeze_face_backbone=self.config.freeze_face_image_backbone,
                face_feature_mode=self.config.face_feature_mode,
                eye_token_grid=self.config.eye_token_grid,
                eye_feature_stage=self.config.eye_feature_stage,
                eye_feature_mode=self.config.eye_feature_mode,
                eye_stem_mode=self.config.eye_stem_mode,
                use_binocular_self_attention=(
                    self.config.use_binocular_self_attention
                ),
                binocular_attention_heads=self.config.binocular_attention_heads,
                binocular_attention_dropout=(
                    self.config.binocular_attention_dropout
                ),
                use_iris_auxiliary=self.config.use_eye_iris_auxiliary,
                iris_auxiliary_feature_stage=(
                    self.config.eye_iris_auxiliary_feature_stage
                ),
                iris_auxiliary_hidden_channels=(
                    self.config.eye_iris_auxiliary_hidden_channels
                ),
                use_keypoint_auxiliary=(
                    self.config.use_eye_keypoint_auxiliary
                ),
                keypoint_count=self.config.eye_keypoint_count,
                keypoint_auxiliary_feature_stage=(
                    self.config.eye_keypoint_auxiliary_feature_stage
                ),
                keypoint_auxiliary_hidden_channels=(
                    self.config.eye_keypoint_auxiliary_hidden_channels
                ),
                keypoint_probability_temperature=(
                    self.config.eye_keypoint_probability_temperature
                ),
                keypoint_heatmap_size=self.config.eye_keypoint_heatmap_size,
                keypoint_predict_variance_correction=(
                    self.config.eye_keypoint_predict_variance_correction
                ),
                keypoint_max_abs_log_std_correction=(
                    self.config.eye_keypoint_max_abs_log_std_correction
                ),
                keypoint_shape_embedding_dim=(
                    self.config.eye_keypoint_shape_embedding_dim
                ),
                use_landmark_guided_fusion=(
                    self.config.use_landmark_guided_eye_fusion
                ),
                landmark_guided_fusion_heads=(
                    self.config.landmark_guided_fusion_heads
                ),
                landmark_guided_fusion_dropout=(
                    self.config.landmark_guided_fusion_dropout
                ),
                landmark_guided_fusion_gate_init=(
                    self.config.landmark_guided_fusion_gate_init
                ),
                landmark_guided_fusion_gate_max=(
                    self.config.landmark_guided_fusion_gate_max
                ),
                landmark_guided_quality_logit_scale=(
                    self.config.landmark_guided_quality_logit_scale
                ),
                landmark_guided_uncertainty_scale_norm=(
                    self.config.landmark_guided_uncertainty_scale_norm
                ),
                landmark_guided_min_quality=(
                    self.config.landmark_guided_min_quality
                ),
                landmark_guided_detach_probability_for_sampling=(
                    self.config.landmark_guided_detach_probability_for_sampling
                ),
                landmark_guided_detach_landmark_statistics=(
                    self.config.landmark_guided_detach_landmark_statistics
                ),
                landmark_guided_detach_quality=(
                    self.config.landmark_guided_detach_quality
                ),
                use_low_dof_eye_template=self.config.use_low_dof_eye_template,
                eye_template_image_size=self.config.eye_template_image_size,
                eye_template_pitch_range_deg=(
                    self.config.eye_template_pitch_range_deg
                ),
                eye_template_yaw_range_deg=(
                    self.config.eye_template_yaw_range_deg
                ),
                eye_template_pitch_bins=self.config.eye_template_pitch_bins,
                eye_template_yaw_bins=self.config.eye_template_yaw_bins,
                eye_template_posterior_temperature=(
                    self.config.eye_template_posterior_temperature
                ),
                eye_template_covariance_floor_px=(
                    self.config.eye_template_covariance_floor_px
                ),
                eye_template_min_canthus_distance_px=(
                    self.config.eye_template_min_canthus_distance_px
                ),
                eye_template_min_geometry_confidence=(
                    self.config.eye_template_min_geometry_confidence
                ),
                eye_template_center_energy_weight=(
                    self.config.eye_template_center_energy_weight
                ),
                eye_template_rim_energy_weight=(
                    self.config.eye_template_rim_energy_weight
                ),
                eye_template_angle_prior_weight=(
                    self.config.eye_template_angle_prior_weight
                ),
                eye_template_learnable_parameters=(
                    self.config.eye_template_learnable_parameters
                ),
                eye_template_use_reference_uncertainty=(
                    self.config.eye_template_use_reference_uncertainty
                ),
                eye_template_use_aperture_visibility=(
                    self.config.eye_template_use_aperture_visibility
                ),
                token_pooling=self.config.visual_token_pooling,
                token_pooling_hidden_dim=self.config.visual_token_pooling_hidden_dim,
                token_pooling_dropout=self.config.visual_token_pooling_dropout,
            )
            if self.config.use_face_image
            else None
        )
        self.eye_branch = (
            None
            if self.config.use_face_image
            else EyeBranch(
                per_eye_dim=self.config.per_eye_embedding_dim,
                embedding_dim=self.config.eye_embedding_dim,
                dropout=self.config.branch_dropout,
                share_encoder=self.config.share_eye_encoder,
                backbone=self.config.eye_backbone,
                backbone_weights=self.config.eye_backbone_weights,
            )
        )
        self.crop_cam_branch = (
            VectorBranch(
                name="crop_cam_vec",
                input_dim=self.config.crop_cam_dim,
                embedding_dim=self.config.crop_cam_embedding_dim,
                hidden_dims=self.config.crop_cam_hidden_dims,
                dropout=self.config.branch_dropout,
            )
            if self.config.use_crop_cam
            else None
        )
        self.scene_branch = (
            None
            if self.uses_gaze_geometry
            else VectorBranch(
                name="scene_vec",
                input_dim=self.config.scene_dim,
                embedding_dim=self.config.scene_embedding_dim,
                hidden_dims=self.config.scene_hidden_dims,
                dropout=self.config.branch_dropout,
            )
        )
        self.eye_geometry_quality_gate = (
            EyeGeometryQualityGate(
                quality_dim=self.config.eye_geometry_quality_dim,
                hidden_dim=self.config.eye_geometry_gate_hidden_dim,
            )
            if (
                self.config.use_eye_geometry
                and self.config.eye_geometry_gate_mode
                == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
            )
            else None
        )
        use_quality_gate = (
            self.config.use_eye_geometry
            and self.config.eye_geometry_gate_mode != EYE_GEOMETRY_GATE_NONE
        )

        fusion_input_dim = (
            (self.config.face_embedding_dim if self.uses_deca_features else 0)
            + (
                self.config.visual_embedding_dim
                if self.config.use_face_image
                else self.config.eye_embedding_dim
            )
            + (
                self.config.crop_cam_embedding_dim
                if self.config.use_crop_cam
                else 0
            )
            + (0 if self.uses_gaze_geometry else self.config.scene_embedding_dim)
            + (
                self.config.eye_geometry_dim
                if self.config.use_eye_geometry
                else 0
            )
            + (1 if use_quality_gate else 0)
        )
        fusion_output_dim = self.config.fusion_hidden_dims[-1]
        self.direct_uv_film_fusion = (
            DirectUVFiLMFusion(
                input_dim=fusion_input_dim,
                fusion_hidden_dims=self.config.fusion_hidden_dims,
                scale_hidden_dims=(
                    self.config.virtual_distance_film_hidden_dims
                ),
                scale_embedding_dim=(
                    self.config.virtual_distance_film_embedding_dim
                ),
                dropout=self.config.fusion_dropout,
                delta_max=self.config.virtual_distance_film_delta_max,
                use_pose_condition=self.config.use_virtual_pose_film,
                pose_dim=self.config.virtual_pose_dim,
            )
            if (
                self.config.use_virtual_distance_film
                or self.config.use_virtual_pose_film
            )
            else None
        )
        self.fusion_mlp = (
            None
            if self.direct_uv_film_fusion is not None
            else make_mlp(
                input_dim=fusion_input_dim,
                hidden_dims=self.config.fusion_hidden_dims[:-1],
                output_dim=fusion_output_dim,
                dropout=self.config.fusion_dropout,
                input_layer_norm=False,
                activate_output=True,
            )
        )
        self.uv_head = (
            None
            if self.uses_gaze_geometry
            else nn.Linear(fusion_output_dim, self.config.uv_dim)
        )
        self.pitch_to_v_mapper = (
            PitchToTableVResidual(
                scene_dim=self.config.scene_dim,
                pose_dim=(
                    self.config.virtual_pose_dim
                    if self.config.use_virtual_pose_film
                    else 0
                ),
                hidden_dims=self.config.pitch_to_v_hidden_dims,
                max_abs_delta_v_mm=self.config.pitch_to_v_max_abs_delta_mm,
                pitch_scale_deg=self.config.pitch_to_v_pitch_scale_deg,
                uncertainty_scale_deg=(
                    self.config.pitch_to_v_uncertainty_scale_deg
                ),
                disagreement_scale_deg=(
                    self.config.pitch_to_v_disagreement_scale_deg
                ),
                min_pitch_std_deg=self.config.pitch_to_v_min_pitch_std_deg,
                min_active_gate=self.config.pitch_to_v_min_active_gate,
                detach_uncertainty_gate=(
                    self.config.pitch_to_v_detach_uncertainty_gate
                ),
                table_distance_scale_mm=self.config.table_distance_scale_mm,
                use_camera_table_sensitivity=(
                    self.config.pitch_to_v_use_camera_table_sensitivity
                ),
                min_abs_table_normal_z=(
                    self.config.pitch_to_v_min_abs_table_normal_z
                ),
            )
            if self.config.use_pitch_to_v_residual
            else None
        )
        self.vertical_geometry_feature_extractor = (
            VerticalEyeGeometryFeatureExtractor(
                eye_image_size=self.config.eye_template_image_size
            )
            if self.config.use_vertical_geometry_residual
            else None
        )
        self.vertical_geometry_residual = (
            VerticalGeometryVResidual(
                geometry_dim=VERTICAL_EYE_GEOMETRY_FEATURE_DIM,
                fused_feature_dim=fusion_output_dim,
                scene_dim=self.config.scene_dim,
                pose_dim=(
                    self.config.virtual_pose_dim
                    if self.config.use_virtual_pose_film
                    else 0
                ),
                hidden_dims=self.config.vertical_geometry_residual_hidden_dims,
                max_abs_delta_v_mm=(
                    self.config.vertical_geometry_residual_max_abs_delta_mm
                ),
                min_training_gate=(
                    self.config.vertical_geometry_residual_min_training_gate
                ),
                min_inference_confidence=(
                    self.config.vertical_geometry_residual_min_inference_confidence
                ),
            )
            if self.config.use_vertical_geometry_residual
            else None
        )
        self.gaze_head = (
            nn.Linear(fusion_output_dim, self.config.gaze_dim)
            if self.uses_gaze_geometry
            else None
        )
        self.gaze_uncertainty_head = (
            nn.Linear(fusion_output_dim, 3)
            if self.uses_gaze_geometry and self.config.use_gaze_uncertainty
            else None
        )
        if self.uses_gaze_geometry:
            depth_config = DepthHypothesisConfig(
                mode=self.config.depth_distribution_mode,
                num_bins=self.config.depth_num_bins,
                range_sigma=self.config.depth_range_sigma,
                log_sigma_min=self.config.depth_log_sigma_min,
                log_sigma_max=self.config.depth_log_sigma_max,
            )
            self.depth_hypothesis_generator = PnpDepthHypothesisGenerator(
                depth_config
            )
            self.ray_table_geometry = RayTableGeometry(
                RayTableGeometryConfig(
                    table_distance_scale_mm=self.config.table_distance_scale_mm,
                    min_abs_normal_dot_gaze=(
                        self.config.geometry_min_abs_normal_dot_gaze
                    ),
                    min_lambda_mm=self.config.geometry_min_lambda_mm,
                    max_lambda_mm=self.config.geometry_max_lambda_mm,
                )
            )
            self.depth_reweighter = (
                DepthHypothesisReweighter(
                    context_dim=fusion_output_dim,
                    quality_dim=self.config.eye_geometry_quality_dim,
                    hidden_dims=self.config.depth_reweighter_hidden_dims,
                    dropout=self.config.depth_reweighter_dropout,
                    detach_context=self.config.depth_reweighter_detach_context,
                )
                if self.config.depth_distribution_mode
                == DEPTH_DISTRIBUTION_LEARNED_REWEIGHT
                else None
            )
            self.depth_correction_head = (
                DepthCorrectionHead(
                    geometry_dim=self.config.eye_geometry_dim,
                    quality_dim=self.config.eye_geometry_quality_dim,
                    hidden_dims=self.config.depth_correction_hidden_dims,
                    dropout=self.config.depth_correction_dropout,
                    range_sigma=self.config.depth_range_sigma,
                    log_sigma_min=self.config.depth_log_sigma_min,
                    log_sigma_max=self.config.depth_log_sigma_max,
                )
                if self.config.use_depth_correction
                else None
            )
        else:
            self.depth_hypothesis_generator = None
            self.ray_table_geometry = None
            self.depth_reweighter = None
            self.depth_correction_head = None

        self.reset_parameters()

    def predict_eye_landmarks(
        self,
        left_eye: Tensor,
        right_eye: Tensor,
        *,
        include_template: bool = True,
    ) -> dict[str, Tensor]:
        """Predict eye landmarks without running face fusion or the UV head."""

        if self.visual_encoder is None:
            raise RuntimeError(
                "Eye-landmark prediction requires use_face_image=True."
            )
        return self.visual_encoder.predict_eye_landmarks(
            left_eye,
            right_eye,
            include_template=include_template,
        )

    def forward(
        self,
        batch: Mapping[str, object] | None = None,
        *,
        deca_feat: Tensor | None = None,
        face: Tensor | None = None,
        left_eye: Tensor | None = None,
        right_eye: Tensor | None = None,
        crop_cam_vec: Tensor | None = None,
        scene_vec: Tensor | None = None,
        eye_geometry_vec: Tensor | None = None,
        eye_geometry_quality_vec: Tensor | None = None,
        eye_geometry_confidence: Tensor | None = None,
        eye_geometry_valid_mask: Tensor | None = None,
        raw_eye_geometry_mm: Tensor | None = None,
        depth_log_scale_sigma: Tensor | None = None,
        pnp_geometry_vec: Tensor | None = None,
        pnp_quality_vec: Tensor | None = None,
        rotation_n_from_c: Tensor | None = None,
        virtual_log_scale_normalized: Tensor | None = None,
        virtual_camera_pose_table: Tensor | None = None,
        return_features: bool = False,
        return_auxiliary: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Predict table-local gaze ``uv`` in the configured target space.

        The model can be called either with a batch dictionary or explicit
        tensors. Batch dictionaries should contain ``left_eye``, ``right_eye``,
        and ``scene_vec``. One of the keys in :data:`DECA_BATCH_KEYS` is also
        required unless ``deca_feature_representation='none'``. ``face`` is
        additionally required when
        ``config.use_face_image`` is enabled. ``crop_cam_vec`` is required when
        ``config.use_crop_cam`` is enabled. When
        ``config.use_eye_geometry`` is enabled, the batch must additionally
        contain a normalized ``eye_geometry_vec``. Virtual-camera gaze
        prediction additionally requires the per-sample ``rotation_n_from_c``.
        Direct-UV FiLM uses either the training-normalized scalar
        ``virtual_log_scale_normalized`` or, when virtual pose FiLM is enabled,
        ``virtual_camera_pose_table``.

        The default ModelV1 DataLoader uses z-score-normalized targets. Use
        ``UVTargetNormalizer.denormalize`` to convert this output to millimeters.
        """

        if batch is not None:
            if self.uses_deca_features and deca_feat is None:
                deca_feat = get_required_tensor(batch, DECA_BATCH_KEYS)
            if self.config.use_face_image and face is None:
                face = get_required_tensor(batch, ("face",))
            if left_eye is None:
                left_eye = get_required_tensor(batch, ("left_eye",))
            if right_eye is None:
                right_eye = get_required_tensor(batch, ("right_eye",))
            if self.config.use_crop_cam and crop_cam_vec is None:
                crop_cam_vec = get_required_tensor(batch, ("crop_cam_vec",))
            if scene_vec is None:
                scene_vec = get_required_tensor(
                    batch,
                    ("table_frame7", "scene_vec")
                    if self.uses_gaze_geometry
                    else (
                        ("table_frame7_n", "scene_vec")
                        if (
                            self.config.use_virtual_distance_film
                            or self.config.use_virtual_pose_film
                        )
                        else ("scene_vec",)
                    ),
                )
            if (
                self.config.use_virtual_distance_film
                and not self.config.use_virtual_pose_film
                and virtual_log_scale_normalized is None
            ):
                virtual_log_scale_normalized = get_required_tensor(
                    batch,
                    ("virtual_log_scale_normalized",),
                )
            if (
                self.config.use_virtual_pose_film
                and virtual_camera_pose_table is None
            ):
                virtual_camera_pose_table = get_required_tensor(
                    batch,
                    ("virtual_camera_pose_table",),
                )
            if self.uses_gaze_geometry:
                if (
                    self.config.gaze_prediction_frame
                    == GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA
                    and rotation_n_from_c is None
                ):
                    rotation_n_from_c = get_required_tensor(
                        batch,
                        ("rotation_n_from_c",),
                    )
                if raw_eye_geometry_mm is None:
                    raw_eye_geometry_mm = get_required_tensor(
                        batch,
                        ("raw_eye_geometry_mm",),
                    )
                if depth_log_scale_sigma is None:
                    depth_log_scale_sigma = get_required_tensor(
                        batch,
                        ("depth_log_scale_sigma",),
                    )
                if self.depth_correction_head is not None and pnp_geometry_vec is None:
                    pnp_geometry_vec = get_required_tensor(
                        batch,
                        ("pnp_geometry_vec",),
                    )
                if (
                    self.depth_reweighter is not None
                    or self.depth_correction_head is not None
                ) and pnp_quality_vec is None:
                    pnp_quality_vec = get_required_tensor(
                        batch,
                        ("pnp_quality_vec",),
                    )
            if self.config.use_eye_geometry and eye_geometry_vec is None:
                eye_geometry_vec = get_required_tensor(batch, ("eye_geometry_vec",))
            if self._uses_eye_geometry_quality_gate:
                if eye_geometry_confidence is None:
                    eye_geometry_confidence = get_required_tensor(
                        batch,
                        ("eye_geometry_confidence",),
                    )
                if eye_geometry_valid_mask is None:
                    eye_geometry_valid_mask = get_required_tensor(
                        batch,
                        ("eye_geometry_valid_mask",),
                    )
            if (
                self.config.eye_geometry_gate_mode
                == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
                and eye_geometry_quality_vec is None
            ):
                eye_geometry_quality_vec = get_required_tensor(
                    batch,
                    ("eye_geometry_quality_vec",),
                )

        if self.uses_deca_features and deca_feat is None:
            raise ValueError("Missing deca_feat for face_branch.")
        if self.config.use_face_image and face is None:
            raise ValueError("Missing face while use_face_image=True.")
        if left_eye is None:
            raise ValueError("Missing left_eye for eye_branch.")
        if right_eye is None:
            raise ValueError("Missing right_eye for eye_branch.")
        if self.config.use_crop_cam and crop_cam_vec is None:
            raise ValueError("Missing crop_cam_vec for crop_cam_branch.")
        if scene_vec is None:
            raise ValueError("Missing scene_vec/TableFrame7.")
        if (
            self.config.use_virtual_distance_film
            and not self.config.use_virtual_pose_film
            and virtual_log_scale_normalized is None
        ):
            raise ValueError(
                "Missing virtual_log_scale_normalized for direct-UV FiLM."
            )
        if (
            self.config.use_virtual_pose_film
            and virtual_camera_pose_table is None
        ):
            raise ValueError(
                "Missing virtual_camera_pose_table for virtual-camera pose FiLM."
            )
        if self.uses_gaze_geometry and raw_eye_geometry_mm is None:
            raise ValueError(
                "Missing raw_eye_geometry_mm for V4 parameter-free geometry."
            )
        if (
            self.uses_gaze_geometry
            and self.config.gaze_prediction_frame
            == GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA
            and rotation_n_from_c is None
        ):
            raise ValueError(
                "Missing rotation_n_from_c for virtual-camera gaze prediction."
            )
        if self.uses_gaze_geometry and depth_log_scale_sigma is None:
            raise ValueError("Missing depth_log_scale_sigma for V4 depth hypotheses.")
        if self.depth_reweighter is not None and pnp_quality_vec is None:
            raise ValueError("Missing pnp_quality_vec for learned depth reweighting.")
        if self.depth_correction_head is not None:
            if pnp_geometry_vec is None:
                raise ValueError("Missing pnp_geometry_vec for depth correction.")
            if pnp_quality_vec is None:
                raise ValueError("Missing pnp_quality_vec for depth correction.")
        if self.config.use_eye_geometry and eye_geometry_vec is None:
            raise ValueError(
                "Missing eye_geometry_vec while use_eye_geometry=True."
            )
        if self._uses_eye_geometry_quality_gate:
            if eye_geometry_confidence is None:
                raise ValueError(
                    "Missing eye_geometry_confidence while quality gate is enabled."
                )
            if eye_geometry_valid_mask is None:
                raise ValueError(
                    "Missing eye_geometry_valid_mask while quality gate is enabled."
                )
        if (
            self.config.eye_geometry_gate_mode
            == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
            and eye_geometry_quality_vec is None
        ):
            raise ValueError(
                "Missing eye_geometry_quality_vec in learned_residual mode."
            )

        face_features = None
        if self.face_branch is not None:
            assert deca_feat is not None
            face_features = self.face_branch(deca_feat)
        visual_debug_features: dict[str, Tensor] = {}
        if self.visual_encoder is not None:
            assert face is not None
            visual_outputs = self.visual_encoder(
                face,
                left_eye,
                right_eye,
                return_attention=return_features,
                return_auxiliary=(
                    return_auxiliary
                    or self.pitch_to_v_mapper is not None
                    or self.vertical_geometry_residual is not None
                ),
            )
            face_image_features = visual_outputs["face_image_features"]
            eye_features = visual_outputs["eye_features"]
            visual_features = visual_outputs["visual_features"]
            visual_debug_features = {
                key: value
                for key, value in visual_outputs.items()
                if key.startswith("eye_template_")
                or key.startswith("landmark_guided_")
                or key
                in {
                    "face_tokens",
                    "eye_tokens",
                    "pre_binocular_eye_tokens",
                    "pre_landmark_eye_tokens",
                    "cross_attended_tokens",
                    "cross_attention_weights",
                    "binocular_self_attention_weights",
                    "token_pooling_weights",
                    "left_iris_heatmap",
                    "right_iris_heatmap",
                    "left_eye_keypoint_logits",
                    "left_eye_keypoint_probability_maps",
                    "left_eye_keypoint_mean_xy",
                    "left_eye_keypoint_covariance",
                    "left_eye_keypoint_covariance_heatmap",
                    "left_eye_keypoint_variance_xy",
                    "left_eye_keypoint_log_std_correction",
                    "left_eye_keypoint_visibility_logits",
                    "left_eye_keypoint_visibility_probability",
                    "left_eye_keypoint_entropy",
                    "left_eye_keypoint_confidence",
                    "left_eye_keypoint_canonical_shape",
                    "left_eye_keypoint_shape_embedding",
                    "right_eye_keypoint_logits",
                    "right_eye_keypoint_probability_maps",
                    "right_eye_keypoint_mean_xy",
                    "right_eye_keypoint_covariance",
                    "right_eye_keypoint_covariance_heatmap",
                    "right_eye_keypoint_variance_xy",
                    "right_eye_keypoint_log_std_correction",
                    "right_eye_keypoint_visibility_logits",
                    "right_eye_keypoint_visibility_probability",
                    "right_eye_keypoint_entropy",
                    "right_eye_keypoint_confidence",
                    "right_eye_keypoint_canonical_shape",
                    "right_eye_keypoint_shape_embedding",
                }
            }
        else:
            assert self.eye_branch is not None
            face_image_features = None
            eye_features = self.eye_branch(left_eye, right_eye)
            visual_features = eye_features
        crop_cam_features = (
            self.crop_cam_branch(crop_cam_vec)
            if self.crop_cam_branch is not None and crop_cam_vec is not None
            else None
        )
        scene_features = (
            self.scene_branch(scene_vec)
            if self.scene_branch is not None
            else None
        )

        fusion_parts = [visual_features]
        if face_features is not None:
            fusion_parts.insert(0, face_features)
        if crop_cam_features is not None:
            fusion_parts.insert(2, crop_cam_features)
        if scene_features is not None:
            fusion_parts.append(scene_features)
        normalized_eye_geometry = None
        eye_geometry_for_fusion = None
        eye_geometry_gate = None
        eye_geometry_base_gate = None
        eye_geometry_gate_delta = None
        normalized_eye_geometry_quality = None
        if self.config.use_eye_geometry:
            assert eye_geometry_vec is not None
            normalized_eye_geometry = ensure_vector_batch(
                eye_geometry_vec,
                "eye_geometry_vec",
            )
            if normalized_eye_geometry.shape[-1] != self.config.eye_geometry_dim:
                raise ValueError(
                    "Expected eye_geometry_vec dim "
                    f"{self.config.eye_geometry_dim}, got "
                    f"{normalized_eye_geometry.shape[-1]}"
                )
            eye_geometry_for_fusion = normalized_eye_geometry.float()
            if self._uses_eye_geometry_quality_gate:
                assert eye_geometry_confidence is not None
                assert eye_geometry_valid_mask is not None
                confidence = self._validate_gate_scalar(
                    eye_geometry_confidence,
                    "eye_geometry_confidence",
                ).to(device=eye_geometry_for_fusion.device)
                valid_mask = self._validate_gate_scalar(
                    eye_geometry_valid_mask,
                    "eye_geometry_valid_mask",
                ).to(device=eye_geometry_for_fusion.device)
                geometry_batch_size = eye_geometry_for_fusion.shape[0]
                if (
                    confidence.shape[0] != geometry_batch_size
                    or valid_mask.shape[0] != geometry_batch_size
                ):
                    raise ValueError(
                        "Eye geometry, confidence, and valid mask must have "
                        "the same batch size."
                    )
                if torch.any(confidence < 0) or torch.any(confidence > 1):
                    raise ValueError(
                        "eye_geometry_confidence values must lie in [0, 1]."
                    )
                if torch.any(valid_mask < 0) or torch.any(valid_mask > 1):
                    raise ValueError(
                        "eye_geometry_valid_mask values must lie in [0, 1]."
                    )
                confidence = confidence.to(dtype=eye_geometry_for_fusion.dtype)
                valid_mask = valid_mask.to(dtype=eye_geometry_for_fusion.dtype)
                eye_geometry_base_gate = confidence.pow(1.0 / 3.0).clamp(
                    min=self.config.eye_geometry_gate_min,
                    max=self.config.eye_geometry_gate_max,
                )
                if (
                    self.config.eye_geometry_gate_mode
                    == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
                ):
                    assert eye_geometry_quality_vec is not None
                    assert self.eye_geometry_quality_gate is not None
                    normalized_eye_geometry_quality = ensure_vector_batch(
                        eye_geometry_quality_vec,
                        "eye_geometry_quality_vec",
                    )
                    if (
                        normalized_eye_geometry_quality.shape[0]
                        != geometry_batch_size
                    ):
                        raise ValueError(
                            "Eye geometry and quality vector must have the "
                            "same batch size."
                        )
                    gate_residual_raw = self.eye_geometry_quality_gate(
                        normalized_eye_geometry_quality
                    )
                    eye_geometry_gate_delta = (
                        self.config.eye_geometry_gate_delta_max
                        * torch.tanh(gate_residual_raw)
                    )
                    eye_geometry_gate = valid_mask * torch.sigmoid(
                        torch.logit(eye_geometry_base_gate)
                        + eye_geometry_gate_delta
                    )
                else:
                    eye_geometry_gate = valid_mask * eye_geometry_base_gate
                eye_geometry_for_fusion = (
                    eye_geometry_gate * eye_geometry_for_fusion
                )
            fusion_parts.append(eye_geometry_for_fusion)
            if eye_geometry_gate is not None:
                fusion_parts.append(eye_geometry_gate)
        fusion_input = torch.cat(fusion_parts, dim=-1)
        virtual_film_features: dict[str, Tensor] = {}
        if self.direct_uv_film_fusion is not None:
            fused_features, virtual_film_features = self.direct_uv_film_fusion(
                fusion_input,
                virtual_log_scale_normalized,
                virtual_camera_pose_table=virtual_camera_pose_table,
            )
        else:
            assert self.fusion_mlp is not None
            fused_features = self.fusion_mlp(fusion_input)
        if self.uses_gaze_geometry:
            assert self.gaze_head is not None
            assert self.depth_hypothesis_generator is not None
            assert self.ray_table_geometry is not None
            assert raw_eye_geometry_mm is not None
            gaze_raw = self.gaze_head(fused_features)
            gaze_norm = torch.linalg.vector_norm(
                gaze_raw,
                dim=-1,
                keepdim=True,
            )
            gaze_fallback = torch.zeros_like(gaze_raw)
            gaze_fallback[:, 2] = 1.0
            gaze_direction_prediction = torch.where(
                gaze_norm > 1e-8,
                gaze_raw / gaze_norm.clamp_min(1e-8),
                gaze_fallback,
            )
            gaze_cholesky = (
                tangent_cholesky_from_raw(
                    self.gaze_uncertainty_head(fused_features),
                    min_std_rad=self.config.gaze_uncertainty_min_std_rad,
                    max_std_rad=self.config.gaze_uncertainty_max_std_rad,
                )
                if self.gaze_uncertainty_head is not None
                else None
            )
            gaze_samples_prediction, gaze_sample_weights = tangent_sigma_points(
                gaze_direction_prediction,
                gaze_cholesky,
            )
            batch_size, gaze_sample_count, _ = gaze_samples_prediction.shape
            rotation_normalized_from_camera = None
            gaze_direction_n = None
            gaze_samples_n = None
            if (
                self.config.gaze_prediction_frame
                == GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA
            ):
                assert rotation_n_from_c is not None
                rotation_normalized_from_camera = ensure_rotation_batch(
                    rotation_n_from_c,
                    "rotation_n_from_c",
                    batch_size,
                ).to(
                    device=gaze_direction_prediction.device,
                    dtype=gaze_direction_prediction.dtype,
                )
                rotation_camera_from_normalized = (
                    rotation_normalized_from_camera.transpose(-1, -2)
                )
                gaze_direction_n = gaze_direction_prediction
                gaze_samples_n = gaze_samples_prediction
                gaze_direction_c = torch.bmm(
                    rotation_camera_from_normalized,
                    gaze_direction_n.unsqueeze(-1),
                ).squeeze(-1)
                gaze_samples_c = torch.einsum(
                    "bij,bkj->bki",
                    rotation_camera_from_normalized,
                    gaze_samples_n,
                )
            else:
                gaze_direction_c = gaze_direction_prediction
                gaze_samples_c = gaze_samples_prediction
            depth_correction_outputs: dict[str, Tensor] = {}
            eye_geometry_for_hypotheses = raw_eye_geometry_mm
            if self.depth_correction_head is not None:
                assert pnp_geometry_vec is not None
                assert pnp_quality_vec is not None
                assert depth_log_scale_sigma is not None
                depth_correction_outputs = self.depth_correction_head(
                    pnp_geometry_vec,
                    pnp_quality_vec,
                    raw_eye_geometry_mm,
                    depth_log_scale_sigma,
                )
                eye_geometry_for_hypotheses = depth_correction_outputs[
                    "corrected_eye_geometry_mm"
                ]
            depth_outputs = self.depth_hypothesis_generator(
                eye_geometry_for_hypotheses,
                depth_log_scale_sigma,
            )
            weight_outputs: dict[str, Tensor] = {}
            if self.depth_reweighter is not None:
                assert pnp_quality_vec is not None
                weight_outputs = self.depth_reweighter(
                    fused_features,
                    ensure_vector_batch(pnp_quality_vec, "pnp_quality_vec").float(),
                    depth_outputs["depth_log_scales"],
                    depth_outputs["depth_prior_log_weights"],
                )
                depth_weights = weight_outputs["depth_posterior_weights"]
            else:
                depth_weights = depth_outputs["depth_prior_weights"]
            # Flatten the gaze-sample dimension into the batch dimension so
            # the existing parameter-free ray/table module remains unchanged.
            depth_count = depth_outputs["eye_hypotheses_c_mm"].shape[1]
            geometry_flat = self.ray_table_geometry(
                gaze_samples_c.reshape(batch_size * gaze_sample_count, 3),
                depth_outputs["eye_hypotheses_c_mm"]
                .unsqueeze(1)
                .expand(-1, gaze_sample_count, -1, -1)
                .reshape(batch_size * gaze_sample_count, depth_count, 3),
                scene_vec.unsqueeze(1)
                .expand(-1, gaze_sample_count, -1)
                .reshape(batch_size * gaze_sample_count, scene_vec.shape[-1]),
            )
            joint_weights = (
                gaze_sample_weights.unsqueeze(-1) * depth_weights.unsqueeze(1)
            ).reshape(batch_size, gaze_sample_count * depth_count)
            uv_hypotheses = geometry_flat["uv_hypotheses_mm"].reshape(
                batch_size,
                gaze_sample_count * depth_count,
                2,
            )
            ray_finite_mask = geometry_flat["ray_finite_mask"].reshape(
                batch_size,
                gaze_sample_count * depth_count,
            )
            ray_valid_mask = geometry_flat["ray_valid_mask"].reshape(
                batch_size,
                gaze_sample_count * depth_count,
            )
            statistics = mixture_uv_statistics(
                uv_hypotheses,
                joint_weights,
                # A clamped near-parallel ray is finite but not a physically
                # meaningful UV hypothesis. Excluding it prevents a handful
                # of such rays from dominating the propagated covariance.
                ray_valid_mask,
            )
            geometry_outputs = {
                # These table-frame values are repeated for every gaze sample.
                key: value.reshape(batch_size, gaze_sample_count, *value.shape[1:])[:, 0]
                for key, value in geometry_flat.items()
                if key in {"e1_c", "e2_c", "n_c", "d_c_mm", "origin_c_mm"}
            }
            geometry_outputs.update(
                {
                    "gaze_direction_c": geometry_flat["gaze_direction_c"].reshape(
                        batch_size, gaze_sample_count, 3
                    )[:, 0],
                    "gaze_sample_directions_c": geometry_flat[
                        "gaze_direction_c"
                    ].reshape(batch_size, gaze_sample_count, 3),
                    "gaze_was_flipped": geometry_flat["gaze_was_flipped"].reshape(
                        batch_size, gaze_sample_count, 1
                    )[:, 0],
                    "gaze_sample_was_flipped": geometry_flat[
                        "gaze_was_flipped"
                    ].reshape(batch_size, gaze_sample_count, 1),
                    "uv_hypotheses_mm": uv_hypotheses,
                    "intersection_hypotheses_c_mm": geometry_flat[
                        "intersection_hypotheses_c_mm"
                    ].reshape(batch_size, gaze_sample_count * depth_count, 3),
                    "lambda_hypotheses_mm": geometry_flat[
                        "lambda_hypotheses_mm"
                    ].reshape(batch_size, gaze_sample_count * depth_count),
                    "ray_valid_mask": ray_valid_mask,
                    "ray_finite_mask": ray_finite_mask,
                    "normal_dot_gaze": geometry_flat["normal_dot_gaze"].reshape(
                        batch_size, gaze_sample_count, 1
                    ),
                }
            )
            statistics["geometry_any_physically_valid"] = ray_valid_mask.any(dim=-1)
            virtual_camera_outputs: dict[str, Tensor] = {}
            if gaze_direction_n is not None and gaze_samples_n is not None:
                assert rotation_normalized_from_camera is not None
                virtual_camera_outputs = {
                    "rotation_n_from_c": rotation_normalized_from_camera,
                    "rotation_c_from_n": rotation_normalized_from_camera.transpose(
                        -1, -2
                    ),
                    "gaze_axis_direction_n": gaze_direction_n,
                    "gaze_mean_direction_n": gaze_direction_n,
                    "gaze_sigma_points_n": gaze_samples_n,
                }
            return {
                "uv": statistics["uv_mean_mm"],
                "gaze_axis_direction_c": gaze_direction_c,
                "gaze_mean_direction_c": gaze_direction_c,
                "gaze_tangent_cholesky": (
                    gaze_cholesky
                    if gaze_cholesky is not None
                    else gaze_direction_c.new_zeros((batch_size, 2, 2))
                ),
                "gaze_tangent_covariance_rad2": (
                    gaze_cholesky @ gaze_cholesky.transpose(-1, -2)
                    if gaze_cholesky is not None
                    else gaze_direction_c.new_zeros((batch_size, 2, 2))
                ),
                "gaze_sigma_points_c": gaze_samples_c,
                "gaze_sigma_point_weights": gaze_sample_weights,
                "geometry_min_lambda_mm": gaze_direction_c.new_tensor(
                    self.config.geometry_min_lambda_mm
                ),
                "geometry_max_lambda_mm": gaze_direction_c.new_tensor(
                    self.config.geometry_max_lambda_mm
                ),
                "geometry_min_abs_normal_dot_gaze": gaze_direction_c.new_tensor(
                    self.config.geometry_min_abs_normal_dot_gaze
                ),
                **virtual_camera_outputs,
                **depth_outputs,
                **depth_correction_outputs,
                **geometry_outputs,
                **weight_outputs,
                **statistics,
                "face_features": (
                    face_features
                    if face_features is not None
                    else visual_features.new_empty((visual_features.shape[0], 0))
                ),
                "eye_features": eye_features,
                "visual_features": visual_features,
                "fused_features": fused_features,
                **visual_debug_features,
            }

        assert self.uv_head is not None
        uv = self.uv_head(fused_features)
        vertical_residual_outputs: dict[str, Tensor] = {}
        if self.pitch_to_v_mapper is not None:
            required_template_names = (
                "eye_template_pitch_rad",
                "eye_template_pitch_variance",
                "eye_template_geometry_confidence",
                "eye_template_valid_mask",
            )
            missing_template_names = [
                name
                for name in required_template_names
                if name not in visual_debug_features
            ]
            if missing_template_names:
                raise RuntimeError(
                    "Pitch-to-v mapper is missing eye-template outputs: "
                    f"{missing_template_names}."
                )
            mapping_outputs = self.pitch_to_v_mapper(
                pitch_rad=visual_debug_features["eye_template_pitch_rad"],
                pitch_variance=visual_debug_features[
                    "eye_template_pitch_variance"
                ],
                geometry_confidence=visual_debug_features[
                    "eye_template_geometry_confidence"
                ],
                geometry_valid_mask=visual_debug_features[
                    "eye_template_valid_mask"
                ],
                scene_vec=scene_vec,
                virtual_camera_pose_table=virtual_camera_pose_table,
            )
            vertical_residual_outputs = {
                f"vertical_{name}": value
                for name, value in mapping_outputs.items()
            }

        elif self.vertical_geometry_residual is not None:
            if self.vertical_geometry_feature_extractor is None:
                raise RuntimeError("Vertical geometry feature extractor is missing.")
            geometry_outputs = self.vertical_geometry_feature_extractor(
                stack_binocular_outputs(visual_debug_features, "mean_xy"),
                stack_binocular_outputs(visual_debug_features, "covariance"),
                stack_binocular_outputs(visual_debug_features, "visibility_probability"),
                stack_binocular_outputs(visual_debug_features, "entropy"),
            )
            mapping_outputs = self.vertical_geometry_residual(
                geometry_features=geometry_outputs["features"],
                geometry_confidence=geometry_outputs["geometry_confidence"],
                fused_features=fused_features,
                scene_vec=scene_vec,
                base_v_normalized=uv[:, 1:2],
                virtual_camera_pose_table=virtual_camera_pose_table,
            )
            vertical_residual_outputs = {
                f"vertical_{name}": value
                for name, value in mapping_outputs.items()
            }
            vertical_residual_outputs.update(
                {
                    f"vertical_geometry_{name}": value
                    for name, value in geometry_outputs.items()
                }
            )
        if (
            return_features
            or return_auxiliary
            or self.pitch_to_v_mapper is not None
            or self.vertical_geometry_residual is not None
        ):
            features = {
                "uv": uv,
                "face_features": (
                    face_features
                    if face_features is not None
                    else visual_features.new_empty((visual_features.shape[0], 0))
                ),
                "deca_features": (
                    face_features
                    if face_features is not None
                    else visual_features.new_empty((visual_features.shape[0], 0))
                ),
                "eye_features": eye_features,
                "visual_features": visual_features,
                "fused_features": fused_features,
            }
            if scene_features is not None:
                features["scene_features"] = scene_features
            if face_image_features is not None:
                features["face_image_features"] = face_image_features
            features.update(visual_debug_features)
            features.update(vertical_residual_outputs)
            features.update(virtual_film_features)
            if crop_cam_features is not None:
                features["crop_cam_features"] = crop_cam_features
            if normalized_eye_geometry is not None:
                features["eye_geometry_vec"] = normalized_eye_geometry
            if eye_geometry_for_fusion is not None:
                features["eye_geometry_for_fusion"] = eye_geometry_for_fusion
            if eye_geometry_gate is not None:
                features["eye_geometry_gate"] = eye_geometry_gate
            if eye_geometry_base_gate is not None:
                features["eye_geometry_base_gate"] = eye_geometry_base_gate
            if eye_geometry_gate_delta is not None:
                features["eye_geometry_gate_delta"] = eye_geometry_gate_delta
            if normalized_eye_geometry_quality is not None:
                features["eye_geometry_quality_vec"] = (
                    normalized_eye_geometry_quality
                )
            return features
        return uv

    @property
    def _uses_eye_geometry_quality_gate(self) -> bool:
        return (
            self.config.use_eye_geometry
            and self.config.eye_geometry_gate_mode != EYE_GEOMETRY_GATE_NONE
        )

    @staticmethod
    def _validate_gate_scalar(value: Tensor, name: str) -> Tensor:
        value = ensure_vector_batch(value, name).float()
        if value.shape[-1] != 1:
            raise ValueError(f"{name} must have shape [B, 1].")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values.")
        return value

    def eye_resnet_encoders(self) -> tuple[ResNetEyeImageEncoder, ...]:
        """Return the unique torchvision ResNet eye encoders."""

        unique: list[ResNetEyeImageEncoder] = []
        seen: set[int] = set()
        encoders = (
            (self.visual_encoder.eye_encoder,)
            if self.visual_encoder is not None
            else (
                self.eye_branch.resnet_encoders()
                if self.eye_branch is not None
                else ()
            )
        )
        for encoder in encoders:
            if id(encoder) not in seen:
                unique.append(encoder)
                seen.add(id(encoder))
        return tuple(unique)

    def eye_backbone_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return unique eye-ResNet backbone parameters, excluding projections."""

        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for encoder in self.eye_resnet_encoders():
            for parameter in encoder.backbone.parameters():
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        return tuple(parameters)

    def set_eye_backbone_train_from(self, stage: str | None) -> None:
        """Apply the same dynamic freeze boundary to every eye ResNet."""

        encoders = self.eye_resnet_encoders()
        if not encoders:
            raise ValueError(
                "Eye-backbone freeze scheduling requires a torchvision ResNet "
                f"backbone, got {self.config.eye_backbone!r}."
            )
        for encoder in encoders:
            encoder.set_backbone_train_from(stage)

    def reset_parameters(self) -> None:
        """Initialize trainable weights after all branches are built."""

        for module in self.modules():
            if getattr(module, "_preserve_existing_parameters", False):
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.eye_geometry_quality_gate is not None:
            self.eye_geometry_quality_gate.reset_output_to_zero()
        if self.depth_correction_head is not None:
            self.depth_correction_head.reset_output_to_identity()
        if self.direct_uv_film_fusion is not None:
            self.direct_uv_film_fusion.reset_to_identity()
        if self.pitch_to_v_mapper is not None:
            self.pitch_to_v_mapper.reset_output_to_zero()
        if self.vertical_geometry_residual is not None:
            self.vertical_geometry_residual.reset_output_to_zero()
        if self.visual_encoder is not None:
            self.visual_encoder.reset_token_pooling_to_mean()
            if self.visual_encoder.landmark_guided_fusion is not None:
                self.visual_encoder.landmark_guided_fusion.reset_gate()
        if self.gaze_uncertainty_head is not None:
            # A default Linear bias is zero, which would map through sigmoid
            # to the midpoint of the allowed range. Start narrowly instead,
            # then let NLL expand uncertainty when UV residuals require it.
            initial_fraction = (
                self.config.gaze_uncertainty_init_std_rad
                - self.config.gaze_uncertainty_min_std_rad
            ) / (
                self.config.gaze_uncertainty_max_std_rad
                - self.config.gaze_uncertainty_min_std_rad
            )
            initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
            nn.init.zeros_(self.gaze_uncertainty_head.weight)
            with torch.no_grad():
                self.gaze_uncertainty_head.bias.copy_(
                    torch.tensor(
                        (initial_logit, 0.0, initial_logit),
                        dtype=self.gaze_uncertainty_head.bias.dtype,
                    )
                )


def build_modelv1(config: ModelV1Config | None = None) -> ModelV1:
    """Factory used by training scripts."""

    return ModelV1(config)


def canonical_eye_backbone(name: str) -> str:
    normalized = (
        str(name)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace("*", "x")
        .replace("×", "x")
    )
    aliases = {
        "compact": COMPACT_EYE_BACKBONE,
        "compactcnn": COMPACT_EYE_BACKBONE,
        "cnn": COMPACT_EYE_BACKBONE,
        "resnet18": "resnet18",
        "resnet183x3": SMALL_IMAGE_RESNET18_BACKBONE,
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
        "resnet152": "resnet152",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_EYE_BACKBONES)
        raise ValueError(f"Unknown eye_backbone={name!r}; expected one of: {supported}") from exc


def canonical_eye_feature_stage(value: str) -> str:
    """Return the ResNet stage used as the eye spatial-token source."""

    stage = str(value).strip().lower().replace("-", "")
    if stage not in {"layer3", "layer4"}:
        raise ValueError(
            "eye_feature_stage must be 'layer3' or 'layer4', got "
            f"{value!r}."
        )
    return stage


def canonical_eye_feature_mode(value: str) -> str:
    """Return the eye spatial-feature extraction topology."""

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "single": EYE_FEATURE_MODE_SINGLE_STAGE,
        "single_stage": EYE_FEATURE_MODE_SINGLE_STAGE,
        "multiscale_fpn24": EYE_FEATURE_MODE_MULTISCALE_FPN24,
        "fpn24": EYE_FEATURE_MODE_MULTISCALE_FPN24,
        "h3": EYE_FEATURE_MODE_MULTISCALE_FPN24,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "eye_feature_mode must be 'single_stage' or 'multiscale_fpn24', got "
            f"{value!r}."
        ) from exc


def canonical_eye_stem_mode(value: str) -> str:
    """Return the ResNet input-stem topology for an eye image."""

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "standard": EYE_STEM_MODE_STANDARD,
        "imagenet": EYE_STEM_MODE_STANDARD,
        "small_3x3_s2_no_maxpool": EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL,
        "small_stem": EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL,
        "h5": EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "eye_stem_mode must be 'standard' or 'small_3x3_s2_no_maxpool', "
            f"got {value!r}."
        ) from exc


def canonical_eye_iris_auxiliary_feature_stage(value: str) -> str:
    """Return the ResNet stage used by the iris heatmap auxiliary head."""

    stage = str(value).strip().lower().replace("-", "")
    if stage not in {"layer2", "layer3"}:
        raise ValueError(
            "eye_iris_auxiliary_feature_stage must be 'layer2' or 'layer3', "
            f"got {value!r}."
        )
    return stage


def canonical_visual_token_pooling(value: str) -> str:
    """Return the aggregation rule for fused binocular eye tokens."""

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "mean": VISUAL_TOKEN_POOLING_MEAN,
        "average": VISUAL_TOKEN_POOLING_MEAN,
        "learned": VISUAL_TOKEN_POOLING_LEARNED,
        "attention": VISUAL_TOKEN_POOLING_LEARNED,
        "learned_attention": VISUAL_TOKEN_POOLING_LEARNED,
        "h4": VISUAL_TOKEN_POOLING_LEARNED,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "visual_token_pooling must be 'mean' or 'learned', got "
            f"{value!r}."
        ) from exc


def resnet_stage_feature_channels(backbone: nn.Module, stage: str) -> int:
    """Return the output-channel count of one torchvision ResNet stage."""

    if stage == "layer4":
        return int(backbone.fc.in_features)  # type: ignore[attr-defined]
    block = backbone.layer3[-1]  # type: ignore[attr-defined]
    norm = getattr(block, "bn3", None) or getattr(block, "bn2", None)
    if not isinstance(norm, nn.BatchNorm2d):
        raise TypeError("Could not infer the output channels of ResNet layer3.")
    return int(norm.num_features)


def canonical_resnet_stage(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "")
    aliases = {
        "none": "none",
        "false": "none",
        "stem": "stem",
        "layer1": "layer1",
        "layer2": "layer2",
        "layer3": "layer3",
        "layer4": "layer4",
        "all": "layer4",
        "backbone": "layer4",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(RESNET_STAGE_NAMES)
        raise ValueError(
            f"Unknown ResNet stage {value!r}; expected one of: {supported}"
        ) from exc


def resolve_resnet_weights(backbone: str, weights: str | None) -> Any:
    if weights is None:
        return None
    text = str(weights).strip()
    if text.lower() in NO_WEIGHT_VALUES:
        return None
    _, weights_enum = RESNET_EYE_BACKBONES[backbone]
    if text.upper() == "DEFAULT":
        return weights_enum.DEFAULT
    try:
        return weights_enum[text]
    except KeyError as exc:
        valid = ", ".join(["DEFAULT", *(item.name for item in weights_enum)])
        raise ValueError(
            f"Unknown eye_backbone_weights={weights!r} for {backbone}; expected one of: {valid}"
        ) from exc


def named_resnet_stages(
    backbone: nn.Module,
) -> tuple[tuple[str, tuple[nn.Module, ...]], ...]:
    """Return the canonical stem/layer1..4 module groups of a ResNet."""

    return (
        ("stem", (backbone.conv1, backbone.bn1)),
        ("layer1", (backbone.layer1,)),
        ("layer2", (backbone.layer2,)),
        ("layer3", (backbone.layer3,)),
        ("layer4", (backbone.layer4,)),
    )


def replace_first_conv(conv: nn.Conv2d, in_channels: int, preserve_rgb: bool) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    if preserve_rgb:
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
    return new_conv


def replace_resnet_small_image_stem(
    conv: nn.Conv2d,
    in_channels: int,
    preserve_rgb: bool,
) -> nn.Conv2d:
    """Build a CIFAR-style ResNet stem while retaining pretrained weights when possible."""

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    if preserve_rgb:
        with torch.no_grad():
            center_weights = conv.weight[:, :, 2:5, 2:5]
            if in_channels == 3:
                new_conv.weight.copy_(center_weights)
            elif in_channels == 1:
                new_conv.weight.copy_(center_weights.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
    return new_conv


def replace_resnet_h5_small_stem(
    conv: nn.Conv2d,
    in_channels: int,
    preserve_rgb: bool,
) -> nn.Conv2d:
    """Build H5's stride-2 3x3 stem from an ImageNet 7x7 stem when available."""

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    )
    if not preserve_rgb:
        return new_conv
    with torch.no_grad():
        center_weights = conv.weight[:, :, 2:5, 2:5]
        if in_channels == 3:
            source_weights = center_weights
        elif in_channels == 1:
            source_weights = center_weights.mean(dim=1, keepdim=True)
        else:
            nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
            return new_conv
        # Preserve each output filter's energy after the 7x7 -> 3x3 crop,
        # reducing the feature-scale shift seen by the inherited BN1 affine.
        source_norm = conv.weight.flatten(start_dim=1).norm(dim=1, keepdim=True)
        target_norm = source_weights.flatten(start_dim=1).norm(dim=1, keepdim=True)
        scale = (source_norm / target_norm.clamp_min(torch.finfo(source_weights.dtype).eps))
        new_conv.weight.copy_(source_weights * scale.view(-1, 1, 1, 1))
    return new_conv


def mark_preserve_parameters(module: nn.Module) -> None:
    for child in module.modules():
        setattr(child, "_preserve_existing_parameters", True)


def make_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    dropout: float,
    *,
    input_layer_norm: bool,
    activate_output: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    if input_layer_norm:
        layers.append(nn.LayerNorm(input_dim))

    dims = (input_dim, *hidden_dims, output_dim)
    for idx in range(len(dims) - 1):
        in_dim = dims[idx]
        out_dim = dims[idx + 1]
        is_last = idx == len(dims) - 2
        layers.append(nn.Linear(in_dim, out_dim))
        if not is_last or activate_output:
            layers.extend([nn.LayerNorm(out_dim), nn.SiLU(inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def ensure_vector_batch(x: Tensor, name: str) -> Tensor:
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim > 2:
        return x.flatten(start_dim=1)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D tensor, got shape {tuple(x.shape)}")
    return x


def ensure_rotation_batch(
    rotation: Tensor,
    name: str,
    batch_size: int,
) -> Tensor:
    if rotation.ndim == 2:
        rotation = rotation.unsqueeze(0)
    if rotation.ndim != 3 or rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"{name} must have shape [3, 3] or [B, 3, 3], got "
            f"{tuple(rotation.shape)}."
        )
    if rotation.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch size {rotation.shape[0]} does not match "
            f"gaze batch size {batch_size}."
        )
    # The validation matmul must stay in float32. Calling ``rotation.float()``
    # alone is insufficient under CUDA AMP: autocast would still execute the
    # matmul in float16, while ``identity`` remains float32, and ``allclose``
    # rejects the mixed dtypes before comparing their values.
    device_type = rotation.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        rotation_float = rotation.float()
        if not torch.isfinite(rotation_float).all():
            raise ValueError(f"{name} must contain only finite values.")
        identity = torch.eye(
            3,
            device=rotation_float.device,
            dtype=torch.float32,
        ).expand(batch_size, -1, -1)
        gram = rotation_float @ rotation_float.transpose(-1, -2)
        if not torch.allclose(
            gram,
            identity,
            atol=5e-4,
            rtol=5e-4,
        ):
            raise ValueError(f"{name} must contain orthonormal rotation matrices.")
        determinant = torch.linalg.det(rotation_float)
        if not torch.allclose(
            determinant,
            torch.ones_like(determinant),
            atol=5e-4,
            rtol=5e-4,
        ):
            raise ValueError(f"{name} must contain rotations with determinant +1.")
    return rotation_float


def ensure_image_batch(image: Tensor, name: str) -> Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(
            f"{name} image must be a CHW or BCHW tensor, got shape {tuple(image.shape)}"
        )
    return image


def get_required_tensor(batch: Mapping[str, object], keys: tuple[str, ...]) -> Tensor:
    for key in keys:
        if key not in batch:
            continue
        value = batch[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Batch key {key!r} must contain a torch.Tensor.")
        return value
    raise KeyError(f"Batch is missing one of: {', '.join(keys)}")
