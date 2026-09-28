"""Multi-branch ModelV1 gaze regressor."""

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
    ProbabilisticEyeLandmarkHead,
)


PREDICTION_MODE_DIRECT_UV = "direct_uv"
PREDICTION_MODE_GAZE_GEOMETRY = "gaze_geometry"
PREDICTION_MODES = (PREDICTION_MODE_DIRECT_UV, PREDICTION_MODE_GAZE_GEOMETRY)
GAZE_PREDICTION_FRAME_CAMERA = "camera"
GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA = "virtual_camera"
GAZE_PREDICTION_FRAMES = (
    GAZE_PREDICTION_FRAME_CAMERA,
    GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA,
)
COMPACT_EYE_BACKBONE = "cnn"
DEFAULT_EYE_BACKBONE = "resnet18"
SMALL_IMAGE_RESNET18_BACKBONE = "resnet18_3x3"
FACE_IMAGE_PRETRAINED_DATASET = "vggface2"
FACE_IMAGE_BACKBONE = "inception_resnet_v1"
# FACE_IMAGE_FEATURE_CHANNELS = 1792
# FACE_IMAGE_TOKEN_COUNT = 9
# FACE_FEATURE_MODE_NATIVE_3X3 = "native_3x3"
# FACE_IMAGE_TOKEN_GRID = (3, 3)
FACE_IMAGE_OUTPUT_STAGE = "repeat_2"
FACE_IMAGE_FEATURE_CHANNELS = 896
FACE_IMAGE_TOKEN_COUNT = 64
FACE_FEATURE_MODE_NATIVE_8X8 = "native_8x8"
FACE_IMAGE_TOKEN_GRID = (8, 8)
EYE_IMAGE_FEATURE_CHANNELS = 512
EYE_IMAGE_TOKEN_GRID = (7, 12)
EYE_FEATURE_MODE_SINGLE_STAGE = "single_stage"
EYE_FEATURE_MODE_MULTISCALE_FPN24 = "multiscale_fpn24"
EYE_STEM_MODE_STANDARD = "standard"
EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL = "small_3x3_s2_no_maxpool"
TABLE_FRAME_FILM_DIM = 7
TABLE_FRAME_FILM_HIDDEN_DIMS = (16,)
TABLE_FRAME_FILM_EMBEDDING_DIM = 32
TABLE_FRAME_FILM_DELTA_MAX = 0.1
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


def canonical_gaze_prediction_frame(value: str) -> str:
    frame = str(value).strip().lower()
    if frame not in GAZE_PREDICTION_FRAMES:
        raise ValueError(
            f"Unknown gaze_prediction_frame={value!r}; expected one of "
            f"{GAZE_PREDICTION_FRAMES}."
        )
    return frame


@dataclass(frozen=True)
class ModelV1Config:
    """Shape and width configuration for :class:`ModelV1`."""

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

    use_face_image: bool = False
    freeze_face_image_backbone: bool = False
    eye_embedding_dim: int = 128
    per_eye_embedding_dim: int = 96
    eye_backbone: str = DEFAULT_EYE_BACKBONE
    eye_backbone_weights: str | None = None
    visual_embedding_dim: int = 128
    visual_attention_dim: int = 128
    visual_attention_heads: int = 8
    visual_attention_ffn_dim: int = 256
    visual_attention_dropout: float = 0.1
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
    crop_cam_embedding_dim: int = 64
    crop_cam_hidden_dims: tuple[int, ...] = (128,)
    scene_embedding_dim: int = 64
    scene_hidden_dims: tuple[int, ...] = (128,)
    fusion_hidden_dims: tuple[int, ...] = (256, 128)
    use_table_frame_film: bool = False

    branch_dropout: float = 0.1
    fusion_dropout: float = 0.2
    share_eye_encoder: bool = True

    def __post_init__(self) -> None:
        dims = {
            "crop_cam_dim": self.crop_cam_dim,
            "scene_dim": self.scene_dim,
            "eye_geometry_dim": self.eye_geometry_dim,
            "eye_geometry_quality_dim": self.eye_geometry_quality_dim,
            "eye_geometry_gate_hidden_dim": self.eye_geometry_gate_hidden_dim,
            "uv_dim": self.uv_dim,
            "gaze_dim": self.gaze_dim,
            "eye_embedding_dim": self.eye_embedding_dim,
            "per_eye_embedding_dim": self.per_eye_embedding_dim,
            "visual_embedding_dim": self.visual_embedding_dim,
            "visual_attention_dim": self.visual_attention_dim,
            "visual_attention_heads": self.visual_attention_heads,
            "visual_attention_ffn_dim": self.visual_attention_ffn_dim,
            "eye_keypoint_count": self.eye_keypoint_count,
            "eye_keypoint_auxiliary_hidden_channels": (
                self.eye_keypoint_auxiliary_hidden_channels
            ),
            "eye_keypoint_shape_embedding_dim": (
                self.eye_keypoint_shape_embedding_dim
            ),
            "landmark_guided_fusion_heads": self.landmark_guided_fusion_heads,
            "crop_cam_embedding_dim": self.crop_cam_embedding_dim,
            "scene_embedding_dim": self.scene_embedding_dim,
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
            if value <= 0
        ]
        if non_positive:
            raise ValueError(f"ModelV1Config dimensions must be positive: {non_positive}")
        if not self.fusion_hidden_dims:
            raise ValueError("fusion_hidden_dims must contain at least one layer width.")
        if not isinstance(self.use_table_frame_film, bool):
            raise ValueError("use_table_frame_film must be a boolean.")
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
        if self.branch_dropout < 0 or self.fusion_dropout < 0:
            raise ValueError("branch_dropout and fusion_dropout must be non-negative.")
        if not 0 <= self.visual_attention_dropout < 1:
            raise ValueError("visual_attention_dropout must be in [0, 1).")
        if self.visual_attention_dim % self.visual_attention_heads != 0:
            raise ValueError(
                "visual_attention_dim must be divisible by visual_attention_heads."
            )
        object.__setattr__(
            self,
            "eye_keypoint_auxiliary_feature_stage",
            canonical_eye_auxiliary_feature_stage(
                self.eye_keypoint_auxiliary_feature_stage
            ),
        )
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
        if self.use_face_image and self.eye_backbone not in RESNET_EYE_BACKBONES:
            raise ValueError(
                "Cross-attention visual fusion requires a torchvision ResNet "
                f"eye backbone, got {self.eye_backbone!r}."
            )
        if self.use_face_image and self.eye_backbone != "resnet18":
            raise ValueError(
                "The finalized Eye 7x12 token path requires eye_backbone='resnet18'."
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
        if self.use_face_image and not self.share_eye_encoder:
            raise ValueError(
                "Cross-attention visual fusion requires share_eye_encoder=True."
            )
        if not isinstance(self.use_eye_geometry, bool):
            raise ValueError("use_eye_geometry must be a boolean.")
        if not isinstance(self.use_crop_cam, bool):
            raise ValueError("use_crop_cam must be a boolean.")
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
            if self.use_table_frame_film:
                raise ValueError(
                    "TableFrame FiLM is only supported by direct_uv mode."
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
                self.use_table_frame_film
                and self.scene_representation != SCENE_REPRESENTATION_TABLE_FRAME7
            ):
                raise ValueError(
                    "TableFrame FiLM requires "
                    "scene_representation='table_frame7'."
                )
            if self.use_table_frame_film and len(self.fusion_hidden_dims) < 2:
                raise ValueError(
                    "Direct-UV TableFrame FiLM requires fusion_hidden_dims to contain "
                    "at least the FiLM width and output width."
                )
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
            layer4_projected = self.fpn_layer4(layer4)
            layer4_aligned = F.interpolate(
                layer4_projected.float(),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).to(dtype=layer4_projected.dtype)
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


class InceptionResnetFaceEncoder(nn.Module):
    """Return VGGFace2 Inception-ResNet spatial features for a 160x160 face."""

    def __init__(
        self,
        *,
        freeze_backbone: bool = False,
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
        self.output_grid = FACE_IMAGE_TOKEN_GRID
        self.output_stage = "repeat_2"
        self.backbone = InceptionResnetV1(
            pretrained=self.pretrained_dataset,
            classify=False,
        )
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
            # "mixed_7a",
            # "repeat_3",
            # "block8",
        ):
            x = getattr(self.backbone, stage_name)(x)
        # if tuple(x.shape[-2:]) != self.output_grid:
        #     raise ValueError(
        #         f"Expected a {self.output_grid[0]}x{self.output_grid[1]} "
        #         "Inception-ResNet face feature map, got "
        #         f"{tuple(x.shape[-2:])}."
            # )
        return x

    def train(self, mode: bool = True) -> InceptionResnetFaceEncoder:
        super().train(mode)
        if self.freeze_backbone:
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
    ) -> None:
        super().__init__()
        self.eye_feature_stage = "layer3"
        self.eye_feature_mode = EYE_FEATURE_MODE_MULTISCALE_FPN24
        self.eye_stem_mode = EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL
        self.use_keypoint_auxiliary = bool(use_keypoint_auxiliary)
        self.use_landmark_guided_fusion = bool(use_landmark_guided_fusion)
        if self.use_landmark_guided_fusion and not self.use_keypoint_auxiliary:
            raise ValueError(
                "use_landmark_guided_fusion requires use_keypoint_auxiliary=True."
            )
        self.face_token_grid = FACE_IMAGE_TOKEN_GRID
        self.face_token_count = math.prod(self.face_token_grid)
        self.keypoint_auxiliary_feature_stage = (
            canonical_eye_auxiliary_feature_stage(
                keypoint_auxiliary_feature_stage
            )
        )
        self.eye_token_grid = EYE_IMAGE_TOKEN_GRID
        self.binocular_eye_token_count = 2 * math.prod(self.eye_token_grid)
        self.face_encoder = InceptionResnetFaceEncoder(
            freeze_backbone=freeze_face_backbone,
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
        self.attention_norm = nn.LayerNorm(attention_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, attention_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(attention_dim)
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
            )  # 输出两眼的关键点相关信息，包括坐标、均值、方差、可见性等

        face_tokens = self.face_projection(self._spatial_tokens(face_map))  # [64, 1792, 3, 3] --> [64, 9, 128]
        left_tokens = self.eye_projection(self._spatial_tokens(left_eye_map))  # [64, 128, 7, 12] --> [64, 84, 128]
        right_tokens = self.eye_projection(self._spatial_tokens(right_eye_map))  # [64, 128, 7, 12] --> [64, 84, 128]
        face_tokens = face_tokens + self.face_position  # [64, 9, 128]
        left_count = left_tokens.shape[1]  # left_count=84
        left_tokens = (
            left_tokens
            + self.eye_position[:, :left_count]
            + self.left_eye_type
        )  # [64, 84, 128] + [1, 84, 128] + [1, 1, 128] = [64, 84, 128]
        right_tokens = (
            right_tokens
            + self.eye_position[:, left_count:]
            + self.right_eye_type
        )  # [64, 84, 128] + [1, 84, 128] + [1, 1, 128] = [64, 84, 128]
        eye_tokens = torch.cat((left_tokens, right_tokens), dim=1)   # [64, 168, 128]
        binocular_tokens = eye_tokens  # [64, 168, 128]
        landmark_fusion_outputs: dict[str, Tensor] = {}
        pre_landmark_eye_tokens: Tensor | None = None
        if self.landmark_guided_fusion is not None:
            pre_landmark_eye_tokens = binocular_tokens  # [64, 168, 128]
            fusion_outputs = self.landmark_guided_fusion(
                binocular_tokens,
                left_layer2=left_eye_maps["layer2"],
                left_layer3=left_eye_maps["layer3"],
                right_layer2=right_eye_maps["layer2"],
                right_layer3=right_eye_maps["layer3"],
                predictions=landmark_outputs,
                return_attention=return_attention,
            )
            binocular_tokens = fusion_outputs["refined_eye_tokens"]  # [64, 168, 128]
            landmark_fusion_outputs = {
                f"landmark_guided_{name}": value
                for name, value in fusion_outputs.items()
                if name not in {"refined_eye_tokens", "landmark_tokens"}
            }
        # attended, attention_weights = self.cross_attention(
        #     query=binocular_tokens,
        #     key=face_tokens,
        #     value=face_tokens,
        #     need_weights=return_attention,
        #     average_attn_weights=False,
        # )
        # attended = self.attention_norm(binocular_tokens + attended)  # 残差连接 + LayerNorm;[64, 168, 128]
        attended, attention_weights = self.cross_attention(
            query=face_tokens,
            key=binocular_tokens,
            value=binocular_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
            )
        attended = self.attention_norm(face_tokens + attended)

        fused_tokens = self.ffn_norm(attended + self.ffn(attended))  # FFN 前馈网络进一步融合，然后做LayerNorm;[64, 168, 128]
        pooled_tokens = fused_tokens.mean(dim=1)  # 对所有眼 token 做平均池化;[64, 128]
        visual_features = self.output_projection(pooled_tokens)  # 输出投影,视觉特征;[64, 128]

        output = {
            "visual_features": visual_features,  # 视觉特征;[64, 128]
            "face_image_features": face_tokens.mean(dim=1),  # 脸特征;[64, 128]
            "eye_features": binocular_tokens.mean(dim=1),  # 眼特征;[64, 128]
        }
        if return_auxiliary and self.eye_keypoint_head is not None:
            output.update(landmark_outputs)  # landmark_outputs:左右眼特征图进行关键点预测得到的原始输出，是关键点预测结果
        output.update(landmark_fusion_outputs)  # landmark_fusion_outputs：“关键点如何参与眼部特征融合”的状态，是关键点参与融合后的状态
        if return_attention and attention_weights is not None:
            output.update(
                {
                    "face_tokens": face_tokens,
                    "eye_tokens": binocular_tokens,
                    "cross_attended_tokens": fused_tokens,
                    "cross_attention_weights": attention_weights,
                }
            )
            if pre_landmark_eye_tokens is not None:
                output["pre_landmark_eye_tokens"] = pre_landmark_eye_tokens  # [64, 168, 128]
        return output

    def _eye_landmarks_from_feature_maps(
        self,
        left_eye_maps: Mapping[str, Tensor],
        right_eye_maps: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        # if self.eye_keypoint_head is None:
        #     raise RuntimeError("The probabilistic eye-landmark head is disabled.")
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
        return output

    def predict_eye_landmarks(
        self,
        left_eye: Tensor,
        right_eye: Tensor,
    ) -> dict[str, Tensor]:
        """Run only the shared eye encoder and probabilistic landmark head."""

        left_eye_maps = self.eye_encoder.forward_feature_maps(left_eye)
        right_eye_maps = self.eye_encoder.forward_feature_maps(right_eye)
        return self._eye_landmarks_from_feature_maps(
            left_eye_maps,
            right_eye_maps,
        )

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


class DirectUVTableFrameFiLMFusion(nn.Module):
    """Condition the first direct-UV fusion layer on TableFrame7."""

    def __init__(
        self,
        *,
        input_dim: int,
        fusion_hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if len(fusion_hidden_dims) < 2:
            raise ValueError(
                "fusion_hidden_dims must contain a FiLM width and output width."
            )
        self.film_dim = fusion_hidden_dims[0]
        self.output_dim = fusion_hidden_dims[-1]
        self.delta_max = TABLE_FRAME_FILM_DELTA_MAX
        self.pre_film = make_mlp(
            input_dim=input_dim,
            hidden_dims=(),
            output_dim=self.film_dim,
            dropout=dropout,
            input_layer_norm=False,
            activate_output=True,
        )
        self.condition_encoder = make_mlp(
            input_dim=TABLE_FRAME_FILM_DIM,
            hidden_dims=TABLE_FRAME_FILM_HIDDEN_DIMS,
            output_dim=TABLE_FRAME_FILM_EMBEDDING_DIM,
            dropout=0.0,
            input_layer_norm=True,
            activate_output=True,
        )
        self.film_parameters = nn.Linear(
            TABLE_FRAME_FILM_EMBEDDING_DIM,
            2 * self.film_dim,
        )
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
        table_frame7_n: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        frame = ensure_vector_batch(
            table_frame7_n,
            "table_frame7_n",
        ).float()
        # if frame.shape[-1] != TABLE_FRAME_FILM_DIM:
        #     raise ValueError(
        #         f"table_frame7_n must have shape [B, {TABLE_FRAME_FILM_DIM}]."
        #     )
        # if frame.shape[0] != fusion_input.shape[0]:
        #     raise ValueError(
        #         "TableFrame7 and fusion features must share the batch dimension."
        #     )
        # if not torch.isfinite(frame).all():
        #     raise ValueError("table_frame7_n must contain only finite values.")
        condition = frame.to(device=fusion_input.device, dtype=fusion_input.dtype)

        hidden = self.pre_film(fusion_input)
        condition_embedding = self.condition_encoder(condition)
        gamma_raw, beta_raw = self.film_parameters(condition_embedding).chunk(2, dim=-1)
        gamma = self.delta_max * torch.tanh(gamma_raw)
        beta = self.delta_max * torch.tanh(beta_raw)
        modulated = (1.0 + gamma) * hidden + beta
        output = self.post_film(modulated)
        features = {
            "table_frame_film_condition": condition,
            "table_frame_film_condition_embedding": condition_embedding,
            "table_frame_film_gamma": gamma,
            "table_frame_film_beta": beta,
            "table_frame_film_features": modulated,
        }
        features["table_frame_embedding"] = condition_embedding
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
            (
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
        self.direct_uv_table_frame_film_fusion = (
            DirectUVTableFrameFiLMFusion(
                input_dim=fusion_input_dim,
                fusion_hidden_dims=self.config.fusion_hidden_dims,
                dropout=self.config.fusion_dropout,
            )
            if self.config.use_table_frame_film
            else None
        )
        self.fusion_mlp = (
            None
            if self.direct_uv_table_frame_film_fusion is not None
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
    ) -> dict[str, Tensor]:
        """Predict eye landmarks without running face fusion or the UV head."""

        # if self.visual_encoder is None:
        #     raise RuntimeError(
        #         "Eye-landmark prediction requires use_face_image=True."
        #     )
        return self.visual_encoder.predict_eye_landmarks(
            left_eye,
            right_eye,
        )

    def forward(
        self,
        batch: Mapping[str, object] | None = None,
        *,
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
        table_frame7_n: Tensor | None = None,
        return_features: bool = False,
        return_auxiliary: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Predict table-local gaze ``uv`` in the configured target space.

        The model can be called either with a batch dictionary or explicit
        tensors. Batch dictionaries should contain ``left_eye``, ``right_eye``,
        and ``scene_vec``. ``face`` is additionally required when
        ``config.use_face_image`` is enabled. ``crop_cam_vec`` is required when
        ``config.use_crop_cam`` is enabled. When
        ``config.use_eye_geometry`` is enabled, the batch must additionally
        contain a normalized ``eye_geometry_vec``. Virtual-camera gaze
        prediction additionally requires the per-sample ``rotation_n_from_c``.
        Direct-UV TableFrame FiLM additionally requires ``table_frame7_n``.

        The default ModelV1 DataLoader uses z-score-normalized targets. Use
        ``UVTargetNormalizer.denormalize`` to convert this output to millimeters.
        """

        # if batch is not None:
        #     if self.config.use_face_image and face is None:
        #         face = get_required_tensor(batch, ("face",))
        #     if left_eye is None:
        #         left_eye = get_required_tensor(batch, ("left_eye",))
        #     if right_eye is None:
        #         right_eye = get_required_tensor(batch, ("right_eye",))
        #     if self.config.use_crop_cam and crop_cam_vec is None:
        #         crop_cam_vec = get_required_tensor(batch, ("crop_cam_vec",))
        #     if scene_vec is None:
        #         scene_vec = get_required_tensor(
        #             batch,
        #             ("table_frame7", "scene_vec")
        #             if self.uses_gaze_geometry
        #             else (
        #                 ("table_frame7_n", "scene_vec")
        #                 if self.config.use_table_frame_film
        #                 else ("scene_vec",)
        #             ),
        #         )
        #     if (
        #         self.config.use_table_frame_film
        #         and table_frame7_n is None
        #     ):
        #         table_frame7_n = get_required_tensor(
        #             batch,
        #             ("table_frame7_n",),
        #         )
        #     if self.uses_gaze_geometry:
        #         if (
        #             self.config.gaze_prediction_frame
        #             == GAZE_PREDICTION_FRAME_VIRTUAL_CAMERA
        #             and rotation_n_from_c is None
        #         ):
        #             rotation_n_from_c = get_required_tensor(
        #                 batch,
        #                 ("rotation_n_from_c",),
        #             )
        #         if raw_eye_geometry_mm is None:
        #             raw_eye_geometry_mm = get_required_tensor(
        #                 batch,
        #                 ("raw_eye_geometry_mm",),
        #             )
        #         if depth_log_scale_sigma is None:
        #             depth_log_scale_sigma = get_required_tensor(
        #                 batch,
        #                 ("depth_log_scale_sigma",),
        #             )
        #         if self.depth_correction_head is not None and pnp_geometry_vec is None:
        #             pnp_geometry_vec = get_required_tensor(
        #                 batch,
        #                 ("pnp_geometry_vec",),
        #             )
        #         if (
        #             self.depth_reweighter is not None
        #             or self.depth_correction_head is not None
        #         ) and pnp_quality_vec is None:
        #             pnp_quality_vec = get_required_tensor(
        #                 batch,
        #                 ("pnp_quality_vec",),
        #             )
        #     if self.config.use_eye_geometry and eye_geometry_vec is None:
        #         eye_geometry_vec = get_required_tensor(batch, ("eye_geometry_vec",))
        #     if self._uses_eye_geometry_quality_gate:
        #         if eye_geometry_confidence is None:
        #             eye_geometry_confidence = get_required_tensor(
        #                 batch,
        #                 ("eye_geometry_confidence",),
        #             )
        #         if eye_geometry_valid_mask is None:
        #             eye_geometry_valid_mask = get_required_tensor(
        #                 batch,
        #                 ("eye_geometry_valid_mask",),
        #             )
        #     if (
        #         self.config.eye_geometry_gate_mode
        #         == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
        #         and eye_geometry_quality_vec is None
        #     ):
        #         eye_geometry_quality_vec = get_required_tensor(
        #             batch,
        #             ("eye_geometry_quality_vec",),
        #         )
        face = get_required_tensor(batch, ("face",))
        left_eye = get_required_tensor(batch, ("left_eye",))
        right_eye = get_required_tensor(batch, ("right_eye",))
        table_frame7_n = get_required_tensor(batch, ("table_frame7_n",))
        scene_vec = table_frame7_n

        visual_debug_features: dict[str, Tensor] = {}
        visual_outputs = self.visual_encoder(
            face,
            left_eye,
            right_eye,
            return_attention=return_features,
            return_auxiliary=(
                return_auxiliary
            ),
        )
        face_image_features = visual_outputs["face_image_features"]  # 脸部 Token 平均池化后的特征;[64, 128]
        eye_features = visual_outputs["eye_features"]  # 眼部 Token 平均池化后的特征;[64, 128]
        visual_features = visual_outputs["visual_features"]  # 脸部与眼部经过主交叉注意力、FFN 和池化后的最终视觉特征;[64, 128]
        visual_debug_features = {
            key: value
            for key, value in visual_outputs.items()
            if key.startswith("landmark_guided_")
            or key
            in {
                "left_eye_keypoint_logits",
                "left_eye_keypoint_mean_xy",
                "left_eye_keypoint_covariance",
                "left_eye_keypoint_visibility_logits",
                "left_eye_keypoint_shape_embedding",

                "right_eye_keypoint_logits",
                "right_eye_keypoint_mean_xy",
                "right_eye_keypoint_covariance",
                "right_eye_keypoint_visibility_logits",
                "right_eye_keypoint_shape_embedding",
            }
        }
        # crop_cam_features = (
        #     self.crop_cam_branch(crop_cam_vec)
        #     if self.crop_cam_branch is not None and crop_cam_vec is not None
        #     else None
        # )
        scene_features = (
            self.scene_branch(scene_vec)  # [64, 7]
            if self.scene_branch is not None
            else None
        )  # [64, 64]

        fusion_parts = [visual_features]
        # if crop_cam_features is not None:
        #     fusion_parts.insert(2, crop_cam_features)
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
        fusion_input = torch.cat(fusion_parts, dim=-1)  # [64, 192]
        table_frame_film_features: dict[str, Tensor] = {}
        if self.direct_uv_table_frame_film_fusion is not None:
            assert table_frame7_n is not None
            fused_features, table_frame_film_features = self.direct_uv_table_frame_film_fusion(
                fusion_input,
                table_frame7_n,
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
                "eye_features": eye_features,
                "visual_features": visual_features,
                "fused_features": fused_features,
                **visual_debug_features,
            }

        assert self.uv_head is not None
        uv = self.uv_head(fused_features)
        if (
            return_features
            or return_auxiliary
        ):
            features = {
                "uv": uv,
                "eye_features": eye_features,
                "visual_features": visual_features,
                "fused_features": fused_features,
            }
            if scene_features is not None:
                features["scene_features"] = scene_features
            if face_image_features is not None:
                features["face_image_features"] = face_image_features
            features.update(visual_debug_features)
            features.update(table_frame_film_features)
            # if crop_cam_features is not None:
            #     features["crop_cam_features"] = crop_cam_features
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
        if self.direct_uv_table_frame_film_fusion is not None:
            self.direct_uv_table_frame_film_fusion.reset_to_identity()
        if self.visual_encoder is not None:
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


def canonical_eye_auxiliary_feature_stage(value: str) -> str:
    """Return the ResNet stage used by the eye-landmark auxiliary head."""

    stage = str(value).strip().lower().replace("-", "")
    if stage not in {"layer2", "layer3"}:
        raise ValueError(
            "eye_keypoint_auxiliary_feature_stage must be 'layer2' or 'layer3', "
            f"got {value!r}."
        )
    return stage


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
