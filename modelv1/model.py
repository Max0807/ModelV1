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
BINOCULAR_EYE_TOKEN_COUNT = 8
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
    eye_embedding_dim: int = 128
    per_eye_embedding_dim: int = 96
    eye_backbone: str = DEFAULT_EYE_BACKBONE
    eye_backbone_weights: str | None = None
    visual_embedding_dim: int = 128
    visual_attention_dim: int = 128
    visual_attention_heads: int = 8
    visual_attention_ffn_dim: int = 256
    visual_attention_dropout: float = 0.1
    crop_cam_embedding_dim: int = 64
    crop_cam_hidden_dims: tuple[int, ...] = (128,)
    scene_embedding_dim: int = 64
    scene_hidden_dims: tuple[int, ...] = (128,)
    fusion_hidden_dims: tuple[int, ...] = (256, 128)

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
        if self.branch_dropout < 0 or self.fusion_dropout < 0:
            raise ValueError("branch_dropout and fusion_dropout must be non-negative.")
        if not 0 <= self.visual_attention_dropout < 1:
            raise ValueError("visual_attention_dropout must be in [0, 1).")
        if self.visual_attention_dim % self.visual_attention_heads != 0:
            raise ValueError(
                "visual_attention_dim must be divisible by visual_attention_heads."
            )
        if self.use_face_image and self.eye_backbone not in RESNET_EYE_BACKBONES:
            raise ValueError(
                "Cross-attention visual fusion requires a torchvision ResNet "
                f"eye backbone, got {self.eye_backbone!r}."
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
        elif self.use_depth_correction:
            raise ValueError(
                "use_depth_correction=True requires prediction_mode='gaze_geometry'."
            )
        elif self.gaze_prediction_frame != GAZE_PREDICTION_FRAME_CAMERA:
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
    ) -> None:
        super().__init__()
        self.backbone_name = canonical_eye_backbone(backbone)
        resolved_weights = resolve_resnet_weights(self.backbone_name, weights)
        builder, _ = RESNET_EYE_BACKBONES[self.backbone_name]
        resnet = builder(weights=resolved_weights)
        if self.backbone_name == SMALL_IMAGE_RESNET18_BACKBONE:
            resnet.conv1 = replace_resnet_small_image_stem(
                resnet.conv1,
                in_channels,
                bool(resolved_weights),
            )
            resnet.maxpool = nn.Identity()
        elif in_channels != 3:
            resnet.conv1 = replace_first_conv(resnet.conv1, in_channels, bool(resolved_weights))
        feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet
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
        """Return the layer4 feature map before ResNet global pooling."""

        image = ensure_image_batch(image, "eye")
        x = image.float()
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        return self.backbone.layer4(x)

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
            "mixed_7a",
            "repeat_3",
            "block8",
        ):
            x = getattr(self.backbone, stage_name)(x)
        return x

    def train(self, mode: bool = True) -> InceptionResnetFaceEncoder:
        super().train(mode)
        if self.freeze_backbone:
            # A frozen pretrained backbone must also keep all BatchNorm
            # running statistics fixed.
            self.backbone.eval()
        return self


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
    ) -> None:
        super().__init__()
        self.face_encoder = InceptionResnetFaceEncoder(
            freeze_backbone=freeze_face_backbone,
        )
        self.eye_encoder = ResNetEyeImageEncoder(
            backbone=eye_backbone,
            embedding_dim=None,
            dropout=dropout,
            weights=eye_backbone_weights,
        )
        self.face_projection = nn.Sequential(
            nn.Linear(FACE_IMAGE_FEATURE_CHANNELS, attention_dim),
            nn.LayerNorm(attention_dim),
        )
        self.eye_projection = nn.Sequential(
            nn.Linear(EYE_IMAGE_FEATURE_CHANNELS, attention_dim),
            nn.LayerNorm(attention_dim),
        )
        self.face_position = nn.Parameter(
            torch.zeros(1, FACE_IMAGE_TOKEN_COUNT, attention_dim)
        )
        self.eye_position = nn.Parameter(
            torch.zeros(1, BINOCULAR_EYE_TOKEN_COUNT, attention_dim)
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
    ) -> dict[str, Tensor]:
        face_map = self.face_encoder(face)
        left_eye_map = self.eye_encoder.forward_feature_map(left_eye)
        right_eye_map = self.eye_encoder.forward_feature_map(right_eye)

        face_tokens = self.face_projection(self._spatial_tokens(face_map))
        left_tokens = self.eye_projection(self._spatial_tokens(left_eye_map))
        right_tokens = self.eye_projection(self._spatial_tokens(right_eye_map))
        if face_tokens.shape[1] != FACE_IMAGE_TOKEN_COUNT:
            raise ValueError(
                "Expected a 3x3 Inception-ResNet face feature map for 160x160 "
                f"input, got {tuple(face_map.shape[-2:])}."
            )
        if left_tokens.shape[1] + right_tokens.shape[1] != BINOCULAR_EYE_TOKEN_COUNT:
            raise ValueError(
                "Expected two 2x2 ResNet eye feature maps for 60x36 inputs, got "
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
        attended, attention_weights = self.cross_attention(
            query=eye_tokens,
            key=face_tokens,
            value=face_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        attended = self.attention_norm(eye_tokens + attended)
        fused_tokens = self.ffn_norm(attended + self.ffn(attended))
        visual_features = self.output_projection(fused_tokens.mean(dim=1))

        output = {
            "visual_features": visual_features,
            "face_image_features": face_tokens.mean(dim=1),
            "eye_features": eye_tokens.mean(dim=1),
        }
        if return_attention and attention_weights is not None:
            output.update(
                {
                    "face_tokens": face_tokens,
                    "eye_tokens": eye_tokens,
                    "cross_attended_tokens": fused_tokens,
                    "cross_attention_weights": attention_weights,
                }
            )
        return output


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
        self.fusion_mlp = make_mlp(
            input_dim=fusion_input_dim,
            hidden_dims=self.config.fusion_hidden_dims[:-1],
            output_dim=fusion_output_dim,
            dropout=self.config.fusion_dropout,
            input_layer_norm=False,
            activate_output=True,
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
        return_features: bool = False,
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
                    else ("scene_vec",),
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
            )
            face_image_features = visual_outputs["face_image_features"]
            eye_features = visual_outputs["eye_features"]
            visual_features = visual_outputs["visual_features"]
            visual_debug_features = {
                key: value
                for key, value in visual_outputs.items()
                if key
                in {
                    "face_tokens",
                    "eye_tokens",
                    "cross_attended_tokens",
                    "cross_attention_weights",
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

        if return_features:
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
