"""Multi-branch ModelV1 gaze regressor.

The first ModelV1 revision keeps DECA outside the training graph: DECA is run
offline, frozen features are saved with each sample, and the face branch consumes
those feature vectors through ``deca_feat``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torchvision import models

from modelv1.geometry_gate import (
    EYE_GEOMETRY_GATE_LEARNED_RESIDUAL,
    EYE_GEOMETRY_GATE_NONE,
    canonical_eye_geometry_gate_mode,
)
from modelv1.deca_cache import (
    DECA_FEATURE_REPRESENTATION_FULL236,
    DECA_FEATURE_REPRESENTATION_GEOMETRY156,
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
    canonical_scene_representation,
    scene_representation_dim,
)


DEFAULT_DECA_FEATURE_DIM = 236
DECA_BATCH_KEYS = (
    "deca_feat",
    "deca_features",
    "face_deca_feat",
    "face_deca_features",
)
COMPACT_EYE_BACKBONE = "cnn"
DEFAULT_EYE_BACKBONE = "resnet18"
SMALL_IMAGE_RESNET18_BACKBONE = "resnet18_3x3"
DEFAULT_FACE_IMAGE_BACKBONE = "resnet18"
RESNET_EYE_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    SMALL_IMAGE_RESNET18_BACKBONE: (models.resnet18, models.ResNet18_Weights),
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "resnet101": (models.resnet101, models.ResNet101_Weights),
    "resnet152": (models.resnet152, models.ResNet152_Weights),
}
RESNET_FACE_IMAGE_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
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
SUPPORTED_FACE_IMAGE_BACKBONES = tuple(RESNET_FACE_IMAGE_BACKBONES)
FACE_IMAGE_FREEZE_POINTS = (
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

    face_embedding_dim: int = 128
    face_hidden_dims: tuple[int, ...] = (256,)
    use_face_image: bool = False
    face_image_embedding_dim: int = 128
    face_image_backbone: str = DEFAULT_FACE_IMAGE_BACKBONE
    face_image_backbone_weights: str | None = None
    face_image_freeze_until: str = "none"
    eye_embedding_dim: int = 128
    per_eye_embedding_dim: int = 96
    eye_backbone: str = DEFAULT_EYE_BACKBONE
    eye_backbone_weights: str | None = None
    visual_embedding_dim: int = 128
    visual_fusion_hidden_dims: tuple[int, ...] = (256,)
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
            "face_embedding_dim": self.face_embedding_dim,
            "face_image_embedding_dim": self.face_image_embedding_dim,
            "eye_embedding_dim": self.eye_embedding_dim,
            "per_eye_embedding_dim": self.per_eye_embedding_dim,
            "visual_embedding_dim": self.visual_embedding_dim,
            "crop_cam_embedding_dim": self.crop_cam_embedding_dim,
            "scene_embedding_dim": self.scene_embedding_dim,
        }
        object.__setattr__(self, "eye_backbone", canonical_eye_backbone(self.eye_backbone))
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
            "face_image_backbone",
            canonical_face_image_backbone(self.face_image_backbone),
        )
        object.__setattr__(
            self,
            "face_image_freeze_until",
            canonical_face_image_freeze_point(self.face_image_freeze_until),
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
        if (
            self.face_image_backbone_weights is not None
            and str(self.face_image_backbone_weights).strip().lower()
            not in NO_WEIGHT_VALUES
        ):
            resolve_face_image_resnet_weights(
                self.face_image_backbone,
                self.face_image_backbone_weights,
            )
        non_positive = [name for name, value in dims.items() if value <= 0]
        if non_positive:
            raise ValueError(f"ModelV1Config dimensions must be positive: {non_positive}")
        if not self.visual_fusion_hidden_dims:
            raise ValueError(
                "visual_fusion_hidden_dims must contain at least one layer width."
            )
        if not self.fusion_hidden_dims:
            raise ValueError("fusion_hidden_dims must contain at least one layer width.")
        if self.branch_dropout < 0 or self.fusion_dropout < 0:
            raise ValueError("Dropout values must be non-negative.")
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
        embedding_dim: int,
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
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        if resolved_weights is not None:
            mark_preserve_parameters(self.backbone)

    def forward(self, image: Tensor) -> Tensor:
        image = ensure_image_batch(image, "eye")
        return self.proj(self.backbone(image.float()))


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


class FaceImageEncoder(nn.Module):
    """Extract a compact appearance embedding from a 224x224 RGB face crop."""

    def __init__(
        self,
        embedding_dim: int,
        dropout: float,
        *,
        backbone: str = DEFAULT_FACE_IMAGE_BACKBONE,
        backbone_weights: str | None = None,
        freeze_until: str = "none",
    ) -> None:
        super().__init__()
        self.backbone_name = canonical_face_image_backbone(backbone)
        self.freeze_until = canonical_face_image_freeze_point(freeze_until)
        resolved_weights = resolve_face_image_resnet_weights(
            self.backbone_name,
            backbone_weights,
        )
        builder, _ = RESNET_FACE_IMAGE_BACKBONES[self.backbone_name]
        resnet = builder(weights=resolved_weights)
        feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self._frozen_backbone_modules = freeze_resnet_until(
            self.backbone,
            self.freeze_until,
        )
        if resolved_weights is not None:
            mark_preserve_parameters(self.backbone)

    def forward(self, image: Tensor) -> Tensor:
        image = ensure_image_batch(image, "face")
        return self.proj(self.backbone(image.float()))

    def train(self, mode: bool = True) -> FaceImageEncoder:
        super().train(mode)
        if mode:
            # Frozen BatchNorm layers must not update running statistics.
            for module in self._frozen_backbone_modules:
                module.eval()
        return self


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


class ModelV1(nn.Module):
    """Multi-branch gaze model with optional crop/camera and eye geometry."""

    def __init__(self, config: ModelV1Config | None = None) -> None:
        super().__init__()
        self.config = config or ModelV1Config()

        self.face_branch = (
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
        self.face_image_encoder = (
            FaceImageEncoder(
                embedding_dim=self.config.face_image_embedding_dim,
                dropout=self.config.branch_dropout,
                backbone=self.config.face_image_backbone,
                backbone_weights=self.config.face_image_backbone_weights,
                freeze_until=self.config.face_image_freeze_until,
            )
            if self.config.use_face_image
            else None
        )
        self.eye_branch = EyeBranch(
            per_eye_dim=self.config.per_eye_embedding_dim,
            embedding_dim=self.config.eye_embedding_dim,
            dropout=self.config.branch_dropout,
            share_encoder=self.config.share_eye_encoder,
            backbone=self.config.eye_backbone,
            backbone_weights=self.config.eye_backbone_weights,
        )
        self.visual_fusion = (
            make_mlp(
                input_dim=(
                    self.config.face_image_embedding_dim
                    + self.config.eye_embedding_dim
                ),
                hidden_dims=self.config.visual_fusion_hidden_dims,
                output_dim=self.config.visual_embedding_dim,
                dropout=self.config.branch_dropout,
                input_layer_norm=True,
                activate_output=True,
            )
            if self.config.use_face_image
            else None
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
        self.scene_branch = VectorBranch(
            name="scene_vec",
            input_dim=self.config.scene_dim,
            embedding_dim=self.config.scene_embedding_dim,
            hidden_dims=self.config.scene_hidden_dims,
            dropout=self.config.branch_dropout,
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
            self.config.face_embedding_dim
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
            + self.config.scene_embedding_dim
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
        self.uv_head = nn.Linear(fusion_output_dim, self.config.uv_dim)

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
        return_features: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Predict table-local gaze ``uv`` in the configured target space.

        The model can be called either with a batch dictionary or explicit
        tensors. Batch dictionaries should contain ``left_eye``, ``right_eye``,
        ``scene_vec``, and one of the keys in :data:`DECA_BATCH_KEYS` for
        offline DECA features. ``face`` is additionally required when
        ``config.use_face_image`` is enabled. ``crop_cam_vec`` is required when
        ``config.use_crop_cam`` is enabled. When
        ``config.use_eye_geometry`` is enabled, the batch must additionally
        contain a normalized ``eye_geometry_vec``.

        The default ModelV1 DataLoader uses z-score-normalized targets. Use
        ``UVTargetNormalizer.denormalize`` to convert this output to millimeters.
        """

        if batch is not None:
            if deca_feat is None:
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
                scene_vec = get_required_tensor(batch, ("scene_vec",))
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

        if deca_feat is None:
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
            raise ValueError("Missing scene_vec for scene_branch.")
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

        face_features = self.face_branch(deca_feat)
        face_image_features = (
            self.face_image_encoder(face)
            if self.face_image_encoder is not None and face is not None
            else None
        )
        eye_features = self.eye_branch(left_eye, right_eye)
        visual_features = (
            self.visual_fusion(
                torch.cat([face_image_features, eye_features], dim=-1)
            )
            if (
                self.visual_fusion is not None
                and face_image_features is not None
            )
            else eye_features
        )
        crop_cam_features = (
            self.crop_cam_branch(crop_cam_vec)
            if self.crop_cam_branch is not None and crop_cam_vec is not None
            else None
        )
        scene_features = self.scene_branch(scene_vec)

        fusion_parts = [
            face_features,
            visual_features,
            scene_features,
        ]
        if crop_cam_features is not None:
            fusion_parts.insert(2, crop_cam_features)
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
        uv = self.uv_head(fused_features)

        if return_features:
            features = {
                "uv": uv,
                "face_features": face_features,
                "deca_features": face_features,
                "eye_features": eye_features,
                "visual_features": visual_features,
                "scene_features": scene_features,
                "fused_features": fused_features,
            }
            if face_image_features is not None:
                features["face_image_features"] = face_image_features
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


def canonical_face_image_backbone(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_FACE_IMAGE_BACKBONES)
        raise ValueError(
            f"Unknown face_image_backbone={name!r}; expected one of: {supported}"
        ) from exc


def canonical_face_image_freeze_point(value: str) -> str:
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
        supported = ", ".join(FACE_IMAGE_FREEZE_POINTS)
        raise ValueError(
            "Unknown face_image_freeze_until="
            f"{value!r}; expected one of: {supported}"
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


def resolve_face_image_resnet_weights(backbone: str, weights: str | None) -> Any:
    if weights is None:
        return None
    text = str(weights).strip()
    if text.lower() in NO_WEIGHT_VALUES:
        return None
    _, weights_enum = RESNET_FACE_IMAGE_BACKBONES[backbone]
    if text.upper() == "DEFAULT":
        return weights_enum.DEFAULT
    try:
        return weights_enum[text]
    except KeyError as exc:
        valid = ", ".join(["DEFAULT", *(item.name for item in weights_enum)])
        raise ValueError(
            "Unknown face_image_backbone_weights="
            f"{weights!r} for {backbone}; expected one of: {valid}"
        ) from exc


def freeze_resnet_until(
    backbone: nn.Module,
    freeze_until: str,
) -> tuple[nn.Module, ...]:
    freeze_until = canonical_face_image_freeze_point(freeze_until)
    if freeze_until == "none":
        return ()

    named_stages = (
        ("stem", (backbone.conv1, backbone.bn1)),
        ("layer1", (backbone.layer1,)),
        ("layer2", (backbone.layer2,)),
        ("layer3", (backbone.layer3,)),
        ("layer4", (backbone.layer4,)),
    )
    frozen_modules: list[nn.Module] = []
    for stage_name, modules in named_stages:
        for module in modules:
            module.requires_grad_(False)
            frozen_modules.append(module)
        if stage_name == freeze_until:
            break
    return tuple(frozen_modules)


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
