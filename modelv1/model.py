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


DEFAULT_DECA_FEATURE_DIM = 236
DECA_BATCH_KEYS = (
    "deca_feat",
    "deca_features",
    "face_deca_feat",
    "face_deca_features",
)
COMPACT_EYE_BACKBONE = "cnn"
RESNET_EYE_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "resnet101": (models.resnet101, models.ResNet101_Weights),
    "resnet152": (models.resnet152, models.ResNet152_Weights),
}
SUPPORTED_EYE_BACKBONES = (
    COMPACT_EYE_BACKBONE,
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
)
NO_WEIGHT_VALUES = {"", "none", "null", "false", "random", "scratch"}


@dataclass(frozen=True)
class ModelV1Config:
    """Shape and width configuration for :class:`ModelV1`."""

    deca_feature_dim: int = DEFAULT_DECA_FEATURE_DIM
    crop_cam_dim: int = 36
    scene_dim: int = 25
    uv_dim: int = 2

    face_embedding_dim: int = 128
    face_hidden_dims: tuple[int, ...] = (256,)
    eye_embedding_dim: int = 128
    per_eye_embedding_dim: int = 96
    eye_backbone: str = COMPACT_EYE_BACKBONE
    eye_backbone_weights: str | None = None
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
            "crop_cam_dim": self.crop_cam_dim,
            "scene_dim": self.scene_dim,
            "uv_dim": self.uv_dim,
            "face_embedding_dim": self.face_embedding_dim,
            "eye_embedding_dim": self.eye_embedding_dim,
            "per_eye_embedding_dim": self.per_eye_embedding_dim,
            "crop_cam_embedding_dim": self.crop_cam_embedding_dim,
            "scene_embedding_dim": self.scene_embedding_dim,
        }
        object.__setattr__(self, "eye_backbone", canonical_eye_backbone(self.eye_backbone))
        if (
            self.eye_backbone not in RESNET_EYE_BACKBONES
            and self.eye_backbone_weights is not None
            and str(self.eye_backbone_weights).strip().lower() not in NO_WEIGHT_VALUES
        ):
            raise ValueError(
                "eye_backbone_weights is only supported for torchvision ResNet eye backbones."
            )
        non_positive = [name for name, value in dims.items() if value <= 0]
        if non_positive:
            raise ValueError(f"ModelV1Config dimensions must be positive: {non_positive}")
        if not self.fusion_hidden_dims:
            raise ValueError("fusion_hidden_dims must contain at least one layer width.")
        if self.branch_dropout < 0 or self.fusion_dropout < 0:
            raise ValueError("Dropout values must be non-negative.")


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
        if in_channels != 3:
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


class EyeBranch(nn.Module):
    """Encode left/right eye crops and fuse them into one eye embedding."""

    def __init__(
        self,
        per_eye_dim: int,
        embedding_dim: int,
        dropout: float,
        share_encoder: bool = True,
        backbone: str = COMPACT_EYE_BACKBONE,
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


class ModelV1(nn.Module):
    """V1 gaze model with face, eye, crop/camera, scene, fusion, and uv head."""

    def __init__(self, config: ModelV1Config | None = None) -> None:
        super().__init__()
        self.config = config or ModelV1Config()

        self.face_branch = FaceBranch(
            input_dim=self.config.deca_feature_dim,
            embedding_dim=self.config.face_embedding_dim,
            hidden_dims=self.config.face_hidden_dims,
            dropout=self.config.branch_dropout,
            detach_input=self.config.detach_deca_features,
        )
        self.eye_branch = EyeBranch(
            per_eye_dim=self.config.per_eye_embedding_dim,
            embedding_dim=self.config.eye_embedding_dim,
            dropout=self.config.branch_dropout,
            share_encoder=self.config.share_eye_encoder,
            backbone=self.config.eye_backbone,
            backbone_weights=self.config.eye_backbone_weights,
        )
        self.crop_cam_branch = VectorBranch(
            name="crop_cam_vec",
            input_dim=self.config.crop_cam_dim,
            embedding_dim=self.config.crop_cam_embedding_dim,
            hidden_dims=self.config.crop_cam_hidden_dims,
            dropout=self.config.branch_dropout,
        )
        self.scene_branch = VectorBranch(
            name="scene_vec",
            input_dim=self.config.scene_dim,
            embedding_dim=self.config.scene_embedding_dim,
            hidden_dims=self.config.scene_hidden_dims,
            dropout=self.config.branch_dropout,
        )

        fusion_input_dim = (
            self.config.face_embedding_dim
            + self.config.eye_embedding_dim
            + self.config.crop_cam_embedding_dim
            + self.config.scene_embedding_dim
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
        left_eye: Tensor | None = None,
        right_eye: Tensor | None = None,
        crop_cam_vec: Tensor | None = None,
        scene_vec: Tensor | None = None,
        return_features: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Predict table-local gaze ``uv`` in the configured target space.

        The model can be called either with a batch dictionary or explicit
        tensors. Batch dictionaries should contain ``left_eye``, ``right_eye``,
        ``crop_cam_vec``, ``scene_vec``, and one of the keys in
        :data:`DECA_BATCH_KEYS` for offline DECA features.

        The default ModelV1 DataLoader uses z-score-normalized targets. Use
        ``UVTargetNormalizer.denormalize`` to convert this output to millimeters.
        """

        if batch is not None:
            if deca_feat is None:
                deca_feat = get_required_tensor(batch, DECA_BATCH_KEYS)
            if left_eye is None:
                left_eye = get_required_tensor(batch, ("left_eye",))
            if right_eye is None:
                right_eye = get_required_tensor(batch, ("right_eye",))
            if crop_cam_vec is None:
                crop_cam_vec = get_required_tensor(batch, ("crop_cam_vec",))
            if scene_vec is None:
                scene_vec = get_required_tensor(batch, ("scene_vec",))

        if deca_feat is None:
            raise ValueError("Missing deca_feat for face_branch.")
        if left_eye is None:
            raise ValueError("Missing left_eye for eye_branch.")
        if right_eye is None:
            raise ValueError("Missing right_eye for eye_branch.")
        if crop_cam_vec is None:
            raise ValueError("Missing crop_cam_vec for crop_cam_branch.")
        if scene_vec is None:
            raise ValueError("Missing scene_vec for scene_branch.")

        face_features = self.face_branch(deca_feat)
        eye_features = self.eye_branch(left_eye, right_eye)
        crop_cam_features = self.crop_cam_branch(crop_cam_vec)
        scene_features = self.scene_branch(scene_vec)

        fusion_input = torch.cat(
            [face_features, eye_features, crop_cam_features, scene_features],
            dim=-1,
        )
        fused_features = self.fusion_mlp(fusion_input)
        uv = self.uv_head(fused_features)

        if return_features:
            return {
                "uv": uv,
                "face_features": face_features,
                "eye_features": eye_features,
                "crop_cam_features": crop_cam_features,
                "scene_features": scene_features,
                "fused_features": fused_features,
            }
        return uv

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


def build_modelv1(config: ModelV1Config | None = None) -> ModelV1:
    """Factory used by training scripts."""

    return ModelV1(config)


def canonical_eye_backbone(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "compact": COMPACT_EYE_BACKBONE,
        "compactcnn": COMPACT_EYE_BACKBONE,
        "cnn": COMPACT_EYE_BACKBONE,
        "resnet18": "resnet18",
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
