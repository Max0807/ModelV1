"""Standalone Blaze-style direct-UV model for architecture ablations.

This module is a PyTorch adaptation of the BlazeBlock topology used by
WebEyeTrack's BlazeGaze model.  It deliberately keeps all Blaze-specific code
outside :mod:`modelv1.model`; only the existing scene and TableFrame7-FiLM
components are reused so the ablation retains the same non-visual prediction
path as ModelV1.

Upstream reference:
    https://github.com/RedForestAI/WebEyeTrack
    python/webeyetrack/blazegaze.py

The upstream project is MIT licensed.  Its license text is retained under
``third_party/webeyetrack/LICENSE``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modelv1.model import (
    PREDICTION_MODE_DIRECT_UV,
    DirectUVTableFrameFiLMFusion,
    ModelV1Config,
    ResNetEyeImageEncoder,
    VectorBranch,
    ensure_image_batch,
    get_required_tensor,
)
from modelv1.scene import SCENE_REPRESENTATION_TABLE_FRAME7


@dataclass(frozen=True)
class BlazeStyleModelConfig(ModelV1Config):
    """Blaze-only options layered on the shared direct-UV configuration."""

    use_reconstruction_auxiliary: bool = False
    reconstruction_sample_limit: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.use_reconstruction_auxiliary, bool):
            raise ValueError("use_reconstruction_auxiliary must be a boolean.")
        if self.reconstruction_sample_limit <= 0:
            raise ValueError("reconstruction_sample_limit must be positive.")


def blaze_embedding_consistency_loss(
    embeddings: Tensor,
    uv_targets_mm: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Match pairwise latent distances to normalized pairwise gaze distances.

    This follows WebEyeTrack's embedding-consistency objective: gaze distances
    are normalized by the largest distance in the batch while embedding
    distances remain in their learned scale.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [B, D].")
    if uv_targets_mm.ndim != 2 or uv_targets_mm.shape[-1] != 2:
        raise ValueError("uv_targets_mm must have shape [B, 2].")
    if embeddings.shape[0] != uv_targets_mm.shape[0]:
        raise ValueError("embeddings and uv_targets_mm batch sizes must match.")
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0

    embedding_distances = torch.cdist(embeddings.float(), embeddings.float(), p=2)
    gaze_distances = torch.cdist(
        uv_targets_mm.float(),
        uv_targets_mm.float(),
        p=2,
    )
    normalized_gaze_distances = gaze_distances / gaze_distances.max().clamp_min(eps)
    return F.mse_loss(embedding_distances, normalized_gaze_distances)


class BlazeBlock(nn.Module):
    """Single depthwise-separable Blaze residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"BlazeBlock stride must be 1 or 2, got {stride}.")
        if stride == 1 and in_channels != out_channels:
            raise ValueError(
                "Stride-1 BlazeBlock requires matching input/output channels."
            )

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.residual_pool = (
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            if stride == 2
            else nn.Identity()
        )
        self.residual_projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            if stride == 2
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual_projection(self.residual_pool(x))
        features = self.pointwise(self.depthwise(x))
        return self.activation(features + residual)


class DoubleBlazeBlock(nn.Module):
    """Two-stage depthwise-separable Blaze residual block."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(
                f"DoubleBlazeBlock stride must be 1 or 2, got {stride}."
            )
        if stride == 1 and in_channels != out_channels:
            raise ValueError(
                "Stride-1 DoubleBlazeBlock requires matching input/output channels."
            )

        self.depthwise1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=in_channels,
            bias=True,
        )
        self.pointwise1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.depthwise2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=5,
            padding=2,
            groups=hidden_channels,
            bias=True,
        )
        self.pointwise2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )
        self.residual_pool = (
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            if stride == 2
            else nn.Identity()
        )
        self.residual_projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            if stride == 2
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual_projection(self.residual_pool(x))
        features = self.pointwise1(self.depthwise1(x))
        features = self.activation(features)
        features = self.pointwise2(self.depthwise2(features))
        return self.activation(features + residual)


class BlazeImageEncoder(nn.Module):
    """BlazeGaze CNN encoder with a fixed, modality-specific input canvas."""

    def __init__(self, input_size: tuple[int, int]) -> None:
        super().__init__()
        height, width = input_size
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(
                "BlazeImageEncoder input dimensions must be divisible by 64; "
                f"got {input_size}."
            )
        self.input_size = (int(height), int(width))
        self.output_spatial_shape = (32, height // 64, width // 64)
        self.output_dim = math.prod(self.output_spatial_shape)

        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 48, stride=2),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48),
            DoubleBlazeBlock(48, 24, 96, stride=2),
            DoubleBlazeBlock(96, 24, 96),
            DoubleBlazeBlock(96, 24, 96),
            DoubleBlazeBlock(96, 24, 96, stride=2),
            DoubleBlazeBlock(96, 24, 96),
            DoubleBlazeBlock(96, 24, 96),
        )
        self.squeeze = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: Tensor) -> Tensor:
        image = ensure_image_batch(image, "blaze_image")
        if image.shape[1] != 3:
            raise ValueError(
                f"BlazeImageEncoder expects RGB input, got {image.shape[1]} channels."
            )
        if tuple(image.shape[-2:]) != self.input_size:
            image = F.interpolate(
                image,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        features = self.squeeze(self.blocks(self.stem(image.float())))
        flattened = torch.flatten(features, start_dim=1)
        if flattened.shape[-1] != self.output_dim:
            raise RuntimeError(
                "Unexpected Blaze encoder output size: "
                f"expected {self.output_dim}, got {flattened.shape[-1]}."
            )
        return flattened


class BlazeStyleVisualEncoder(nn.Module):
    """Task-aligned Blaze encoder for the existing face/left/right inputs."""

    def __init__(self, output_dim: int, dropout: float) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if dropout < 0:
            raise ValueError("dropout must be non-negative.")

        # The paper's single 128x512 eye strip is represented here by two
        # weight-shared 128x256 eye canvases.  The current 160x160 normalized
        # face is retained through a separate 128x128 Blaze branch so the
        # ablation uses all image modalities available to ModelV1.
        self.face_encoder = BlazeImageEncoder((128, 128))
        self.eye_encoder = BlazeImageEncoder((128, 256))
        self.landmark_guided_fusion = None

        self.face_projection = nn.Sequential(
            nn.Linear(self.face_encoder.output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.eye_projection = nn.Sequential(
            nn.Linear(2 * self.eye_encoder.output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        face: Tensor,
        left_eye: Tensor,
        right_eye: Tensor,
        *,
        return_attention: bool = False,
        return_auxiliary: bool = False,
    ) -> dict[str, Tensor]:
        del return_attention, return_auxiliary
        face_features = self.face_projection(self.face_encoder(face))
        left_features = self.eye_encoder(left_eye)
        right_features = self.eye_encoder(right_eye)
        left_spatial = left_features.reshape(
            left_features.shape[0],
            *self.eye_encoder.output_spatial_shape,
        )
        right_spatial = right_features.reshape(
            right_features.shape[0],
            *self.eye_encoder.output_spatial_shape,
        )
        blaze_eye_latent = torch.cat(
            (left_spatial, right_spatial),
            dim=-1,
        ).flatten(start_dim=1)
        eye_features = self.eye_projection(blaze_eye_latent)
        visual_features = self.output_projection(
            torch.cat((face_features, eye_features), dim=-1)
        )
        return {
            "face_image_features": face_features,
            "eye_features": eye_features,
            "visual_features": visual_features,
            "blaze_eye_latent": blaze_eye_latent,
        }


class BlazeDecoderBlock(nn.Module):
    """Transpose-convolution block used by the eye-strip decoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=1 if stride == 2 else 0,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class BlazeEyeStripDecoder(nn.Module):
    """Decode the 512D binocular latent into a 128x512 RGB eye strip."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            BlazeDecoderBlock(32, 96),
            BlazeDecoderBlock(96, 96, stride=2),
            BlazeDecoderBlock(96, 96),
            BlazeDecoderBlock(96, 96, stride=2),
            BlazeDecoderBlock(96, 48),
            BlazeDecoderBlock(48, 48, stride=2),
            BlazeDecoderBlock(48, 48),
            BlazeDecoderBlock(48, 48, stride=2),
            BlazeDecoderBlock(48, 24),
            BlazeDecoderBlock(24, 24, stride=2),
            BlazeDecoderBlock(24, 24),
            BlazeDecoderBlock(24, 24, stride=2),
        )
        self.output = nn.Sequential(
            nn.Conv2d(24, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[-1] != 512:
            raise ValueError("BlazeEyeStripDecoder expects latent shape [B, 512].")
        spatial = latent.reshape(latent.shape[0], 32, 2, 8)
        return self.output(self.blocks(spatial))


class BlazeStyleModel(nn.Module):
    """Standalone direct-UV model sharing ModelV1's scene/FiLM prediction path."""

    def __init__(self, config: BlazeStyleModelConfig) -> None:
        super().__init__()
        self.config = config
        self.uses_gaze_geometry = False
        self.depth_reweighter = None
        self.depth_correction_head = None
        self._validate_config()

        self.visual_encoder = BlazeStyleVisualEncoder(
            output_dim=config.visual_embedding_dim,
            dropout=config.branch_dropout,
        )
        self.scene_branch = VectorBranch(
            name="scene_vec",
            input_dim=config.scene_dim,
            embedding_dim=config.scene_embedding_dim,
            hidden_dims=config.scene_hidden_dims,
            dropout=config.branch_dropout,
        )
        fusion_input_dim = config.visual_embedding_dim + config.scene_embedding_dim
        self.direct_uv_table_frame_film_fusion = DirectUVTableFrameFiLMFusion(
            input_dim=fusion_input_dim,
            fusion_hidden_dims=config.fusion_hidden_dims,
            dropout=config.fusion_dropout,
        )
        self.uv_head = nn.Linear(config.fusion_hidden_dims[-1], config.uv_dim)
        self.reconstruction_decoder = (
            BlazeEyeStripDecoder()
            if config.use_reconstruction_auxiliary
            else None
        )
        self.reset_parameters()

    def _validate_config(self) -> None:
        violations: list[str] = []
        if self.config.prediction_mode != PREDICTION_MODE_DIRECT_UV:
            violations.append("prediction_mode must be 'direct_uv'")
        if not self.config.use_face_image:
            violations.append("use_face_image must be true")
        if self.config.use_crop_cam:
            violations.append("use_crop_cam must be false")
        if self.config.use_eye_geometry:
            violations.append("use_eye_geometry must be false")
        if self.config.scene_representation != SCENE_REPRESENTATION_TABLE_FRAME7:
            violations.append("scene_representation must be 'table_frame7'")
        if not self.config.use_table_frame_film:
            violations.append("use_table_frame_film must be true")
        if self.config.use_eye_keypoint_auxiliary:
            violations.append("use_eye_keypoint_auxiliary must be false")
        if self.config.use_landmark_guided_eye_fusion:
            violations.append("use_landmark_guided_eye_fusion must be false")
        if violations:
            raise ValueError(
                "BlazeStyleModel configuration is invalid: "
                + "; ".join(violations)
                + "."
            )

    def forward(
        self,
        batch: Mapping[str, object] | None = None,
        *,
        face: Tensor | None = None,
        left_eye: Tensor | None = None,
        right_eye: Tensor | None = None,
        table_frame7_n: Tensor | None = None,
        return_features: bool = False,
        return_auxiliary: bool = False,
        **_: object,
    ) -> Tensor | dict[str, Tensor]:
        if batch is not None:
            if face is None:
                face = get_required_tensor(batch, ("face",))
            if left_eye is None:
                left_eye = get_required_tensor(batch, ("left_eye",))
            if right_eye is None:
                right_eye = get_required_tensor(batch, ("right_eye",))
            if table_frame7_n is None:
                table_frame7_n = get_required_tensor(batch, ("table_frame7_n",))
        if face is None or left_eye is None or right_eye is None:
            raise ValueError("BlazeStyleModel requires face, left_eye, and right_eye.")
        if table_frame7_n is None:
            raise ValueError("BlazeStyleModel requires table_frame7_n.")

        visual_outputs = self.visual_encoder(
            face,
            left_eye,
            right_eye,
            return_attention=return_features,
            return_auxiliary=return_auxiliary,
        )
        scene_features = self.scene_branch(table_frame7_n)
        fusion_input = torch.cat(
            (visual_outputs["visual_features"], scene_features),
            dim=-1,
        )
        fused_features, film_features = self.direct_uv_table_frame_film_fusion(
            fusion_input,
            table_frame7_n,
        )
        uv = self.uv_head(fused_features)
        if not (return_features or return_auxiliary):
            return uv
        outputs = {
            "uv": uv,
            **visual_outputs,
            "scene_features": scene_features,
            "fused_features": fused_features,
            **film_features,
        }
        if return_auxiliary and self.reconstruction_decoder is not None:
            sample_count = min(
                int(self.config.reconstruction_sample_limit),
                int(visual_outputs["blaze_eye_latent"].shape[0]),
            )
            outputs["reconstructed_eye_strip"] = self.reconstruction_decoder(
                visual_outputs["blaze_eye_latent"][:sample_count]
            )
            outputs["reconstruction_target_eye_strip"] = (
                self.build_reconstruction_target(
                    left_eye[:sample_count],
                    right_eye[:sample_count],
                )
            )
        return outputs

    @staticmethod
    def build_reconstruction_target(left_eye: Tensor, right_eye: Tensor) -> Tensor:
        """Build the paper-shaped [B, 3, 128, 512] eye strip in RGB [0, 1]."""

        mean = left_eye.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = left_eye.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

        def prepare(eye: Tensor) -> Tensor:
            eye = ensure_image_batch(eye, "reconstruction_eye")
            eye = (eye * std + mean).clamp(0.0, 1.0)
            return F.interpolate(
                eye,
                size=(128, 256),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        return torch.cat((prepare(left_eye), prepare(right_eye)), dim=-1)

    def eye_resnet_encoders(self) -> tuple[ResNetEyeImageEncoder, ...]:
        """Return no ResNet encoders; Blaze uses a single base LR."""

        return ()

    def eye_backbone_parameters(self) -> tuple[nn.Parameter, ...]:
        return ()

    def set_eye_backbone_train_from(self, stage: str | None) -> None:
        del stage
        raise ValueError("BlazeStyleModel does not support the ResNet schedule.")

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.direct_uv_table_frame_film_fusion.reset_to_identity()
