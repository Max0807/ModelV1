"""Native PyTorch implementation of the released BlazeGaze Stage-1 model.

This module intentionally does not import :mod:`modelv1`.  It mirrors
``WebEyeTrack/python/webeyetrack/blazegaze.py``: a single 128x512 RGB eye
patch is encoded to 512 dimensions, concatenated with ``head_vector`` and
``face_origin_3d``, and passed through a 16-16-2 gaze MLP.  The decoder is an
auxiliary Stage-1 component and is excluded from inference parameter counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


NATIVE_IMAGE_HEIGHT = 128
NATIVE_IMAGE_WIDTH = 512
NATIVE_EMBEDDING_SIZE = 512


@dataclass(frozen=True)
class NativeBlazeGazeConfig:
    image_height: int = NATIVE_IMAGE_HEIGHT
    image_width: int = NATIVE_IMAGE_WIDTH
    embedding_size: int = NATIVE_EMBEDDING_SIZE
    head_vector_dim: int = 3
    face_origin_dim: int = 3
    use_decoder: bool = True

    def __post_init__(self) -> None:
        expected = (
            NATIVE_IMAGE_HEIGHT,
            NATIVE_IMAGE_WIDTH,
            NATIVE_EMBEDDING_SIZE,
            3,
            3,
        )
        actual = (
            self.image_height,
            self.image_width,
            self.embedding_size,
            self.head_vector_dim,
            self.face_origin_dim,
        )
        if actual != expected:
            raise ValueError(
                "Native BlazeGaze has a fixed image/embedding/pose contract; "
                f"expected {expected}, got {actual}."
            )


def _native_batch_norm(channels: int) -> nn.BatchNorm2d:
    # Keras BatchNormalization defaults: epsilon=1e-3, momentum=0.99.
    # PyTorch's momentum is the update fraction, hence 1 - 0.99 = 0.01.
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)


class NativeBlazeBlock(nn.Module):
    """Single depthwise-separable Blaze residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("NativeBlazeBlock stride must be 1 or 2.")
        if stride == 1 and in_channels != out_channels:
            raise ValueError(
                "A stride-1 NativeBlazeBlock requires equal channel counts."
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
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.residual_pool = (
            nn.MaxPool2d(kernel_size=2, stride=2)
            if stride == 2
            else nn.Identity()
        )
        self.residual_projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if stride == 2
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, values: Tensor) -> Tensor:
        residual = self.residual_projection(self.residual_pool(values))
        features = self.pointwise(self.depthwise(values))
        return self.activation(features + residual)


class NativeDoubleBlazeBlock(nn.Module):
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
            raise ValueError("NativeDoubleBlazeBlock stride must be 1 or 2.")
        if stride == 1 and in_channels != out_channels:
            raise ValueError(
                "A stride-1 NativeDoubleBlazeBlock requires equal channel counts."
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
        self.pointwise1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.depthwise2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=5,
            padding=2,
            groups=hidden_channels,
            bias=True,
        )
        self.pointwise2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.residual_pool = (
            nn.MaxPool2d(kernel_size=2, stride=2)
            if stride == 2
            else nn.Identity()
        )
        self.residual_projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if stride == 2
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, values: Tensor) -> Tensor:
        residual = self.residual_projection(self.residual_pool(values))
        features = self.activation(self.pointwise1(self.depthwise1(values)))
        features = self.pointwise2(self.depthwise2(features))
        return self.activation(features + residual)


class NativeBlazeGazeEncoder(nn.Module):
    """Released 128x512 Blaze encoder with a fixed 512D output."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            NativeBlazeBlock(24, 24),
            NativeBlazeBlock(24, 24),
            NativeBlazeBlock(24, 48, stride=2),
            NativeBlazeBlock(48, 48),
            NativeBlazeBlock(48, 48),
            NativeDoubleBlazeBlock(48, 24, 96, stride=2),
            NativeDoubleBlazeBlock(96, 24, 96),
            NativeDoubleBlazeBlock(96, 24, 96),
            NativeDoubleBlazeBlock(96, 24, 96, stride=2),
            NativeDoubleBlazeBlock(96, 24, 96),
            NativeDoubleBlazeBlock(96, 24, 96),
        )
        self.squeeze = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, stride=2, padding=1),
            _native_batch_norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            _native_batch_norm(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (
            3,
            NATIVE_IMAGE_HEIGHT,
            NATIVE_IMAGE_WIDTH,
        ):
            raise ValueError(
                "Native BlazeGaze expects image shape [B, 3, 128, 512]; "
                f"got {tuple(image.shape)}."
            )
        embedding = torch.flatten(
            self.squeeze(self.blocks(self.stem(image.float()))), start_dim=1
        )
        if embedding.shape[-1] != NATIVE_EMBEDDING_SIZE:
            raise RuntimeError(
                "Native BlazeGaze encoder produced an invalid embedding shape: "
                f"{tuple(embedding.shape)}."
            )
        return embedding


class NativeBlazeGazeDecoderBlock(nn.Module):
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
            ),
            _native_batch_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.block(values)


class NativeBlazeGazeDecoder(nn.Module):
    """Released auxiliary decoder reconstructing the 128x512 RGB input."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            NativeBlazeGazeDecoderBlock(32, 96),
            NativeBlazeGazeDecoderBlock(96, 96, stride=2),
            NativeBlazeGazeDecoderBlock(96, 96),
            NativeBlazeGazeDecoderBlock(96, 96, stride=2),
            NativeBlazeGazeDecoderBlock(96, 48),
            NativeBlazeGazeDecoderBlock(48, 48, stride=2),
            NativeBlazeGazeDecoderBlock(48, 48),
            NativeBlazeGazeDecoderBlock(48, 48, stride=2),
            NativeBlazeGazeDecoderBlock(48, 24),
            NativeBlazeGazeDecoderBlock(24, 24, stride=2),
            NativeBlazeGazeDecoderBlock(24, 24),
            NativeBlazeGazeDecoderBlock(24, 24, stride=2),
        )
        self.output = nn.Sequential(
            nn.Conv2d(24, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        if embedding.ndim != 2 or embedding.shape[-1] != NATIVE_EMBEDDING_SIZE:
            raise ValueError("Native decoder expects embedding shape [B, 512].")
        spatial = embedding.reshape(embedding.shape[0], 32, 2, 8)
        return self.output(self.blocks(spatial))


class NativeBlazeGazeMLP(nn.Module):
    """Released 16-16-2 gaze estimator."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(NATIVE_EMBEDDING_SIZE + 3 + 3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
        )

    def forward(
        self,
        embedding: Tensor,
        head_vector: Tensor,
        face_origin_3d: Tensor,
    ) -> Tensor:
        batch_size = embedding.shape[0]
        if head_vector.shape != (batch_size, 3):
            raise ValueError("head_vector must have shape [B, 3].")
        if face_origin_3d.shape != (batch_size, 3):
            raise ValueError("face_origin_3d must have shape [B, 3].")
        return self.layers(
            torch.cat(
                (
                    embedding,
                    head_vector.to(embedding),
                    face_origin_3d.to(embedding),
                ),
                dim=-1,
            )
        )


class NativeBlazeGaze(nn.Module):
    """Complete supervised Stage-1 BlazeGaze model."""

    def __init__(self, config: NativeBlazeGazeConfig | None = None) -> None:
        super().__init__()
        self.config = config or NativeBlazeGazeConfig()
        self.encoder = NativeBlazeGazeEncoder()
        self.gaze_mlp = NativeBlazeGazeMLP()
        self.decoder = NativeBlazeGazeDecoder() if self.config.use_decoder else None
        self.reset_parameters()

    def forward(
        self,
        image: Tensor,
        head_vector: Tensor,
        face_origin_3d: Tensor,
        *,
        return_reconstruction: bool = True,
    ) -> Mapping[str, Tensor]:
        embedding = self.encoder(image)
        outputs: dict[str, Tensor] = {
            "embedding": embedding,
            "pog_pred": self.gaze_mlp(
                embedding,
                head_vector,
                face_origin_3d,
            ),
        }
        if return_reconstruction:
            if self.decoder is None:
                raise RuntimeError(
                    "return_reconstruction=True requires config.use_decoder=True."
                )
            outputs["reconstruction"] = self.decoder(embedding)
        return outputs

    def predict(
        self,
        image: Tensor,
        head_vector: Tensor,
        face_origin_3d: Tensor,
    ) -> Tensor:
        return self.forward(
            image,
            head_vector,
            face_origin_3d,
            return_reconstruction=False,
        )["pog_pred"]

    def inference_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for module in (self.encoder, self.gaze_mlp)
            for parameter in module.parameters()
        )

    def reset_parameters(self) -> None:
        # Keras Conv/Dense defaults use Glorot-uniform kernels and zero biases.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
