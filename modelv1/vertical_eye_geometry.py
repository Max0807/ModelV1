"""High-resolution probabilistic eye landmarks for vertical gaze geometry."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modelv1.data.eye_geometry_pseudo_labels import (
    EYE_PSEUDO_IRIS_CENTER_INDEX,
    EYE_PSEUDO_POINT_COUNT,
)
from modelv1.geometry import unpack_table_frame7


DEFAULT_LANDMARK_HEATMAP_SIZE = (28, 45)
OUTER_CANTHUS_INDEX = 0
INNER_CANTHUS_INDEX = 5
IRIS_RIM_SLICE = slice(EYE_PSEUDO_IRIS_CENTER_INDEX + 1, EYE_PSEUDO_POINT_COUNT)


class ProbabilisticEyeLandmarkHead(nn.Module):
    """Fuse ResNet layer2/layer3 and predict landmark distributions.

    The input maps remain substantially finer than the final gaze tokens.
    For a 56x90 eye crop, layer2/layer3 are 14x23 and 7x12. They are fused at
    layer2 resolution and decoded to the configured high-resolution heatmap.
    """

    def __init__(
        self,
        *,
        layer2_channels: int = 128,
        layer3_channels: int = 256,
        hidden_channels: int = 128,
        keypoint_count: int = EYE_PSEUDO_POINT_COUNT,
        output_size: tuple[int, int] = DEFAULT_LANDMARK_HEATMAP_SIZE,
        temperature: float = 1.0,
        predict_variance_correction: bool = True,
        max_abs_log_std_correction: float = 1.5,
        shape_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        if min(
            layer2_channels,
            layer3_channels,
            hidden_channels,
            keypoint_count,
            shape_embedding_dim,
        ) <= 0:
            raise ValueError("Eye-landmark head dimensions must be positive.")
        if keypoint_count != EYE_PSEUDO_POINT_COUNT:
            raise ValueError(
                "keypoint_count must match the 15-point pseudo-label schema."
            )
        if len(output_size) != 2 or min(output_size) <= 1:
            raise ValueError("output_size must be a two-item spatial size > 1.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if max_abs_log_std_correction <= 0:
            raise ValueError("max_abs_log_std_correction must be positive.")
        self.keypoint_count = int(keypoint_count)
        self.output_size = tuple(int(value) for value in output_size)
        self.temperature = float(temperature)
        self.predict_variance_correction = bool(predict_variance_correction)
        self.max_abs_log_std_correction = float(max_abs_log_std_correction)

        self.layer2_projection = _conv_norm_activation(
            layer2_channels, hidden_channels, kernel_size=1
        )
        self.layer3_projection = _conv_norm_activation(
            layer3_channels, hidden_channels, kernel_size=1
        )
        self.fusion = _conv_norm_activation(
            2 * hidden_channels, hidden_channels, kernel_size=3
        )
        self.heatmap_predictor = nn.Sequential(
            _conv_norm_activation(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, self.keypoint_count, kernel_size=1),
        )
        self.visibility_predictor = nn.Linear(hidden_channels, self.keypoint_count)
        self.variance_predictor = (
            nn.Linear(hidden_channels, 2 * self.keypoint_count)
            if self.predict_variance_correction
            else None
        )
        self.shape_encoder = nn.Sequential(
            nn.Linear(2 * self.keypoint_count, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, shape_embedding_dim),
        )

    def forward(self, layer2: Tensor, layer3: Tensor) -> dict[str, Tensor]:
        _validate_feature_map(layer2, "layer2")
        _validate_feature_map(layer3, "layer3")
        if layer2.shape[0] != layer3.shape[0]:
            raise ValueError("layer2 and layer3 must share the batch dimension.")
        layer2_features = self.layer2_projection(layer2)
        layer3_features = F.interpolate(
            self.layer3_projection(layer3),
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.fusion(torch.cat((layer2_features, layer3_features), dim=1))
        decoded = F.interpolate(
            fused,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        logits = self.heatmap_predictor(decoded)
        pooled = F.adaptive_avg_pool2d(fused, output_size=1).flatten(start_dim=1)
        visibility_logits = self.visibility_predictor(pooled)
        moments = spatial_probability_moments(logits, self.temperature)

        covariance_heatmap = moments["covariance"]
        if self.variance_predictor is None:
            log_std_correction = logits.new_zeros(
                (logits.shape[0], self.keypoint_count, 2), dtype=torch.float32
            )
            covariance = covariance_heatmap
        else:
            raw_correction = self.variance_predictor(pooled).reshape(
                logits.shape[0], self.keypoint_count, 2
            )
            log_std_correction = self.max_abs_log_std_correction * torch.tanh(
                raw_correction.float()
            )
            std_scale = torch.exp(log_std_correction)
            covariance = (
                covariance_heatmap
                * std_scale.unsqueeze(-1)
                * std_scale.unsqueeze(-2)
            )
        canonical_shape = canonicalize_eye_landmark_shape(moments["mean_xy"])
        shape_embedding = F.normalize(
            self.shape_encoder(canonical_shape.flatten(start_dim=1)),
            dim=-1,
            eps=1e-8,
        )
        return {
            "logits": logits,
            **moments,
            "covariance_heatmap": covariance_heatmap,
            "covariance": covariance,
            "variance_xy": torch.diagonal(covariance, dim1=-2, dim2=-1),
            "log_std_correction": log_std_correction,
            "visibility_logits": visibility_logits,
            "visibility_probability": torch.sigmoid(visibility_logits.float()),
            "canonical_shape": canonical_shape,
            "shape_embedding": shape_embedding,
        }


class ProbabilisticLandmarkTokenizer(nn.Module):
    """Turn landmark posteriors and shallow eye maps into anatomical tokens."""

    _STATISTIC_DIM = 7

    def __init__(
        self,
        *,
        token_dim: int,
        keypoint_count: int = EYE_PSEUDO_POINT_COUNT,
        layer2_channels: int = 128,
        layer3_channels: int = 256,
        uncertainty_scale_norm: float = 0.25,
        min_quality: float = 0.1,
        detach_probability_for_sampling: bool = True,
        detach_landmark_statistics: bool = True,
    ) -> None:
        super().__init__()
        if min(token_dim, keypoint_count, layer2_channels, layer3_channels) <= 0:
            raise ValueError("Landmark-tokenizer dimensions must be positive.")
        if keypoint_count != EYE_PSEUDO_POINT_COUNT:
            raise ValueError(
                "keypoint_count must match the 15-point pseudo-label schema."
            )
        if uncertainty_scale_norm <= 0:
            raise ValueError("uncertainty_scale_norm must be positive.")
        if not 0 <= min_quality < 1:
            raise ValueError("min_quality must lie in [0, 1).")
        self.keypoint_count = int(keypoint_count)
        self.token_dim = int(token_dim)
        self.uncertainty_scale_norm = float(uncertainty_scale_norm)
        self.min_quality = float(min_quality)
        self.detach_probability_for_sampling = bool(
            detach_probability_for_sampling
        )
        self.detach_landmark_statistics = bool(detach_landmark_statistics)

        self.layer2_projection = nn.Conv2d(
            layer2_channels, token_dim, kernel_size=1, bias=False
        )
        self.layer3_projection = nn.Conv2d(
            layer3_channels, token_dim, kernel_size=1, bias=False
        )
        self.statistics_projection = nn.Sequential(
            nn.LayerNorm(self._STATISTIC_DIM),
            nn.Linear(self._STATISTIC_DIM, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.token_fusion = nn.Sequential(
            nn.Linear(3 * token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.point_embedding = nn.Parameter(
            torch.zeros(1, self.keypoint_count, token_dim)
        )
        self.eye_type_embedding = nn.Parameter(torch.zeros(2, 1, token_dim))
        self.output_norm = nn.LayerNorm(token_dim)
        nn.init.trunc_normal_(self.point_embedding, std=0.02)
        nn.init.trunc_normal_(self.eye_type_embedding, std=0.02)

    @staticmethod
    def _normalized_probability_at_size(
        probability_maps: Tensor,
        spatial_size: tuple[int, int],
        *,
        detach: bool,
        dtype: torch.dtype,
    ) -> Tensor:
        probabilities = torch.nan_to_num(
            F.interpolate(
                probability_maps.float(),
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        probabilities = probabilities / probabilities.sum(
            dim=(-1, -2), keepdim=True
        ).clamp_min(1e-8)
        if detach:
            probabilities = probabilities.detach()
        return probabilities.to(dtype=dtype)

    def _appearance_tokens(
        self,
        probability_maps: Tensor,
        feature_map: Tensor,
        projection: nn.Module,
    ) -> Tensor:
        projected = projection(feature_map)
        probabilities = self._normalized_probability_at_size(
            probability_maps,
            tuple(projected.shape[-2:]),
            detach=self.detach_probability_for_sampling,
            dtype=projected.dtype,
        )
        return torch.einsum("bkhw,bchw->bkc", probabilities, projected)

    def _tokenize_eye(
        self,
        *,
        layer2: Tensor,
        layer3: Tensor,
        probability_maps: Tensor,
        mean_xy: Tensor,
        covariance: Tensor,
        visibility_probability: Tensor,
        entropy: Tensor,
        eye_index: int,
    ) -> tuple[Tensor, Tensor]:
        batch_size = layer2.shape[0]
        expected_points = (batch_size, self.keypoint_count)
        if probability_maps.ndim != 4 or probability_maps.shape[:2] != expected_points:
            raise ValueError("probability_maps must have shape [B, 15, H, W].")
        if mean_xy.shape != (*expected_points, 2):
            raise ValueError("mean_xy must have shape [B, 15, 2].")
        if covariance.shape != (*expected_points, 2, 2):
            raise ValueError("covariance must have shape [B, 15, 2, 2].")
        if visibility_probability.shape != expected_points:
            raise ValueError(
                "visibility_probability must have shape [B, 15]."
            )
        if entropy.shape != expected_points:
            raise ValueError("entropy must have shape [B, 15].")
        if eye_index not in {0, 1}:
            raise ValueError("eye_index must be zero (left) or one (right).")

        layer2_appearance = self._appearance_tokens(
            probability_maps, layer2, self.layer2_projection
        )
        layer3_appearance = self._appearance_tokens(
            probability_maps, layer3, self.layer3_projection
        )
        covariance = 0.5 * (
            covariance.float() + covariance.float().transpose(-1, -2)
        )
        statistics = torch.cat(
            (
                mean_xy.float(),
                covariance[..., 0, 0:1],
                covariance[..., 0, 1:2],
                covariance[..., 1, 1:2],
                visibility_probability.float().unsqueeze(-1),
                entropy.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        statistics = torch.nan_to_num(
            statistics, nan=0.0, posinf=1.0, neginf=-1.0
        )
        if self.detach_landmark_statistics:
            statistics = statistics.detach()
        statistics_features = self.statistics_projection(statistics).to(
            dtype=layer2_appearance.dtype
        )
        tokens = self.token_fusion(
            torch.cat(
                (layer2_appearance, layer3_appearance, statistics_features),
                dim=-1,
            )
        )
        tokens = self.output_norm(
            tokens
            + self.point_embedding.to(dtype=tokens.dtype)
            + self.eye_type_embedding[eye_index : eye_index + 1].to(
                dtype=tokens.dtype
            )
        )

        posterior_std = torch.diagonal(
            covariance, dim1=-2, dim2=-1
        ).clamp_min(0.0).sum(dim=-1).sqrt()
        visibility = visibility_probability.float().clamp(0.0, 1.0)
        normalized_entropy = entropy.float().clamp(0.0, 1.0)
        raw_quality = (
            visibility
            * (1.0 - 0.5 * normalized_entropy)
            * torch.exp(-posterior_std / self.uncertainty_scale_norm)
        ).clamp(0.0, 1.0)
        quality = self.min_quality + (1.0 - self.min_quality) * raw_quality
        quality = torch.nan_to_num(
            quality,
            nan=self.min_quality,
            posinf=1.0,
            neginf=self.min_quality,
        )
        return tokens, quality

    def forward(
        self,
        *,
        left_layer2: Tensor,
        left_layer3: Tensor,
        right_layer2: Tensor,
        right_layer3: Tensor,
        predictions: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        left_tokens, left_quality = self._tokenize_eye(
            layer2=left_layer2,
            layer3=left_layer3,
            probability_maps=predictions["left_eye_keypoint_probability_maps"],
            mean_xy=predictions["left_eye_keypoint_mean_xy"],
            covariance=predictions["left_eye_keypoint_covariance"],
            visibility_probability=predictions[
                "left_eye_keypoint_visibility_probability"
            ],
            entropy=predictions["left_eye_keypoint_entropy"],
            eye_index=0,
        )
        right_tokens, right_quality = self._tokenize_eye(
            layer2=right_layer2,
            layer3=right_layer3,
            probability_maps=predictions["right_eye_keypoint_probability_maps"],
            mean_xy=predictions["right_eye_keypoint_mean_xy"],
            covariance=predictions["right_eye_keypoint_covariance"],
            visibility_probability=predictions[
                "right_eye_keypoint_visibility_probability"
            ],
            entropy=predictions["right_eye_keypoint_entropy"],
            eye_index=1,
        )
        return {
            "tokens": torch.cat((left_tokens, right_tokens), dim=1),
            "quality": torch.cat((left_quality, right_quality), dim=1),
            "left_tokens": left_tokens,
            "right_tokens": right_tokens,
            "left_quality": left_quality,
            "right_quality": right_quality,
        }


class LandmarkGuidedEyeFusion(nn.Module):
    """Inject probabilistic eye anatomy before face fusion."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        keypoint_count: int = EYE_PSEUDO_POINT_COUNT,
        gate_init: float = 0.0,
        gate_max: float = 1.0,
        quality_logit_scale: float = 1.0,
        uncertainty_scale_norm: float = 0.25,
        min_quality: float = 0.1,
        detach_probability_for_sampling: bool = True,
        detach_landmark_statistics: bool = True,
        detach_quality: bool = True,
    ) -> None:
        super().__init__()
        if min(token_dim, num_heads) <= 0:
            raise ValueError("Fusion token/head dimensions must be positive.")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if gate_max <= 0 or not -gate_max < gate_init < gate_max:
            raise ValueError(
                "gate_init must lie strictly inside (-gate_max, gate_max)."
            )
        if quality_logit_scale < 0:
            raise ValueError("quality_logit_scale must be non-negative.")
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.gate_init = float(gate_init)
        self.gate_max = float(gate_max)
        self.quality_logit_scale = float(quality_logit_scale)
        self.detach_quality = bool(detach_quality)

        self.tokenizer = ProbabilisticLandmarkTokenizer(
            token_dim=token_dim,
            keypoint_count=keypoint_count,
            uncertainty_scale_norm=uncertainty_scale_norm,
            min_quality=min_quality,
            detach_probability_for_sampling=detach_probability_for_sampling,
            detach_landmark_statistics=detach_landmark_statistics,
        )
        self.quality_embedding = nn.Linear(1, token_dim, bias=False)
        self.query_norm = nn.LayerNorm(token_dim)
        self.landmark_norm = nn.LayerNorm(token_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.delta_norm = nn.LayerNorm(token_dim)
        self.delta_dropout = nn.Dropout(dropout)
        self.raw_gate = nn.Parameter(torch.empty(()))
        self.reset_gate()

    def reset_gate(self) -> None:
        normalized = self.gate_init / self.gate_max
        with torch.no_grad():
            self.raw_gate.fill_(math.atanh(normalized))

    def forward(
        self,
        eye_tokens: Tensor,
        *,
        left_layer2: Tensor,
        left_layer3: Tensor,
        right_layer2: Tensor,
        right_layer3: Tensor,
        predictions: Mapping[str, Tensor],
        return_attention: bool = False,
    ) -> dict[str, Tensor]:
        if eye_tokens.ndim != 3 or eye_tokens.shape[-1] != self.token_dim:
            raise ValueError("eye_tokens must have shape [B, N, token_dim].")
        tokenized = self.tokenizer(
            left_layer2=left_layer2,
            left_layer3=left_layer3,
            right_layer2=right_layer2,
            right_layer3=right_layer3,
            predictions=predictions,
        )
        landmark_tokens = tokenized["tokens"]
        quality = tokenized["quality"]
        quality_for_fusion = quality.detach() if self.detach_quality else quality
        query = self.query_norm(eye_tokens)
        landmark_key = self.landmark_norm(
            landmark_tokens
            + self.quality_embedding(
                quality_for_fusion.to(dtype=landmark_tokens.dtype).unsqueeze(-1)
            )
        )
        landmark_value = landmark_tokens * quality_for_fusion.to(
            dtype=landmark_tokens.dtype
        ).unsqueeze(-1)

        attention_bias = self.quality_logit_scale * torch.log(
            quality_for_fusion.clamp_min(1e-4)
        )
        attention_bias = (
            attention_bias[:, None, None, :]
            .expand(-1, self.num_heads, eye_tokens.shape[1], -1)
            .reshape(
                eye_tokens.shape[0] * self.num_heads,
                eye_tokens.shape[1],
                landmark_tokens.shape[1],
            )
            .to(dtype=query.dtype)
        )
        attended, attention_weights = self.cross_attention(
            query=query,
            key=landmark_key,
            value=landmark_value,
            attn_mask=attention_bias,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        delta = self.delta_dropout(self.delta_norm(attended))
        global_gate = self.gate_max * torch.tanh(self.raw_gate)
        sample_quality = quality_for_fusion.mean(dim=-1, keepdim=True)
        applied_gate = global_gate * sample_quality
        applied_delta = applied_gate.unsqueeze(-1) * delta
        refined_eye_tokens = eye_tokens + applied_delta

        output = {
            "refined_eye_tokens": refined_eye_tokens,
            "landmark_tokens": landmark_tokens,
            "landmark_quality": quality,
            "sample_quality": sample_quality,
            "global_gate": global_gate.expand(eye_tokens.shape[0], 1),
            "applied_gate": applied_gate,
            "delta_norm": torch.linalg.vector_norm(
                applied_delta.float(), dim=-1
            ).mean(dim=-1, keepdim=True),
        }
        if return_attention and attention_weights is not None:
            output["attention_weights"] = attention_weights
        return output


def _conv_norm_activation(
    input_channels: int,
    output_channels: int,
    *,
    kernel_size: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm2d(output_channels),
        nn.SiLU(inplace=True),
    )


def _validate_feature_map(value: Tensor, name: str) -> None:
    if value.ndim != 4 or min(value.shape[-2:]) <= 0:
        raise ValueError(f"{name} must have shape [B, C, H, W].")


def spatial_probability_moments(
    logits: Tensor,
    temperature: float,
) -> dict[str, Tensor]:
    """Return spatial probabilities, normalized means, and full covariance."""

    if logits.ndim != 4:
        raise ValueError("Landmark logits must have shape [B, K, H, W].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    batch_size, keypoint_count, height, width = logits.shape
    probabilities = torch.softmax(
        logits.float().flatten(start_dim=2) / float(temperature), dim=-1
    )
    y_coordinates, x_coordinates = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=logits.device),
        torch.linspace(0.0, 1.0, width, device=logits.device),
        indexing="ij",
    )
    coordinate_grid = torch.stack((x_coordinates, y_coordinates), dim=-1).reshape(
        height * width, 2
    )
    mean_xy = torch.einsum("bkn,nc->bkc", probabilities, coordinate_grid)
    centered = coordinate_grid[None, None] - mean_xy[:, :, None]
    covariance = torch.einsum(
        "bkn,bknc,bknd->bkcd", probabilities, centered, centered
    )
    log_probabilities = torch.log(probabilities.clamp_min(1e-12))
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    if height * width > 1:
        entropy = entropy / math.log(height * width)
    return {
        "probability_maps": probabilities.reshape(
            batch_size, keypoint_count, height, width
        ),
        "mean_xy": mean_xy,
        "covariance": covariance,
        "variance_xy": torch.diagonal(covariance, dim1=-2, dim2=-1),
        "entropy": entropy,
        "confidence": (1.0 - entropy).clamp(0.0, 1.0),
    }


def canonicalize_eye_landmark_shape(points_xy: Tensor) -> Tensor:
    """Map either anatomical eye to an outer-to-inner canonical eye frame."""

    if points_xy.ndim != 3 or points_xy.shape[1:] != (EYE_PSEUDO_POINT_COUNT, 2):
        raise ValueError("points_xy must have shape [B, 15, 2].")
    outer = points_xy[:, OUTER_CANTHUS_INDEX]
    inner = points_xy[:, INNER_CANTHUS_INDEX]
    origin = 0.5 * (outer + inner)
    x_vector = inner - outer
    scale = torch.linalg.vector_norm(x_vector, dim=-1, keepdim=True).clamp_min(1e-4)
    x_axis = x_vector / scale
    y_axis = torch.stack((-x_axis[:, 1], x_axis[:, 0]), dim=-1)
    upper = points_xy[:, 2]
    lower = points_xy[:, 8]
    orientation = torch.sign(((lower - upper) * y_axis).sum(dim=-1, keepdim=True))
    orientation = torch.where(orientation == 0, torch.ones_like(orientation), orientation)
    y_axis = y_axis * orientation
    centered = points_xy - origin[:, None]
    local_x = (centered * x_axis[:, None]).sum(dim=-1) / scale
    local_y = (centered * y_axis[:, None]).sum(dim=-1) / scale
    return torch.stack((local_x, local_y), dim=-1)


def probabilistic_landmark_supervision_losses(
    *,
    predicted_mean_xy: Tensor,
    heatmap_logits: Tensor,
    visibility_logits: Tensor,
    target_xy: Tensor,
    valid_mask: Tensor,
    teacher_quality: Tensor,
    visibility_supervision_weight: Tensor,
    huber_delta_norm: float,
    heatmap_sigma_min_px: float,
    heatmap_sigma_max_px: float,
) -> dict[str, Tensor]:
    """Coordinate, Gaussian-distribution, and visibility supervision."""

    if huber_delta_norm <= 0:
        raise ValueError("huber_delta_norm must be positive.")
    if not 0 < heatmap_sigma_min_px <= heatmap_sigma_max_px:
        raise ValueError("Heatmap sigma bounds must satisfy 0 < min <= max.")
    batch_size, eye_count, keypoint_count, _ = target_xy.shape
    if eye_count != 2 or keypoint_count != EYE_PSEUDO_POINT_COUNT:
        raise ValueError("target_xy must have shape [B, 2, 15, 2].")
    if predicted_mean_xy.shape != target_xy.shape:
        raise ValueError("predicted_mean_xy must match target_xy.")
    if valid_mask.shape != target_xy.shape[:-1]:
        raise ValueError("valid_mask must have shape [B, 2, 15].")
    if teacher_quality.shape != (batch_size, 2):
        raise ValueError("teacher_quality must have shape [B, 2].")
    if visibility_supervision_weight.shape != (batch_size, 2):
        raise ValueError("visibility_supervision_weight must have shape [B, 2].")
    if heatmap_logits.ndim != 5 or heatmap_logits.shape[:3] != (
        batch_size,
        2,
        keypoint_count,
    ):
        raise ValueError("heatmap_logits must have shape [B, 2, 15, H, W].")
    if visibility_logits.shape != valid_mask.shape:
        raise ValueError("visibility_logits must have shape [B, 2, 15].")

    targets = target_xy.to(device=predicted_mean_xy.device, dtype=torch.float32)
    valid = valid_mask.to(device=predicted_mean_xy.device, dtype=torch.float32)
    quality = teacher_quality.to(device=predicted_mean_xy.device, dtype=torch.float32)
    visibility_weight = visibility_supervision_weight.to(
        device=predicted_mean_xy.device, dtype=torch.float32
    )
    point_weight = valid * quality.unsqueeze(-1)
    denominator = point_weight.sum().clamp_min(1.0)
    coordinate_loss = F.smooth_l1_loss(
        predicted_mean_xy.float(),
        targets,
        beta=huber_delta_norm,
        reduction="none",
    ).sum(dim=-1)
    coordinate_loss = (coordinate_loss * point_weight).sum() / denominator

    _, _, _, height, width = heatmap_logits.shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=heatmap_logits.device, dtype=torch.float32),
        torch.arange(width, device=heatmap_logits.device, dtype=torch.float32),
        indexing="ij",
    )
    centers_px = targets.clone()
    centers_px[..., 0] *= max(width - 1, 1)
    centers_px[..., 1] *= max(height - 1, 1)
    sigma = heatmap_sigma_min_px + (1.0 - quality) * (
        heatmap_sigma_max_px - heatmap_sigma_min_px
    )
    distance_sq = (
        (x_grid[None, None, None] - centers_px[..., 0, None, None]).square()
        + (y_grid[None, None, None] - centers_px[..., 1, None, None]).square()
    )
    target_distribution = torch.exp(
        -0.5 * distance_sq / sigma[:, :, None, None, None].square()
    )
    target_distribution = target_distribution / target_distribution.sum(
        dim=(-1, -2), keepdim=True
    ).clamp_min(1e-8)
    log_probability = F.log_softmax(
        heatmap_logits.float().flatten(start_dim=-2), dim=-1
    ).reshape_as(heatmap_logits)
    heatmap_cross_entropy = -(
        target_distribution * log_probability
    ).sum(dim=(-1, -2))
    heatmap_loss = (heatmap_cross_entropy * point_weight).sum() / denominator

    visibility_bce = F.binary_cross_entropy_with_logits(
        visibility_logits.float(), valid, reduction="none"
    )
    visibility_point_weight = visibility_weight.unsqueeze(-1).expand_as(valid)
    visibility_loss = (
        visibility_bce * visibility_point_weight
    ).sum() / visibility_point_weight.sum().clamp_min(1.0)
    return {
        "coordinate_loss": coordinate_loss,
        "heatmap_loss": heatmap_loss,
        "visibility_loss": visibility_loss,
        "effective_point_weight": point_weight.sum(),
    }


def mirror_shape_consistency_loss(
    left_shape_embedding: Tensor,
    right_shape_embedding: Tensor,
    eye_weight: Tensor,
) -> Tensor:
    """Match canonical shared shape embeddings, never raw left/right coordinates."""

    if left_shape_embedding.shape != right_shape_embedding.shape:
        raise ValueError("Left/right shape embeddings must share shape.")
    if eye_weight.shape != (left_shape_embedding.shape[0], 2):
        raise ValueError("eye_weight must have shape [B, 2].")
    sample_weight = eye_weight.to(
        device=left_shape_embedding.device, dtype=torch.float32
    ).prod(dim=-1)
    cosine_distance = 1.0 - F.cosine_similarity(
        left_shape_embedding.float(), right_shape_embedding.float(), dim=-1
    )
    return (cosine_distance * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)


def sample_eye_affine_matrices(
    batch_size: int,
    *,
    device: torch.device,
    max_rotation_deg: float,
    max_translation_norm: float,
    scale_min: float,
    scale_max: float,
    deterministic_sign: int | None = None,
) -> Tensor:
    """Sample forward transforms from original to augmented crop coordinates."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_rotation_deg < 0 or not 0 <= max_translation_norm < 0.5:
        raise ValueError("Invalid affine rotation/translation range.")
    if not 0 < scale_min <= scale_max:
        raise ValueError("Affine scale bounds must satisfy 0 < min <= max.")
    if deterministic_sign not in (None, -1, 1):
        raise ValueError("deterministic_sign must be None, -1, or 1.")
    if deterministic_sign is None:
        angle_deg = torch.empty(batch_size, device=device).uniform_(
            -max_rotation_deg, max_rotation_deg
        )
        translation = torch.empty(batch_size, 2, device=device).uniform_(
            -max_translation_norm, max_translation_norm
        )
        scale = torch.empty(batch_size, device=device).uniform_(scale_min, scale_max)
    else:
        sign = float(deterministic_sign)
        angle_deg = torch.full(
            (batch_size,), sign * 0.7 * max_rotation_deg, device=device
        )
        translation = torch.tensor(
            [sign * 0.5 * max_translation_norm, -sign * 0.35 * max_translation_norm],
            device=device,
        ).expand(batch_size, -1)
        scale = torch.ones(batch_size, device=device)
    angle = torch.deg2rad(angle_deg)
    cosine = torch.cos(angle) * scale
    sine = torch.sin(angle) * scale
    linear = torch.stack(
        (
            torch.stack((cosine, -sine), dim=-1),
            torch.stack((sine, cosine), dim=-1),
        ),
        dim=-2,
    )
    center = torch.full((batch_size, 2), 0.5, device=device)
    offset = center + translation - torch.bmm(
        linear, center.unsqueeze(-1)
    ).squeeze(-1)
    matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    matrix[:, :2, :2] = linear
    matrix[:, :2, 2] = offset
    return matrix


def warp_eye_images(images: Tensor, forward_matrix: Tensor) -> Tensor:
    """Warp normalized eye images using a known original-to-augmented matrix."""

    if images.ndim != 4 or forward_matrix.shape != (images.shape[0], 3, 3):
        raise ValueError("Expected images [B,C,H,W] and matrices [B,3,3].")
    batch_size, _, height, width = images.shape
    y_coordinates, x_coordinates = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=images.device),
        torch.linspace(0.0, 1.0, width, device=images.device),
        indexing="ij",
    )
    output_grid = torch.stack(
        (x_coordinates, y_coordinates, torch.ones_like(x_coordinates)), dim=-1
    ).reshape(1, height * width, 3).expand(batch_size, -1, -1)
    inverse = torch.linalg.inv(forward_matrix.float())
    source = torch.bmm(output_grid, inverse.transpose(1, 2))[..., :2]
    sampling_grid = (2.0 * source - 1.0).reshape(batch_size, height, width, 2)
    return F.grid_sample(
        images,
        sampling_grid.to(dtype=images.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def transform_normalized_points(points_xy: Tensor, matrix: Tensor) -> Tensor:
    if points_xy.ndim < 3 or points_xy.shape[0] != matrix.shape[0]:
        raise ValueError("Points and matrices must share a batch dimension.")
    homogeneous = torch.cat((points_xy, torch.ones_like(points_xy[..., :1])), dim=-1)
    return torch.einsum("bij,b...j->b...i", matrix.float(), homogeneous.float())[..., :2]


def equivariant_landmark_consistency_loss(
    view1_mean_xy: Tensor,
    view2_mean_xy: Tensor,
    view1_matrix: Tensor,
    view2_matrix: Tensor,
    eye_weight: Tensor,
    *,
    huber_delta_norm: float,
) -> dict[str, Tensor]:
    """Compare two augmented predictions after mapping both back to the crop."""

    if view1_mean_xy.shape != view2_mean_xy.shape:
        raise ValueError("Equivariant landmark predictions must share shape.")
    if view1_mean_xy.ndim != 4 or view1_mean_xy.shape[1:] != (
        2,
        EYE_PSEUDO_POINT_COUNT,
        2,
    ):
        raise ValueError("Equivariant means must have shape [B, 2, 15, 2].")
    if eye_weight.shape != (view1_mean_xy.shape[0], 2):
        raise ValueError("eye_weight must have shape [B, 2].")
    inverse1 = torch.linalg.inv(view1_matrix.float())
    inverse2 = torch.linalg.inv(view2_matrix.float())
    canonical1 = transform_normalized_points(view1_mean_xy, inverse1)
    canonical2 = transform_normalized_points(view2_mean_xy, inverse2)
    in_bounds = (
        (canonical1 >= 0.0).all(dim=-1)
        & (canonical1 <= 1.0).all(dim=-1)
        & (canonical2 >= 0.0).all(dim=-1)
        & (canonical2 <= 1.0).all(dim=-1)
    ).float()
    weight = eye_weight.to(device=canonical1.device, dtype=torch.float32).unsqueeze(-1)
    weight = weight * in_bounds
    per_point = F.smooth_l1_loss(
        canonical1,
        canonical2,
        beta=huber_delta_norm,
        reduction="none",
    ).sum(dim=-1)
    denominator = weight.sum().clamp_min(1.0)
    return {
        "loss": (per_point * weight).sum() / denominator,
        "mean_error_norm": (
            torch.linalg.vector_norm(canonical1 - canonical2, dim=-1) * weight
        ).sum()
        / denominator,
    }


def eye_landmark_validation_metrics(
    predicted_mean_xy: Tensor,
    target_xy: Tensor,
    valid_mask: Tensor,
    teacher_quality: Tensor,
    *,
    eye_image_size: tuple[int, int],
) -> dict[str, Tensor]:
    """Return validation-only pseudo-label accuracy and calibration metrics."""

    if predicted_mean_xy.shape != target_xy.shape or predicted_mean_xy.ndim != 4:
        raise ValueError("Predicted/target landmarks must share shape [B,2,K,2].")
    width, height = eye_image_size
    pixel_scale = predicted_mean_xy.new_tensor((width, height))
    predicted_px = predicted_mean_xy.float() * pixel_scale
    target_px = target_xy.float() * pixel_scale
    valid = valid_mask.float()
    point_error_px = torch.linalg.vector_norm(predicted_px - target_px, dim=-1)
    canthus_distance_px = torch.linalg.vector_norm(
        target_px[:, :, INNER_CANTHUS_INDEX]
        - target_px[:, :, OUTER_CANTHUS_INDEX],
        dim=-1,
    ).clamp_min(1.0)
    nme_per_point = point_error_px / canthus_distance_px.unsqueeze(-1)
    scale_valid = (
        valid[:, :, OUTER_CANTHUS_INDEX]
        * valid[:, :, INNER_CANTHUS_INDEX]
    )
    nme_valid = valid * scale_valid.unsqueeze(-1)
    nme_denominator = nme_valid.sum().clamp_min(1.0)
    nme = (nme_per_point * nme_valid).sum() / nme_denominator

    point_denominator = valid.sum().clamp_min(1.0)

    iris_valid = valid[:, :, EYE_PSEUDO_IRIS_CENTER_INDEX]
    iris_error = point_error_px[:, :, EYE_PSEUDO_IRIS_CENTER_INDEX]
    iris_center_error_px = (iris_error * iris_valid).sum() / iris_valid.sum().clamp_min(1.0)
    vertical_error_px = (
        (predicted_px[..., 1] - target_px[..., 1]).abs() * valid
    ).sum() / point_denominator

    rim_valid = valid[:, :, IRIS_RIM_SLICE].prod(dim=-1) * iris_valid
    teacher_center = target_px[:, :, EYE_PSEUDO_IRIS_CENTER_INDEX]
    teacher_rim = target_px[:, :, IRIS_RIM_SLICE]
    predicted_rim = predicted_px[:, :, IRIS_RIM_SLICE]
    teacher_delta = teacher_rim - teacher_center.unsqueeze(-2)
    teacher_covariance = torch.einsum(
        "beqi,beqj->beij", teacher_delta, teacher_delta
    ) / teacher_delta.shape[-2]
    identity = torch.eye(2, device=target_px.device).reshape(1, 1, 2, 2)
    teacher_inverse = torch.linalg.inv(teacher_covariance + identity * 1e-3)
    predicted_delta = predicted_rim - teacher_center.unsqueeze(-2)
    predicted_radius_sq = 0.5 * torch.einsum(
        "beqi,beij,beqj->beq",
        predicted_delta,
        teacher_inverse,
        predicted_delta,
    ).clamp_min(0.0)
    teacher_radius_sq = 0.5 * torch.einsum(
        "beqi,beij,beqj->beq",
        teacher_delta,
        teacher_inverse,
        teacher_delta,
    ).clamp_min(0.0)
    ellipse_error = (
        torch.sqrt(predicted_radius_sq + 1e-12)
        - torch.sqrt(teacher_radius_sq + 1e-12)
    ).abs().mean(dim=-1)
    ellipse_reprojection_error = (
        ellipse_error * rim_valid
    ).sum() / rim_valid.sum().clamp_min(1.0)

    eye_valid_count = nme_valid.sum(dim=-1)
    per_eye_error = (
        (nme_per_point * nme_valid).sum(dim=-1)
        / eye_valid_count.clamp_min(1.0)
    )
    correlation_mask = eye_valid_count > 0
    quality = teacher_quality.float()[correlation_mask]
    errors = per_eye_error[correlation_mask]
    if quality.numel() >= 2:
        centered_quality = quality - quality.mean()
        centered_error = errors - errors.mean()
        correlation = (
            (centered_quality * centered_error).mean()
            / (
                centered_quality.square().mean().sqrt()
                * centered_error.square().mean().sqrt()
            ).clamp_min(1e-8)
        )
    else:
        correlation = predicted_mean_xy.new_zeros(())
    return {
        "eye_landmark_nme": nme,
        "eye_landmark_iris_center_error_px": iris_center_error_px,
        "eye_landmark_vertical_error_px": vertical_error_px,
        "eye_landmark_ellipse_reprojection_error": ellipse_reprojection_error,
        "eye_landmark_teacher_quality_error_corr": correlation,
    }


class LowDOFDifferentiableEyeballTemplate(nn.Module):
    """Fit a weak-perspective spherical eyeball with only yaw and pitch free.

    The ten canthus/eyelid landmarks define an isotropic pixel-space eye frame.
    The iris centre and four unordered rim points score a bounded yaw/pitch
    candidate grid. A spatial posterior, rather than an argmin, keeps the fit
    differentiable and exposes angular covariance for later uncertainty gates.
    """

    def __init__(
        self,
        *,
        eye_image_size: tuple[int, int] = (90, 56),
        pitch_range_deg: tuple[float, float] = (-35.0, 35.0),
        yaw_range_deg: tuple[float, float] = (-45.0, 45.0),
        pitch_bins: int = 29,
        yaw_bins: int = 31,
        posterior_temperature: float = 0.25,
        covariance_floor_px: float = 0.75,
        min_canthus_distance_px: float = 4.0,
        min_geometry_confidence: float = 0.1,
        center_energy_weight: float = 1.0,
        rim_energy_weight: float = 1.0,
        angle_prior_weight: float = 0.05,
        learnable_template: bool = True,
        use_reference_uncertainty: bool = True,
        use_aperture_visibility: bool = True,
    ) -> None:
        super().__init__()
        if len(eye_image_size) != 2 or min(eye_image_size) <= 1:
            raise ValueError("eye_image_size must be (width, height), both > 1.")
        if len(pitch_range_deg) != 2 or pitch_range_deg[0] >= pitch_range_deg[1]:
            raise ValueError("pitch_range_deg must be an increasing pair.")
        if len(yaw_range_deg) != 2 or yaw_range_deg[0] >= yaw_range_deg[1]:
            raise ValueError("yaw_range_deg must be an increasing pair.")
        if pitch_bins < 3 or yaw_bins < 3:
            raise ValueError("pitch_bins and yaw_bins must be at least three.")
        if posterior_temperature <= 0 or covariance_floor_px <= 0:
            raise ValueError("Posterior temperature and covariance floor must be positive.")
        if min_canthus_distance_px <= 0:
            raise ValueError("min_canthus_distance_px must be positive.")
        if not 0 <= min_geometry_confidence <= 1:
            raise ValueError("min_geometry_confidence must lie in [0, 1].")
        if min(center_energy_weight, rim_energy_weight, angle_prior_weight) < 0:
            raise ValueError("Eyeball-template energy weights must be non-negative.")

        self.eye_image_size = tuple(int(value) for value in eye_image_size)
        self.posterior_temperature = float(posterior_temperature)
        self.covariance_floor_px = float(covariance_floor_px)
        self.min_canthus_distance_px = float(min_canthus_distance_px)
        self.min_geometry_confidence = float(min_geometry_confidence)
        self.center_energy_weight = float(center_energy_weight)
        self.rim_energy_weight = float(rim_energy_weight)
        self.angle_prior_weight = float(angle_prior_weight)
        self.use_reference_uncertainty = bool(use_reference_uncertainty)
        self.use_aperture_visibility = bool(use_aperture_visibility)
        self.pitch_bins = int(pitch_bins)
        self.yaw_bins = int(yaw_bins)

        pitch_values = torch.linspace(*pitch_range_deg, pitch_bins)
        yaw_values = torch.linspace(*yaw_range_deg, yaw_bins)
        pitch_grid_deg, yaw_grid_deg = torch.meshgrid(
            pitch_values,
            yaw_values,
            indexing="ij",
        )
        pitch_grid = torch.deg2rad(pitch_grid_deg.reshape(-1))
        yaw_grid = torch.deg2rad(yaw_grid_deg.reshape(-1))
        candidate_angles = torch.stack((yaw_grid, pitch_grid), dim=-1)
        candidate_gaze = torch.stack(
            (
                torch.sin(yaw_grid) * torch.cos(pitch_grid),
                torch.sin(pitch_grid),
                torch.cos(yaw_grid) * torch.cos(pitch_grid),
            ),
            dim=-1,
        )
        self.register_buffer("candidate_angles_rad", candidate_angles)
        self.register_buffer("candidate_gaze", candidate_gaze)
        self.register_buffer("pitch_values_rad", torch.deg2rad(pitch_values))
        self.register_buffer("yaw_values_rad", torch.deg2rad(yaw_values))
        yaw_prior_std = math.radians(30.0)
        pitch_prior_std = math.radians(25.0)
        candidate_prior = 0.5 * (
            (yaw_grid / yaw_prior_std).square()
            + (pitch_grid / pitch_prior_std).square()
        )
        self.register_buffer("candidate_angle_prior", candidate_prior)

        self.raw_eyeball_radius = nn.Parameter(
            _inverse_bounded_value(0.55, 0.40, 0.70),
            requires_grad=learnable_template,
        )
        self.raw_iris_radius = nn.Parameter(
            _inverse_bounded_value(0.105, 0.06, 0.18),
            requires_grad=learnable_template,
        )
        self.raw_center_x_offset = nn.Parameter(
            torch.zeros(()), requires_grad=learnable_template
        )
        self.raw_center_y_offset = nn.Parameter(
            torch.zeros(()), requires_grad=learnable_template
        )

    def template_parameters(self) -> dict[str, Tensor]:
        return {
            "eyeball_radius_canthus_ratio": _bounded_value(
                self.raw_eyeball_radius, 0.40, 0.70
            ),
            "iris_radius_canthus_ratio": _bounded_value(
                self.raw_iris_radius, 0.06, 0.18
            ),
            "center_x_offset_canthus_ratio": 0.05
            * torch.tanh(self.raw_center_x_offset),
            "center_y_offset_canthus_ratio": 0.12
            * torch.tanh(self.raw_center_y_offset),
        }

    def forward(
        self,
        landmark_mean_xy: Tensor,
        landmark_covariance: Tensor,
        visibility_probability: Tensor,
    ) -> dict[str, Tensor]:
        if landmark_mean_xy.ndim != 4 or landmark_mean_xy.shape[1:] != (
            2,
            EYE_PSEUDO_POINT_COUNT,
            2,
        ):
            raise ValueError("landmark_mean_xy must have shape [B, 2, 15, 2].")
        if landmark_covariance.shape != (*landmark_mean_xy.shape, 2):
            raise ValueError(
                "landmark_covariance must have shape [B, 2, 15, 2, 2]."
            )
        if visibility_probability.shape != landmark_mean_xy.shape[:-1]:
            raise ValueError(
                "visibility_probability must have shape [B, 2, 15]."
            )

        batch_size = landmark_mean_xy.shape[0]
        eye_count = batch_size * 2
        means = landmark_mean_xy.float().reshape(
            eye_count, EYE_PSEUDO_POINT_COUNT, 2
        )
        covariance = landmark_covariance.float().reshape(
            eye_count, EYE_PSEUDO_POINT_COUNT, 2, 2
        )
        visibility = visibility_probability.float().reshape(
            eye_count, EYE_PSEUDO_POINT_COUNT
        ).clamp(0.0, 1.0)
        width, height = self.eye_image_size
        pixel_scale = means.new_tensor((width - 1.0, height - 1.0))
        points_px = means * pixel_scale
        covariance_px = (
            covariance
            * pixel_scale.reshape(1, 1, 2, 1)
            * pixel_scale.reshape(1, 1, 1, 2)
        )

        reference = _estimate_probabilistic_eye_reference_frame(
            points_px,
            covariance_px,
            visibility,
            covariance_floor_px=self.covariance_floor_px,
            use_uncertainty=self.use_reference_uncertainty,
        )
        origin_px = reference["origin_px"]
        basis = reference["basis"]
        canthus_distance_px = reference["canthus_distance_px"]
        relative_px = points_px - origin_px.unsqueeze(1)
        local_points = (
            torch.einsum("nij,nqj->nqi", basis.transpose(1, 2), relative_px)
            / canthus_distance_px[:, None, None]
        )

        upper_local = _weighted_point_mean(
            local_points[:, 1:5], reference["point_weight"][:, 1:5]
        )
        lower_local = _weighted_point_mean(
            local_points[:, 6:10], reference["point_weight"][:, 6:10]
        )
        aperture_center_local = 0.5 * (upper_local + lower_local)
        parameters = self.template_parameters()
        eyeball_center_local = aperture_center_local + torch.stack(
            (
                parameters["center_x_offset_canthus_ratio"].expand(eye_count),
                parameters["center_y_offset_canthus_ratio"].expand(eye_count),
            ),
            dim=-1,
        )
        eyeball_radius_ratio = parameters[
            "eyeball_radius_canthus_ratio"
        ]
        iris_radius_ratio = parameters["iris_radius_canthus_ratio"]

        if self.use_aperture_visibility:
            iris_aperture_visibility = _soft_iris_aperture_visibility(
                local_points[:, 1:5],
                local_points[:, 6:10],
                local_points[:, EYE_PSEUDO_IRIS_CENTER_INDEX:],
            )
        else:
            iris_aperture_visibility = visibility.new_ones((eye_count, 5))
        iris_center_visibility = (
            visibility[:, EYE_PSEUDO_IRIS_CENTER_INDEX]
            * iris_aperture_visibility[:, 0]
        )
        rim_visibility = (
            visibility[:, IRIS_RIM_SLICE]
            * iris_aperture_visibility[:, 1:]
        )
        (
            iris_center_px,
            iris_center_variance_px2,
            iris_center_effective_visibility,
        ) = _observed_iris_center(
            points_px,
            covariance_px,
            iris_center_visibility,
            rim_visibility,
            covariance_floor_px=self.covariance_floor_px,
        )
        rim_px = points_px[:, IRIS_RIM_SLICE]
        rim_variance_px2 = 0.5 * torch.diagonal(
            covariance_px[:, IRIS_RIM_SLICE], dim1=-2, dim2=-1
        ).sum(dim=-1)

        candidate_gxy = self.candidate_gaze[:, :2]
        candidate_center_local = (
            eyeball_center_local[:, None]
            + eyeball_radius_ratio * candidate_gxy[None]
        )
        candidate_center_px = origin_px[:, None] + canthus_distance_px[
            :, None, None
        ] * torch.einsum("nij,ncj->nci", basis, candidate_center_local)
        center_delta_px = candidate_center_px - iris_center_px[:, None]
        center_scale_px2 = iris_center_variance_px2 + (
            0.02 * canthus_distance_px
        ).square()
        center_normalized_sq = center_delta_px.square().sum(dim=-1) / (
            center_scale_px2[:, None].clamp_min(1e-4)
        )
        center_energy = torch.sqrt(1.0 + center_normalized_sq) - 1.0

        candidate_covariance_local = _projected_iris_covariance(
            candidate_gxy,
            iris_radius_ratio,
        )
        candidate_covariance_px = canthus_distance_px[:, None, None, None].square() * (
            basis[:, None]
            @ candidate_covariance_local[None]
            @ basis.transpose(1, 2)[:, None]
        )
        identity = torch.eye(2, device=means.device, dtype=torch.float32)
        candidate_covariance_inverse = torch.linalg.inv(
            candidate_covariance_px
            + self.covariance_floor_px**2 * identity.reshape(1, 1, 2, 2)
        )
        rim_delta_px = rim_px[:, None] - candidate_center_px[:, :, None]
        rim_radius = torch.sqrt(
            0.5
            * torch.einsum(
                "ncqi,ncij,ncqj->ncq",
                rim_delta_px,
                candidate_covariance_inverse,
                rim_delta_px,
            ).clamp_min(1e-10)
        )
        rim_uncertainty_scale = (
            0.02 * canthus_distance_px[:, None]
        ).square().clamp_min(1e-4)
        rim_weight = rim_visibility / (
            1.0 + rim_variance_px2 / rim_uncertainty_scale
        )
        rim_residual = torch.sqrt(
            1.0 + ((rim_radius - 1.0) / 0.15).square()
        ) - 1.0
        rim_energy = (
            rim_residual * rim_weight[:, None]
        ).sum(dim=-1) / rim_weight.sum(dim=-1, keepdim=True).clamp_min(0.1)
        data_energy = (
            self.center_energy_weight * center_energy
            + self.rim_energy_weight * rim_energy
        )
        total_energy = data_energy + (
            self.angle_prior_weight * self.candidate_angle_prior[None]
        )
        posterior = torch.softmax(
            -total_energy / self.posterior_temperature,
            dim=-1,
        )

        gaze_direction = F.normalize(
            posterior @ self.candidate_gaze,
            dim=-1,
            eps=1e-8,
        )
        yaw_rad = torch.atan2(gaze_direction[:, 0], gaze_direction[:, 2])
        pitch_rad = torch.asin(gaze_direction[:, 1].clamp(-1.0, 1.0))
        fitted_angles = torch.stack((yaw_rad, pitch_rad), dim=-1)
        angle_delta = self.candidate_angles_rad[None] - fitted_angles[:, None]
        angle_covariance = torch.einsum(
            "nc,nci,ncj->nij", posterior, angle_delta, angle_delta
        )

        fitted_gxy = gaze_direction[:, :2]
        fitted_center_local = (
            eyeball_center_local + eyeball_radius_ratio * fitted_gxy
        )
        fitted_center_px = origin_px + canthus_distance_px[:, None] * torch.einsum(
            "nij,nj->ni", basis, fitted_center_local
        )
        fitted_covariance_local = _projected_iris_covariance(
            fitted_gxy,
            iris_radius_ratio,
        )
        fitted_covariance_px = canthus_distance_px[:, None, None].square() * (
            basis @ fitted_covariance_local @ basis.transpose(1, 2)
        )
        fitted_inverse = torch.linalg.inv(
            fitted_covariance_px
            + self.covariance_floor_px**2 * identity.reshape(1, 2, 2)
        )
        fitted_rim_delta = rim_px - fitted_center_px[:, None]
        fitted_rim_radius = torch.sqrt(
            0.5
            * torch.einsum(
                "nqi,nij,nqj->nq",
                fitted_rim_delta,
                fitted_inverse,
                fitted_rim_delta,
            ).clamp_min(1e-10)
        )
        fitted_rim_error = (
            (fitted_rim_radius - 1.0).abs() * rim_weight
        ).sum(dim=-1) / rim_weight.sum(dim=-1).clamp_min(0.1)
        fitted_center_error = torch.linalg.vector_norm(
            fitted_center_px - iris_center_px, dim=-1
        ) / canthus_distance_px
        reprojection_error = (
            fitted_center_error + iris_radius_ratio * fitted_rim_error
        ).clamp(max=10.0)

        posterior_entropy = -(
            posterior * torch.log(posterior.clamp_min(1e-12))
        ).sum(dim=-1) / math.log(posterior.shape[-1])
        iris_confidence_product = (
            iris_center_effective_visibility * rim_visibility.mean(dim=-1)
        )
        iris_confidence = (
            torch.sqrt(iris_confidence_product.clamp_min(1e-12))
            * (iris_confidence_product > 0).to(iris_confidence_product.dtype)
        )
        input_confidence_product = reference["confidence"] * iris_confidence
        input_confidence = (
            torch.sqrt(input_confidence_product.clamp_min(1e-12))
            * (input_confidence_product > 0).to(input_confidence_product.dtype)
        ).clamp(0.0, 1.0)
        fit_confidence = torch.exp(-reprojection_error / 0.08)
        posterior_confidence = (1.0 - posterior_entropy).clamp(0.0, 1.0)
        scale_confidence = torch.sigmoid(
            (canthus_distance_px - self.min_canthus_distance_px) / 2.0
        )
        geometry_confidence = (
            input_confidence
            * fit_confidence
            * (0.25 + 0.75 * posterior_confidence)
            * scale_confidence
        ).clamp(0.0, 1.0)
        valid_mask = (
            (geometry_confidence >= self.min_geometry_confidence)
            & (canthus_distance_px >= self.min_canthus_distance_px)
        ).float()

        fitted_center_xy = fitted_center_px / pixel_scale
        eyeball_center_px = origin_px + canthus_distance_px[:, None] * torch.einsum(
            "nij,nj->ni", basis, eyeball_center_local
        )
        eyeball_center_xy = eyeball_center_px / pixel_scale
        observed_iris_center_xy = iris_center_px / pixel_scale
        fitted_covariance_xy = (
            fitted_covariance_px
            / pixel_scale.reshape(1, 2, 1)
            / pixel_scale.reshape(1, 1, 2)
        )
        expected_fit_energy = (posterior * data_energy).sum(dim=-1)
        observed_iris_center_local = (
            torch.einsum(
                "nij,nj->ni",
                basis.transpose(1, 2),
                iris_center_px - origin_px,
            )
            / canthus_distance_px[:, None]
        )

        def binocular(value: Tensor) -> Tensor:
            return value.reshape(batch_size, 2, *value.shape[1:])

        return {
            "gaze_direction_eye": binocular(gaze_direction),
            "yaw_rad": binocular(yaw_rad),
            "pitch_rad": binocular(pitch_rad),
            "safe_yaw_rad": binocular(yaw_rad * valid_mask),
            "safe_pitch_rad": binocular(pitch_rad * valid_mask),
            "angle_covariance": binocular(angle_covariance),
            "pitch_variance": binocular(angle_covariance[:, 1, 1]),
            "angle_posterior": posterior.reshape(
                batch_size, 2, self.pitch_bins, self.yaw_bins
            ),
            "pitch_probability": posterior.reshape(
                eye_count, self.pitch_bins, self.yaw_bins
            ).sum(dim=-1).reshape(batch_size, 2, self.pitch_bins),
            "eyeball_center_xy": binocular(eyeball_center_xy),
            "eyeball_radius_px": binocular(
                canthus_distance_px * eyeball_radius_ratio
            ),
            "iris_radius_px": binocular(
                canthus_distance_px * iris_radius_ratio
            ),
            "observed_iris_center_xy": binocular(observed_iris_center_xy),
            "observed_iris_center_local": binocular(
                observed_iris_center_local
            ),
            "reference_origin_xy": binocular(origin_px / pixel_scale),
            "reference_basis_image": binocular(basis),
            "reference_confidence": binocular(reference["confidence"]),
            "iris_center_effective_visibility": binocular(
                iris_center_effective_visibility
            ),
            "iris_rim_effective_visibility": binocular(rim_visibility),
            "fitted_iris_center_xy": binocular(fitted_center_xy),
            "fitted_iris_covariance": binocular(fitted_covariance_xy),
            "reprojection_error_norm": binocular(reprojection_error),
            "expected_fit_energy": binocular(expected_fit_energy),
            "posterior_entropy": binocular(posterior_entropy),
            "geometry_confidence": binocular(geometry_confidence),
            "valid_mask": binocular(valid_mask),
            "canthus_distance_px": binocular(canthus_distance_px),
            "eyeball_radius_canthus_ratio": binocular(
                eyeball_radius_ratio.expand(eye_count)
            ),
            "iris_radius_canthus_ratio": binocular(
                iris_radius_ratio.expand(eye_count)
            ),
        }


def low_dof_eye_template_losses(
    template_outputs: Mapping[str, Tensor],
    eye_weight: Tensor,
    *,
    huber_delta_rad: float = math.radians(2.0),
) -> dict[str, Tensor]:
    """Self-supervise template reprojection and binocular pitch agreement."""

    if huber_delta_rad <= 0:
        raise ValueError("huber_delta_rad must be positive.")
    reprojection = template_outputs["reprojection_error_norm"]
    pitch = template_outputs["pitch_rad"]
    confidence = template_outputs["geometry_confidence"]
    if reprojection.ndim != 2 or reprojection.shape[1] != 2:
        raise ValueError("Template outputs must use binocular shape [B, 2].")
    if eye_weight.shape != reprojection.shape:
        raise ValueError("eye_weight must have shape [B, 2].")
    teacher_weight = eye_weight.to(
        device=reprojection.device, dtype=torch.float32
    )
    # Detaching the gate prevents the network from lowering confidence to hide
    # a bad fit. It also keeps an untrained, collapsed canthus prediction from
    # injecting unstable geometry gradients before the landmark head separates.
    weight = teacher_weight * confidence.detach()
    reprojection_loss = (
        reprojection.float() * weight
    ).sum() / weight.sum().clamp_min(1.0)
    pair_weight = weight.prod(dim=-1)
    pitch_difference = pitch[:, 0] - pitch[:, 1]
    pitch_consistency = F.smooth_l1_loss(
        pitch_difference,
        torch.zeros_like(pitch_difference),
        beta=huber_delta_rad,
        reduction="none",
    )
    pitch_consistency_loss = (
        pitch_consistency * pair_weight
    ).sum() / pair_weight.sum().clamp_min(1.0)
    return {
        "reprojection_loss": reprojection_loss,
        "binocular_pitch_consistency_loss": pitch_consistency_loss,
        "effective_eye_weight": weight.sum(),
    }


def low_dof_eye_template_metrics(
    template_outputs: Mapping[str, Tensor],
    eye_weight: Tensor,
) -> dict[str, Tensor]:
    """Summarize fit quality without using gaze or desktop-UV labels."""

    reprojection = template_outputs["reprojection_error_norm"].float()
    pitch = template_outputs["pitch_rad"].float()
    pitch_variance = template_outputs["pitch_variance"].float()
    confidence = template_outputs["geometry_confidence"].float()
    valid = template_outputs["valid_mask"].float()
    reference_confidence = template_outputs["reference_confidence"].float()
    iris_center_visibility = template_outputs[
        "iris_center_effective_visibility"
    ].float()
    iris_rim_visibility = template_outputs[
        "iris_rim_effective_visibility"
    ].float().mean(dim=-1)
    weight = eye_weight.to(device=reprojection.device, dtype=torch.float32)
    denominator = weight.sum().clamp_min(1.0)
    pair_weight = weight.prod(dim=-1)
    pair_denominator = pair_weight.sum().clamp_min(1.0)
    return {
        "eye_template_reprojection_error_norm": (
            reprojection * weight
        ).sum()
        / denominator,
        "eye_template_pitch_std_deg": (
            torch.rad2deg(pitch_variance.clamp_min(0.0).sqrt()) * weight
        ).sum()
        / denominator,
        "eye_template_binocular_pitch_disagreement_deg": (
            torch.rad2deg((pitch[:, 0] - pitch[:, 1]).abs()) * pair_weight
        ).sum()
        / pair_denominator,
        "eye_template_geometry_confidence": (
            confidence * weight
        ).sum()
        / denominator,
        "eye_template_reference_confidence": (
            reference_confidence * weight
        ).sum()
        / denominator,
        "eye_template_iris_effective_visibility": (
            0.5 * (iris_center_visibility + iris_rim_visibility) * weight
        ).sum()
        / denominator,
        "eye_template_valid_rate": (valid * weight).sum() / denominator,
    }



VERTICAL_EYE_GEOMETRY_FEATURE_DIM = 22


class VerticalEyeGeometryFeatureExtractor(nn.Module):
    """Convert 15 probabilistic landmarks into a compact vertical descriptor.

    This deliberately does not infer a 3D pitch.  It exposes the crop-local
    vertical iris displacement, aperture, ellipse shape, uncertainty, and
    visibility that a residual head can combine with global face/scene context.
    """

    def __init__(self, *, eye_image_size: tuple[int, int] = (90, 56)) -> None:
        super().__init__()
        if len(eye_image_size) != 2 or min(eye_image_size) <= 1:
            raise ValueError("eye_image_size must be (width, height) > 1.")
        width, height = (int(value) for value in eye_image_size)
        self.register_buffer(
            "pixel_scale",
            torch.tensor((float(width - 1), float(height - 1))),
            persistent=False,
        )

    def forward(
        self,
        mean_xy: Tensor,
        covariance_xy: Tensor,
        visibility: Tensor,
        entropy: Tensor,
    ) -> dict[str, Tensor]:
        expected_points = (EYE_PSEUDO_POINT_COUNT, 2)
        if mean_xy.ndim != 4 or mean_xy.shape[1:] != (2, *expected_points):
            raise ValueError("mean_xy must have shape [B, 2, 15, 2].")
        if covariance_xy.shape != (*mean_xy.shape[:-1], 2, 2):
            raise ValueError("covariance_xy must have shape [B, 2, 15, 2, 2].")
        if visibility.shape != mean_xy.shape[:-1]:
            raise ValueError("visibility must have shape [B, 2, 15].")
        if entropy.shape != visibility.shape:
            raise ValueError("entropy must have shape [B, 2, 15].")

        pixel_scale = self.pixel_scale.to(device=mean_xy.device, dtype=torch.float32)
        points = mean_xy.float() * pixel_scale.reshape(1, 1, 1, 2)
        covariance = (
            covariance_xy.float()
            * pixel_scale.reshape(1, 1, 1, 2, 1)
            * pixel_scale.reshape(1, 1, 1, 1, 2)
        )
        visibility = visibility.float().clamp(0.0, 1.0)
        entropy = entropy.float().clamp(0.0, 1.0)

        outer = points[:, :, OUTER_CANTHUS_INDEX]
        inner = points[:, :, INNER_CANTHUS_INDEX]
        origin = 0.5 * (outer + inner)
        horizontal = inner - outer
        canthus_distance = torch.linalg.vector_norm(horizontal, dim=-1).clamp_min(1e-4)
        x_axis = horizontal / canthus_distance.unsqueeze(-1)
        y_axis = torch.stack((-x_axis[..., 1], x_axis[..., 0]), dim=-1)
        upper = points[:, :, 1:5].mean(dim=-2)
        lower = points[:, :, 6:10].mean(dim=-2)
        orientation = torch.where(
            ((lower - upper) * y_axis).sum(dim=-1, keepdim=True) >= 0,
            torch.ones_like(canthus_distance).unsqueeze(-1),
            -torch.ones_like(canthus_distance).unsqueeze(-1),
        )
        y_axis = y_axis * orientation
        centered = points - origin.unsqueeze(-2)
        local_x = (centered * x_axis.unsqueeze(-2)).sum(dim=-1) / canthus_distance.unsqueeze(-1)
        local_y = (centered * y_axis.unsqueeze(-2)).sum(dim=-1) / canthus_distance.unsqueeze(-1)

        iris_center_y = local_y[:, :, EYE_PSEUDO_IRIS_CENTER_INDEX]
        upper_y = local_y[:, :, 1:5].mean(dim=-1)
        lower_y = local_y[:, :, 6:10].mean(dim=-1)
        aperture_height = (lower_y - upper_y).clamp_min(0.0)
        rim_x = local_x[:, :, IRIS_RIM_SLICE]
        rim_y = local_y[:, :, IRIS_RIM_SLICE]
        iris_radius_x = 0.5 * (rim_x[:, :, 0].abs() + rim_x[:, :, 2].abs())
        iris_radius_y = 0.5 * (rim_y[:, :, 1].abs() + rim_y[:, :, 3].abs())
        iris_log_aspect = torch.log(
            (iris_radius_y + 1e-4) / (iris_radius_x + 1e-4)
        ).clamp(-3.0, 3.0)

        covariance_trace = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)
        point_std = torch.sqrt(covariance_trace.clamp_min(1e-8)) / canthus_distance.unsqueeze(-1)
        iris_std = point_std[:, :, EYE_PSEUDO_IRIS_CENTER_INDEX]
        iris_visibility = visibility[:, :, EYE_PSEUDO_IRIS_CENTER_INDEX]
        rim_visibility = visibility[:, :, IRIS_RIM_SLICE].mean(dim=-1)
        iris_entropy = entropy[:, :, EYE_PSEUDO_IRIS_CENTER_INDEX]
        anchor_visibility = visibility[:, :, (OUTER_CANTHUS_INDEX, INNER_CANTHUS_INDEX)].mean(dim=-1)
        lid_visibility = 0.5 * (
            visibility[:, :, 1:5].mean(dim=-1)
            + visibility[:, :, 6:10].mean(dim=-1)
        )
        reference_visibility = anchor_visibility * lid_visibility

        per_eye = torch.stack(
            (
                iris_center_y.clamp(-1.5, 1.5),
                aperture_height.clamp(0.0, 1.5),
                iris_radius_y.clamp(0.0, 1.0),
                iris_radius_x.clamp(0.0, 1.0),
                iris_log_aspect,
                iris_visibility,
                rim_visibility,
                iris_std.clamp(0.0, 2.0),
                iris_entropy,
                reference_visibility,
            ),
            dim=-1,
        )
        binocular_mean = per_eye.mean(dim=1)
        binocular_difference = per_eye[:, 0] - per_eye[:, 1]
        iris_disagreement = iris_center_y[:, :1] - iris_center_y[:, 1:2]
        entropy_confidence = (1.0 - iris_entropy).clamp(0.0, 1.0)
        uncertainty_confidence = torch.exp(-iris_std.clamp_min(0.0) / 0.12)
        per_eye_confidence = (
            iris_visibility
            * rim_visibility
            * reference_visibility
            * entropy_confidence
            * uncertainty_confidence
        ).clamp(0.0, 1.0)
        geometry_confidence = per_eye_confidence.mean(dim=-1, keepdim=True)
        features = torch.cat(
            (
                binocular_mean,
                binocular_difference,
                geometry_confidence,
                iris_disagreement.abs(),
            ),
            dim=-1,
        )
        if features.shape[-1] != VERTICAL_EYE_GEOMETRY_FEATURE_DIM:
            raise RuntimeError("Unexpected vertical geometry feature dimension.")
        if not torch.isfinite(features).all():
            raise ValueError("Vertical eye geometry features must be finite.")
        return {
            "features": features,
            "per_eye_features": per_eye,
            "geometry_confidence": geometry_confidence,
            "iris_vertical_disagreement": iris_disagreement.abs(),
            "per_eye_confidence": per_eye_confidence,
        }


class VerticalGeometryVResidual(nn.Module):
    """Low-capacity, confidence-gated residual that can modify only table ``v``.

    During training the gate has a small continuous floor so the branch can
    bootstrap from uncertain early heatmaps.  Evaluation restores a confidence
    threshold and falls back exactly to the base UV estimate when geometry is
    unreliable.
    """

    def __init__(
        self,
        *,
        geometry_dim: int,
        fused_feature_dim: int,
        scene_dim: int,
        pose_dim: int = 0,
        hidden_dims: tuple[int, ...] = (64, 32),
        max_abs_delta_v_mm: float = 250.0,
        min_training_gate: float = 0.10,
        min_inference_confidence: float = 0.10,
    ) -> None:
        super().__init__()
        if min(geometry_dim, fused_feature_dim, scene_dim, max_abs_delta_v_mm) <= 0:
            raise ValueError("Vertical geometry residual dimensions/scales must be positive.")
        if pose_dim < 0 or not hidden_dims or min(hidden_dims) <= 0:
            raise ValueError("Vertical geometry residual hidden dimensions are invalid.")
        if not 0 <= min_training_gate <= 1 or not 0 <= min_inference_confidence <= 1:
            raise ValueError("Vertical geometry gates must lie in [0, 1].")
        self.geometry_dim = int(geometry_dim)
        self.fused_feature_dim = int(fused_feature_dim)
        self.scene_dim = int(scene_dim)
        self.pose_dim = int(pose_dim)
        self.max_abs_delta_v_mm = float(max_abs_delta_v_mm)
        self.min_training_gate = float(min_training_gate)
        self.min_inference_confidence = float(min_inference_confidence)
        input_dim = geometry_dim + fused_feature_dim + scene_dim + pose_dim + 1
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.SiLU(inplace=True)))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 2))
        self.network = nn.Sequential(*layers)
        self.reset_output_to_zero()

    def reset_output_to_zero(self) -> None:
        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(
        self,
        *,
        geometry_features: Tensor,
        geometry_confidence: Tensor,
        fused_features: Tensor,
        scene_vec: Tensor,
        base_v_normalized: Tensor,
        virtual_camera_pose_table: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch_size = geometry_features.shape[0]
        expected = (batch_size, self.geometry_dim)
        if geometry_features.shape != expected:
            raise ValueError(f"geometry_features must have shape {expected}.")
        if geometry_confidence.shape != (batch_size, 1):
            raise ValueError("geometry_confidence must have shape [B, 1].")
        if fused_features.shape != (batch_size, self.fused_feature_dim):
            raise ValueError("fused_features has an unexpected shape.")
        if scene_vec.shape != (batch_size, self.scene_dim):
            raise ValueError("scene_vec has an unexpected shape.")
        if base_v_normalized.shape != (batch_size, 1):
            raise ValueError("base_v_normalized must have shape [B, 1].")
        parts = (
            geometry_features.float(),
            fused_features.float(),
            scene_vec.float(),
            base_v_normalized.detach().float(),
        )
        if self.pose_dim > 0:
            if virtual_camera_pose_table is None or virtual_camera_pose_table.shape != (
                batch_size,
                self.pose_dim,
            ):
                raise ValueError("virtual_camera_pose_table has an unexpected shape.")
            parts = (*parts[:-1], virtual_camera_pose_table.float(), parts[-1])
        output = self.network(torch.cat(parts, dim=-1))
        raw_delta_v_mm = self.max_abs_delta_v_mm * torch.tanh(output[:, :1])
        learned_gate = torch.sigmoid(output[:, 1:2])
        confidence = geometry_confidence.float().clamp(0.0, 1.0)
        training_gate = self.min_training_gate + (1.0 - self.min_training_gate) * confidence
        if self.training:
            confidence_gate = training_gate
            fallback_mask = torch.zeros_like(confidence)
        else:
            fallback_mask = (confidence < self.min_inference_confidence).float()
            confidence_gate = confidence * (1.0 - fallback_mask)
        applied_geometry_gate = confidence_gate * learned_gate
        delta_v_mm = applied_geometry_gate * raw_delta_v_mm
        return {
            "delta_v_mm": delta_v_mm,
            "raw_delta_v_mm": raw_delta_v_mm,
            "geometry_gate": confidence_gate,
            "applied_geometry_gate": applied_geometry_gate,
            "training_gate": training_gate,
            "learned_gate": learned_gate,
            "fallback_mask": fallback_mask,
            "camera_table_vertical_sensitivity_mm_per_rad": torch.zeros_like(delta_v_mm),
            "camera_table_geometry_valid_mask": torch.zeros_like(delta_v_mm),
            "geometric_pitch_delta_v_mm": torch.zeros_like(delta_v_mm),
        }


class PitchToTableVResidual(nn.Module):
    """Map uncertain binocular pitch to a confidence-gated table-v residual.

    The mapper is deliberately low capacity. Scene/virtual-pose context predicts
    only three coefficients for ``p``, ``p^3``, and a bias. It cannot modify
    table ``u`` and starts as an exact zero residual, preserving the baseline.
    """

    def __init__(
        self,
        *,
        scene_dim: int,
        pose_dim: int = 0,
        hidden_dims: tuple[int, ...] = (32, 16),
        max_abs_delta_v_mm: float = 250.0,
        pitch_scale_deg: float = 20.0,
        uncertainty_scale_deg: float = 8.0,
        disagreement_scale_deg: float = 6.0,
        min_pitch_std_deg: float = 0.5,
        min_active_gate: float = 0.05,
        detach_uncertainty_gate: bool = True,
        table_distance_scale_mm: float = 1000.0,
        use_camera_table_sensitivity: bool = True,
        min_abs_table_normal_z: float = 0.1,
    ) -> None:
        super().__init__()
        if scene_dim <= 0 or pose_dim < 0:
            raise ValueError("scene_dim must be positive and pose_dim non-negative.")
        if not hidden_dims or min(hidden_dims) <= 0:
            raise ValueError("hidden_dims must contain positive widths.")
        if min(
            max_abs_delta_v_mm,
            pitch_scale_deg,
            uncertainty_scale_deg,
            disagreement_scale_deg,
            min_pitch_std_deg,
            table_distance_scale_mm,
            min_abs_table_normal_z,
        ) <= 0:
            raise ValueError("Pitch-to-v scales must be positive.")
        if not 0 <= min_active_gate <= 1:
            raise ValueError("min_active_gate must lie in [0, 1].")
        self.scene_dim = int(scene_dim)
        self.pose_dim = int(pose_dim)
        self.max_abs_delta_v_mm = float(max_abs_delta_v_mm)
        self.pitch_scale_rad = math.radians(pitch_scale_deg)
        self.uncertainty_scale_rad = math.radians(uncertainty_scale_deg)
        self.disagreement_scale_rad = math.radians(disagreement_scale_deg)
        self.min_pitch_variance = math.radians(min_pitch_std_deg) ** 2
        self.min_active_gate = float(min_active_gate)
        self.detach_uncertainty_gate = bool(detach_uncertainty_gate)
        self.table_distance_scale_mm = float(table_distance_scale_mm)
        self.use_camera_table_sensitivity = bool(use_camera_table_sensitivity)
        self.min_abs_table_normal_z = float(min_abs_table_normal_z)

        layers: list[nn.Module] = [nn.LayerNorm(scene_dim + pose_dim)]
        input_dim = scene_dim + pose_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                (
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(inplace=True),
                )
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 3))
        self.coefficient_network = nn.Sequential(*layers)
        self.reset_output_to_zero()

    def reset_output_to_zero(self) -> None:
        output = self.coefficient_network[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(
        self,
        *,
        pitch_rad: Tensor,
        pitch_variance: Tensor,
        geometry_confidence: Tensor,
        geometry_valid_mask: Tensor,
        scene_vec: Tensor,
        virtual_camera_pose_table: Tensor | None = None,
    ) -> dict[str, Tensor]:
        expected_shape = pitch_rad.shape
        if pitch_rad.ndim != 2 or pitch_rad.shape[1] != 2:
            raise ValueError("pitch_rad must have shape [B, 2].")
        for name, value in (
            ("pitch_variance", pitch_variance),
            ("geometry_confidence", geometry_confidence),
            ("geometry_valid_mask", geometry_valid_mask),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape [B, 2].")
        if scene_vec.ndim != 2 or scene_vec.shape != (
            pitch_rad.shape[0],
            self.scene_dim,
        ):
            raise ValueError(
                f"scene_vec must have shape [B, {self.scene_dim}]."
            )
        if self.pose_dim > 0:
            if virtual_camera_pose_table is None or virtual_camera_pose_table.shape != (
                pitch_rad.shape[0],
                self.pose_dim,
            ):
                raise ValueError(
                    "virtual_camera_pose_table must match configured pose_dim."
                )
            context = torch.cat(
                (scene_vec.float(), virtual_camera_pose_table.float()), dim=-1
            )
        else:
            context = scene_vec.float()

        pitch = pitch_rad.float()
        variance = pitch_variance.float().clamp_min(self.min_pitch_variance)
        confidence = geometry_confidence.float().clamp(0.0, 1.0)
        valid = geometry_valid_mask.float().clamp(0.0, 1.0)
        eye_quality = confidence * valid
        precision = eye_quality / variance
        precision_sum = precision.sum(dim=-1, keepdim=True)
        fused_pitch = (precision * pitch).sum(dim=-1, keepdim=True) / (
            precision_sum.clamp_min(1e-8)
        )
        has_geometry = (precision_sum > 1e-8).float()
        fused_pitch = fused_pitch * has_geometry
        fused_variance = torch.where(
            precision_sum > 1e-8,
            precision_sum.clamp_min(1e-8).reciprocal(),
            torch.full_like(
                precision_sum,
                4.0 * self.uncertainty_scale_rad**2,
            ),
        )
        pitch_disagreement = (pitch[:, :1] - pitch[:, 1:2]).abs()
        pair_support = (eye_quality[:, :1] * eye_quality[:, 1:2] > 0).float()
        confidence_union = 1.0 - (
            (1.0 - eye_quality[:, :1]) * (1.0 - eye_quality[:, 1:2])
        )
        uncertainty_gate = torch.exp(
            -0.5 * fused_variance / self.uncertainty_scale_rad**2
        )
        disagreement_gate = torch.where(
            pair_support > 0,
            torch.exp(
                -0.5
                * (pitch_disagreement / self.disagreement_scale_rad).square()
            ),
            torch.ones_like(pitch_disagreement),
        )
        geometry_gate = (
            has_geometry
            * confidence_union
            * uncertainty_gate
            * disagreement_gate
        ).clamp(0.0, 1.0)

        normalized_pitch = (fused_pitch / self.pitch_scale_rad).clamp(-2.0, 2.0)
        if self.use_camera_table_sensitivity:
            (
                camera_table_vertical_sensitivity,
                camera_table_geometry_valid,
            ) = _camera_table_vertical_sensitivity(
                scene_vec,
                distance_scale_mm=self.table_distance_scale_mm,
                min_abs_normal_z=self.min_abs_table_normal_z,
                max_abs_sensitivity_mm_per_rad=(
                    4.0 * self.max_abs_delta_v_mm / self.pitch_scale_rad
                ),
            )
            geometry_gate = geometry_gate * camera_table_geometry_valid
            geometric_pitch_delta_v_mm = (
                camera_table_vertical_sensitivity * fused_pitch
            )
            normalized_geometric_pitch = (
                geometric_pitch_delta_v_mm / self.max_abs_delta_v_mm
            ).clamp(-2.0, 2.0)
        else:
            camera_table_vertical_sensitivity = torch.zeros_like(fused_pitch)
            camera_table_geometry_valid = torch.ones_like(fused_pitch)
            geometric_pitch_delta_v_mm = torch.zeros_like(fused_pitch)
            normalized_geometric_pitch = normalized_pitch
        coefficients = self.coefficient_network(context)
        mapping_value = (
            coefficients[:, :1] * normalized_geometric_pitch
            + coefficients[:, 1:2] * normalized_pitch.pow(3)
            + coefficients[:, 2:3]
        )
        raw_delta_v_mm = self.max_abs_delta_v_mm * torch.tanh(mapping_value)
        fallback_mask = (geometry_gate < self.min_active_gate).float()
        active_geometry_gate = torch.where(
            fallback_mask > 0,
            torch.zeros_like(geometry_gate),
            geometry_gate,
        )
        applied_gate = (
            active_geometry_gate.detach()
            if self.detach_uncertainty_gate
            else active_geometry_gate
        )
        delta_v_mm = applied_gate * raw_delta_v_mm
        return {
            "delta_v_mm": delta_v_mm,
            "raw_delta_v_mm": raw_delta_v_mm,
            "geometry_gate": geometry_gate,
            "applied_geometry_gate": active_geometry_gate,
            "fallback_mask": fallback_mask,
            "fused_pitch_rad": fused_pitch,
            "fused_pitch_variance": fused_variance,
            "pitch_disagreement_rad": pitch_disagreement,
            "binocular_precision": precision_sum,
            "camera_table_vertical_sensitivity_mm_per_rad": (
                camera_table_vertical_sensitivity
            ),
            "camera_table_geometry_valid_mask": camera_table_geometry_valid,
            "geometric_pitch_delta_v_mm": geometric_pitch_delta_v_mm,
            "mapping_coefficients": coefficients,
        }


def _camera_table_vertical_sensitivity(
    table_frame7: Tensor,
    *,
    distance_scale_mm: float,
    min_abs_normal_z: float,
    max_abs_sensitivity_mm_per_rad: float,
) -> tuple[Tensor, Tensor]:
    """Return local table-v displacement per camera vertical ray radian.

    The derivative is evaluated around the virtual-camera optical axis.  The
    learned mapper still calibrates eye-local pitch, but its linear basis now
    carries the correct per-sample camera/table scale and orientation in mm.
    """

    table = unpack_table_frame7(
        table_frame7,
        distance_scale_mm=distance_scale_mm,
    )
    normal_z = table["n_c"][:, 2:3]
    normal_y = table["n_c"][:, 1:2]
    distance = table["d_c_mm"]
    valid = (
        (normal_z.abs() >= min_abs_normal_z)
        & (distance.abs() >= 1e-3)
    ).float()
    safe_sign = torch.where(normal_z >= 0, 1.0, -1.0)
    safe_normal_z = torch.where(
        normal_z.abs() >= min_abs_normal_z,
        normal_z,
        safe_sign * min_abs_normal_z,
    )
    forward_sign = torch.sign(distance / safe_normal_z)
    forward_sign = torch.where(
        forward_sign == 0,
        torch.ones_like(forward_sign),
        forward_sign,
    )
    denominator = safe_normal_z * forward_sign
    ray_distance = distance / denominator
    ray_distance_derivative = -distance * normal_y / denominator.square()
    e2 = table["e2_c"]
    sensitivity = (
        ray_distance * e2[:, 1:2]
        + ray_distance_derivative * forward_sign * e2[:, 2:3]
    )
    sensitivity = sensitivity.clamp(
        -max_abs_sensitivity_mm_per_rad,
        max_abs_sensitivity_mm_per_rad,
    )
    return sensitivity * valid, valid


def pitch_to_v_residual_regularization(
    delta_v_mm: Tensor,
    *,
    max_abs_delta_v_mm: float,
) -> Tensor:
    """Penalize large corrections without supplying additional labels."""

    if max_abs_delta_v_mm <= 0:
        raise ValueError("max_abs_delta_v_mm must be positive.")
    if delta_v_mm.ndim != 2 or delta_v_mm.shape[1] != 1:
        raise ValueError("delta_v_mm must have shape [B, 1].")
    return (delta_v_mm.float() / max_abs_delta_v_mm).square().mean()


def _bounded_value(raw: Tensor, minimum: float, maximum: float) -> Tensor:
    return minimum + (maximum - minimum) * torch.sigmoid(raw)


def _inverse_bounded_value(value: float, minimum: float, maximum: float) -> Tensor:
    ratio = (value - minimum) / (maximum - minimum)
    return torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)


def _weighted_point_mean(points: Tensor, weight: Tensor) -> Tensor:
    denominator = weight.sum(dim=-1, keepdim=True)
    weighted = (points * weight.unsqueeze(-1)).sum(dim=-2) / denominator.clamp_min(
        1e-6
    )
    fallback = points.mean(dim=-2)
    return torch.where(denominator > 1e-6, weighted, fallback)


def _estimate_probabilistic_eye_reference_frame(
    points_px: Tensor,
    covariance_px: Tensor,
    visibility: Tensor,
    *,
    covariance_floor_px: float,
    use_uncertainty: bool,
) -> dict[str, Tensor]:
    """Estimate the canthus/lid frame while downweighting uncertain anchors."""

    point_variance = 0.5 * torch.diagonal(
        covariance_px, dim1=-2, dim2=-1
    ).sum(dim=-1)
    if use_uncertainty:
        point_weight = visibility / (
            point_variance + covariance_floor_px**2
        ).clamp_min(1e-6)
    else:
        point_weight = visibility

    outer = points_px[:, OUTER_CANTHUS_INDEX]
    inner = points_px[:, INNER_CANTHUS_INDEX]
    origin_px = 0.5 * (outer + inner)
    horizontal = inner - outer
    canthus_distance_px = torch.linalg.vector_norm(
        horizontal, dim=-1
    ).clamp_min(1e-4)
    x_axis = horizontal / canthus_distance_px.unsqueeze(-1)
    perpendicular = torch.stack((-x_axis[:, 1], x_axis[:, 0]), dim=-1)
    upper_px = _weighted_point_mean(
        points_px[:, 1:5], point_weight[:, 1:5]
    )
    lower_px = _weighted_point_mean(
        points_px[:, 6:10], point_weight[:, 6:10]
    )
    vertical_sign = torch.where(
        ((lower_px - upper_px) * perpendicular).sum(dim=-1) >= 0,
        torch.ones_like(canthus_distance_px),
        -torch.ones_like(canthus_distance_px),
    )
    y_axis = perpendicular * vertical_sign.unsqueeze(-1)
    basis = torch.stack((x_axis, y_axis), dim=-1)

    anchor_visibility_product = (
        visibility[:, OUTER_CANTHUS_INDEX]
        * visibility[:, INNER_CANTHUS_INDEX]
    )
    anchor_visibility = (
        torch.sqrt(anchor_visibility_product.clamp_min(1e-12))
        * (anchor_visibility_product > 0).to(anchor_visibility_product.dtype)
    )
    lid_visibility_product = (
        visibility[:, 1:5].mean(dim=-1)
        * visibility[:, 6:10].mean(dim=-1)
    )
    lid_visibility = (
        torch.sqrt(lid_visibility_product.clamp_min(1e-12))
        * (lid_visibility_product > 0).to(lid_visibility_product.dtype)
    )
    if use_uncertainty:
        anchor_variance = 0.5 * (
            point_variance[:, OUTER_CANTHUS_INDEX]
            + point_variance[:, INNER_CANTHUS_INDEX]
        )
        upper_variance = point_weight[:, 1:5].sum(dim=-1).clamp_min(1e-6).reciprocal()
        lower_variance = point_weight[:, 6:10].sum(dim=-1).clamp_min(1e-6).reciprocal()
        reference_std_norm = torch.sqrt(
            anchor_variance + upper_variance + lower_variance
        ) / canthus_distance_px
        uncertainty_confidence = torch.exp(
            -0.5 * (reference_std_norm / 0.08).square()
        )
    else:
        reference_std_norm = torch.zeros_like(canthus_distance_px)
        uncertainty_confidence = torch.ones_like(canthus_distance_px)
    confidence_product = (
        anchor_visibility * lid_visibility * uncertainty_confidence
    )
    confidence = (
        confidence_product.clamp_min(1e-12).pow(1.0 / 3.0)
        * (confidence_product > 0).to(confidence_product.dtype)
    ).clamp(0.0, 1.0)
    return {
        "origin_px": origin_px,
        "basis": basis,
        "canthus_distance_px": canthus_distance_px,
        "point_weight": point_weight,
        "reference_std_norm": reference_std_norm,
        "confidence": confidence,
    }


def _soft_iris_aperture_visibility(
    upper_lid_local: Tensor,
    lower_lid_local: Tensor,
    iris_points_local: Tensor,
    *,
    horizontal_bandwidth: float = 0.12,
    vertical_softness: float = 0.025,
) -> Tensor:
    """Softly reduce support for iris points predicted behind an eyelid."""

    query_x = iris_points_local[..., 0]

    def interpolate_lid_y(lid: Tensor) -> Tensor:
        distance = (
            query_x.unsqueeze(-1) - lid[:, None, :, 0]
        ) / horizontal_bandwidth
        weight = torch.softmax(-distance.square(), dim=-1)
        return (weight * lid[:, None, :, 1]).sum(dim=-1)

    upper_y = interpolate_lid_y(upper_lid_local)
    lower_y = interpolate_lid_y(lower_lid_local)
    top = torch.minimum(upper_y, lower_y)
    bottom = torch.maximum(upper_y, lower_y)
    visible_from_top = torch.sigmoid(
        (iris_points_local[..., 1] - top) / vertical_softness
    )
    visible_from_bottom = torch.sigmoid(
        (bottom - iris_points_local[..., 1]) / vertical_softness
    )
    return (visible_from_top * visible_from_bottom).clamp(0.0, 1.0)


def _observed_iris_center(
    points_px: Tensor,
    covariance_px: Tensor,
    center_visibility: Tensor,
    rim_visibility: Tensor,
    *,
    covariance_floor_px: float,
) -> tuple[Tensor, Tensor, Tensor]:
    direct = points_px[:, EYE_PSEUDO_IRIS_CENTER_INDEX]
    direct_variance = 0.5 * torch.diagonal(
        covariance_px[:, EYE_PSEUDO_IRIS_CENTER_INDEX], dim1=-2, dim2=-1
    ).sum(dim=-1)
    rim = points_px[:, IRIS_RIM_SLICE]
    rim_weight_sum = rim_visibility.sum(dim=-1, keepdim=True)
    rim_alpha = rim_visibility / rim_weight_sum.clamp_min(1e-6)
    rim_center = (rim * rim_alpha.unsqueeze(-1)).sum(dim=-2)
    rim_covariance = torch.einsum(
        "nq,nqij->nij",
        rim_alpha.square(),
        covariance_px[:, IRIS_RIM_SLICE],
    )
    rim_center_variance = 0.5 * torch.diagonal(
        rim_covariance, dim1=-2, dim2=-1
    ).sum(dim=-1)
    floor_variance = covariance_floor_px**2
    direct_precision = center_visibility / (
        direct_variance + floor_variance
    ).clamp_min(1e-6)
    rim_precision = rim_visibility.mean(dim=-1) / (
        rim_center_variance + floor_variance
    ).clamp_min(1e-6)
    total_precision = direct_precision + rim_precision
    center = (
        direct * direct_precision.unsqueeze(-1)
        + rim_center * rim_precision.unsqueeze(-1)
    ) / total_precision.unsqueeze(-1).clamp_min(1e-6)
    fallback = 0.5 * (direct + rim_center)
    center = torch.where(total_precision[:, None] > 1e-6, center, fallback)
    variance = total_precision.clamp_min(1e-6).reciprocal()
    effective_visibility = 1.0 - (
        (1.0 - center_visibility) * (1.0 - rim_visibility.mean(dim=-1))
    )
    return center, variance, effective_visibility.clamp(0.0, 1.0)


def _projected_iris_covariance(
    gaze_xy: Tensor,
    iris_radius: Tensor,
) -> Tensor:
    identity = torch.eye(2, device=gaze_xy.device, dtype=torch.float32)
    projection = identity - gaze_xy.unsqueeze(-1) * gaze_xy.unsqueeze(-2)
    return 0.5 * iris_radius.square() * projection


def stack_binocular_outputs(
    outputs: Mapping[str, Tensor],
    suffix: str,
) -> Tensor:
    """Stack left/right model outputs into anatomical eye dimension 1."""

    return torch.stack(
        (outputs[f"left_eye_keypoint_{suffix}"], outputs[f"right_eye_keypoint_{suffix}"]),
        dim=1,
    )
