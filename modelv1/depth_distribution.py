"""PnP-centred eye-depth hypotheses and optional learned posterior weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


DEPTH_DISTRIBUTION_POINT = "point"
DEPTH_DISTRIBUTION_FIXED_PRIOR = "fixed_prior"
DEPTH_DISTRIBUTION_LEARNED_REWEIGHT = "learned_reweight"
DEPTH_DISTRIBUTION_MODES = (
    DEPTH_DISTRIBUTION_POINT,
    DEPTH_DISTRIBUTION_FIXED_PRIOR,
    DEPTH_DISTRIBUTION_LEARNED_REWEIGHT,
)


def canonical_depth_distribution_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in DEPTH_DISTRIBUTION_MODES:
        raise ValueError(
            f"Unknown depth_distribution_mode={value!r}; expected one of "
            f"{', '.join(DEPTH_DISTRIBUTION_MODES)}."
        )
    return mode


@dataclass(frozen=True)
class DepthHypothesisConfig:
    mode: str
    num_bins: int
    range_sigma: float
    log_sigma_min: float
    log_sigma_max: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", canonical_depth_distribution_mode(self.mode))
        if self.mode == DEPTH_DISTRIBUTION_POINT:
            if self.num_bins != 1:
                raise ValueError("point depth mode requires num_bins=1.")
        else:
            if self.num_bins < 3 or self.num_bins % 2 == 0:
                raise ValueError("Distribution depth modes require an odd num_bins >= 3.")
            if self.range_sigma <= 0:
                raise ValueError("range_sigma must be positive.")
            if not 0 < self.log_sigma_min <= self.log_sigma_max:
                raise ValueError(
                    "log_sigma_min/max must satisfy 0 < min <= max for distribution modes."
                )


class PnpDepthHypothesisGenerator(nn.Module):
    """Create a shared binocular log-scale distribution around raw PnP eyes."""

    def __init__(self, config: DepthHypothesisConfig) -> None:
        super().__init__()
        self.config = config
        if config.mode == DEPTH_DISTRIBUTION_POINT:
            sigma_nodes = torch.zeros(1, dtype=torch.float32)
        else:
            sigma_nodes = torch.linspace(
                -config.range_sigma,
                config.range_sigma,
                config.num_bins,
                dtype=torch.float32,
            )
        prior_logits = -0.5 * sigma_nodes.square()
        prior_log_weights = torch.log_softmax(prior_logits, dim=0)
        self.register_buffer("sigma_nodes", sigma_nodes, persistent=True)
        self.register_buffer("base_prior_log_weights", prior_log_weights, persistent=True)

    def forward(
        self,
        raw_eye_geometry_mm: Tensor,
        depth_log_scale_sigma: Tensor | None,
    ) -> dict[str, Tensor]:
        eyes = torch.as_tensor(raw_eye_geometry_mm)
        if eyes.ndim == 1:
            eyes = eyes.unsqueeze(0)
        if eyes.ndim != 2 or eyes.shape[-1] != 6:
            raise ValueError(
                "raw_eye_geometry_mm must have shape [B, 6], "
                f"got {tuple(eyes.shape)}"
            )
        eyes = eyes.float()
        if not torch.isfinite(eyes).all():
            raise ValueError("raw_eye_geometry_mm must contain only finite values.")
        left = eyes[:, :3]
        right = eyes[:, 3:]
        if torch.any(left[:, 2] <= 0) or torch.any(right[:, 2] <= 0):
            raise ValueError("PnP eye depths must be positive.")

        batch_size = eyes.shape[0]
        if self.config.mode == DEPTH_DISTRIBUTION_POINT:
            sigma = torch.zeros((batch_size, 1), device=eyes.device, dtype=eyes.dtype)
        else:
            if depth_log_scale_sigma is None:
                raise ValueError(
                    f"depth_log_scale_sigma is required for mode={self.config.mode!r}."
                )
            sigma = torch.as_tensor(
                depth_log_scale_sigma,
                device=eyes.device,
                dtype=eyes.dtype,
            )
            if sigma.ndim == 1:
                sigma = sigma.unsqueeze(-1)
            if sigma.shape != (batch_size, 1):
                raise ValueError(
                    "depth_log_scale_sigma must have shape [B, 1], "
                    f"got {tuple(sigma.shape)}"
                )
            if not torch.isfinite(sigma).all() or torch.any(sigma < 0):
                raise ValueError(
                    "depth_log_scale_sigma must be finite and non-negative."
                )
            sigma = sigma.clamp(
                min=self.config.log_sigma_min,
                max=self.config.log_sigma_max,
            )

        nodes = self.sigma_nodes.to(device=eyes.device, dtype=eyes.dtype)
        log_scales = sigma * nodes.unsqueeze(0)
        scales = torch.exp(log_scales)
        left_hypotheses = left.unsqueeze(1) * scales.unsqueeze(-1)
        right_hypotheses = right.unsqueeze(1) * scales.unsqueeze(-1)
        centre_hypotheses = 0.5 * (left_hypotheses + right_hypotheses)
        prior_log_weights = self.base_prior_log_weights.to(
            device=eyes.device,
            dtype=eyes.dtype,
        ).unsqueeze(0).expand(batch_size, -1)
        return {
            "depth_log_scale_sigma": sigma,
            "depth_log_scales": log_scales,
            "depth_scales": scales,
            "depth_prior_log_weights": prior_log_weights,
            "depth_prior_weights": prior_log_weights.exp(),
            "left_eye_hypotheses_c_mm": left_hypotheses,
            "right_eye_hypotheses_c_mm": right_hypotheses,
            "eye_hypotheses_c_mm": centre_hypotheses,
        }


class DepthHypothesisReweighter(nn.Module):
    """Shared candidate scorer that updates, but does not replace, a PnP prior."""

    def __init__(
        self,
        *,
        context_dim: int,
        quality_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        detach_context: bool,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or quality_dim <= 0:
            raise ValueError("context_dim and quality_dim must be positive.")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if dropout < 0:
            raise ValueError("dropout must be non-negative.")
        self.context_dim = context_dim
        self.quality_dim = quality_dim
        self.detach_context = detach_context
        widths = (context_dim + quality_dim + 2, *hidden_dims, 1)
        layers: list[nn.Module] = []
        for index, (input_width, output_width) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(input_width, output_width))
            if index < len(widths) - 2:
                layers.extend((nn.SiLU(inplace=True), nn.Dropout(dropout)))
        self.scorer = nn.Sequential(*layers)

    def forward(
        self,
        context: Tensor,
        pnp_quality: Tensor,
        depth_log_scales: Tensor,
        prior_log_weights: Tensor,
    ) -> dict[str, Tensor]:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context must have shape [B, {self.context_dim}], got {tuple(context.shape)}"
            )
        if pnp_quality.ndim != 2 or pnp_quality.shape[-1] != self.quality_dim:
            raise ValueError(
                f"pnp_quality must have shape [B, {self.quality_dim}], "
                f"got {tuple(pnp_quality.shape)}"
            )
        if depth_log_scales.ndim != 2 or prior_log_weights.shape != depth_log_scales.shape:
            raise ValueError("Depth log-scales and prior log-weights must share shape [B, K].")
        if context.shape[0] != depth_log_scales.shape[0] or pnp_quality.shape[0] != context.shape[0]:
            raise ValueError("Reweighter inputs must share the batch dimension.")
        context = context.detach() if self.detach_context else context
        candidate_count = depth_log_scales.shape[1]
        expanded_context = context.unsqueeze(1).expand(-1, candidate_count, -1)
        expanded_quality = pnp_quality.unsqueeze(1).expand(-1, candidate_count, -1)
        candidate_features = torch.cat(
            (
                expanded_context,
                expanded_quality,
                depth_log_scales.unsqueeze(-1),
                prior_log_weights.unsqueeze(-1),
            ),
            dim=-1,
        )
        residual_logits = self.scorer(candidate_features).squeeze(-1)
        posterior_log_weights = torch.log_softmax(
            prior_log_weights + residual_logits,
            dim=-1,
        )
        return {
            "depth_weight_logit_residual": residual_logits,
            "depth_posterior_log_weights": posterior_log_weights,
            "depth_posterior_weights": posterior_log_weights.exp(),
        }


def mixture_uv_statistics(
    uv_hypotheses_mm: Tensor,
    weights: Tensor,
    valid_mask: Tensor,
) -> dict[str, Tensor]:
    """Return normalized valid weights, UV mean/covariance, and MAP hypothesis."""

    device_type = torch.as_tensor(uv_hypotheses_mm).device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return _mixture_uv_statistics_float32(
            uv_hypotheses_mm,
            weights,
            valid_mask,
        )


def _mixture_uv_statistics_float32(
    uv_hypotheses_mm: Tensor,
    weights: Tensor,
    valid_mask: Tensor,
) -> dict[str, Tensor]:
    """Float32 implementation for :func:`mixture_uv_statistics`."""

    if uv_hypotheses_mm.ndim != 3 or uv_hypotheses_mm.shape[-1] != 2:
        raise ValueError("uv_hypotheses_mm must have shape [B, K, 2].")
    if weights.shape != uv_hypotheses_mm.shape[:2] or valid_mask.shape != weights.shape:
        raise ValueError("weights and valid_mask must have shape [B, K].")
    uv_hypotheses_mm = uv_hypotheses_mm.float()
    weights = weights.to(device=uv_hypotheses_mm.device, dtype=torch.float32)
    masked_weights = weights * valid_mask.to(dtype=weights.dtype)
    weight_sum = masked_weights.sum(dim=-1, keepdim=True)
    any_valid = weight_sum.squeeze(-1) > 0
    normalized = masked_weights / weight_sum.clamp_min(torch.finfo(weights.dtype).eps)
    uv_mean = (normalized.unsqueeze(-1) * uv_hypotheses_mm).sum(dim=1)
    centered = uv_hypotheses_mm - uv_mean.unsqueeze(1)
    covariance = torch.einsum(
        "bk,bki,bkj->bij",
        normalized,
        centered,
        centered,
    )
    map_index = normalized.argmax(dim=-1)
    batch_index = torch.arange(uv_hypotheses_mm.shape[0], device=uv_hypotheses_mm.device)
    uv_map = uv_hypotheses_mm[batch_index, map_index]
    return {
        "depth_effective_weights": normalized,
        "uv_mean_mm": uv_mean,
        "uv_covariance_mm2": covariance,
        "uv_map_mm": uv_map,
        "depth_map_index": map_index,
        "geometry_any_valid": any_valid,
    }
