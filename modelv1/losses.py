"""Loss and millimeter-space metrics for normalized ModelV1 UV predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .data.normalization import UVTargetNormalizer, validate_uv_tensor


@dataclass(frozen=True)
class UVLossConfig:
    """Configuration for the V1 primary UV regression objective."""

    beta_mm: float = 30.0

    def __post_init__(self) -> None:
        if self.beta_mm <= 0:
            raise ValueError("beta_mm must be positive.")


class UVRegressionLoss(nn.Module):
    """Per-axis Smooth L1 loss for normalized model outputs.

    The UV head and target are compared in the normalized space. The Huber
    transition is converted from ``beta_mm`` independently for u and v, so it
    still begins at the same physical coordinate error in each axis.
    """

    def __init__(
        self,
        normalizer: UVTargetNormalizer,
        config: UVLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.config = config or UVLossConfig()

    def forward(self, uv_pred: Tensor, uv_target: Tensor) -> Tensor:
        uv_pred = validate_uv_tensor(uv_pred, "uv_pred")
        uv_target = validate_uv_tensor(uv_target, "uv_target").to(
            device=uv_pred.device, dtype=uv_pred.dtype
        )
        error = (uv_pred - uv_target).abs()
        beta = self.normalizer.normalized_beta(self.config.beta_mm, uv_pred)
        return torch.where(
            error < beta,
            0.5 * error.square() / beta,
            error - 0.5 * beta,
        ).mean()

    @torch.no_grad()
    def metrics(self, uv_pred: Tensor, uv_gt_mm: Tensor) -> dict[str, Tensor]:
        return compute_uv_metrics(uv_pred, uv_gt_mm, self.normalizer)


@torch.no_grad()
def compute_uv_metrics(
    uv_pred: Tensor,
    uv_gt_mm: Tensor,
    normalizer: UVTargetNormalizer,
) -> dict[str, Tensor]:
    """Return physically interpretable validation metrics in millimeters."""

    uv_pred = validate_uv_tensor(uv_pred, "uv_pred").float()
    uv_gt_mm = validate_uv_tensor(uv_gt_mm, "uv_gt_mm").to(
        device=uv_pred.device, dtype=torch.float32
    )
    uv_pred_mm = normalizer.denormalize(uv_pred)
    error_mm = uv_pred_mm - uv_gt_mm
    abs_error_mm = error_mm.abs()
    epe_mm = torch.linalg.vector_norm(error_mm, dim=-1)
    return {
        "epe_mm": epe_mm.mean(),
        "median_epe_mm": epe_mm.median(),
        "mae_u_mm": abs_error_mm[..., 0].mean(),
        "mae_v_mm": abs_error_mm[..., 1].mean(),
    }


def batch_uv_targets(batch: Mapping[str, object]) -> tuple[Tensor, Tensor]:
    """Extract normalized and millimeter UV targets from one collated batch."""

    try:
        uv_target = batch["uv_target"]
        uv_gt = batch["uv_gt"]
    except KeyError as exc:
        raise KeyError("Batch must contain uv_target and uv_gt.") from exc
    if not torch.is_tensor(uv_target) or not torch.is_tensor(uv_gt):
        raise TypeError("uv_target and uv_gt must be torch tensors.")
    return validate_uv_tensor(uv_target, "uv_target"), validate_uv_tensor(uv_gt, "uv_gt")
