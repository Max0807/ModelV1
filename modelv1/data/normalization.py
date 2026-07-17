"""Training-set normalization for table-local gaze targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


UV_DIM = 2
UV_TARGET_COLUMNS = ("uv_gt_u_mm", "uv_gt_v_mm")


@dataclass(frozen=True)
class UVTargetNormalizer:
    """Per-axis z-score transform fitted only on training targets in millimeters."""

    mean_mm: Tensor
    std_mm: Tensor

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean_mm, dtype=torch.float32).flatten().clone()
        std = torch.as_tensor(self.std_mm, dtype=torch.float32).flatten().clone()
        if mean.shape != (UV_DIM,) or std.shape != (UV_DIM,):
            raise ValueError(
                f"UV normalization tensors must have shape ({UV_DIM},), "
                f"got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("UV normalization statistics must be finite.")
        if torch.any(std <= 0):
            raise ValueError("UV normalization std_mm must be strictly positive.")
        object.__setattr__(self, "mean_mm", mean)
        object.__setattr__(self, "std_mm", std)

    @classmethod
    def fit(cls, uv_mm: Tensor, min_std_mm: float = 1e-6) -> "UVTargetNormalizer":
        """Fit from an ``[N, 2]`` tensor of training targets only."""

        uv_mm = torch.as_tensor(uv_mm, dtype=torch.float32)
        if uv_mm.ndim != 2 or uv_mm.shape[-1] != UV_DIM or uv_mm.shape[0] == 0:
            raise ValueError(f"uv_mm must have shape [N, {UV_DIM}] with N > 0")
        if not torch.isfinite(uv_mm).all():
            raise ValueError("Cannot fit UV normalizer from non-finite targets.")
        if min_std_mm <= 0:
            raise ValueError("min_std_mm must be positive.")

        mean = uv_mm.mean(dim=0)
        std = uv_mm.std(dim=0, unbiased=False).clamp_min(min_std_mm)
        return cls(mean_mm=mean, std_mm=std)

    def normalize(self, uv_mm: Tensor) -> Tensor:
        """Map millimeter coordinates to the training-target z-score space."""

        uv_mm = validate_uv_tensor(uv_mm, "uv_mm")
        mean, std = self._for(uv_mm)
        return (uv_mm - mean) / std

    def denormalize(self, uv_normalized: Tensor) -> Tensor:
        """Map normalized coordinates back to millimeters."""

        uv_normalized = validate_uv_tensor(uv_normalized, "uv_normalized")
        mean, std = self._for(uv_normalized)
        return uv_normalized * std + mean

    def state_dict(self) -> dict[str, Tensor]:
        """Checkpoint-ready tensors for restoring the exact target space."""

        return {"mean_mm": self.mean_mm.clone(), "std_mm": self.std_mm.clone()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Tensor]) -> "UVTargetNormalizer":
        try:
            return cls(mean_mm=state["mean_mm"], std_mm=state["std_mm"])
        except KeyError as exc:
            raise KeyError("UV normalizer state requires mean_mm and std_mm.") from exc

    def normalized_beta(self, beta_mm: float, like: Tensor) -> Tensor:
        """Convert one physical Huber transition to per-axis target-space units."""

        if beta_mm <= 0:
            raise ValueError("beta_mm must be positive.")
        _, std = self._for(like)
        return beta_mm / std

    def _for(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self.mean_mm.to(device=tensor.device, dtype=tensor.dtype),
            self.std_mm.to(device=tensor.device, dtype=tensor.dtype),
        )


def fit_uv_target_normalizer(rows: list[dict[str, str]]) -> UVTargetNormalizer:
    """Fit from CSV rows without loading images or DECA features."""

    targets = []
    for row in rows:
        try:
            targets.append([float(row[column]) for column in UV_TARGET_COLUMNS])
        except KeyError as exc:
            raise KeyError(f"Dataset row is missing target column: {exc.args[0]}") from exc
        except ValueError as exc:
            raise ValueError(f"Dataset row has invalid UV target: {row.get('sample_id')!r}") from exc
    return UVTargetNormalizer.fit(torch.tensor(targets, dtype=torch.float32))


def validate_uv_tensor(uv: Tensor, name: str) -> Tensor:
    if not torch.is_tensor(uv):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if uv.ndim < 1 or uv.shape[-1] != UV_DIM:
        raise ValueError(f"{name} must end with dimension {UV_DIM}, got {tuple(uv.shape)}")
    return uv
