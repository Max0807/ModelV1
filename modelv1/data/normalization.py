"""Training-set normalization for table-local gaze targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


UV_DIM = 2
UV_TARGET_COLUMNS = ("uv_gt_u_mm", "uv_gt_v_mm")
EYE_GEOMETRY_DIM = 6
EYE_GEOMETRY_QUALITY_DIM = 4
VIRTUAL_DISTANCE_SCALE_DIM = 1


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


@dataclass(frozen=True)
class VirtualDistanceScaleNormalizer:
    """Training-fitted z-score normalization for ``log(distance_scale)``."""

    mean_log_scale: Tensor
    std_log_scale: Tensor

    def __post_init__(self) -> None:
        mean = torch.as_tensor(
            self.mean_log_scale,
            dtype=torch.float32,
        ).flatten().clone()
        std = torch.as_tensor(
            self.std_log_scale,
            dtype=torch.float32,
        ).flatten().clone()
        expected = (VIRTUAL_DISTANCE_SCALE_DIM,)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                "Virtual-distance normalization tensors must have shape "
                f"{expected}, got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError(
                "Virtual-distance normalization statistics must be finite."
            )
        if torch.any(std <= 0):
            raise ValueError(
                "Virtual-distance log-scale std must be strictly positive."
            )
        object.__setattr__(self, "mean_log_scale", mean)
        object.__setattr__(self, "std_log_scale", std)

    @classmethod
    def fit(
        cls,
        distance_scale: Tensor,
        min_std: float = 1e-6,
    ) -> "VirtualDistanceScaleNormalizer":
        scale = validate_virtual_distance_scale_tensor(
            torch.as_tensor(distance_scale, dtype=torch.float32),
            "distance_scale",
        )
        if scale.ndim != 2 or scale.shape[0] == 0:
            raise ValueError("distance_scale must have shape [N, 1] with N > 0.")
        if min_std <= 0:
            raise ValueError("min_std must be positive.")
        log_scale = torch.log(scale)
        return cls(
            mean_log_scale=log_scale.mean(dim=0),
            std_log_scale=log_scale.std(dim=0, unbiased=False).clamp_min(min_std),
        )

    def normalize(self, distance_scale: Tensor) -> Tensor:
        scale = validate_virtual_distance_scale_tensor(
            distance_scale,
            "distance_scale",
        )
        mean = self.mean_log_scale.to(device=scale.device, dtype=scale.dtype)
        std = self.std_log_scale.to(device=scale.device, dtype=scale.dtype)
        return (torch.log(scale) - mean) / std

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "mean_log_scale": self.mean_log_scale.clone(),
            "std_log_scale": self.std_log_scale.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Tensor],
    ) -> "VirtualDistanceScaleNormalizer":
        try:
            return cls(
                mean_log_scale=state["mean_log_scale"],
                std_log_scale=state["std_log_scale"],
            )
        except KeyError as exc:
            raise KeyError(
                "Virtual-distance normalizer state requires mean_log_scale "
                "and std_log_scale."
            ) from exc


@dataclass(frozen=True)
class EyeGeometryNormalizer:
    """Per-dimension z-score transform fitted on training eye geometry only."""

    mean: Tensor
    std: Tensor

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=torch.float32).flatten().clone()
        std = torch.as_tensor(self.std, dtype=torch.float32).flatten().clone()
        expected = (EYE_GEOMETRY_DIM,)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                "Eye geometry normalization tensors must have shape "
                f"{expected}, got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Eye geometry normalization statistics must be finite.")
        if torch.any(std <= 0):
            raise ValueError("Eye geometry normalization std must be strictly positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def fit(
        cls,
        geometry: Tensor,
        min_std: float = 1e-6,
    ) -> "EyeGeometryNormalizer":
        """Fit from an ``[N, 6]`` tensor of training samples only."""

        geometry = torch.as_tensor(geometry, dtype=torch.float32)
        if (
            geometry.ndim != 2
            or geometry.shape[-1] != EYE_GEOMETRY_DIM
            or geometry.shape[0] == 0
        ):
            raise ValueError(
                f"geometry must have shape [N, {EYE_GEOMETRY_DIM}] with N > 0"
            )
        if not torch.isfinite(geometry).all():
            raise ValueError("Cannot fit eye geometry normalizer from non-finite values.")
        if min_std <= 0:
            raise ValueError("min_std must be positive.")
        return cls(
            mean=geometry.mean(dim=0),
            std=geometry.std(dim=0, unbiased=False).clamp_min(min_std),
        )

    def normalize(self, geometry: Tensor) -> Tensor:
        geometry = validate_eye_geometry_tensor(geometry, "geometry")
        mean = self.mean.to(device=geometry.device, dtype=geometry.dtype)
        std = self.std.to(device=geometry.device, dtype=geometry.dtype)
        return (geometry - mean) / std

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean.clone(), "std": self.std.clone()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Tensor]) -> "EyeGeometryNormalizer":
        try:
            return cls(mean=state["mean"], std=state["std"])
        except KeyError as exc:
            raise KeyError("Eye geometry normalizer state requires mean and std.") from exc


@dataclass(frozen=True)
class EyeGeometryQualityNormalizer:
    """Per-dimension z-score transform fitted on training PnP quality only."""

    mean: Tensor
    std: Tensor

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=torch.float32).flatten().clone()
        std = torch.as_tensor(self.std, dtype=torch.float32).flatten().clone()
        expected = (EYE_GEOMETRY_QUALITY_DIM,)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                "Eye geometry quality normalization tensors must have shape "
                f"{expected}, got mean={tuple(mean.shape)}, std={tuple(std.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError(
                "Eye geometry quality normalization statistics must be finite."
            )
        if torch.any(std <= 0):
            raise ValueError(
                "Eye geometry quality normalization std must be strictly positive."
            )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def fit(
        cls,
        quality: Tensor,
        min_std: float = 1e-6,
    ) -> "EyeGeometryQualityNormalizer":
        """Fit from an ``[N, 4]`` tensor of valid training samples only."""

        quality = torch.as_tensor(quality, dtype=torch.float32)
        if (
            quality.ndim != 2
            or quality.shape[-1] != EYE_GEOMETRY_QUALITY_DIM
            or quality.shape[0] == 0
        ):
            raise ValueError(
                "quality must have shape "
                f"[N, {EYE_GEOMETRY_QUALITY_DIM}] with N > 0"
            )
        if not torch.isfinite(quality).all():
            raise ValueError(
                "Cannot fit eye geometry quality normalizer from non-finite values."
            )
        if min_std <= 0:
            raise ValueError("min_std must be positive.")
        return cls(
            mean=quality.mean(dim=0),
            std=quality.std(dim=0, unbiased=False).clamp_min(min_std),
        )

    def normalize(self, quality: Tensor) -> Tensor:
        quality = validate_eye_geometry_quality_tensor(quality, "quality")
        mean = self.mean.to(device=quality.device, dtype=quality.dtype)
        std = self.std.to(device=quality.device, dtype=quality.dtype)
        return (quality - mean) / std

    def state_dict(self) -> dict[str, Tensor]:
        return {"mean": self.mean.clone(), "std": self.std.clone()}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Tensor],
    ) -> "EyeGeometryQualityNormalizer":
        try:
            return cls(mean=state["mean"], std=state["std"])
        except KeyError as exc:
            raise KeyError(
                "Eye geometry quality normalizer state requires mean and std."
            ) from exc


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


def validate_virtual_distance_scale_tensor(scale: Tensor, name: str) -> Tensor:
    if not torch.is_tensor(scale):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if scale.ndim < 1 or scale.shape[-1] != VIRTUAL_DISTANCE_SCALE_DIM:
        raise ValueError(
            f"{name} must end with dimension {VIRTUAL_DISTANCE_SCALE_DIM}, "
            f"got {tuple(scale.shape)}"
        )
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError(f"{name} must contain only finite positive values.")
    return scale


def validate_eye_geometry_tensor(geometry: Tensor, name: str) -> Tensor:
    if not torch.is_tensor(geometry):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if geometry.ndim < 1 or geometry.shape[-1] != EYE_GEOMETRY_DIM:
        raise ValueError(
            f"{name} must end with dimension {EYE_GEOMETRY_DIM}, "
            f"got {tuple(geometry.shape)}"
        )
    return geometry


def validate_eye_geometry_quality_tensor(quality: Tensor, name: str) -> Tensor:
    if not torch.is_tensor(quality):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if quality.ndim < 1 or quality.shape[-1] != EYE_GEOMETRY_QUALITY_DIM:
        raise ValueError(
            f"{name} must end with dimension {EYE_GEOMETRY_QUALITY_DIM}, "
            f"got {tuple(quality.shape)}"
        )
    return quality
