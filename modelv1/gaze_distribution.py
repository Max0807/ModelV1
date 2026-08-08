"""Camera-gaze distributions represented on the unit sphere.

The learned gaze mean remains a camera-coordinate unit vector. Uncertainty
lives in its two-dimensional tangent plane, avoiding an invalid unconstrained
3D covariance for a direction on the unit sphere.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def tangent_basis(direction_c: Tensor) -> Tensor:
    """Return a stable orthonormal tangent basis with shape ``[B, 3, 2]``."""

    if direction_c.ndim != 2 or direction_c.shape[-1] != 3:
        raise ValueError("direction_c must have shape [B, 3].")
    mean = F.normalize(direction_c, dim=-1, eps=1e-8)
    seed = F.one_hot(mean.abs().argmin(dim=-1), num_classes=3).to(mean.dtype)
    first = F.normalize(
        seed - (seed * mean).sum(dim=-1, keepdim=True) * mean,
        dim=-1,
        eps=1e-8,
    )
    second = F.normalize(torch.cross(mean, first, dim=-1), dim=-1, eps=1e-8)
    return torch.stack((first, second), dim=-1)


def tangent_cholesky_from_raw(
    raw: Tensor,
    *,
    min_std_rad: float,
    max_std_rad: float,
) -> Tensor:
    """Map three unconstrained values to a bounded 2D Cholesky factor."""

    if raw.ndim != 2 or raw.shape[-1] != 3:
        raise ValueError("raw gaze uncertainty must have shape [B, 3].")
    if not 0 < min_std_rad <= max_std_rad:
        raise ValueError("Expected 0 < min_std_rad <= max_std_rad.")
    # Keep the distribution parameterization in float32 even when the visual
    # backbone is evaluated under AMP.
    raw = raw.float()
    std_first = min_std_rad + (max_std_rad - min_std_rad) * torch.sigmoid(raw[:, 0])
    std_second = min_std_rad + (max_std_rad - min_std_rad) * torch.sigmoid(raw[:, 2])
    correlation = 0.999 * torch.tanh(raw[:, 1])
    diagonal_second = std_second * torch.sqrt(
        (1.0 - correlation.square()).clamp_min(1e-8)
    )
    zeros = torch.zeros_like(std_first)
    return torch.stack(
        (
            torch.stack((std_first, zeros), dim=-1),
            torch.stack((correlation * std_second, diagonal_second), dim=-1),
        ),
        dim=-2,
    )


def unit_sphere_exp_map(mean: Tensor, tangent_offset: Tensor) -> Tensor:
    """Map tangent vectors ``[B, S, 3]`` to unit directions around ``mean``."""

    if mean.ndim != 2 or mean.shape[-1] != 3:
        raise ValueError("mean must have shape [B, 3].")
    if (
        tangent_offset.ndim != 3
        or tangent_offset.shape[0] != mean.shape[0]
        or tangent_offset.shape[-1] != 3
    ):
        raise ValueError("tangent_offset must have shape [B, S, 3].")
    theta = torch.linalg.vector_norm(tangent_offset, dim=-1, keepdim=True)
    samples = (
        torch.cos(theta) * mean.unsqueeze(1)
        + torch.sinc(theta / torch.pi) * tangent_offset
    )
    return F.normalize(samples, dim=-1, eps=1e-8)


def tangent_sigma_points(
    mean_direction_c: Tensor,
    cholesky: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Return deterministic camera-gaze samples and positive moment weights.

    With uncertainty this returns a centre point (weight 0.5) and four
    symmetric points (each weight 0.125). The ``±2 L[:, i]`` offsets reproduce
    the given tangent covariance before nonlinear geometry is applied.
    """

    mean = F.normalize(mean_direction_c.float(), dim=-1, eps=1e-8)
    if cholesky is not None:
        cholesky = cholesky.float()
    if cholesky is None:
        return mean.unsqueeze(1), mean.new_ones((mean.shape[0], 1))
    if cholesky.shape != (mean.shape[0], 2, 2):
        raise ValueError("cholesky must have shape [B, 2, 2].")
    basis = tangent_basis(mean)
    offset_2d = torch.stack(
        (
            torch.zeros_like(cholesky[:, 0]),
            2.0 * cholesky[:, :, 0],
            -2.0 * cholesky[:, :, 0],
            2.0 * cholesky[:, :, 1],
            -2.0 * cholesky[:, :, 1],
        ),
        dim=1,
    )
    offset_3d = torch.einsum("bij,bsj->bsi", basis, offset_2d)
    samples = unit_sphere_exp_map(mean, offset_3d)
    weights = mean.new_tensor((0.5, 0.125, 0.125, 0.125, 0.125)).expand(
        mean.shape[0], -1
    )
    return samples, weights
