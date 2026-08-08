"""Differentiable, parameter-free gaze ray/table geometry.

The learned model predicts only a camera-space gaze direction.  This module
uses physical eye positions and the per-frame TableFrame representation to
compute table-local UV coordinates.  It deliberately owns no trainable
parameters; autograd is still enabled so UV losses can update the gaze model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RayTableGeometryConfig:
    """Numerical and physical validity limits for ray/plane intersection."""

    table_distance_scale_mm: float
    min_abs_normal_dot_gaze: float
    min_lambda_mm: float
    max_lambda_mm: float

    def __post_init__(self) -> None:
        if self.table_distance_scale_mm <= 0:
            raise ValueError("table_distance_scale_mm must be positive.")
        if not 0 < self.min_abs_normal_dot_gaze < 1:
            raise ValueError("min_abs_normal_dot_gaze must lie in (0, 1).")
        if self.min_lambda_mm < 0:
            raise ValueError("min_lambda_mm must be non-negative.")
        if self.max_lambda_mm <= self.min_lambda_mm:
            raise ValueError("max_lambda_mm must exceed min_lambda_mm.")


def _vector_batch(value: Tensor, name: str, width: int) -> Tensor:
    value = torch.as_tensor(value)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[-1] != width:
        raise ValueError(f"{name} must have shape [B, {width}], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value.float()


def unpack_table_frame7(
    table_frame7: Tensor,
    *,
    distance_scale_mm: float,
) -> dict[str, Tensor]:
    """Return an orthonormal camera-space table frame from ``[e1,e2,d]``.

    ``table_frame7[..., 6]`` is stored in scaled units (metres in the current
    dataset).  The returned signed plane distance and projection origin are in
    millimetres.  The UV origin is the orthogonal projection of the camera
    centre onto the table plane, ``origin_c = d_c * n_c``.
    """

    frame = _vector_batch(table_frame7, "table_frame7", 7)
    e1_raw = frame[:, :3]
    e2_raw = frame[:, 3:6]
    e1_c = F.normalize(e1_raw, dim=-1, eps=1e-8)
    # Preserve a right-handed frame while removing small numeric drift.
    e2_orthogonal = e2_raw - (e2_raw * e1_c).sum(dim=-1, keepdim=True) * e1_c
    e2_c = F.normalize(e2_orthogonal, dim=-1, eps=1e-8)
    n_c = F.normalize(torch.cross(e1_c, e2_c, dim=-1), dim=-1, eps=1e-8)
    d_c_mm = frame[:, 6:7] * float(distance_scale_mm)
    origin_c_mm = d_c_mm * n_c
    return {
        "e1_c": e1_c,
        "e2_c": e2_c,
        "n_c": n_c,
        "d_c_mm": d_c_mm,
        "origin_c_mm": origin_c_mm,
    }


class RayTableGeometry(nn.Module):
    """Intersect gaze rays with a table and return camera-projection-local UV."""

    def __init__(self, config: RayTableGeometryConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        gaze_direction_c: Tensor,
        eye_hypotheses_c_mm: Tensor,
        table_frame7: Tensor,
    ) -> dict[str, Tensor]:
        """Run ray/plane calculations in float32, outside AMP autocast.

        Intersections may be measured in tens of thousands of millimetres for
        near-parallel rays. Their squares are later used for UV covariance and
        overflow float16 even when all inputs are otherwise finite.
        """

        device_type = torch.as_tensor(gaze_direction_c).device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return self._forward_float32(
                gaze_direction_c,
                eye_hypotheses_c_mm,
                table_frame7,
            )

    def _forward_float32(
        self,
        gaze_direction_c: Tensor,
        eye_hypotheses_c_mm: Tensor,
        table_frame7: Tensor,
    ) -> dict[str, Tensor]:
        gaze = _vector_batch(gaze_direction_c, "gaze_direction_c", 3)
        gaze = F.normalize(gaze, dim=-1, eps=1e-8)
        eyes = torch.as_tensor(
            eye_hypotheses_c_mm,
            device=gaze.device,
            dtype=gaze.dtype,
        )
        if eyes.ndim == 2 and eyes.shape[-1] == 3:
            eyes = eyes.unsqueeze(1)
        if eyes.ndim != 3 or eyes.shape[-1] != 3:
            raise ValueError(
                "eye_hypotheses_c_mm must have shape [B, K, 3], "
                f"got {tuple(eyes.shape)}"
            )
        if eyes.shape[0] != gaze.shape[0]:
            raise ValueError("gaze and eye hypotheses must share the batch dimension.")
        if not torch.isfinite(eyes).all():
            raise ValueError("eye_hypotheses_c_mm must contain only finite values.")

        table = unpack_table_frame7(
            table_frame7.to(device=gaze.device, dtype=gaze.dtype),
            distance_scale_mm=self.config.table_distance_scale_mm,
        )
        if table["n_c"].shape[0] != gaze.shape[0]:
            raise ValueError("table_frame7 and gaze must share the batch dimension.")

        n_c = table["n_c"]
        raw_denominator = (n_c * gaze).sum(dim=-1, keepdim=True)
        # A line/plane UV loss cannot distinguish g from -g.  Because every
        # target in this project lies on the known table, orient the predicted
        # gaze axis toward the plane using the centre (baseline-scale) PnP eye
        # hypothesis. This is parameter-free and happens only inside geometry;
        # TableFrame7 never enters the learned gaze network.
        reference_eye = eyes[:, eyes.shape[1] // 2]
        reference_plane_numerator = table["d_c_mm"] - (
            n_c * reference_eye
        ).sum(dim=-1, keepdim=True)
        gaze_was_flipped = (
            reference_plane_numerator * raw_denominator < 0
        )
        gaze = torch.where(gaze_was_flipped, -gaze, gaze)
        denominator = (n_c * gaze).sum(dim=-1, keepdim=True)
        denominator_sign = torch.where(
            denominator < 0,
            -torch.ones_like(denominator),
            torch.ones_like(denominator),
        )
        safe_denominator = torch.where(
            denominator.abs() >= self.config.min_abs_normal_dot_gaze,
            denominator,
            denominator_sign * self.config.min_abs_normal_dot_gaze,
        )

        plane_offset = table["d_c_mm"].unsqueeze(1)
        eye_normal_coordinate = torch.einsum("bi,bki->bk", n_c, eyes).unsqueeze(-1)
        lambdas = (plane_offset - eye_normal_coordinate) / safe_denominator.unsqueeze(1)
        intersections = eyes + lambdas * gaze.unsqueeze(1)
        displacement = intersections - table["origin_c_mm"].unsqueeze(1)
        u = torch.einsum("bi,bki->bk", table["e1_c"], displacement)
        v = torch.einsum("bi,bki->bk", table["e2_c"], displacement)
        uv = torch.stack((u, v), dim=-1)

        finite = (
            torch.isfinite(lambdas.squeeze(-1))
            & torch.isfinite(intersections).all(dim=-1)
            & torch.isfinite(uv).all(dim=-1)
        )
        valid = (
            finite
            & (denominator.abs() >= self.config.min_abs_normal_dot_gaze)
            & (lambdas.squeeze(-1) > self.config.min_lambda_mm)
            & (lambdas.squeeze(-1) <= self.config.max_lambda_mm)
        )
        return {
            "gaze_direction_c": gaze,
            "gaze_was_flipped": gaze_was_flipped,
            "uv_hypotheses_mm": uv,
            "intersection_hypotheses_c_mm": intersections,
            "lambda_hypotheses_mm": lambdas.squeeze(-1),
            "ray_valid_mask": valid,
            "ray_finite_mask": finite,
            "normal_dot_gaze": denominator,
            **table,
        }
