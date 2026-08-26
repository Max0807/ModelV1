"""Inference helpers for converting ModelV1 UV outputs back to millimeters."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .data.normalization import UVTargetNormalizer
from .geometry import virtual_camera_xy_to_table_uv


def compose_table_uv_with_vertical_residual(
    uv_base_table_mm: Tensor,
    output: Mapping[str, object],
) -> Tensor:
    """Apply an optional physical v residual while preserving table u."""

    if uv_base_table_mm.ndim != 2 or uv_base_table_mm.shape[1] != 2:
        raise ValueError("uv_base_table_mm must have shape [B, 2].")
    delta_v_mm = output.get("vertical_delta_v_mm")
    if delta_v_mm is None:
        return uv_base_table_mm
    if not torch.is_tensor(delta_v_mm) or delta_v_mm.shape != (
        uv_base_table_mm.shape[0],
        1,
    ):
        raise ValueError("vertical_delta_v_mm must have shape [B, 1].")
    return torch.stack(
        (
            uv_base_table_mm[:, 0],
            uv_base_table_mm[:, 1]
            + delta_v_mm.to(uv_base_table_mm)[:, 0],
        ),
        dim=-1,
    )


@torch.inference_mode()
def predict_uv_mm(
    model: nn.Module,
    batch: Mapping[str, object],
    normalizer: UVTargetNormalizer | None = None,
) -> Tensor:
    """Run one batch and return table-local UV predictions in millimeters."""

    was_training = model.training
    model.eval()
    try:
        output = model(batch)
        if isinstance(output, Mapping):
            uv_mean_mm = output.get("uv_mean_mm")
            if torch.is_tensor(uv_mean_mm):
                return uv_mean_mm
            uv = output.get("uv")
            if not torch.is_tensor(uv):
                raise TypeError("Model output mapping must contain tensor uv.")
            if normalizer is None:
                raise ValueError("Direct-UV inference requires a UV normalizer.")
            base_mm = normalizer.denormalize(uv.float())
            model_config = getattr(model, "config", None)
            target_frame = getattr(
                model_config, "pitch_to_v_target_frame", "table_local"
            )
            if target_frame == "virtual_camera":
                table_frame7_n = batch.get("table_frame7_n")
                if not torch.is_tensor(table_frame7_n):
                    raise ValueError(
                        "Virtual-camera pitch-to-v inference requires table_frame7_n."
                    )
                base_table_mm = virtual_camera_xy_to_table_uv(
                    base_mm,
                    table_frame7_n.to(device=base_mm.device),
                    distance_scale_mm=float(
                        getattr(model_config, "table_distance_scale_mm", 1000.0)
                    ),
                )
            else:
                base_table_mm = base_mm
            return compose_table_uv_with_vertical_residual(
                base_table_mm,
                output,
            )
        if not torch.is_tensor(output):
            raise TypeError("Model inference must return a UV tensor or V4 mapping.")
        if normalizer is None:
            raise ValueError("Legacy direct_uv inference requires a UV normalizer.")
        return normalizer.denormalize(output)
    finally:
        model.train(was_training)
