"""Inference helpers for converting ModelV1 UV outputs back to millimeters."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .data.normalization import UVTargetNormalizer
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
            return normalizer.denormalize(uv.float())
        if not torch.is_tensor(output):
            raise TypeError("Model inference must return a UV tensor or V4 mapping.")
        if normalizer is None:
            raise ValueError("Legacy direct_uv inference requires a UV normalizer.")
        return normalizer.denormalize(output)
    finally:
        model.train(was_training)
