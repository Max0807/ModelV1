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
    normalizer: UVTargetNormalizer,
) -> Tensor:
    """Run one batch and return table-local UV predictions in millimeters."""

    was_training = model.training
    model.eval()
    try:
        uv_normalized = model(batch)
        if not torch.is_tensor(uv_normalized):
            raise TypeError("Model inference must return a UV tensor.")
        return normalizer.denormalize(uv_normalized)
    finally:
        model.train(was_training)
