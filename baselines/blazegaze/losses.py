"""Losses and metric conversion for supervised Native BlazeGaze."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data_adapter import PogBoundsMM


@dataclass(frozen=True)
class NativeBlazeGazeLossConfig:
    gaze_weight: float = 1.0
    reconstruction_weight: float = 1.0
    consistency_weight: float = 0.2

    def __post_init__(self) -> None:
        values = (
            self.gaze_weight,
            self.reconstruction_weight,
            self.consistency_weight,
        )
        if any(value < 0 for value in values):
            raise ValueError("Native BlazeGaze loss weights must be non-negative.")
        if not any(value > 0 for value in values):
            raise ValueError("At least one Native BlazeGaze loss must be enabled.")


def _validate_gaze_tensors(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            "Gaze prediction and target must share shape [B, 2]; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    if prediction.shape[-1] != 2:
        raise ValueError("Gaze prediction and target must have two coordinates.")


def weighted_gaze_l2_loss(
    prediction: Tensor,
    target: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Released weighted squared L2 gaze loss.

    This intentionally returns ``mean(weight * squared_L2)`` rather than
    dividing by the sum of weights, matching the released TensorFlow helper.
    """

    _validate_gaze_tensors(prediction, target)
    per_sample = (prediction - target.to(prediction)).square().sum(dim=-1)
    if sample_weight is not None:
        if sample_weight.shape != (prediction.shape[0],):
            raise ValueError("sample_weight must have shape [B].")
        weight = sample_weight.to(per_sample)
        if not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("sample_weight must be finite and non-negative.")
        per_sample = per_sample * weight
    return per_sample.mean()


def embedding_consistency_loss(
    embedding: Tensor,
    pog_target: Tensor,
    sample_weight: Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Released pairwise embedding-to-gaze distance consistency loss."""

    if embedding.ndim != 2:
        raise ValueError("embedding must have shape [B, D].")
    if pog_target.shape != (embedding.shape[0], 2):
        raise ValueError("pog_target must have shape [B, 2].")
    embedding_float = embedding.float()
    target_float = pog_target.to(device=embedding.device, dtype=torch.float32)
    embedding_distances = torch.cdist(embedding_float, embedding_float, p=2)
    gaze_distances = torch.cdist(target_float, target_float, p=2)
    normalized_gaze_distances = gaze_distances / (
        gaze_distances.max() + float(eps)
    )
    loss_matrix = (embedding_distances - normalized_gaze_distances).square()
    if sample_weight is not None:
        if sample_weight.shape != (embedding.shape[0],):
            raise ValueError("sample_weight must have shape [B].")
        weight = sample_weight.to(loss_matrix)
        if not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("sample_weight must be finite and non-negative.")
        loss_matrix = loss_matrix * weight[:, None] * weight[None, :]
    return loss_matrix.mean()


class NativeBlazeGazeStage1Loss(nn.Module):
    """Joint Stage-1 objective: gaze + reconstruction + consistency."""

    def __init__(self, config: NativeBlazeGazeLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or NativeBlazeGazeLossConfig()

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        *,
        image_target: Tensor,
        pog_target: Tensor,
        sample_weight: Tensor | None = None,
    ) -> dict[str, Tensor]:
        prediction = outputs.get("pog_pred")
        embedding = outputs.get("embedding")
        if not torch.is_tensor(prediction) or not torch.is_tensor(embedding):
            raise KeyError("Native model outputs require pog_pred and embedding.")
        gaze = weighted_gaze_l2_loss(prediction, pog_target, sample_weight)
        consistency = embedding_consistency_loss(
            embedding,
            pog_target,
            sample_weight,
        )
        reconstruction = outputs.get("reconstruction")
        if self.config.reconstruction_weight > 0:
            if not torch.is_tensor(reconstruction):
                raise KeyError(
                    "A positive reconstruction weight requires reconstruction output."
                )
            if reconstruction.shape != image_target.shape:
                raise ValueError(
                    "Reconstruction and input image shapes differ: "
                    f"{tuple(reconstruction.shape)} vs {tuple(image_target.shape)}."
                )
            reconstruction_loss = F.mse_loss(
                reconstruction.float(), image_target.float()
            )
        else:
            reconstruction_loss = embedding.sum() * 0.0
        total = (
            self.config.gaze_weight * gaze
            + self.config.reconstruction_weight * reconstruction_loss
            + self.config.consistency_weight * consistency
        )
        return {
            "loss": total,
            "gaze_l2": gaze,
            "reconstruction_mse": reconstruction_loss,
            "embedding_consistency": consistency,
        }


def denormalize_pog_mm(pog_centered: Tensor, bounds: PogBoundsMM) -> Tensor:
    """Convert centered ``[-0.5, 0.5]`` BlazeGaze coordinates to millimetres."""

    if pog_centered.ndim != 2 or pog_centered.shape[-1] != 2:
        raise ValueError("pog_centered must have shape [B, 2].")
    lower = pog_centered.new_tensor((bounds.u_min, bounds.v_min))
    extent = pog_centered.new_tensor(
        (bounds.u_max - bounds.u_min, bounds.v_max - bounds.v_min)
    )
    return (pog_centered + 0.5) * extent + lower


@torch.no_grad()
def mean_pog_error_mm(
    prediction: Tensor,
    target_mm: Tensor,
    bounds: PogBoundsMM,
) -> Tensor:
    prediction_mm = denormalize_pog_mm(prediction.float(), bounds)
    return torch.linalg.vector_norm(
        prediction_mm - target_mm.to(prediction_mm), dim=-1
    ).mean()
