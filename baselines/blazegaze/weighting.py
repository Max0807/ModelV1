"""Optional released 30x30 inverse-frequency weighting for BlazeGaze."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from .data_adapter import BlazeGazeDataAdapter


@dataclass(frozen=True)
class OfficialInverseFrequencyGrid:
    """The released ``1 / (histogram + eps)`` 30x30 weighting scheme.

    The grid must be fitted on the training split only. Its normalization is
    intentionally identical to WebEyeTrack's ``output_space.py``.
    """

    weights: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray

    @classmethod
    def fit(
        cls,
        centered_pog: Tensor | np.ndarray,
        *,
        bins: int = 30,
        eps: float = 1e-6,
    ) -> "OfficialInverseFrequencyGrid":
        values = np.asarray(centered_pog, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("centered_pog must have non-empty shape [N, 2].")
        if bins <= 0 or eps <= 0:
            raise ValueError("bins and eps must be positive.")
        if not np.isfinite(values).all():
            raise ValueError("centered_pog contains non-finite values.")
        histogram, x_edges, y_edges = np.histogram2d(
            values[:, 0],
            values[:, 1],
            bins=bins,
            range=[[-0.5, 0.5], [-0.5, 0.5]],
        )
        weights = 1.0 / (histogram + eps)
        weights /= weights.sum()
        return cls(weights=weights, x_edges=x_edges, y_edges=y_edges)

    def lookup(self, centered_pog: Tensor | np.ndarray) -> np.ndarray:
        values = np.asarray(centered_pog, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("centered_pog must have shape [N, 2].")
        x_indices = np.digitize(values[:, 0], self.x_edges) - 1
        y_indices = np.digitize(values[:, 1], self.y_edges) - 1
        x_indices = np.clip(x_indices, 0, self.weights.shape[0] - 1)
        y_indices = np.clip(y_indices, 0, self.weights.shape[1] - 1)
        return self.weights[x_indices, y_indices].astype(np.float32)

    def assign(self, adapter: BlazeGazeDataAdapter) -> None:
        targets = torch.stack(
            [adapter.pog_norm_at(index) for index in range(len(adapter))]
        )
        weights = self.lookup(targets.numpy())
        adapter.set_sample_weights(
            {
                adapter.sample_id_at(index): float(weights[index])
                for index in range(len(adapter))
            }
        )


def fit_official_weight_grid(
    adapters: Iterable[BlazeGazeDataAdapter],
    *,
    bins: int = 30,
) -> OfficialInverseFrequencyGrid:
    selected = tuple(adapters)
    if not selected:
        raise ValueError("At least one training adapter is required.")
    targets = torch.cat(
        [
            torch.stack(
                [adapter.pog_norm_at(index) for index in range(len(adapter))]
            )
            for adapter in selected
        ],
        dim=0,
    )
    return OfficialInverseFrequencyGrid.fit(targets.numpy(), bins=bins)
