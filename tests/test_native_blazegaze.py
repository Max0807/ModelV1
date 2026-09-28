from __future__ import annotations

import numpy as np
import pytest
import torch

from baselines.blazegaze.data_adapter import PogBoundsMM
from baselines.blazegaze.losses import (
    NativeBlazeGazeLossConfig,
    NativeBlazeGazeStage1Loss,
    denormalize_pog_mm,
    embedding_consistency_loss,
    weighted_gaze_l2_loss,
)
from baselines.blazegaze.model import NativeBlazeGaze
from baselines.blazegaze.weighting import OfficialInverseFrequencyGrid
from scripts.train_native_blazegaze import _wandb_epoch_metrics


def test_native_model_matches_released_shapes_and_parameter_count() -> None:
    model = NativeBlazeGaze().eval()
    image = torch.rand(1, 3, 128, 512)
    head_vector = torch.rand(1, 3)
    face_origin = torch.rand(1, 3)

    with torch.no_grad():
        outputs = model(image, head_vector, face_origin)

    assert outputs["embedding"].shape == (1, 512)
    assert outputs["pog_pred"].shape == (1, 2)
    assert outputs["reconstruction"].shape == image.shape
    assert model.inference_parameter_count() == 155_826
    assert sum(parameter.numel() for parameter in model.parameters()) == 564_573


def test_weighted_gaze_loss_matches_released_mean_definition() -> None:
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros_like(prediction)
    weights = torch.tensor([0.5, 2.0])

    loss = weighted_gaze_l2_loss(prediction, target, weights)

    assert loss.item() == pytest.approx((5.0 * 0.5 + 25.0 * 2.0) / 2.0)


def test_consistency_loss_applies_pairwise_sample_weights() -> None:
    embedding = torch.tensor([[0.0], [2.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    weights = torch.tensor([1.0, 3.0])

    loss = embedding_consistency_loss(
        embedding, target, weights, eps=1e-6
    )
    normalized = 1.0 / (1.0 + 1e-6)
    expected = 2.0 * 3.0 * (2.0 - normalized) ** 2 / 4.0

    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_stage1_loss_combines_all_three_objectives() -> None:
    criterion = NativeBlazeGazeStage1Loss(
        NativeBlazeGazeLossConfig(
            gaze_weight=1.0,
            reconstruction_weight=2.0,
            consistency_weight=0.5,
        )
    )
    image = torch.zeros(2, 3, 2, 2)
    outputs = {
        "pog_pred": torch.ones(2, 2),
        "embedding": torch.tensor([[0.0], [1.0]], requires_grad=True),
        "reconstruction": torch.ones_like(image, requires_grad=True),
    }
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    losses = criterion(
        outputs,
        image_target=image,
        pog_target=target,
        sample_weight=torch.ones(2),
    )

    expected = (
        losses["gaze_l2"]
        + 2.0 * losses["reconstruction_mse"]
        + 0.5 * losses["embedding_consistency"]
    )
    assert torch.allclose(losses["loss"], expected)
    losses["loss"].backward()


def test_pog_denormalization_uses_fixed_physical_bounds() -> None:
    bounds = PogBoundsMM(-500.0, -100.0, -250.0, 350.0)
    centered = torch.tensor([[-0.5, -0.5], [0.5, 0.5], [0.0, 0.0]])

    actual = denormalize_pog_mm(centered, bounds)

    expected = torch.tensor(
        [[-500.0, -250.0], [-100.0, 350.0], [-300.0, 50.0]]
    )
    assert torch.equal(actual, expected)


def test_official_weight_grid_matches_inverse_frequency_formula() -> None:
    targets = np.array(
        [[-0.49, -0.49], [-0.48, -0.48], [0.49, 0.49]], dtype=np.float32
    )
    grid = OfficialInverseFrequencyGrid.fit(targets, bins=30)
    looked_up = grid.lookup(targets)

    assert grid.weights.shape == (30, 30)
    assert grid.weights.sum() == pytest.approx(1.0)
    assert looked_up[0] == pytest.approx(looked_up[1])
    assert looked_up[2] > looked_up[0]


def test_wandb_metrics_match_local_training_record() -> None:
    record = {
        "epoch": 3,
        "learning_rate": 0.0009,
        "train_loss": 1.1,
        "train_gaze_l2": 0.2,
        "train_reconstruction_mse": 0.3,
        "train_embedding_consistency": 0.4,
        "train_epe_mm": 20.0,
        "val_loss": 1.2,
        "val_gaze_l2": 0.25,
        "val_reconstruction_mse": 0.35,
        "val_embedding_consistency": 0.45,
        "val_epe_mm": 21.0,
        "epoch_seconds": 12.5,
    }

    metrics = _wandb_epoch_metrics(
        record,
        best_val_loss=1.0,
        best_val_epe_mm=19.0,
        best_epoch=2,
    )

    assert metrics["epoch"] == 3
    assert metrics["optimizer/learning_rate"] == pytest.approx(0.0009)
    assert metrics["train/reconstruction_mse"] == pytest.approx(0.3)
    assert metrics["val/embedding_consistency"] == pytest.approx(0.45)
    assert metrics["checkpoint/best_val_loss"] == pytest.approx(1.0)
    assert metrics["checkpoint/best_val_epe_mm"] == pytest.approx(19.0)
    assert metrics["checkpoint/best_epoch"] == 2
