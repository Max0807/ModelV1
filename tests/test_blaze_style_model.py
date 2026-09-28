from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from modelv1.blaze_style_model import (
    BlazeBlock,
    BlazeStyleModel,
    BlazeStyleModelConfig,
    DoubleBlazeBlock,
    blaze_embedding_consistency_loss,
)
from modelv1.model_factory import (
    MODEL_ARCHITECTURE_BLAZE_STYLE,
    build_experiment_model,
    canonical_model_architecture,
)


def blaze_config() -> BlazeStyleModelConfig:
    return BlazeStyleModelConfig(
        prediction_mode="direct_uv",
        use_face_image=True,
        use_crop_cam=False,
        scene_representation="table_frame7",
        scene_dim=7,
        use_table_frame_film=True,
        use_eye_keypoint_auxiliary=False,
        use_landmark_guided_eye_fusion=False,
        use_eye_geometry=False,
        eye_geometry_representation="raw_eye6d",
        fusion_hidden_dims=(256, 128),
        eye_backbone="resnet18",
        eye_backbone_weights=None,
    )


def test_blaze_blocks_preserve_or_downsample_shapes() -> None:
    x = torch.randn(2, 24, 28, 45, requires_grad=True)
    same = BlazeBlock(24, 24)(x)
    down = BlazeBlock(24, 48, stride=2)(same)
    doubled = DoubleBlazeBlock(48, 24, 96, stride=2)(down)

    assert same.shape == (2, 24, 28, 45)
    assert down.shape == (2, 48, 14, 23)
    assert doubled.shape == (2, 96, 7, 12)
    doubled.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_blaze_style_model_matches_direct_uv_contract() -> None:
    model = BlazeStyleModel(blaze_config()).eval()
    batch = {
        "face": torch.randn(2, 3, 160, 160),
        "left_eye": torch.randn(2, 3, 56, 90),
        "right_eye": torch.randn(2, 3, 56, 90),
        "table_frame7_n": torch.randn(2, 7),
    }

    with torch.no_grad():
        output = model(batch, return_features=True)

    assert output["uv"].shape == (2, 2)
    assert output["face_image_features"].shape == (2, 128)
    assert output["eye_features"].shape == (2, 128)
    assert output["visual_features"].shape == (2, 128)
    assert output["scene_features"].shape == (2, 64)
    assert output["fused_features"].shape == (2, 128)
    assert output["blaze_eye_latent"].shape == (2, 512)
    assert model.eye_resnet_encoders() == ()


def test_blaze_reconstruction_matches_paper_eye_strip_shape() -> None:
    config = replace(
        blaze_config(),
        use_reconstruction_auxiliary=True,
        reconstruction_sample_limit=2,
    )
    model = BlazeStyleModel(config).eval()
    batch = {
        "face": torch.randn(3, 3, 160, 160),
        "left_eye": torch.randn(3, 3, 56, 90),
        "right_eye": torch.randn(3, 3, 56, 90),
        "table_frame7_n": torch.randn(3, 7),
    }

    with torch.no_grad():
        output = model(batch, return_auxiliary=True)

    reconstruction = output["reconstructed_eye_strip"]
    target = output["reconstruction_target_eye_strip"]
    assert reconstruction.shape == (2, 3, 128, 512)
    assert target.shape == reconstruction.shape
    assert torch.all((reconstruction >= 0.0) & (reconstruction <= 1.0))
    assert torch.all((target >= 0.0) & (target <= 1.0))


def test_blaze_embedding_consistency_matches_reference_definition() -> None:
    embeddings = torch.tensor([[0.0], [1.0]])
    uv_targets_mm = torch.tensor([[0.0, 0.0], [100.0, 0.0]])

    loss = blaze_embedding_consistency_loss(embeddings, uv_targets_mm)

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_blaze_auxiliary_losses_backpropagate_into_encoder_and_decoder() -> None:
    config = replace(
        blaze_config(),
        use_reconstruction_auxiliary=True,
        reconstruction_sample_limit=2,
    )
    model = BlazeStyleModel(config).train()
    batch = {
        "face": torch.randn(2, 3, 160, 160),
        "left_eye": torch.randn(2, 3, 56, 90),
        "right_eye": torch.randn(2, 3, 56, 90),
        "table_frame7_n": torch.randn(2, 7),
    }

    output = model(batch, return_auxiliary=True)
    reconstruction_loss = torch.nn.functional.mse_loss(
        output["reconstructed_eye_strip"],
        output["reconstruction_target_eye_strip"],
    )
    consistency_loss = blaze_embedding_consistency_loss(
        output["blaze_eye_latent"],
        torch.tensor([[0.0, 0.0], [100.0, 0.0]]),
    )
    (reconstruction_loss + 0.2 * consistency_loss).backward()

    decoder_gradient = model.reconstruction_decoder.output[0].weight.grad
    encoder_gradient = model.visual_encoder.eye_encoder.stem[0].weight.grad
    assert decoder_gradient is not None
    assert encoder_gradient is not None
    assert torch.isfinite(decoder_gradient).all()
    assert torch.isfinite(encoder_gradient).all()


def test_blaze_style_model_backpropagates_through_visual_and_film_paths() -> None:
    model = BlazeStyleModel(blaze_config()).train()
    batch = {
        "face": torch.randn(2, 3, 160, 160),
        "left_eye": torch.randn(2, 3, 56, 90),
        "right_eye": torch.randn(2, 3, 56, 90),
        "table_frame7_n": torch.randn(2, 7),
    }

    loss = model(batch).square().mean()
    loss.backward()

    visual_gradient = model.visual_encoder.face_encoder.stem[0].weight.grad
    film_gradient = (
        model.direct_uv_table_frame_film_fusion.film_parameters.weight.grad
    )
    assert visual_gradient is not None
    assert film_gradient is not None
    assert torch.isfinite(visual_gradient).all()
    assert torch.isfinite(film_gradient).all()


def test_factory_builds_standalone_blaze_model() -> None:
    assert canonical_model_architecture("blaze-style") == "blaze_style"
    model = build_experiment_model(MODEL_ARCHITECTURE_BLAZE_STYLE, blaze_config())
    assert isinstance(model, BlazeStyleModel)


def test_blaze_style_rejects_auxiliary_landmark_path() -> None:
    config = blaze_config()
    object.__setattr__(config, "use_eye_keypoint_auxiliary", True)
    with pytest.raises(ValueError, match="use_eye_keypoint_auxiliary"):
        BlazeStyleModel(config)
