from __future__ import annotations

import torch

from modelv1.blaze_maml import (
    BlazeMAMLGazeHead,
    adapt_blaze_maml_head,
    blaze_maml_l2_loss,
    functional_head_prediction,
    sample_support_query_indices,
)
from modelv1.blaze_style_model import BlazeStyleModel, BlazeStyleModelConfig


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
        fusion_hidden_dims=(64, 32),
        eye_backbone="resnet18",
        eye_backbone_weights=None,
        branch_dropout=0.0,
        fusion_dropout=0.0,
    )


def test_maml_head_matches_blaze_prediction_path() -> None:
    model = BlazeStyleModel(blaze_config()).eval()
    model.requires_grad_(False)
    head = BlazeMAMLGazeHead.from_blaze_style_model(model).eval()
    batch = {
        "face": torch.randn(3, 3, 160, 160),
        "left_eye": torch.randn(3, 3, 56, 90),
        "right_eye": torch.randn(3, 3, 56, 90),
        "table_frame7_n": torch.randn(3, 7),
    }

    with torch.no_grad():
        output = model(batch, return_features=True)
        head_prediction = head(
            output["visual_features"],
            batch["table_frame7_n"],
        )

    assert torch.allclose(head_prediction, output["uv"], atol=1e-6, rtol=1e-6)
    assert all(parameter.requires_grad for parameter in head.parameters())


def test_first_order_maml_query_loss_backpropagates_to_meta_head() -> None:
    model = BlazeStyleModel(blaze_config()).eval()
    head = BlazeMAMLGazeHead.from_blaze_style_model(model).train()
    head.set_adaptable_components(("uv_head",))
    support_visual = torch.randn(4, model.config.visual_embedding_dim)
    support_scene = torch.randn(4, 7)
    support_targets = torch.randn(4, 2)
    query_visual = torch.randn(5, model.config.visual_embedding_dim)
    query_scene = torch.randn(5, 7)
    query_targets = torch.randn(5, 2)

    adapted, support_loss = adapt_blaze_maml_head(
        head,
        support_visual,
        support_scene,
        support_targets,
        inner_lr=1e-4,
        inner_steps=2,
        first_order=True,
    )
    query_predictions = functional_head_prediction(
        head,
        adapted,
        query_visual,
        query_scene,
    )
    query_loss = blaze_maml_l2_loss(query_predictions, query_targets)
    query_loss.backward()

    assert torch.isfinite(support_loss)
    assert torch.isfinite(query_loss)
    adaptable = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    gradients = [parameter.grad for parameter in adaptable]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
        if gradient is not None
    )
    assert sum(parameter.numel() for parameter in adaptable) == 66


def test_maml_episode_indices_are_disjoint_and_reproducible() -> None:
    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)
    candidates = tuple(range(30))

    support_a, query_a = sample_support_query_indices(
        candidates,
        support_size=9,
        query_size=10,
        generator=first_generator,
    )
    support_b, query_b = sample_support_query_indices(
        candidates,
        support_size=9,
        query_size=10,
        generator=second_generator,
    )

    assert torch.equal(support_a, support_b)
    assert torch.equal(query_a, query_b)
    assert set(support_a.tolist()).isdisjoint(query_a.tolist())
