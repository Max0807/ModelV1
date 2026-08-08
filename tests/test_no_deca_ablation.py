from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from modelv1.deca_cache import (
    DECA_FEATURE_REPRESENTATIONS,
    deca_feature_representation_dim,
    select_deca_feature_representation,
)
from modelv1.model import ModelV1, ModelV1Config


class _FakeInceptionResnetV1(nn.Module):
    def __init__(self, *, pretrained: str, classify: bool) -> None:
        super().__init__()
        self.conv2d_1a = nn.AdaptiveAvgPool2d((3, 3))
        for name in (
            "conv2d_2a",
            "conv2d_2b",
            "maxpool_3a",
            "conv2d_3b",
            "conv2d_4a",
            "conv2d_4b",
            "repeat_1",
            "mixed_6a",
            "repeat_2",
            "mixed_7a",
            "repeat_3",
        ):
            setattr(self, name, nn.Identity())
        self.block8 = nn.Conv2d(3, 1792, kernel_size=1)
        self.logits = nn.Identity()


def _fake_facenet_module() -> types.ModuleType:
    module = types.ModuleType("facenet_pytorch")
    module.InceptionResnetV1 = _FakeInceptionResnetV1  # type: ignore[attr-defined]
    return module


class NoDecaAblationTests(unittest.TestCase):
    def test_none_is_a_zero_width_representation(self) -> None:
        self.assertIn("none", DECA_FEATURE_REPRESENTATIONS)
        self.assertEqual(deca_feature_representation_dim("none"), 0)
        selected = select_deca_feature_representation(
            np.zeros((236,), dtype=np.float32),
            "none",
        )
        self.assertEqual(selected.shape, (0,))

    def test_model_runs_without_deca_tensor_and_keeps_cross_attention(self) -> None:
        config = ModelV1Config(
            deca_feature_representation="none",
            deca_feature_dim=0,
            use_face_image=True,
            eye_backbone="resnet18",
            eye_backbone_weights=None,
            use_crop_cam=False,
            visual_attention_dropout=0.0,
        )
        with patch.dict(sys.modules, {"facenet_pytorch": _fake_facenet_module()}):
            model = ModelV1(config)

        self.assertIsNone(model.face_branch)
        output = model(
            {
                "face": torch.randn(2, 3, 160, 160),
                "left_eye": torch.randn(2, 3, 36, 60),
                "right_eye": torch.randn(2, 3, 36, 60),
                "scene_vec": torch.randn(2, 25),
            },
            return_features=True,
        )
        self.assertEqual(tuple(output["uv"].shape), (2, 2))
        self.assertEqual(tuple(output["face_features"].shape), (2, 0))
        self.assertEqual(tuple(output["deca_features"].shape), (2, 0))
        self.assertEqual(tuple(output["visual_features"].shape), (2, 128))
        self.assertEqual(tuple(output["cross_attention_weights"].shape), (2, 8, 8, 9))
        output["uv"].square().mean().backward()
        self.assertIsNotNone(model.visual_encoder.cross_attention.in_proj_weight.grad)

    def test_none_requires_zero_feature_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires deca_feature_dim=0"):
            ModelV1Config(
                deca_feature_representation="none",
                deca_feature_dim=156,
            )


if __name__ == "__main__":
    unittest.main()
