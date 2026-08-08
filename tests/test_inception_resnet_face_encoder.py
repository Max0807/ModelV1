from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from modelv1.data.dataset import FACENET_MEAN, FACENET_STD, FACE_SIZE, ModelV1Dataset
from modelv1.model import (
    CrossAttentionVisualEncoder,
    InceptionResnetFaceEncoder,
    ModelV1Config,
)


class _FakeInceptionResnetV1(nn.Module):
    def __init__(self, *, pretrained: str, classify: bool) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.classify = classify
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
        self.logits = nn.Linear(1, 1)


def _fake_facenet_module() -> types.ModuleType:
    module = types.ModuleType("facenet_pytorch")
    module.InceptionResnetV1 = _FakeInceptionResnetV1  # type: ignore[attr-defined]
    return module


class InceptionResnetFaceEncoderTests(unittest.TestCase):
    def test_uses_vggface2_backbone_and_returns_spatial_features(self) -> None:
        with patch.dict(sys.modules, {"facenet_pytorch": _fake_facenet_module()}):
            encoder = InceptionResnetFaceEncoder().eval()

        self.assertEqual(encoder.backbone.pretrained, "vggface2")
        self.assertFalse(encoder.backbone.classify)
        self.assertIsInstance(encoder.backbone.logits, nn.Identity)
        output = encoder(torch.randn(2, 3, 160, 160))
        self.assertEqual(tuple(output.shape), (2, 1792, 3, 3))

    def test_frozen_backbone_stays_in_eval_mode(self) -> None:
        with patch.dict(sys.modules, {"facenet_pytorch": _fake_facenet_module()}):
            encoder = InceptionResnetFaceEncoder(
                freeze_backbone=True,
            )

        encoder.train()
        self.assertFalse(encoder.backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.backbone.parameters()))

    def test_cross_attention_shapes_and_gradients(self) -> None:
        with patch.dict(sys.modules, {"facenet_pytorch": _fake_facenet_module()}):
            encoder = CrossAttentionVisualEncoder(
                attention_dim=128,
                output_dim=128,
                num_heads=8,
                ffn_dim=256,
                dropout=0.0,
                eye_backbone="resnet18",
                eye_backbone_weights=None,
                freeze_face_backbone=False,
            )

        output = encoder(
            torch.randn(2, 3, 160, 160),
            torch.randn(2, 3, 36, 60),
            torch.randn(2, 3, 36, 60),
            return_attention=True,
        )
        self.assertEqual(tuple(output["face_tokens"].shape), (2, 9, 128))
        self.assertEqual(tuple(output["eye_tokens"].shape), (2, 8, 128))
        self.assertEqual(tuple(output["visual_features"].shape), (2, 128))
        self.assertEqual(
            tuple(output["cross_attention_weights"].shape),
            (2, 8, 8, 9),
        )
        output["visual_features"].square().mean().backward()
        self.assertIsNotNone(encoder.face_projection[0].weight.grad)
        self.assertIsNotNone(encoder.eye_projection[0].weight.grad)
        self.assertIsNotNone(encoder.cross_attention.in_proj_weight.grad)

    def test_face_input_uses_facenet_size_and_fixed_standardization(self) -> None:
        self.assertEqual(FACE_SIZE, (160, 160))
        black = torch.zeros(3, 2, 2)
        white = torch.ones(3, 2, 2)
        torch.testing.assert_close(
            ModelV1Dataset._normalize_image(black, FACENET_MEAN, FACENET_STD),
            torch.full_like(black, -1.0),
        )
        torch.testing.assert_close(
            ModelV1Dataset._normalize_image(white, FACENET_MEAN, FACENET_STD),
            torch.ones_like(white),
        )

    def test_cross_attention_configuration_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "torchvision ResNet"):
            ModelV1Config(use_face_image=True, eye_backbone="cnn")
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            ModelV1Config(
                use_face_image=True,
                visual_attention_dim=130,
                visual_attention_heads=8,
            )


if __name__ == "__main__":
    unittest.main()
