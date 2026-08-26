from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from modelv1.data.dataset import (
    FACENET_MEAN,
    FACENET_STD,
    FACE_SIZE,
    ModelV1Dataset,
    iris_centres_in_eye_crops,
)
from modelv1.model import (
    CrossAttentionVisualEncoder,
    InceptionResnetFaceEncoder,
    ModelV1,
    ModelV1Config,
)


class _FakeBasicConv2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeMixed7a(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch0 = _FakeBasicConv2d()
        self.branch1 = nn.Sequential(nn.Identity(), _FakeBasicConv2d())
        self.branch2 = nn.Sequential(nn.Identity(), nn.Identity(), _FakeBasicConv2d())
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = (4, 4) if self.branch3.padding == 1 else (3, 3)
        return F.adaptive_avg_pool2d(x, output_size)


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
            "repeat_3",
        ):
            setattr(self, name, nn.Identity())
        self.mixed_7a = _FakeMixed7a()
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

    def test_f1_padded_mixed7a_returns_4x4_spatial_features(self) -> None:
        with patch.dict(sys.modules, {"facenet_pytorch": _fake_facenet_module()}):
            encoder = InceptionResnetFaceEncoder(feature_mode="f1").eval()

        self.assertEqual(encoder.feature_mode, "padded_mixed7a_4x4")
        self.assertEqual(encoder.output_grid, (4, 4))
        self.assertEqual(encoder.backbone.mixed_7a.branch3.padding, 1)
        output = encoder(torch.randn(2, 3, 160, 160))
        self.assertEqual(tuple(output.shape), (2, 1792, 4, 4))

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

    def test_cross_attention_uses_f1_face_4x4_tokens(self) -> None:
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
                face_feature_mode="f1",
            )

        output = encoder(
            torch.randn(2, 3, 160, 160),
            torch.randn(2, 3, 36, 60),
            torch.randn(2, 3, 36, 60),
            return_attention=True,
        )
        self.assertEqual(tuple(output["face_tokens"].shape), (2, 16, 128))
        self.assertEqual(tuple(output["eye_tokens"].shape), (2, 8, 128))
        self.assertEqual(
            tuple(output["cross_attention_weights"].shape),
            (2, 8, 8, 16),
        )

    def test_cross_attention_keeps_the_native_90x56_token_grid_for_h1(self) -> None:
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
                eye_token_grid=(2, 3),
            )

        output = encoder(
            torch.randn(2, 3, 160, 160),
            torch.randn(2, 3, 56, 90),
            torch.randn(2, 3, 56, 90),
            return_attention=True,
        )
        self.assertEqual(tuple(output["eye_tokens"].shape), (2, 12, 128))
        self.assertEqual(tuple(output["visual_features"].shape), (2, 128))

    def test_cross_attention_uses_layer3_4x6_tokens_for_h2(self) -> None:
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
                eye_feature_stage="layer3",
                eye_token_grid=(4, 6),
            )

        output = encoder(
            torch.randn(2, 3, 160, 160),
            torch.randn(2, 3, 56, 90),
            torch.randn(2, 3, 56, 90),
            return_attention=True,
        )
        self.assertEqual(tuple(output["eye_tokens"].shape), (2, 48, 128))
        self.assertEqual(
            tuple(output["cross_attention_weights"].shape),
            (2, 8, 48, 9),
        )
        self.assertEqual(tuple(output["visual_features"].shape), (2, 128))

    def test_cross_attention_fuses_three_resnet_scales_for_h3(self) -> None:
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
                eye_feature_mode="multiscale_fpn24",
                eye_token_grid=(4, 6),
            )

        eye_map = encoder.eye_encoder.forward_feature_map(torch.randn(2, 3, 56, 90))
        self.assertEqual(tuple(eye_map.shape), (2, 128, 4, 6))
        self.assertEqual(encoder.eye_encoder.feature_map_channels, 128)
        output = encoder(
            torch.randn(2, 3, 160, 160),
            torch.randn(2, 3, 56, 90),
            torch.randn(2, 3, 56, 90),
            return_attention=True,
        )
        self.assertEqual(tuple(output["eye_tokens"].shape), (2, 48, 128))
        self.assertEqual(
            tuple(output["cross_attention_weights"].shape),
            (2, 8, 48, 9),
        )
        output["visual_features"].square().mean().backward()
        self.assertIsNotNone(encoder.eye_encoder.fpn_fusion[0].weight.grad)

    def test_cross_attention_uses_h5_small_stem_fpn24(self) -> None:
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
                eye_stem_mode="small_3x3_s2_no_maxpool",
                eye_feature_mode="multiscale_fpn24",
                eye_token_grid=(4, 6),
            )

        self.assertEqual(encoder.eye_encoder.backbone.conv1.kernel_size, (3, 3))
        self.assertEqual(encoder.eye_encoder.backbone.conv1.stride, (2, 2))
        self.assertIsInstance(encoder.eye_encoder.backbone.maxpool, nn.Identity)
        eye_map = encoder.eye_encoder.forward_feature_map(torch.randn(2, 3, 56, 90))
        self.assertEqual(tuple(eye_map.shape), (2, 128, 4, 6))
        output = encoder(
            torch.randn(2, 3, 160, 160),
            torch.randn(2, 3, 56, 90),
            torch.randn(2, 3, 56, 90),
            return_attention=True,
        )
        self.assertEqual(tuple(output["eye_tokens"].shape), (2, 48, 128))
        self.assertEqual(
            tuple(output["cross_attention_weights"].shape),
            (2, 8, 48, 9),
        )
        output["visual_features"].square().mean().backward()
        self.assertIsNotNone(encoder.eye_encoder.fpn_layer2[0].weight.grad)

    def test_cross_attention_uses_h6_binocular_self_attention(self) -> None:
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
                eye_stem_mode="small_3x3_s2_no_maxpool",
                eye_feature_mode="multiscale_fpn24",
                eye_token_grid=(4, 6),
                use_binocular_self_attention=True,
                binocular_attention_heads=4,
                binocular_attention_dropout=0.0,
            )

        output = encoder(
            torch.randn(2, 3, 160, 160),
            torch.randn(2, 3, 56, 90),
            torch.randn(2, 3, 56, 90),
            return_attention=True,
        )
        self.assertEqual(tuple(output["eye_tokens"].shape), (2, 48, 128))
        self.assertEqual(
            tuple(output["binocular_self_attention_weights"].shape),
            (2, 4, 48, 48),
        )
        self.assertEqual(
            tuple(output["cross_attention_weights"].shape),
            (2, 8, 48, 9),
        )
        output["visual_features"].square().mean().backward()
        self.assertIsNotNone(encoder.binocular_self_attention.in_proj_weight.grad)

    def test_cross_attention_returns_h7_iris_heatmaps_from_h5_layer3(self) -> None:
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
                eye_stem_mode="small_3x3_s2_no_maxpool",
                eye_feature_mode="multiscale_fpn24",
                eye_token_grid=(4, 6),
                use_iris_auxiliary=True,
                iris_auxiliary_feature_stage="layer3",
                iris_auxiliary_hidden_channels=64,
            )

        output = encoder(
            torch.randn(2, 3, 160, 160),
            torch.randn(2, 3, 56, 90),
            torch.randn(2, 3, 56, 90),
            return_auxiliary=True,
        )
        self.assertIsNone(encoder.binocular_self_attention)
        self.assertEqual(tuple(output["visual_features"].shape), (2, 128))
        self.assertEqual(tuple(output["left_iris_heatmap"].shape), (2, 1, 7, 12))
        self.assertEqual(tuple(output["right_iris_heatmap"].shape), (2, 1, 7, 12))
        (
            output["visual_features"].square().mean()
            + output["left_iris_heatmap"].square().mean()
            + output["right_iris_heatmap"].square().mean()
        ).backward()
        self.assertIsNotNone(encoder.iris_heatmap_head[0].weight.grad)

    def test_cross_attention_returns_probabilistic_15_point_eye_heads(self) -> None:
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
                eye_stem_mode="small_3x3_s2_no_maxpool",
                eye_feature_mode="multiscale_fpn24",
                eye_token_grid=(4, 6),
                use_keypoint_auxiliary=True,
                keypoint_count=15,
                keypoint_auxiliary_feature_stage="layer3",
                keypoint_auxiliary_hidden_channels=64,
                use_low_dof_eye_template=True,
            )

        output = encoder(
            torch.randn(2, 3, 160, 160),
            torch.randn(2, 3, 56, 90),
            torch.randn(2, 3, 56, 90),
            return_auxiliary=True,
        )

        for side in ("left", "right"):
            self.assertEqual(
                tuple(output[f"{side}_eye_keypoint_probability_maps"].shape),
                (2, 15, 28, 45),
            )
            self.assertEqual(
                tuple(output[f"{side}_eye_keypoint_mean_xy"].shape),
                (2, 15, 2),
            )
            self.assertEqual(
                tuple(output[f"{side}_eye_keypoint_covariance"].shape),
                (2, 15, 2, 2),
            )
            self.assertEqual(
                tuple(output[f"{side}_eye_keypoint_visibility_logits"].shape),
                (2, 15),
            )
        output["left_eye_keypoint_mean_xy"].square().mean().backward()
        self.assertIsNotNone(
            encoder.eye_keypoint_head.heatmap_predictor[-1].weight.grad
        )
        self.assertEqual(tuple(output["eye_template_pitch_rad"].shape), (2, 2))
        self.assertEqual(
            tuple(output["eye_template_angle_covariance"].shape),
            (2, 2, 2, 2),
        )
        self.assertEqual(
            tuple(output["eye_template_geometry_confidence"].shape),
            (2, 2),
        )

    def test_cross_attention_learns_h4_token_pooling_weights(self) -> None:
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
                eye_feature_mode="multiscale_fpn24",
                eye_token_grid=(4, 6),
                token_pooling="learned",
                token_pooling_hidden_dim=64,
                token_pooling_dropout=0.0,
            )

        output = encoder(
            torch.randn(2, 3, 160, 160),
            torch.randn(2, 3, 56, 90),
            torch.randn(2, 3, 56, 90),
            return_attention=True,
        )
        weights = output["token_pooling_weights"]
        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertEqual(tuple(weights.shape), (2, 48))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
        # Zero initialization makes H4 start as H3 mean pooling, then the
        # score head can learn which local regions deserve more weight.
        torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 48.0))
        output["visual_features"].square().mean().backward()
        self.assertIsNotNone(encoder.token_score[-1].weight.grad)

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

    def test_iris_centres_are_assigned_by_crop_membership(self) -> None:
        row = {
            "left_eye_bbox_x": "100",
            "left_eye_bbox_y": "200",
            "left_eye_bbox_w": "90",
            "left_eye_bbox_h": "54",
            "right_eye_bbox_x": "300",
            "right_eye_bbox_y": "210",
            "right_eye_bbox_w": "90",
            "right_eye_bbox_h": "54",
        }
        centres, valid = iris_centres_in_eye_crops(
            row,
            ((145.0, 227.0), (345.0, 237.0)),
        )
        torch.testing.assert_close(valid, torch.ones(2))
        torch.testing.assert_close(centres[0], torch.tensor((0.5, 0.5)))
        torch.testing.assert_close(centres[1], torch.tensor((0.5, 0.5)))

    def test_modelv1_returns_h7_maps_when_requested(self) -> None:
        config = ModelV1Config(
            deca_feature_representation="none",
            deca_feature_dim=0,
            use_face_image=True,
            use_crop_cam=False,
            eye_backbone="resnet18",
            eye_backbone_weights=None,
            eye_stem_mode="small_3x3_s2_no_maxpool",
            eye_feature_mode="multiscale_fpn24",
            eye_token_grid=(4, 6),
            use_eye_iris_auxiliary=True,
        )
        with patch.dict(sys.modules, {"facenet_pytorch": _fake_facenet_module()}):
            model = ModelV1(config)
        output = model(
            face=torch.randn(2, 3, 160, 160),
            left_eye=torch.randn(2, 3, 56, 90),
            right_eye=torch.randn(2, 3, 56, 90),
            scene_vec=torch.randn(2, 25),
            return_auxiliary=True,
        )
        self.assertIsInstance(output, dict)
        assert isinstance(output, dict)
        self.assertEqual(tuple(output["uv"].shape), (2, 2))
        self.assertEqual(tuple(output["left_iris_heatmap"].shape), (2, 1, 7, 12))

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
