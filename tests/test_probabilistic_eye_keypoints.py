from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from modelv1.data.normalization import UVTargetNormalizer
from modelv1.losses import DirectTableUVLoss, probabilistic_eye_keypoint_nll
from modelv1.model import ModelV1Config, ProbabilisticEyeKeypointHead
from modelv1.vertical_eye_geometry import (
    LandmarkGuidedEyeFusion,
    ProbabilisticEyeLandmarkHead,
    eye_landmark_validation_metrics,
    equivariant_landmark_consistency_loss,
    probabilistic_landmark_supervision_losses,
    sample_eye_affine_matrices,
    transform_normalized_points,
)
from scripts.train_modelv1 import run_epoch


class ProbabilisticEyeKeypointHeadTests(unittest.TestCase):
    @staticmethod
    def _landmark_predictions(batch_size: int) -> dict[str, torch.Tensor]:
        probability = torch.softmax(
            torch.randn(batch_size, 15, 8 * 12), dim=-1
        ).reshape(batch_size, 15, 8, 12)
        covariance = (
            torch.eye(2)
            .reshape(1, 1, 2, 2)
            .expand(batch_size, 15, 2, 2)
            * 0.01
        )
        output: dict[str, torch.Tensor] = {}
        for side in ("left", "right"):
            output.update(
                {
                    f"{side}_eye_keypoint_probability_maps": probability,
                    f"{side}_eye_keypoint_mean_xy": torch.rand(
                        batch_size, 15, 2
                    ),
                    f"{side}_eye_keypoint_covariance": covariance,
                    f"{side}_eye_keypoint_visibility_probability": torch.full(
                        (batch_size, 15), 0.8
                    ),
                    f"{side}_eye_keypoint_entropy": torch.full(
                        (batch_size, 15), 0.3
                    ),
                }
            )
        return output

    def test_landmark_guided_fusion_starts_as_exact_identity(self) -> None:
        batch_size = 2
        fusion = LandmarkGuidedEyeFusion(
            token_dim=16,
            num_heads=4,
            dropout=0.0,
            gate_init=0.0,
        ).eval()
        eye_tokens = torch.randn(batch_size, 48, 16, requires_grad=True)
        output = fusion(
            eye_tokens,
            left_layer2=torch.randn(batch_size, 128, 14, 23),
            left_layer3=torch.randn(batch_size, 256, 7, 12),
            right_layer2=torch.randn(batch_size, 128, 14, 23),
            right_layer3=torch.randn(batch_size, 256, 7, 12),
            predictions=self._landmark_predictions(batch_size),
            return_attention=True,
        )

        torch.testing.assert_close(output["refined_eye_tokens"], eye_tokens)
        self.assertEqual(tuple(output["landmark_tokens"].shape), (2, 30, 16))
        self.assertEqual(tuple(output["landmark_quality"].shape), (2, 30))
        self.assertEqual(tuple(output["attention_weights"].shape), (2, 4, 48, 30))
        torch.testing.assert_close(output["global_gate"], torch.zeros((2, 1)))
        self.assertTrue(torch.isfinite(output["refined_eye_tokens"]).all())

        probe = torch.randn_like(output["refined_eye_tokens"])
        (output["refined_eye_tokens"] * probe).sum().backward()
        self.assertIsNotNone(fusion.raw_gate.grad)
        self.assertTrue(torch.isfinite(fusion.raw_gate.grad))

    def test_high_resolution_head_fuses_layer2_and_layer3(self) -> None:
        head = ProbabilisticEyeLandmarkHead(
            layer2_channels=8,
            layer3_channels=16,
            hidden_channels=8,
            output_size=(28, 45),
            shape_embedding_dim=6,
        ).eval()

        output = head(
            torch.randn(2, 8, 14, 23),
            torch.randn(2, 16, 7, 12),
        )

        self.assertEqual(tuple(output["logits"].shape), (2, 15, 28, 45))
        self.assertEqual(tuple(output["mean_xy"].shape), (2, 15, 2))
        self.assertEqual(tuple(output["covariance"].shape), (2, 15, 2, 2))
        self.assertEqual(tuple(output["visibility_logits"].shape), (2, 15))
        self.assertEqual(tuple(output["shape_embedding"].shape), (2, 6))
        torch.testing.assert_close(
            output["probability_maps"].sum(dim=(-1, -2)),
            torch.ones((2, 15)),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(torch.all(output["variance_xy"] > 0))

    def test_multiterm_landmark_supervision_is_finite_and_differentiable(self) -> None:
        mean = torch.full((1, 2, 15, 2), 0.5, requires_grad=True)
        logits = torch.randn((1, 2, 15, 12, 20), requires_grad=True)
        visibility = torch.zeros((1, 2, 15), requires_grad=True)
        target = torch.full((1, 2, 15, 2), 0.55)
        valid = torch.ones((1, 2, 15))
        quality = torch.tensor([[1.0, 0.5]])

        losses = probabilistic_landmark_supervision_losses(
            predicted_mean_xy=mean,
            heatmap_logits=logits,
            visibility_logits=visibility,
            target_xy=target,
            valid_mask=valid,
            teacher_quality=quality,
            visibility_supervision_weight=torch.ones((1, 2)),
            huber_delta_norm=0.02,
            heatmap_sigma_min_px=1.0,
            heatmap_sigma_max_px=3.0,
        )
        total = (
            losses["coordinate_loss"]
            + losses["heatmap_loss"]
            + losses["visibility_loss"]
        )
        total.backward()

        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(mean.grad)
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(visibility.grad)

    def test_known_affine_transform_has_zero_canonical_equivariance_error(self) -> None:
        points = torch.rand((2, 2, 15, 2)) * 0.4 + 0.3
        matrix1 = sample_eye_affine_matrices(
            2,
            device=torch.device("cpu"),
            max_rotation_deg=3.0,
            max_translation_norm=0.02,
            scale_min=1.0,
            scale_max=1.0,
            deterministic_sign=-1,
        )
        matrix2 = sample_eye_affine_matrices(
            2,
            device=torch.device("cpu"),
            max_rotation_deg=3.0,
            max_translation_norm=0.02,
            scale_min=1.0,
            scale_max=1.0,
            deterministic_sign=1,
        )
        output = equivariant_landmark_consistency_loss(
            transform_normalized_points(points, matrix1),
            transform_normalized_points(points, matrix2),
            matrix1,
            matrix2,
            torch.ones((2, 2)),
            huber_delta_norm=0.02,
        )

        torch.testing.assert_close(output["loss"], torch.zeros(()), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            output["mean_error_norm"], torch.zeros(()), atol=1e-6, rtol=0
        )

    def test_validation_metrics_are_zero_for_exact_predictions(self) -> None:
        points = torch.rand((2, 2, 15, 2)) * 0.5 + 0.25
        metrics = eye_landmark_validation_metrics(
            points,
            points,
            torch.ones((2, 2, 15)),
            torch.ones((2, 2)),
            eye_image_size=(90, 56),
        )
        for name, value in metrics.items():
            self.assertTrue(torch.isfinite(value), name)
            self.assertLess(abs(float(value)), 1e-5, name)

    def test_head_returns_normalized_means_full_covariance_and_entropy(self) -> None:
        head = ProbabilisticEyeKeypointHead(
            input_channels=8,
            hidden_channels=4,
            keypoint_count=15,
        ).eval()
        for parameter in head.parameters():
            torch.nn.init.zeros_(parameter)

        output = head(torch.randn(2, 8, 7, 12))

        self.assertEqual(tuple(output["logits"].shape), (2, 15, 7, 12))
        self.assertEqual(tuple(output["probability_maps"].shape), (2, 15, 7, 12))
        self.assertEqual(tuple(output["mean_xy"].shape), (2, 15, 2))
        self.assertEqual(tuple(output["covariance"].shape), (2, 15, 2, 2))
        self.assertEqual(tuple(output["entropy"].shape), (2, 15))
        torch.testing.assert_close(
            output["probability_maps"].sum(dim=(-1, -2)),
            torch.ones((2, 15)),
        )
        torch.testing.assert_close(
            output["mean_xy"],
            torch.full((2, 15, 2), 0.5),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output["covariance"],
            output["covariance"].transpose(-1, -2),
        )
        self.assertTrue(torch.all(output["variance_xy"] > 0))
        torch.testing.assert_close(output["entropy"], torch.ones((2, 15)))
        torch.testing.assert_close(output["confidence"], torch.zeros((2, 15)))

    def test_probability_moments_receive_gradients_from_nll(self) -> None:
        head = ProbabilisticEyeKeypointHead(
            input_channels=8,
            hidden_channels=4,
            keypoint_count=15,
        )
        left = head(torch.randn(2, 8, 7, 12))
        right = head(torch.randn(2, 8, 7, 12))
        targets = torch.rand((2, 2, 15, 2))
        valid = torch.ones((2, 2, 15))
        quality = torch.tensor([[1.0, 0.8], [0.5, 0.0]])

        outputs = probabilistic_eye_keypoint_nll(
            left["mean_xy"],
            right["mean_xy"],
            left["covariance"],
            right["covariance"],
            targets,
            valid,
            quality,
            min_std_norm=0.01,
        )
        outputs["loss"].backward()

        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertTrue(torch.isfinite(outputs["mean_error_norm"]))
        self.assertIsNotNone(head.predictor[0].weight.grad)
        self.assertTrue(torch.isfinite(head.predictor[0].weight.grad).all())

    def test_shifted_target_has_larger_nll_and_masks_are_respected(self) -> None:
        means = torch.full((1, 15, 2), 0.5)
        covariance = torch.eye(2).reshape(1, 1, 2, 2).expand(1, 15, 2, 2) * 0.01
        exact_targets = torch.full((1, 2, 15, 2), 0.5)
        shifted_targets = exact_targets.clone()
        shifted_targets[..., 1] = 0.8
        valid = torch.ones((1, 2, 15))
        quality = torch.ones((1, 2))

        exact = probabilistic_eye_keypoint_nll(
            means,
            means,
            covariance,
            covariance,
            exact_targets,
            valid,
            quality,
            min_std_norm=0.01,
        )
        shifted = probabilistic_eye_keypoint_nll(
            means,
            means,
            covariance,
            covariance,
            shifted_targets,
            valid,
            quality,
            min_std_norm=0.01,
        )
        masked = probabilistic_eye_keypoint_nll(
            means,
            means,
            covariance,
            covariance,
            shifted_targets,
            torch.zeros_like(valid),
            quality,
            min_std_norm=0.01,
        )

        self.assertGreater(float(shifted["loss"]), float(exact["loss"]))
        self.assertEqual(float(masked["loss"]), 0.0)
        self.assertEqual(float(masked["effective_weight"]), 0.0)

    def test_model_config_enforces_15_point_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "pseudo-label schema"):
            ModelV1Config(eye_keypoint_count=14)
        with self.assertRaisesRegex(ValueError, "requires use_face_image"):
            ModelV1Config(use_eye_keypoint_auxiliary=True)
        with self.assertRaisesRegex(ValueError, "requires.*keypoint"):
            ModelV1Config(
                use_face_image=True,
                use_landmark_guided_eye_fusion=True,
            )

    def test_training_loop_adds_weighted_probabilistic_keypoint_loss(self) -> None:
        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.uv = nn.Parameter(torch.zeros((1, 2)))
                self.keypoint_mean = nn.Parameter(torch.full((1, 15, 2), 0.5))
                self.config = SimpleNamespace(
                    use_depth_correction=False,
                    use_eye_keypoint_auxiliary=True,
                )
                self._trainable_components = "all"

            def forward(self, batch, *, return_features=False, return_auxiliary=False):
                batch_size = int(batch["uv_gt"].shape[0])
                covariance = (
                    torch.eye(2, device=self.uv.device)
                    .reshape(1, 1, 2, 2)
                    .expand(batch_size, 15, 2, 2)
                    * 0.01
                )
                means = self.keypoint_mean.expand(batch_size, -1, -1)
                return {
                    "uv": self.uv.expand(batch_size, -1),
                    "left_eye_keypoint_mean_xy": means,
                    "right_eye_keypoint_mean_xy": means,
                    "left_eye_keypoint_covariance": covariance,
                    "right_eye_keypoint_covariance": covariance,
                    "left_eye_keypoint_logits": torch.zeros(
                        (batch_size, 15, 12, 20), device=self.uv.device
                    ),
                    "right_eye_keypoint_logits": torch.zeros(
                        (batch_size, 15, 12, 20), device=self.uv.device
                    ),
                    "left_eye_keypoint_visibility_logits": torch.zeros(
                        (batch_size, 15), device=self.uv.device
                    ),
                    "right_eye_keypoint_visibility_logits": torch.zeros(
                        (batch_size, 15), device=self.uv.device
                    ),
                    "left_eye_keypoint_shape_embedding": torch.ones(
                        (batch_size, 4), device=self.uv.device
                    ),
                    "right_eye_keypoint_shape_embedding": torch.ones(
                        (batch_size, 4), device=self.uv.device
                    ),
                }

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = {
            "uv_gt": torch.zeros((2, 2)),
            "uv_gt_table_mm": torch.zeros((2, 2)),
            "eye_pseudo_landmarks_xy": torch.full((2, 2, 15, 2), 0.6),
            "eye_pseudo_landmark_valid_mask": torch.ones((2, 2, 15)),
            "eye_pseudo_effective_quality": torch.ones((2, 2)),
            "eye_pseudo_eye_quality": torch.ones((2, 2)),
            "eye_pseudo_eye_valid_mask": torch.ones((2, 2)),
            "left_eye": torch.zeros((2, 3, 56, 90)),
            "right_eye": torch.zeros((2, 3, 56, 90)),
        }
        criterion = DirectTableUVLoss(
            UVTargetNormalizer(torch.zeros(2), torch.ones(2))
        )

        metrics, sample_count = run_epoch(
            model=model,
            loader=[batch],
            criterion=criterion,
            device=torch.device("cpu"),
            optimizer=optimizer,
            scaler=torch.cuda.amp.GradScaler(enabled=False),
            amp_enabled=False,
            grad_clip_norm=None,
            gate_regularization_weight=0.0,
            eye_keypoint_nll_weight=0.05,
            eye_keypoint_min_std_norm=0.01,
        )

        self.assertEqual(sample_count, 2)
        self.assertGreater(metrics["eye_keypoint_nll"], 0.0)
        self.assertAlmostEqual(
            metrics["weighted_eye_keypoint_nll"],
            0.05 * metrics["eye_keypoint_nll"],
        )
        self.assertEqual(metrics["eye_keypoint_effective_weight_per_sample"], 30.0)
if __name__ == "__main__":
    unittest.main()
