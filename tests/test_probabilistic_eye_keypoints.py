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
    LowDOFDifferentiableEyeballTemplate,
    PitchToTableVResidual,
    ProbabilisticEyeLandmarkHead,
    VerticalEyeGeometryFeatureExtractor,
    VerticalGeometryVResidual,
    eye_landmark_validation_metrics,
    equivariant_landmark_consistency_loss,
    probabilistic_landmark_supervision_losses,
    sample_eye_affine_matrices,
    transform_normalized_points,
)
from scripts.train_modelv1 import run_epoch


class ProbabilisticEyeKeypointHeadTests(unittest.TestCase):
    @staticmethod
    def _fronto_parallel_table_frame(batch_size: int) -> torch.Tensor:
        return torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.7]]
        ).expand(batch_size, -1)

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

    def test_pitch_to_v_mapper_starts_as_exact_baseline(self) -> None:
        mapper = PitchToTableVResidual(
            scene_dim=7,
            hidden_dims=(8,),
            max_abs_delta_v_mm=100.0,
        )
        output = mapper(
            pitch_rad=torch.tensor([[0.2, 0.2], [-0.1, -0.1]]),
            pitch_variance=torch.full((2, 2), 0.001),
            geometry_confidence=torch.ones((2, 2)),
            geometry_valid_mask=torch.ones((2, 2)),
            scene_vec=self._fronto_parallel_table_frame(2),
        )

        torch.testing.assert_close(output["delta_v_mm"], torch.zeros((2, 1)))
        torch.testing.assert_close(
            output["raw_delta_v_mm"], torch.zeros((2, 1))
        )
        self.assertTrue(torch.all(output["geometry_gate"] > 0.0))
        self.assertTrue(torch.all(output["fallback_mask"] == 0.0))

    def test_pitch_to_v_mapper_is_bounded_and_falls_back_without_geometry(self) -> None:
        mapper = PitchToTableVResidual(
            scene_dim=7,
            hidden_dims=(8,),
            max_abs_delta_v_mm=100.0,
        )
        output_layer = mapper.coefficient_network[-1]
        self.assertIsInstance(output_layer, nn.Linear)
        with torch.no_grad():
            output_layer.bias[0] = 1.0
        pitch = torch.tensor([[0.2, 0.2], [0.2, 0.2]], requires_grad=True)
        output = mapper(
            pitch_rad=pitch,
            pitch_variance=torch.full((2, 2), 0.001),
            geometry_confidence=torch.tensor([[1.0, 1.0], [0.01, 0.01]]),
            geometry_valid_mask=torch.ones((2, 2)),
            scene_vec=self._fronto_parallel_table_frame(2),
        )

        self.assertGreater(float(output["delta_v_mm"][0]), 0.0)
        self.assertLessEqual(float(output["delta_v_mm"].abs().max()), 100.0)
        torch.testing.assert_close(output["delta_v_mm"][1], torch.zeros(1))
        self.assertGreater(float(output["geometry_gate"][1]), 0.0)
        self.assertLess(float(output["geometry_gate"][1]), 0.05)
        torch.testing.assert_close(
            output["applied_geometry_gate"][1], torch.zeros(1)
        )
        torch.testing.assert_close(output["fallback_mask"][1], torch.ones(1))
        output["delta_v_mm"].sum().backward()
        self.assertIsNotNone(pitch.grad)
        self.assertTrue(torch.isfinite(pitch.grad).all())

    def test_pitch_to_v_mapper_exposes_camera_table_mm_per_radian(self) -> None:
        mapper = PitchToTableVResidual(
            scene_dim=7,
            hidden_dims=(8,),
            max_abs_delta_v_mm=250.0,
        )
        output = mapper(
            pitch_rad=torch.tensor([[0.1, 0.1]]),
            pitch_variance=torch.full((1, 2), 0.001),
            geometry_confidence=torch.ones((1, 2)),
            geometry_valid_mask=torch.ones((1, 2)),
            scene_vec=self._fronto_parallel_table_frame(1),
        )

        torch.testing.assert_close(
            output["camera_table_vertical_sensitivity_mm_per_rad"],
            torch.tensor([[700.0]]),
        )
        torch.testing.assert_close(
            output["geometric_pitch_delta_v_mm"], torch.tensor([[70.0]])
        )
        torch.testing.assert_close(
            output["camera_table_geometry_valid_mask"], torch.ones((1, 1))
        )

    def test_low_dof_template_recovers_synthetic_positive_pitch(self) -> None:
        template = LowDOFDifferentiableEyeballTemplate(
            eye_image_size=(90, 56),
            pitch_bins=29,
            yaw_bins=31,
            posterior_temperature=0.08,
            covariance_floor_px=0.25,
            min_geometry_confidence=0.0,
            angle_prior_weight=0.0,
            learnable_template=False,
        )
        width, height = 90, 56
        origin = torch.tensor((44.5, 27.5))
        canthus = 44.0
        local = torch.zeros((15, 2))
        local[0] = torch.tensor((-0.5, 0.0))
        local[5] = torch.tensor((0.5, 0.0))
        local[1:5, 0] = torch.tensor((-0.35, -0.15, 0.15, 0.35))
        local[1:5, 1] = -0.12
        local[6:10, 0] = torch.tensor((0.35, 0.15, -0.15, -0.35))
        local[6:10, 1] = 0.12
        pitch = torch.deg2rad(torch.tensor(10.0))
        iris_center = torch.tensor((0.0, 0.55 * torch.sin(pitch)))
        local[10] = iris_center
        iris_radius = 0.105
        projected_vertical_radius = iris_radius * torch.cos(pitch)
        local[11:15] = iris_center + torch.tensor(
            (
                (iris_radius, 0.0),
                (0.0, projected_vertical_radius),
                (-iris_radius, 0.0),
                (0.0, -projected_vertical_radius),
            )
        )
        points_px = origin + canthus * local
        points_xy = points_px / torch.tensor((width - 1.0, height - 1.0))
        means = points_xy.reshape(1, 1, 15, 2).expand(1, 2, 15, 2).clone()
        variance = torch.tensor(
            ((0.25 / (width - 1.0)) ** 2, (0.25 / (height - 1.0)) ** 2)
        )
        covariance = torch.diag(variance).reshape(1, 1, 1, 2, 2).expand(
            1, 2, 15, 2, 2
        )

        output = template(means, covariance, torch.ones((1, 2, 15)))

        recovered_pitch_deg = torch.rad2deg(output["pitch_rad"])
        self.assertTrue(torch.all(recovered_pitch_deg > 5.0))
        self.assertTrue(torch.all(recovered_pitch_deg < 15.0))
        self.assertEqual(tuple(output["angle_covariance"].shape), (1, 2, 2, 2))
        self.assertEqual(tuple(output["angle_posterior"].shape), (1, 2, 29, 31))
        torch.testing.assert_close(
            output["angle_posterior"].sum(dim=(-1, -2)), torch.ones((1, 2))
        )
        self.assertEqual(tuple(output["observed_iris_center_local"].shape), (1, 2, 2))
        self.assertTrue(torch.all(output["reference_confidence"] > 0.0))
        self.assertTrue(torch.isfinite(output["reprojection_error_norm"]).all())

    def test_reference_confidence_uses_landmark_covariance(self) -> None:
        template = LowDOFDifferentiableEyeballTemplate(
            min_geometry_confidence=0.0
        )
        means = torch.rand((1, 2, 15, 2)) * 0.2 + 0.4
        means[:, :, 0] = torch.tensor([0.25, 0.5])
        means[:, :, 5] = torch.tensor([0.75, 0.5])
        means[:, :, 1:5, 1] = 0.45
        means[:, :, 6:10, 1] = 0.55
        low_covariance = (
            torch.eye(2).reshape(1, 1, 1, 2, 2).expand(1, 2, 15, 2, 2)
            * 1e-5
        )
        high_covariance = low_covariance * 1000.0

        low = template(means, low_covariance, torch.ones((1, 2, 15)))
        high = template(means, high_covariance, torch.ones((1, 2, 15)))

        self.assertTrue(
            torch.all(low["reference_confidence"] > high["reference_confidence"])
        )

    def test_eyelid_aperture_downweights_occluded_iris_rim(self) -> None:
        means = torch.full((1, 2, 15, 2), 0.5)
        means[:, :, 0] = torch.tensor([0.2, 0.5])
        means[:, :, 5] = torch.tensor([0.8, 0.5])
        means[:, :, 1:5, 0] = torch.tensor([0.3, 0.4, 0.6, 0.7])
        means[:, :, 1:5, 1] = 0.4
        means[:, :, 6:10, 0] = torch.tensor([0.7, 0.6, 0.4, 0.3])
        means[:, :, 6:10, 1] = 0.6
        means[:, :, 11] = torch.tensor([0.5, 0.2])
        covariance = (
            torch.eye(2).reshape(1, 1, 1, 2, 2).expand(1, 2, 15, 2, 2)
            * 1e-5
        )
        visibility = torch.ones((1, 2, 15))

        gated = LowDOFDifferentiableEyeballTemplate(
            use_aperture_visibility=True
        )(means, covariance, visibility)
        ungated = LowDOFDifferentiableEyeballTemplate(
            use_aperture_visibility=False
        )(means, covariance, visibility)

        self.assertTrue(
            torch.all(
                gated["iris_rim_effective_visibility"][:, :, 0]
                < ungated["iris_rim_effective_visibility"][:, :, 0]
            )
        )

    def test_template_confidence_falls_back_when_all_points_are_invisible(self) -> None:
        template = LowDOFDifferentiableEyeballTemplate(
            min_geometry_confidence=0.1
        )
        means = torch.rand((1, 2, 15, 2), requires_grad=True)
        covariance = torch.eye(2).reshape(1, 1, 1, 2, 2).expand(
            1, 2, 15, 2, 2
        ) * 0.01

        output = template(means, covariance, torch.zeros((1, 2, 15)))

        torch.testing.assert_close(
            output["geometry_confidence"], torch.zeros((1, 2))
        )
        torch.testing.assert_close(output["valid_mask"], torch.zeros((1, 2)))
        torch.testing.assert_close(output["safe_pitch_rad"], torch.zeros((1, 2)))

        output["geometry_confidence"].sum().backward()
        self.assertIsNotNone(means.grad)
        self.assertTrue(torch.isfinite(means.grad).all())

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
                use_low_dof_eye_template=True,
            )
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
                    use_eye_iris_auxiliary=False,
                    use_eye_keypoint_auxiliary=True,
                    use_low_dof_eye_template=True,
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
                template_pitch = torch.stack(
                    (means[:, 10, 1], means[:, 10, 1]), dim=1
                )
                template_reprojection = 0.1 + 0.0 * template_pitch
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
                    "eye_template_reprojection_error_norm": (
                        template_reprojection
                    ),
                    "eye_template_pitch_rad": template_pitch,
                    "eye_template_pitch_variance": torch.full(
                        (batch_size, 2), 0.01, device=self.uv.device
                    ),
                    "eye_template_geometry_confidence": torch.ones(
                        (batch_size, 2), device=self.uv.device
                    ),
                    "eye_template_reference_confidence": torch.ones(
                        (batch_size, 2), device=self.uv.device
                    ),
                    "eye_template_iris_center_effective_visibility": torch.ones(
                        (batch_size, 2), device=self.uv.device
                    ),
                    "eye_template_iris_rim_effective_visibility": torch.ones(
                        (batch_size, 2, 4), device=self.uv.device
                    ),
                    "eye_template_valid_mask": torch.ones(
                        (batch_size, 2), device=self.uv.device
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
            eye_template_reprojection_weight=0.05,
            eye_template_binocular_pitch_weight=0.02,
        )

        self.assertEqual(sample_count, 2)
        self.assertGreater(metrics["eye_keypoint_nll"], 0.0)
        self.assertAlmostEqual(
            metrics["weighted_eye_keypoint_nll"],
            0.05 * metrics["eye_keypoint_nll"],
        )
        self.assertEqual(metrics["eye_keypoint_effective_weight_per_sample"], 30.0)
        self.assertAlmostEqual(metrics["eye_template_reprojection_loss"], 0.1)
        self.assertAlmostEqual(
            metrics["weighted_eye_template_reprojection_loss"], 0.005
        )
        self.assertEqual(
            metrics["eye_template_binocular_pitch_disagreement_deg"], 0.0
        )


    def test_training_loop_supervises_simplified_vertical_residual(self) -> None:
        class TinyVerticalModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.uv = nn.Parameter(torch.zeros((1, 2)))
                self.delta_v = nn.Parameter(torch.full((1, 1), 20.0))
                self.config = SimpleNamespace(
                    use_depth_correction=False,
                    use_eye_iris_auxiliary=False,
                    use_eye_keypoint_auxiliary=False,
                    use_low_dof_eye_template=False,
                    use_pitch_to_v_residual=False,
                    use_vertical_geometry_residual=True,
                    vertical_geometry_residual_max_abs_delta_mm=100.0,
                )
                self._trainable_components = "all"

            def forward(self, batch, *, return_features=False, return_auxiliary=False):
                batch_size = int(batch["uv_gt"].shape[0])
                delta_v = self.delta_v.expand(batch_size, -1)
                return {
                    "uv": self.uv.expand(batch_size, -1),
                    "vertical_delta_v_mm": delta_v,
                    "vertical_training_gate": torch.ones_like(delta_v),
                    "vertical_learned_gate": torch.ones_like(delta_v),
                    "vertical_geometry_gate": torch.ones_like(delta_v),
                    "vertical_geometry_geometry_confidence": torch.ones_like(delta_v),
                    "vertical_fallback_mask": torch.zeros_like(delta_v),
                    "vertical_camera_table_vertical_sensitivity_mm_per_rad": torch.zeros_like(delta_v),
                    "vertical_camera_table_geometry_valid_mask": torch.zeros_like(delta_v),
                    "vertical_geometric_pitch_delta_v_mm": torch.zeros_like(delta_v),
                }

        model = TinyVerticalModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = {"uv_gt": torch.zeros((2, 2)), "uv_gt_table_mm": torch.tensor([[0.0, 10.0], [0.0, 10.0]])}
        criterion = DirectTableUVLoss(UVTargetNormalizer(torch.zeros(2), torch.ones(2)))
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
            vertical_residual_regularization_weight=0.001,
            vertical_residual_supervision_weight=0.25,
            vertical_residual_huber_beta_mm=30.0,
        )
        self.assertEqual(sample_count, 2)
        self.assertGreater(metrics["vertical_residual_supervision_loss"], 0.0)
        self.assertGreater(metrics["vertical_delta_abs_mean_mm"], 0.0)
        self.assertEqual(metrics["vertical_u_invariance_max_mm"], 0.0)

    def test_simplified_vertical_geometry_residual_is_bounded_and_safe(self) -> None:
        points = torch.full((2, 2, 15, 2), 0.5)
        points[:, :, 0, 0] = 0.2
        points[:, :, 5, 0] = 0.8
        points[:, :, 1:5, 1] = 0.4
        points[:, :, 6:10, 1] = 0.6
        points[:, :, 10, 1] = 0.53
        points[:, :, 11] = torch.tensor((0.6, 0.53))
        points[:, :, 12] = torch.tensor((0.5, 0.58))
        points[:, :, 13] = torch.tensor((0.4, 0.53))
        points[:, :, 14] = torch.tensor((0.5, 0.48))
        covariance = torch.eye(2).reshape(1, 1, 1, 2, 2).expand(2, 2, 15, -1, -1) * 1e-4
        visibility = torch.ones((2, 2, 15))
        entropy = torch.zeros((2, 2, 15))
        extractor = VerticalEyeGeometryFeatureExtractor(eye_image_size=(90, 56))
        geometry = extractor(points, covariance, visibility, entropy)
        self.assertEqual(geometry["features"].shape, (2, 22))
        self.assertTrue(torch.isfinite(geometry["features"]).all())

        mapper = VerticalGeometryVResidual(
            geometry_dim=22,
            fused_feature_dim=8,
            scene_dim=7,
            hidden_dims=(8,),
            max_abs_delta_v_mm=100.0,
            min_training_gate=0.1,
            min_inference_confidence=0.1,
        )
        features = geometry["features"].detach().requires_grad_()
        output = mapper(
            geometry_features=features,
            geometry_confidence=geometry["geometry_confidence"],
            fused_features=torch.zeros((2, 8)),
            scene_vec=self._fronto_parallel_table_frame(2),
            base_v_normalized=torch.zeros((2, 1)),
        )
        torch.testing.assert_close(output["delta_v_mm"], torch.zeros((2, 1)))
        output_layer = mapper.network[-1]
        self.assertIsInstance(output_layer, nn.Linear)
        with torch.no_grad():
            output_layer.bias[0] = 1.0
        output = mapper(
            geometry_features=features,
            geometry_confidence=geometry["geometry_confidence"],
            fused_features=torch.zeros((2, 8)),
            scene_vec=self._fronto_parallel_table_frame(2),
            base_v_normalized=torch.zeros((2, 1)),
        )
        self.assertGreater(float(output["delta_v_mm"].abs().max()), 0.0)
        self.assertLessEqual(float(output["delta_v_mm"].abs().max()), 100.0)
        output["delta_v_mm"].sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

        mapper.eval()
        fallback = mapper(
            geometry_features=features.detach(),
            geometry_confidence=torch.zeros((2, 1)),
            fused_features=torch.zeros((2, 8)),
            scene_vec=self._fronto_parallel_table_frame(2),
            base_v_normalized=torch.zeros((2, 1)),
        )
        torch.testing.assert_close(fallback["delta_v_mm"], torch.zeros((2, 1)))
        torch.testing.assert_close(fallback["fallback_mask"], torch.ones((2, 1)))

if __name__ == "__main__":
    unittest.main()
