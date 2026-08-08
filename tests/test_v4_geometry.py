from __future__ import annotations

import unittest

import torch

from modelv1.depth_distribution import (
    DepthHypothesisConfig,
    PnpDepthHypothesisGenerator,
    mixture_uv_statistics,
)
from modelv1.geometry import RayTableGeometry, RayTableGeometryConfig
from modelv1.losses import GazeGeometryLoss, GazeGeometryLossConfig


class RayTableGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layer = RayTableGeometry(
            RayTableGeometryConfig(
                table_distance_scale_mm=1000.0,
                min_abs_normal_dot_gaze=0.02,
                min_lambda_mm=1.0,
                max_lambda_mm=5000.0,
            )
        )

    def test_exact_intersection_uses_camera_projection_origin(self) -> None:
        eye = torch.tensor([[[0.0, 0.0, 500.0]]])
        target = torch.tensor([[120.0, -80.0, 0.0]])
        gaze = torch.nn.functional.normalize(target - eye[:, 0], dim=-1)
        table_frame7 = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        )
        result = self.layer(gaze, eye, table_frame7)
        torch.testing.assert_close(
            result["uv_hypotheses_mm"][:, 0],
            target[:, :2],
            atol=1e-4,
            rtol=1e-5,
        )
        self.assertTrue(bool(result["ray_valid_mask"].all()))
        self.assertEqual(sum(p.numel() for p in self.layer.parameters()), 0)

    def test_reverse_axis_is_oriented_toward_table_without_parameters(self) -> None:
        eye = torch.tensor([[[0.0, 0.0, 500.0]]])
        target = torch.tensor([[120.0, -80.0, 0.0]])
        forward_gaze = torch.nn.functional.normalize(
            target - eye[:, 0],
            dim=-1,
        )
        table_frame7 = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        )
        result = self.layer(-forward_gaze, eye, table_frame7)
        torch.testing.assert_close(result["gaze_direction_c"], forward_gaze)
        self.assertTrue(bool(result["gaze_was_flipped"].all()))
        self.assertTrue(bool(result["ray_valid_mask"].all()))
        self.assertGreater(float(result["lambda_hypotheses_mm"].item()), 0.0)

    def test_common_depth_scale_creates_k17_binocular_midpoints(self) -> None:
        generator = PnpDepthHypothesisGenerator(
            DepthHypothesisConfig(
                mode="fixed_prior",
                num_bins=17,
                range_sigma=3.0,
                log_sigma_min=0.005,
                log_sigma_max=0.20,
            )
        )
        raw_eyes = torch.tensor(
            [[-30.0, 4.0, 600.0, 30.0, 6.0, 604.0]]
        )
        result = generator(raw_eyes, torch.tensor([[0.05]]))
        self.assertEqual(tuple(result["eye_hypotheses_c_mm"].shape), (1, 17, 3))
        torch.testing.assert_close(
            result["eye_hypotheses_c_mm"],
            0.5
            * (
                result["left_eye_hypotheses_c_mm"]
                + result["right_eye_hypotheses_c_mm"]
            ),
        )
        torch.testing.assert_close(
            result["depth_prior_weights"].sum(dim=-1),
            torch.ones(1),
        )
        torch.testing.assert_close(
            result["depth_log_scales"][:, 8],
            torch.zeros(1),
        )

    def test_uv_statistics_report_mean_map_and_covariance(self) -> None:
        uv = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]])
        weights = torch.tensor([[0.25, 0.50, 0.25]])
        valid = torch.ones((1, 3), dtype=torch.bool)
        result = mixture_uv_statistics(uv, weights, valid)
        torch.testing.assert_close(result["uv_mean_mm"], torch.tensor([[10.0, 0.0]]))
        torch.testing.assert_close(result["uv_map_mm"], torch.tensor([[10.0, 0.0]]))
        torch.testing.assert_close(
            result["uv_covariance_mm2"],
            torch.tensor([[[50.0, 0.0], [0.0, 0.0]]]),
        )

    def test_uv_statistics_excludes_nonphysical_hypotheses(self) -> None:
        result = mixture_uv_statistics(
            torch.tensor([[[0.0, 0.0], [30000.0, -30000.0]]]),
            torch.tensor([[0.5, 0.5]]),
            torch.tensor([[True, False]]),
        )
        torch.testing.assert_close(result["uv_mean_mm"], torch.zeros((1, 2)))
        torch.testing.assert_close(
            result["uv_covariance_mm2"], torch.zeros((1, 2, 2))
        )

    def test_uv_gaussian_nll_uses_predicted_covariance(self) -> None:
        criterion = GazeGeometryLoss(
            GazeGeometryLossConfig(
                uv_huber_beta_mm=30.0,
                uv_huber_weight=0.0,
                mixture_nll_weight=0.0,
                mixture_kernel_sigma_mm=30.0,
                gaze_direction_weight=0.0,
                gaze_angular_weight=0.0,
                depth_prior_kl_weight=0.0,
                ray_validity_weight=0.0,
                ray_penalty_scale_mm=100.0,
                uv_gaussian_nll_weight=1.0,
                uv_covariance_floor_mm2=1.0,
            )
        )
        batch = {"uv_gt": torch.tensor([[10.0, 0.0]])}
        common = {
            "uv_mean_mm": torch.zeros((1, 2)),
            "uv_hypotheses_mm": torch.zeros((1, 1, 2)),
            "depth_effective_weights": torch.ones((1, 1)),
            "depth_prior_log_weights": torch.zeros((1, 1)),
        }
        narrow = criterion(
            {**common, "uv_covariance_mm2": torch.eye(2).unsqueeze(0)},
            batch,
        )
        wide = criterion(
            {**common, "uv_covariance_mm2": (100.0 * torch.eye(2)).unsqueeze(0)},
            batch,
        )
        self.assertGreater(float(narrow["uv_gaussian_nll"]), float(wide["uv_gaussian_nll"]))

    def test_direction_constraint_penalizes_only_opposite_hemisphere(self) -> None:
        criterion = GazeGeometryLoss(
            GazeGeometryLossConfig(
                uv_huber_beta_mm=30.0,
                uv_huber_weight=0.0,
                mixture_nll_weight=0.0,
                mixture_kernel_sigma_mm=30.0,
                gaze_direction_weight=1.0,
                gaze_angular_weight=0.0,
                depth_prior_kl_weight=0.0,
                ray_validity_weight=0.0,
                ray_penalty_scale_mm=100.0,
            )
        )
        batch = {
            "uv_gt": torch.zeros((1, 2)),
            "raw_eye_geometry_mm": torch.tensor(
                [[-30.0, 0.0, 600.0, 30.0, 0.0, 600.0]]
            ),
            "gaze_target_camera_mm": torch.tensor([[0.0, 0.0, 1000.0]]),
        }
        common = {
            "uv_mean_mm": torch.zeros((1, 2)),
            "uv_hypotheses_mm": torch.zeros((1, 1, 2)),
            "depth_effective_weights": torch.ones((1, 1)),
            "depth_prior_log_weights": torch.zeros((1, 1)),
        }
        forward = criterion(
            {**common, "gaze_direction_c": torch.tensor([[0.0, 0.0, 1.0]])},
            batch,
        )
        backward = criterion(
            {**common, "gaze_direction_c": torch.tensor([[0.0, 0.0, -1.0]])},
            batch,
        )
        self.assertEqual(float(forward["gaze_direction"]), 0.0)
        self.assertEqual(float(backward["gaze_direction"]), 1.0)

    def test_pseudo_gaze_cosine_loss_has_configured_weak_weight(self) -> None:
        criterion = GazeGeometryLoss(
            GazeGeometryLossConfig(
                uv_huber_beta_mm=30.0,
                uv_huber_weight=0.0,
                mixture_nll_weight=0.0,
                mixture_kernel_sigma_mm=30.0,
                gaze_direction_weight=0.0,
                gaze_angular_weight=100.0,
                depth_prior_kl_weight=0.0,
                ray_validity_weight=0.0,
                ray_penalty_scale_mm=100.0,
            )
        )
        batch = {
            "uv_gt": torch.zeros((1, 2)),
            "raw_eye_geometry_mm": torch.tensor(
                [[-30.0, 0.0, 600.0, 30.0, 0.0, 600.0]]
            ),
            "gaze_target_camera_mm": torch.tensor([[0.0, 0.0, 1000.0]]),
        }
        output = {
            "uv_mean_mm": torch.zeros((1, 2)),
            "uv_hypotheses_mm": torch.zeros((1, 1, 2)),
            "depth_effective_weights": torch.ones((1, 1)),
            "depth_prior_log_weights": torch.zeros((1, 1)),
            "gaze_direction_c": torch.tensor([[1.0, 0.0, 0.0]]),
        }

        losses = criterion(output, batch)

        self.assertAlmostEqual(float(losses["gaze_angular"]), 1.0)
        self.assertAlmostEqual(float(losses["loss"]), 100.0)


if __name__ == "__main__":
    unittest.main()
