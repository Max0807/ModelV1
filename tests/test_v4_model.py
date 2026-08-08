from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from modelv1.losses import GazeGeometryLoss, GazeGeometryLossConfig
from modelv1.model import ModelV1, ModelV1Config, ensure_rotation_batch


class V4ModelTests(unittest.TestCase):
    def _config(self, depth_mode: str, bins: int) -> ModelV1Config:
        return ModelV1Config(
            prediction_mode="gaze_geometry",
            deca_feature_representation="geometry156",
            deca_feature_dim=156,
            use_crop_cam=False,
            scene_representation="table_frame7",
            scene_dim=7,
            use_eye_geometry=False,
            eye_geometry_representation="raw_eye6d",
            eye_backbone="cnn",
            face_embedding_dim=16,
            face_hidden_dims=(16,),
            eye_embedding_dim=16,
            per_eye_embedding_dim=8,
            fusion_hidden_dims=(24, 12),
            depth_distribution_mode=depth_mode,
            depth_num_bins=bins,
            depth_range_sigma=3.0,
            depth_log_sigma_min=0.005,
            depth_log_sigma_max=0.20,
            depth_reweighter_hidden_dims=(8,),
            table_distance_scale_mm=1000.0,
            geometry_min_abs_normal_dot_gaze=0.02,
            geometry_min_lambda_mm=1.0,
            geometry_max_lambda_mm=5000.0,
            branch_dropout=0.0,
            fusion_dropout=0.0,
        )

    def test_table_frame_has_no_learned_branch(self) -> None:
        model = ModelV1(self._config("fixed_prior", 17))
        self.assertIsNone(model.scene_branch)
        self.assertIsNone(model.uv_head)
        self.assertEqual(
            [name for name, _ in model.named_parameters() if "scene" in name],
            [],
        )
        self.assertEqual(
            sum(p.numel() for p in model.ray_table_geometry.parameters()),
            0,
        )

    def test_forward_returns_unit_gaze_and_k17_geometry(self) -> None:
        model = ModelV1(self._config("fixed_prior", 17)).eval()
        batch = {
            "deca_feat": torch.zeros((2, 156)),
            "left_eye": torch.zeros((2, 3, 36, 60)),
            "right_eye": torch.zeros((2, 3, 36, 60)),
            "table_frame7": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
            ).expand(2, -1),
            "raw_eye_geometry_mm": torch.tensor(
                [[-30.0, 0.0, 600.0, 30.0, 0.0, 600.0]]
            ).expand(2, -1),
            "depth_log_scale_sigma": torch.full((2, 1), 0.05),
        }
        with torch.no_grad():
            output = model(batch)
        self.assertEqual(tuple(output["uv_hypotheses_mm"].shape), (2, 17, 2))
        self.assertEqual(tuple(output["uv_covariance_mm2"].shape), (2, 2, 2))
        torch.testing.assert_close(
            torch.linalg.vector_norm(output["gaze_direction_c"], dim=-1),
            torch.ones(2),
        )

    def test_camera_gaze_uncertainty_propagates_to_uv_distribution(self) -> None:
        model = ModelV1(
            replace(
                self._config("point", 1),
                use_gaze_uncertainty=True,
                gaze_uncertainty_min_std_rad=0.01,
                gaze_uncertainty_max_std_rad=0.25,
                gaze_uncertainty_init_std_rad=0.02,
            )
        )
        batch = {
            "deca_feat": torch.randn((2, 156)),
            "left_eye": torch.randn((2, 3, 36, 60)),
            "right_eye": torch.randn((2, 3, 36, 60)),
            "table_frame7": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
            ).expand(2, -1),
            "raw_eye_geometry_mm": torch.tensor(
                [[-30.0, 0.0, 600.0, 30.0, 0.0, 600.0]]
            ).expand(2, -1),
            "depth_log_scale_sigma": torch.full((2, 1), 0.05),
            "uv_gt": torch.tensor([[50.0, -20.0], [80.0, 40.0]]),
            "gaze_target_camera_mm": torch.tensor(
                [[50.0, -20.0, 0.0], [80.0, 40.0, 0.0]]
            ),
        }
        output = model(batch)
        self.assertEqual(tuple(output["gaze_sigma_points_c"].shape), (2, 5, 3))
        self.assertEqual(tuple(output["uv_hypotheses_mm"].shape), (2, 5, 2))
        torch.testing.assert_close(
            torch.diagonal(output["gaze_tangent_cholesky"], dim1=-2, dim2=-1),
            torch.full((2, 2), 0.02),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            output["gaze_sigma_point_weights"].sum(dim=-1), torch.ones(2)
        )
        self.assertTrue(torch.isfinite(output["uv_covariance_mm2"]).all())
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
                uv_covariance_floor_mm2=100.0,
            )
        )
        loss = criterion(output, batch)["loss"]
        loss.backward()
        assert model.gaze_uncertainty_head is not None
        gradient_norm = sum(
            float(parameter.grad.norm())
            for parameter in model.gaze_uncertainty_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_norm, 0.0)

    def test_virtual_camera_gaze_and_sigma_points_rotate_back_to_camera(self) -> None:
        config = replace(
            self._config("point", 1),
            gaze_prediction_frame="virtual_camera",
            use_gaze_uncertainty=True,
            gaze_uncertainty_min_std_rad=0.01,
            gaze_uncertainty_max_std_rad=0.25,
            gaze_uncertainty_init_std_rad=0.02,
        )
        model = ModelV1(config).eval()
        assert model.gaze_head is not None
        with torch.no_grad():
            model.gaze_head.weight.zero_()
            model.gaze_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
        rotation_n_from_c = torch.tensor(
            [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]
        )
        batch = {
            "deca_feat": torch.zeros((1, 156)),
            "left_eye": torch.zeros((1, 3, 36, 60)),
            "right_eye": torch.zeros((1, 3, 36, 60)),
            "table_frame7": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
            ),
            "raw_eye_geometry_mm": torch.tensor(
                [[-30.0, 0.0, 600.0, 30.0, 0.0, 600.0]]
            ),
            "depth_log_scale_sigma": torch.full((1, 1), 0.05),
            "rotation_n_from_c": rotation_n_from_c,
        }
        output = model(batch)
        torch.testing.assert_close(
            output["gaze_axis_direction_n"],
            torch.tensor([[1.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            output["gaze_axis_direction_c"],
            torch.tensor([[0.0, 0.0, -1.0]]),
        )
        expected_sigma_points_c = torch.einsum(
            "bij,bkj->bki",
            rotation_n_from_c.transpose(-1, -2),
            output["gaze_sigma_points_n"],
        )
        torch.testing.assert_close(
            output["gaze_sigma_points_c"],
            expected_sigma_points_c,
        )
        output["uv"].sum().backward()
        self.assertIsNotNone(model.gaze_head.weight.grad)

    def test_virtual_camera_gaze_requires_rotation(self) -> None:
        model = ModelV1(
            replace(
                self._config("point", 1),
                gaze_prediction_frame="virtual_camera",
            )
        )
        batch = {
            "deca_feat": torch.zeros((1, 156)),
            "left_eye": torch.zeros((1, 3, 36, 60)),
            "right_eye": torch.zeros((1, 3, 36, 60)),
            "table_frame7": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
            ),
            "raw_eye_geometry_mm": torch.tensor(
                [[-30.0, 0.0, 600.0, 30.0, 0.0, 600.0]]
            ),
            "depth_log_scale_sigma": torch.full((1, 1), 0.05),
        }
        with self.assertRaisesRegex(KeyError, "rotation_n_from_c"):
            model(batch)

    def test_rotation_validation_stays_float32_inside_amp(self) -> None:
        rotation = torch.eye(3).unsqueeze(0)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            validated = ensure_rotation_batch(
                rotation,
                "rotation_n_from_c",
                batch_size=1,
            )
        self.assertEqual(validated.dtype, torch.float32)
        torch.testing.assert_close(validated, rotation)

    def test_learned_reweight_receives_uv_loss_gradient_without_table_mlp(self) -> None:
        model = ModelV1(self._config("learned_reweight", 17))
        batch = {
            "deca_feat": torch.randn((2, 156)),
            "left_eye": torch.randn((2, 3, 36, 60)),
            "right_eye": torch.randn((2, 3, 36, 60)),
            "table_frame7": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
            ).expand(2, -1),
            "raw_eye_geometry_mm": torch.tensor(
                [[-30.0, 0.0, 600.0, 30.0, 0.0, 600.0]]
            ).expand(2, -1),
            "depth_log_scale_sigma": torch.full((2, 1), 0.05),
            "pnp_quality_vec": torch.zeros((2, 4)),
            "uv_gt": torch.tensor([[50.0, -20.0], [80.0, 40.0]]),
            "gaze_target_camera_mm": torch.tensor(
                [[50.0, -20.0, 0.0], [80.0, 40.0, 0.0]]
            ),
        }
        output = model(batch)
        criterion = GazeGeometryLoss(
            GazeGeometryLossConfig(
                uv_huber_beta_mm=30.0,
                uv_huber_weight=0.5,
                mixture_nll_weight=0.1,
                mixture_kernel_sigma_mm=30.0,
                gaze_direction_weight=1.0,
                gaze_angular_weight=0.0,
                depth_prior_kl_weight=0.01,
                ray_validity_weight=0.0,
                ray_penalty_scale_mm=100.0,
            )
        )
        loss = criterion(output, batch)["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        assert model.depth_reweighter is not None
        gradient_norm = sum(
            float(parameter.grad.norm())
            for parameter in model.depth_reweighter.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_norm, 0.0)

    def test_depth_correction_starts_at_identity_and_is_sigma_bounded(self) -> None:
        config = replace(
            self._config("point", 1),
            use_depth_correction=True,
            depth_correction_hidden_dims=(8,),
            depth_correction_dropout=0.0,
        )
        model = ModelV1(config).eval()
        raw_eyes = torch.tensor(
            [[-30.0, -20.0, 600.0, 30.0, -20.0, 600.0]]
        )
        batch = {
            "deca_feat": torch.zeros((1, 156)),
            "left_eye": torch.zeros((1, 3, 36, 60)),
            "right_eye": torch.zeros((1, 3, 36, 60)),
            "table_frame7": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
            ),
            "raw_eye_geometry_mm": raw_eyes,
            "depth_log_scale_sigma": torch.full((1, 1), 0.05),
            "pnp_geometry_vec": torch.zeros((1, 6)),
            "pnp_quality_vec": torch.zeros((1, 4)),
        }
        with torch.no_grad():
            identity_output = model(batch)
        torch.testing.assert_close(
            identity_output["depth_shared_scale"],
            torch.ones((1, 1)),
        )
        torch.testing.assert_close(
            identity_output["corrected_eye_geometry_mm"],
            raw_eyes,
        )

        assert model.depth_correction_head is not None
        with torch.no_grad():
            model.depth_correction_head.output.bias.fill_(100.0)
            bounded_output = model(batch)
        expected_log_bound = torch.full((1, 1), 3.0 * 0.05)
        torch.testing.assert_close(
            bounded_output["depth_log_scale_correction"],
            expected_log_bound,
        )
        corrected = bounded_output["corrected_eye_geometry_mm"]
        torch.testing.assert_close(
            corrected / raw_eyes,
            bounded_output["depth_shared_scale"].expand_as(raw_eyes),
        )

    def test_depth_correction_receives_uv_gradient_with_frozen_gaze(self) -> None:
        config = replace(
            self._config("point", 1),
            use_depth_correction=True,
            depth_correction_hidden_dims=(8,),
            depth_correction_dropout=0.0,
        )
        model = ModelV1(config)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        assert model.depth_correction_head is not None
        for parameter in model.depth_correction_head.parameters():
            parameter.requires_grad_(True)
        batch = {
            "deca_feat": torch.randn((2, 156)),
            "left_eye": torch.randn((2, 3, 36, 60)),
            "right_eye": torch.randn((2, 3, 36, 60)),
            "table_frame7": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
            ).expand(2, -1),
            "raw_eye_geometry_mm": torch.tensor(
                [[-30.0, -20.0, 600.0, 30.0, -20.0, 600.0]]
            ).expand(2, -1),
            "depth_log_scale_sigma": torch.full((2, 1), 0.05),
            "pnp_geometry_vec": torch.randn((2, 6)),
            "pnp_quality_vec": torch.randn((2, 4)),
            "uv_gt": torch.tensor([[50.0, -20.0], [80.0, 40.0]]),
            "gaze_target_camera_mm": torch.tensor(
                [[50.0, -20.0, 0.0], [80.0, 40.0, 0.0]]
            ),
        }
        output = model(batch)
        criterion = GazeGeometryLoss(
            GazeGeometryLossConfig(
                uv_huber_beta_mm=30.0,
                uv_huber_weight=1.0,
                mixture_nll_weight=0.0,
                mixture_kernel_sigma_mm=30.0,
                gaze_direction_weight=0.0,
                gaze_angular_weight=0.0,
                depth_prior_kl_weight=0.0,
                ray_validity_weight=0.1,
                ray_penalty_scale_mm=100.0,
                depth_correction_prior_weight=0.0,
            )
        )
        loss = criterion(output, batch)["loss"]
        loss.backward()
        gradient_norm = sum(
            float(parameter.grad.norm())
            for parameter in model.depth_correction_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
