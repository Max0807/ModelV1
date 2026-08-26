from __future__ import annotations

import unittest

import numpy as np
import torch

from modelv1.depth_prior.deca_flame import head_local_pose_parameters
from modelv1.depth_prior.pnp import (
    DEFAULT_PNP_MAPPING,
    FLAME_LEFT_EYE_CANTHUS_INDICES,
    FLAME_RIGHT_EYE_CANTHUS_INDICES,
    PNP_GEOMETRY_VERSION,
    PnpCamera,
    PnpConfig,
    _evaluate_pnp_candidate,
    solve_pnp_face_depth,
)
from scripts.generate_virtual_camera_normalized_images import (
    validate_depth_row_for_normalization,
)


try:
    import cv2  # type: ignore

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def synthetic_flame_landmarks() -> np.ndarray:
    landmarks = np.zeros((68, 3), dtype=np.float64)
    values = {
        36: (-45.0, 25.0, 2.0),
        39: (-18.0, 24.0, 7.0),
        42: (18.0, 24.0, 7.0),
        45: (45.0, 25.0, 2.0),
        30: (0.0, 0.0, 35.0),
        48: (-28.0, -35.0, 5.0),
        54: (28.0, -35.0, 5.0),
        8: (0.0, -80.0, -2.0),
    }
    for index, value in values.items():
        landmarks[index] = value
    return landmarks


class PnpSemanticContractTests(unittest.TestCase):
    def test_head_local_flame_pose_removes_global_rotation_only(self) -> None:
        pose = torch.tensor([[0.3, -0.2, 0.1, 0.0, 0.25, -0.05]])
        local = head_local_pose_parameters(pose)
        torch.testing.assert_close(local[:, :3], torch.zeros((1, 3)))
        torch.testing.assert_close(local[:, 3:], pose[:, 3:])
        torch.testing.assert_close(pose[:, :3], torch.tensor([[0.3, -0.2, 0.1]]))

    def test_mapping_preserves_image_side_but_outputs_anatomical_side(self) -> None:
        self.assertEqual(
            [(item.label, item.flame_landmark_index) for item in DEFAULT_PNP_MAPPING],
            [
                ("left_eye_outer", 36),
                ("left_eye_inner", 39),
                ("right_eye_inner", 42),
                ("right_eye_outer", 45),
                ("nose_tip", 30),
                ("mouth_left", 48),
                ("mouth_right", 54),
                ("chin", 8),
            ],
        )
        self.assertEqual(FLAME_LEFT_EYE_CANTHUS_INDICES, (42, 45))
        self.assertEqual(FLAME_RIGHT_EYE_CANTHUS_INDICES, (36, 39))

    def test_virtual_camera_rejects_unversioned_or_behind_camera_pose(self) -> None:
        identity = {
            f"rotation_{row}{column}": str(float(row == column))
            for row in range(3)
            for column in range(3)
        }
        valid = {
            **identity,
            "pnp_geometry_version": PNP_GEOMETRY_VERSION,
            "pnp_min_object_depth_mm": "650",
            "tvec_x_mm": "0",
            "tvec_y_mm": "0",
            "tvec_z_mm": "700",
            "left_eye_camera_z_mm": "695",
            "right_eye_camera_z_mm": "705",
            "reprojection_error_mean_px": "3",
            "reprojection_error_max_px": "6",
            "scale_disagreement_ratio": "0.1",
        }
        validate_depth_row_for_normalization(valid)
        with self.assertRaisesRegex(ValueError, "version_mismatch"):
            validate_depth_row_for_normalization(
                {**valid, "pnp_geometry_version": ""}
            )
        with self.assertRaisesRegex(ValueError, "not_in_front"):
            validate_depth_row_for_normalization(
                {**valid, "pnp_min_object_depth_mm": "-1"}
            )


@unittest.skipUnless(HAS_CV2, "OpenCV is required for solvePnP tests")
class PnpNumericalContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.landmarks = synthetic_flame_landmarks()
        self.camera = PnpCamera.from_focal_length(900.0, 905.0, 640.0, 360.0)

    def test_recovers_head_local_to_camera_transform_in_millimetres(self) -> None:
        rvec_true = np.array([2.9, 0.12, -0.08], dtype=np.float64)
        tvec_true = np.array([25.0, -30.0, 760.0], dtype=np.float64)
        object_points = np.asarray(
            [self.landmarks[item.flame_landmark_index] for item in DEFAULT_PNP_MAPPING]
        )
        image_points = cv2.projectPoints(
            object_points,
            rvec_true,
            tvec_true,
            self.camera.camera_matrix,
            self.camera.dist_coeffs,
        )[0].reshape(-1, 2)
        by_label = {
            item.label: image_points[index]
            for index, item in enumerate(DEFAULT_PNP_MAPPING)
        }

        result = solve_pnp_face_depth(
            by_label,
            self.landmarks,
            self.camera,
            scale_mm_per_flame_unit=1.0,
        )

        self.assertTrue(result.depth_is_plausible)
        self.assertGreater(result.pnp_min_object_depth_mm, 0.0)
        self.assertGreaterEqual(result.pnp_candidate_count, 1)
        self.assertLess(result.reprojection_error_mean_px, 1e-5)
        np.testing.assert_allclose(result.tvec_mm, tvec_true, atol=1e-4)
        expected_rotation = cv2.Rodrigues(rvec_true)[0]
        np.testing.assert_allclose(result.rotation_matrix, expected_rotation, atol=1e-6)
        self.assertEqual(result.as_record()["pnp_geometry_version"], PNP_GEOMETRY_VERSION)

    def test_behind_camera_candidate_is_rejected_even_with_finite_projection(self) -> None:
        object_points = np.asarray(
            [self.landmarks[item.flame_landmark_index] for item in DEFAULT_PNP_MAPPING]
        )
        image_points = np.zeros((len(object_points), 2), dtype=np.float64)
        candidate = _evaluate_pnp_candidate(
            method="ITERATIVE",
            success=True,
            rvec=np.zeros(3),
            tvec=np.array([0.0, 0.0, -700.0]),
            object_points_mm=object_points,
            image_points=image_points,
            camera=self.camera,
            config=PnpConfig(),
        )
        self.assertIsNone(candidate)


if __name__ == "__main__":
    unittest.main()
