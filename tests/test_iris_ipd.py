from __future__ import annotations

import unittest

import numpy as np

from modelv1.depth_prior.iris_ipd import (
    assign_iris_groups_to_anatomical_sides,
    reconstruct_iris_centres_from_ipd,
)


class IrisIpdTests(unittest.TestCase):
    def test_reconstructs_metric_symmetric_iris_points(self) -> None:
        depth_mm = 800.0
        half_ipd_mm = 32.5
        result = reconstruct_iris_centres_from_ipd(
            [-half_ipd_mm / depth_mm, 0.0],
            [half_ipd_mm / depth_mm, 0.0],
            head_left_to_right_direction_flame=[1.0, 0.0, 0.0],
            flame_to_camera_rotation=np.eye(3),
            camera_matrix=np.eye(3),
            dist_coeffs=None,
            ipd_mm=65.0,
        )
        np.testing.assert_allclose(
            result.left_iris_camera_xyz_mm,
            [-half_ipd_mm, 0.0, depth_mm],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.right_iris_camera_xyz_mm,
            [half_ipd_mm, 0.0, depth_mm],
            atol=1e-6,
        )
        self.assertLess(result.baseline_residual_mm, 1e-6)

    def test_assigns_groups_by_anatomical_canthus_projection(self) -> None:
        assigned = assign_iris_groups_to_anatomical_sides(
            {"iris_group_468": [120.0, 50.0], "iris_group_473": [20.0, 50.0]},
            left_canthus_projection_xy=[118.0, 50.0],
            right_canthus_projection_xy=[22.0, 50.0],
        )
        self.assertEqual(assigned["left_group"], "iris_group_468")
        self.assertEqual(assigned["right_group"], "iris_group_473")


if __name__ == "__main__":
    unittest.main()
