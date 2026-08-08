from __future__ import annotations

import unittest

import numpy as np

from modelv1.depth_prior.pnp import compute_eye_canthus_midpoints
from scripts.build_modelv1_dataset import (
    corrected_gaze_camera_fields,
    pnp_fields,
)


class DatasetGeometryRepairTests(unittest.TestCase):
    def test_corrected_gaze_fields_preserve_raw_source_for_audit(self) -> None:
        fields = corrected_gaze_camera_fields(
            [267.0, 36.0, 82.0],
            [236.0, 100.0, 123.0],
        )
        self.assertEqual(fields["gaze_cam_csv_x_mm"], 267.0)
        self.assertEqual(fields["gaze_cam_error_mm"], 0.0)
        self.assertEqual(fields["gaze_cam_csv_raw_x_mm"], 236.0)
        self.assertGreater(float(fields["gaze_cam_raw_error_mm"]), 0.0)

    def test_legacy_pnp_import_swaps_image_sides_to_anatomical_sides(self) -> None:
        fields = pnp_fields(
            {
                "status": "success",
                "left_eye_camera_x_mm": "-40",
                "left_eye_camera_y_mm": "1",
                "left_eye_camera_z_mm": "600",
                "right_eye_camera_x_mm": "40",
                "right_eye_camera_y_mm": "2",
                "right_eye_camera_z_mm": "602",
            }
        )
        self.assertEqual(fields["left_eye_camera_x_mm"], 40.0)
        self.assertEqual(fields["right_eye_camera_x_mm"], -40.0)

    def test_generated_pnp_midpoints_follow_anatomical_eye_sides(self) -> None:
        landmarks = np.zeros((68, 3), dtype=np.float64)
        landmarks[36] = [-2.0, 0.0, 0.0]
        landmarks[39] = [-1.0, 0.0, 0.0]
        landmarks[42] = [1.0, 0.0, 0.0]
        landmarks[45] = [2.0, 0.0, 0.0]

        left, right = compute_eye_canthus_midpoints(landmarks)

        np.testing.assert_allclose(left, [1.5, 0.0, 0.0])
        np.testing.assert_allclose(right, [-1.5, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
