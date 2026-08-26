from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from modelv1.depth_prior.pnp import compute_eye_canthus_midpoints
from scripts.build_modelv1_dataset import (
    DatasetBuildError,
    build_vicon_aligned_table_scene,
    choose_data_log_csv,
    discover_dataset_dirs,
    index_by,
    pnp_fields,
    raw_gaze_camera_audit_fields,
    resolve_table_z_mm,
)


class DatasetGeometryRepairTests(unittest.TestCase):
    def test_raw_gaze_fields_preserve_source_for_audit(self) -> None:
        fields = raw_gaze_camera_audit_fields(
            [267.0, 36.0, 82.0],
            [236.0, 100.0, 123.0],
        )
        self.assertNotIn("gaze_cam_csv_x_mm", fields)
        self.assertNotIn("gaze_cam_error_mm", fields)
        self.assertEqual(fields["gaze_cam_csv_raw_x_mm"], 236.0)
        self.assertGreater(float(fields["gaze_cam_raw_error_mm"]), 0.0)

    def test_vicon_aligned_table_scene_uses_camera_projection_as_uv_origin(self) -> None:
        scene, meta = build_vicon_aligned_table_scene(
            np.eye(3).tolist(),
            [-10.0, -20.0, -30.0],
            [10.0, 20.0, 30.0],
            4.0,
        )
        self.assertEqual(len(scene), 25)
        np.testing.assert_allclose(meta["o_table_w"], [10.0, 20.0, 4.0])
        np.testing.assert_allclose(meta["e1_w"], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(meta["e2_w"], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(meta["n_w"], [0.0, 0.0, 1.0])

        gaze_w = np.asarray([13.0, 18.0, 4.5])
        offset = gaze_w - np.asarray(meta["o_table_w"])
        uv = [
            np.dot(meta["e1_w"], offset),
            np.dot(meta["e2_w"], offset),
        ]
        np.testing.assert_allclose(uv, [3.0, -2.0])

    def test_duplicate_join_key_is_rejected(self) -> None:
        rows = [{"image_name": "same.jpg"}, {"image_name": "same.jpg"}]
        with self.assertRaises(DatasetBuildError):
            index_by(rows, "image_name", label="test rows")

    def test_missing_requested_dataset_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "dataset_dual_rigid_body_13").mkdir()
            with self.assertRaises(DatasetBuildError):
                discover_dataset_dirs(
                    root, {"dataset_dual_rigid_body_4"}
                )

    def test_multiple_data_logs_are_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            (dataset_dir / "data_log_1.csv").touch()
            (dataset_dir / "data_log_2.csv").touch()
            with self.assertRaises(DatasetBuildError):
                choose_data_log_csv(dataset_dir)

    def test_per_sample_table_z_uses_current_target_height(self) -> None:
        self.assertEqual(
            resolve_table_z_mm([1.0, 2.0, 3.5], 99.0, "per-sample-target"),
            3.5,
        )

    def test_per_dataset_table_z_uses_legacy_mean(self) -> None:
        self.assertEqual(
            resolve_table_z_mm([1.0, 2.0, 3.5], 4.25, "per-dataset-mean"),
            4.25,
        )

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
