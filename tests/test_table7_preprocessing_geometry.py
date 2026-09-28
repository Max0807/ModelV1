from __future__ import annotations

import importlib.util

import unittest
from pathlib import Path

import numpy as np

from modelv1.preprocessing import (
    build_metric_head_pose,
    build_table_frame7_n,
    flame_eye_canthus_midpoints,
    normalize_flame_object_points,
)

from modelv1.preprocessing.camera import load_camera_calibration
from scripts.table7_preprocessing_config import (
    DEFAULT_CONFIG_PATH,
    load_table7_preprocessing_config,
)


@unittest.skipIf(importlib.util.find_spec("yaml") is None, "PyYAML is not installed")
class Table7PreprocessingConfigTest(unittest.TestCase):
    def test_default_config_loads_camera_calibration_file(self) -> None:
        config = load_table7_preprocessing_config(DEFAULT_CONFIG_PATH)
        self.assertTrue(config.camera_calibration_file.is_file())
        self.assertEqual(
            config.camera_calibration_file,
            Path("configs/camera/crossgaze_camera.yaml").resolve(),
        )
        camera = load_camera_calibration(config.camera_calibration_file)
        self.assertEqual((camera.image_width_px, camera.image_height_px), (1920, 1080))
        self.assertEqual(len(camera.distortion_coefficients), 5)


class MetricHeadPoseTest(unittest.TestCase):
    def test_translation_uses_iris_midpoint_and_local_offset(self) -> None:
        pose = build_metric_head_pose(
            [10.0, 20.0, 600.0],
            np.eye(3),
            [-0.5, 0.0, 0.2],
            [0.5, 0.0, 0.2],
            ipd_mm=65.0,
        )
        np.testing.assert_allclose(
            pose.eye_proxy_midpoint_head_mm, [0.0, 0.0, 13.0]
        )
        np.testing.assert_allclose(
            pose.head_origin_camera_mm, [10.0, 20.0, 587.0]
        )
        self.assertAlmostEqual(pose.flame_scale_mm_per_unit, 65.0)

    def test_pnp_translation_is_not_an_argument(self) -> None:
        pose = build_metric_head_pose(
            [0.0, 0.0, 600.0],
            np.eye(3),
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            ipd_mm=65.0,
        )
        np.testing.assert_allclose(
            pose.head_origin_camera_mm, [0.0, 0.0, 600.0]
        )


class TableFrame7Test(unittest.TestCase):
    def test_identity_virtual_rotation(self) -> None:
        result = build_table_frame7_n(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            750.0,
            np.eye(3),
        )
        np.testing.assert_allclose(
            result, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75]
        )
        self.assertEqual(result.dtype, np.float32)

    def test_axes_rotate_from_camera_to_normalized(self) -> None:
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        result = build_table_frame7_n(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            500.0,
            rotation,
        )
        np.testing.assert_allclose(
            result, [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.5]
        )


class RotationOnlyPnpGeometryTest(unittest.TestCase):
    def test_object_normalization_removes_uniform_scale_and_translation(self) -> None:
        points = np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.5],
                [0.0, -1.0, -0.5],
            ]
        )
        normalized, _, _ = normalize_flame_object_points(points)
        transformed, _, _ = normalize_flame_object_points(
            points * 1234.5 + np.asarray([9.0, -3.0, 7.0])
        )
        np.testing.assert_allclose(normalized, transformed, atol=1e-12)

    def test_flame_eye_proxy_uses_anatomical_canthus_pairs(self) -> None:
        landmarks = np.zeros((68, 3), dtype=np.float64)
        landmarks[42] = [1.0, 2.0, 3.0]
        landmarks[45] = [3.0, 4.0, 5.0]
        landmarks[36] = [-3.0, 1.0, 2.0]
        landmarks[39] = [-1.0, 3.0, 4.0]
        left, right = flame_eye_canthus_midpoints(landmarks)
        np.testing.assert_allclose(left, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(right, [-2.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()




