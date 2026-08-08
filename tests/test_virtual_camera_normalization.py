from __future__ import annotations

import unittest

import numpy as np

from modelv1.data.virtual_camera import (
    CameraCalibration,
    EyeCropConfig,
    VirtualCameraConfig,
    build_virtual_camera_transform,
    crop_normalized_eye,
    render_shared_normalized_inputs,
)
from scripts.generate_virtual_camera_normalized_images import (
    eye_corners_from_landmark_row,
)


try:
    import cv2  # type: ignore  # noqa: F401

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class VirtualCameraGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.calibration = CameraCalibration(
            camera_matrix=self.camera_matrix,
            distortion_coefficients=np.zeros(5, dtype=np.float64),
        )
        self.config = VirtualCameraConfig(
            output_width=224,
            output_height=224,
            focal_length_px=480.0,
            distance_mm=600.0,
        )

    def build(self):
        return build_virtual_camera_transform(
            calibration=self.calibration,
            head_rotation_camera_from_head=np.eye(3),
            face_center_camera_mm=np.array([40.0, -30.0, 800.0]),
            config=self.config,
        )

    def test_virtual_rotation_is_proper_and_center_maps_to_image_center(self) -> None:
        transform = self.build()
        rotation = transform.rotation_normalized_from_camera
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=10)

        center = transform.face_center_camera_mm
        source_pixel_h = self.camera_matrix @ center
        source_pixel = source_pixel_h[:2] / source_pixel_h[2]
        mapped = transform.transform_undistorted_pixels(source_pixel)
        np.testing.assert_allclose(mapped, np.array([112.0, 112.0]), atol=1e-8)

    def test_direction_round_trip_uses_rotation_only(self) -> None:
        transform = self.build()
        directions_camera = np.array(
            [[0.2, -0.1, 0.97], [-0.4, 0.3, 0.85]], dtype=np.float64
        )
        directions_normalized = transform.direction_camera_to_normalized(
            directions_camera
        )
        recovered = transform.direction_normalized_to_camera(directions_normalized)
        np.testing.assert_allclose(recovered, directions_camera, atol=1e-10)
        np.testing.assert_allclose(
            np.linalg.norm(directions_normalized, axis=1),
            np.linalg.norm(directions_camera, axis=1),
            atol=1e-10,
        )
        self.assertNotEqual(transform.distance_scale, 1.0)

    def test_normalized_head_pose_is_composed_in_the_documented_order(self) -> None:
        angle = 0.3
        rotation_camera_from_head = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        transform = build_virtual_camera_transform(
            calibration=self.calibration,
            head_rotation_camera_from_head=rotation_camera_from_head,
            face_center_camera_mm=np.array([0.0, 0.0, 800.0]),
            config=self.config,
        )
        np.testing.assert_allclose(
            transform.head_rotation_normalized_from_head,
            transform.rotation_normalized_from_camera @ rotation_camera_from_head,
            atol=1e-10,
        )
        # Looking straight at a centered face produces the real-camera frame as N.
        np.testing.assert_allclose(
            transform.rotation_normalized_from_camera,
            np.eye(3),
            atol=1e-10,
        )

    def test_invalid_pose_and_face_center_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            build_virtual_camera_transform(
                self.calibration,
                np.diag([2.0, 1.0, 1.0]),
                np.array([0.0, 0.0, 800.0]),
                self.config,
            )
        with self.assertRaisesRegex(ValueError, "in front"):
            build_virtual_camera_transform(
                self.calibration,
                np.eye(3),
                np.array([0.0, 0.0, -800.0]),
                self.config,
            )

    def test_manifest_record_contains_all_three_matrices(self) -> None:
        record = self.build().as_record()
        for prefix in (
            "rotation_n_from_c",
            "head_rotation_n_from_h",
            "homography_n_from_undistorted",
        ):
            self.assertEqual(
                len([key for key in record if key.startswith(prefix + "_")]),
                9,
            )

    def test_legacy_image_side_eye_columns_are_swapped_to_anatomical_sides(self) -> None:
        row = {
            # Legacy ``left`` is image-left and therefore anatomical right.
            "left_eye_outer_x": "33",
            "left_eye_outer_y": "1033",
            "left_eye_inner_x": "133",
            "left_eye_inner_y": "1133",
            # Legacy ``right`` is image-right and therefore anatomical left.
            "right_eye_inner_x": "362",
            "right_eye_inner_y": "1362",
            "right_eye_outer_x": "263",
            "right_eye_outer_y": "1263",
        }
        anatomical_left, anatomical_right = eye_corners_from_landmark_row(row)
        np.testing.assert_allclose(
            anatomical_left,
            [[263.0, 1263.0], [362.0, 1362.0]],
        )
        np.testing.assert_allclose(
            anatomical_right,
            [[33.0, 1033.0], [133.0, 1133.0]],
        )


@unittest.skipUnless(HAS_CV2, "OpenCV is needed for image-warp tests")
class VirtualCameraImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = VirtualCameraConfig(
            output_width=64,
            output_height=64,
            focal_length_px=80.0,
            distance_mm=600.0,
        )
        self.calibration = CameraCalibration(
            camera_matrix=self.config.camera_matrix,
            distortion_coefficients=np.zeros(5),
        )
        self.transform = build_virtual_camera_transform(
            self.calibration,
            np.eye(3),
            np.array([0.0, 0.0, 600.0]),
            self.config,
        )

    def test_identity_case_render_and_pixel_mapping(self) -> None:
        y, x = np.mgrid[0:64, 0:64]
        image = np.stack([x, y, (x + y) // 2], axis=-1).astype(np.uint8)
        rendered = self.transform.render(image, self.calibration)
        self.assertEqual(rendered.shape, image.shape)
        np.testing.assert_allclose(rendered, image, atol=1)

        points = np.array([[20.0, 30.0], [40.0, 30.0]])
        mapped = self.transform.transform_distorted_pixels(points, self.calibration)
        np.testing.assert_allclose(mapped, points, atol=1e-8)

    def test_eye_crop_has_requested_shape_and_auditable_box(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[25:36, 15:50] = 255
        crop = crop_normalized_eye(
            image,
            np.array([[20.0, 30.0], [40.0, 30.0]]),
            EyeCropConfig(output_width=60, output_height=36, horizontal_span_scale=1.8),
        )
        self.assertEqual(crop.image.shape, (36, 60, 3))
        np.testing.assert_allclose(crop.source_center_xy, [30.0, 30.0])
        np.testing.assert_allclose(crop.source_size_wh, [36.0, 21.6])
        self.assertGreater(int(crop.image.max()), 0)

    def test_shared_renderer_returns_one_face_and_two_eye_patches(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = render_shared_normalized_inputs(
            source_image=image,
            calibration=self.calibration,
            transform=self.transform,
            left_eye_corners_source_xy=np.array([[14.0, 26.0], [26.0, 26.0]]),
            right_eye_corners_source_xy=np.array([[38.0, 26.0], [50.0, 26.0]]),
            eye_crop_config=EyeCropConfig(),
        )
        self.assertEqual(result.face_image.shape, (64, 64, 3))
        self.assertEqual(result.left_eye.image.shape, (36, 60, 3))
        self.assertEqual(result.right_eye.image.shape, (36, 60, 3))


if __name__ == "__main__":
    unittest.main()
