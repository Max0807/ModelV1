"""Shared virtual-camera normalization for gaze images.

This module is deliberately independent from the current Dataset and model code.
It implements the image-space normalization described by Sugano et al./Zhang et
al. style gaze pipelines while keeping direction transforms explicit:

* image points use ``H = K_N S R_NC K_C^-1``;
* gaze directions use rotation only, ``g_N = R_NC g_C``.

Coordinate convention
---------------------
``R_C_H`` maps a vector from the head/PnP frame H to the real camera frame C.
``R_N_C`` maps a vector from the real camera frame C to the virtual camera N.
OpenCV camera axes are assumed: +x right, +y down, +z forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class VirtualCameraDependencyError(RuntimeError):
    """Raised when an optional image-processing dependency is unavailable."""


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise VirtualCameraDependencyError(
            "Virtual-camera image warping requires OpenCV. Install opencv-python "
            "in the environment used for offline preprocessing."
        ) from exc
    return cv2


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _normalize(vector: np.ndarray, name: str, epsilon: float = 1e-9) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= epsilon:
        raise ValueError(f"{name} has near-zero length.")
    return vector / norm


@dataclass(frozen=True)
class CameraCalibration:
    """Real-camera pinhole intrinsics and OpenCV distortion coefficients."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray

    def __post_init__(self) -> None:
        matrix = _finite_array(self.camera_matrix, (3, 3), "camera_matrix")
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64).reshape(-1)
        if distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError(
                "distortion_coefficients must contain 4, 5, 8, 12, or 14 values."
            )
        if not np.isfinite(distortion).all():
            raise ValueError("distortion_coefficients must contain only finite values.")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive.")
        if abs(float(matrix[2, 2])) <= 1e-12:
            raise ValueError("camera_matrix[2, 2] must be non-zero.")
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion.copy())


@dataclass(frozen=True)
class VirtualCameraConfig:
    """Fixed shared virtual-camera parameters used for every sample."""

    output_width: int = 224
    output_height: int = 224
    focal_length_px: float = 480.0
    distance_mm: float = 600.0

    def __post_init__(self) -> None:
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("Virtual-camera output dimensions must be positive.")
        if not np.isfinite(self.focal_length_px) or self.focal_length_px <= 0:
            raise ValueError("focal_length_px must be finite and positive.")
        if not np.isfinite(self.distance_mm) or self.distance_mm <= 0:
            raise ValueError("distance_mm must be finite and positive.")

    @property
    def camera_matrix(self) -> np.ndarray:
        """Return ``K_N`` with the principal point at the output image center."""

        return np.array(
            [
                [self.focal_length_px, 0.0, self.output_width / 2.0],
                [0.0, self.focal_length_px, self.output_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class EyeCropConfig:
    """Axis-aligned eye crop taken from the shared normalized face image."""

    output_width: int = 60
    output_height: int = 36
    horizontal_span_scale: float = 1.8
    minimum_source_width_px: float = 4.0

    def __post_init__(self) -> None:
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("Eye output dimensions must be positive.")
        if not np.isfinite(self.horizontal_span_scale) or self.horizontal_span_scale <= 1:
            raise ValueError("horizontal_span_scale must be finite and greater than 1.")
        if not np.isfinite(self.minimum_source_width_px) or self.minimum_source_width_px <= 0:
            raise ValueError("minimum_source_width_px must be finite and positive.")


@dataclass(frozen=True)
class EyeCropResult:
    image: np.ndarray
    source_center_xy: np.ndarray
    source_size_wh: np.ndarray
    affine_output_from_source: np.ndarray


@dataclass(frozen=True)
class SharedNormalizedInputs:
    face_image: np.ndarray
    left_eye: EyeCropResult
    right_eye: EyeCropResult
    left_eye_corners_normalized: np.ndarray
    right_eye_corners_normalized: np.ndarray


@dataclass(frozen=True)
class VirtualCameraTransform:
    """All deterministic transforms for one image/sample."""

    rotation_normalized_from_camera: np.ndarray
    head_rotation_normalized_from_head: np.ndarray
    homography_normalized_from_undistorted: np.ndarray
    face_center_camera_mm: np.ndarray
    distance_scale: float
    config: VirtualCameraConfig

    def __post_init__(self) -> None:
        rotation = _finite_array(
            self.rotation_normalized_from_camera,
            (3, 3),
            "rotation_normalized_from_camera",
        )
        head_rotation = _finite_array(
            self.head_rotation_normalized_from_head,
            (3, 3),
            "head_rotation_normalized_from_head",
        )
        homography = _finite_array(
            self.homography_normalized_from_undistorted,
            (3, 3),
            "homography_normalized_from_undistorted",
        )
        center = _finite_array(self.face_center_camera_mm, (3,), "face_center_camera_mm")
        if not np.isfinite(self.distance_scale) or self.distance_scale <= 0:
            raise ValueError("distance_scale must be finite and positive.")
        object.__setattr__(self, "rotation_normalized_from_camera", rotation)
        object.__setattr__(self, "head_rotation_normalized_from_head", head_rotation)
        object.__setattr__(self, "homography_normalized_from_undistorted", homography)
        object.__setattr__(self, "face_center_camera_mm", center)

    def direction_camera_to_normalized(self, direction_camera: Any) -> np.ndarray:
        """Rotate C-frame direction(s) to N; distance scaling is not applied."""

        return self._rotate_directions(
            direction_camera,
            self.rotation_normalized_from_camera,
            "direction_camera",
        )

    def direction_normalized_to_camera(self, direction_normalized: Any) -> np.ndarray:
        """Rotate N-frame direction(s) back to C; distance scaling is not applied."""

        return self._rotate_directions(
            direction_normalized,
            self.rotation_normalized_from_camera.T,
            "direction_normalized",
        )

    @staticmethod
    def _rotate_directions(value: Any, rotation: np.ndarray, name: str) -> np.ndarray:
        directions = np.asarray(value, dtype=np.float64)
        if directions.ndim == 1:
            if directions.shape != (3,):
                raise ValueError(f"{name} must have shape (3,) or (..., 3).")
            return rotation @ directions
        if directions.ndim < 2 or directions.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (3,) or (..., 3).")
        return directions @ rotation.T

    def transform_undistorted_pixels(self, pixels_xy: Any) -> np.ndarray:
        """Apply the normalization homography to undistorted source pixels."""

        pixels = np.asarray(pixels_xy, dtype=np.float64)
        if pixels.ndim == 1:
            if pixels.shape != (2,):
                raise ValueError("pixels_xy must have shape (2,) or (..., 2).")
            flat = pixels.reshape(1, 2)
            output_shape: tuple[int, ...] = (2,)
        elif pixels.ndim >= 2 and pixels.shape[-1] == 2:
            flat = pixels.reshape(-1, 2)
            output_shape = pixels.shape
        else:
            raise ValueError("pixels_xy must have shape (2,) or (..., 2).")
        if not np.isfinite(flat).all():
            raise ValueError("pixels_xy must contain only finite values.")
        homogeneous = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
        mapped = homogeneous @ self.homography_normalized_from_undistorted.T
        if np.any(np.abs(mapped[:, 2]) <= 1e-12):
            raise ValueError("Homography mapped a pixel to infinity.")
        mapped_xy = mapped[:, :2] / mapped[:, 2:3]
        return mapped_xy.reshape(output_shape)

    def transform_distorted_pixels(
        self,
        pixels_xy: Any,
        calibration: CameraCalibration,
    ) -> np.ndarray:
        """Undistort source pixels and map them into the normalized image."""

        cv2 = _require_cv2()
        pixels = np.asarray(pixels_xy, dtype=np.float64)
        if pixels.ndim == 1:
            if pixels.shape != (2,):
                raise ValueError("pixels_xy must have shape (2,) or (..., 2).")
            output_shape: tuple[int, ...] = (2,)
            flat = pixels.reshape(-1, 2)
        elif pixels.ndim >= 2 and pixels.shape[-1] == 2:
            output_shape = pixels.shape
            flat = pixels.reshape(-1, 2)
        else:
            raise ValueError("pixels_xy must have shape (2,) or (..., 2).")
        if not np.isfinite(flat).all():
            raise ValueError("pixels_xy must contain only finite values.")
        undistorted = cv2.undistortPoints(
            flat.reshape(-1, 1, 2),
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            P=calibration.camera_matrix,
        ).reshape(-1, 2)
        return self.transform_undistorted_pixels(undistorted).reshape(output_shape)

    def render(
        self,
        source_image: np.ndarray,
        calibration: CameraCalibration,
        interpolation: int | None = None,
    ) -> np.ndarray:
        """Undistort the full source image, then warp it into the shared camera."""

        cv2 = _require_cv2()
        image = np.asarray(source_image)
        if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("source_image must be a non-empty HxW or HxWxC array.")
        interpolation_flag = cv2.INTER_LINEAR if interpolation is None else interpolation
        undistorted = cv2.undistort(
            image,
            calibration.camera_matrix,
            calibration.distortion_coefficients,
        )
        return cv2.warpPerspective(
            undistorted,
            self.homography_normalized_from_undistorted,
            (self.config.output_width, self.config.output_height),
            flags=interpolation_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def as_record(self) -> dict[str, float | int]:
        """Flatten transform/config values for an auditable CSV manifest."""

        record: dict[str, float | int] = {
            "virtual_width": self.config.output_width,
            "virtual_height": self.config.output_height,
            "virtual_focal_length_px": self.config.focal_length_px,
            "virtual_distance_mm": self.config.distance_mm,
            "virtual_distance_scale": self.distance_scale,
        }
        for index, value in enumerate(self.face_center_camera_mm):
            record[f"face_center_camera_{'xyz'[index]}_mm"] = float(value)
        for prefix, matrix in (
            ("rotation_n_from_c", self.rotation_normalized_from_camera),
            ("head_rotation_n_from_h", self.head_rotation_normalized_from_head),
            ("homography_n_from_undistorted", self.homography_normalized_from_undistorted),
        ):
            for row in range(3):
                for column in range(3):
                    record[f"{prefix}_{row}{column}"] = float(matrix[row, column])
        return record


def build_virtual_camera_transform(
    calibration: CameraCalibration,
    head_rotation_camera_from_head: Any,
    face_center_camera_mm: Any,
    config: VirtualCameraConfig | None = None,
) -> VirtualCameraTransform:
    """Build one shared virtual camera centered on the PnP face center.

    The virtual z axis points from the real camera origin to the face center. Its
    x axis follows the projected head x axis, which removes camera-induced roll
    while retaining head yaw/pitch in ``R_N_H``.
    """

    config = config or VirtualCameraConfig()
    rotation_camera_from_head = _finite_array(
        head_rotation_camera_from_head,
        (3, 3),
        "head_rotation_camera_from_head",
    )
    orthogonality = rotation_camera_from_head.T @ rotation_camera_from_head
    if not np.allclose(orthogonality, np.eye(3), atol=1e-4):
        raise ValueError("head_rotation_camera_from_head must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation_camera_from_head), 1.0, atol=1e-4):
        raise ValueError("head_rotation_camera_from_head must have determinant +1.")

    center_camera = _finite_array(face_center_camera_mm, (3,), "face_center_camera_mm")
    center_distance = float(np.linalg.norm(center_camera))
    if center_distance <= 1e-6:
        raise ValueError("face_center_camera_mm must not coincide with the camera origin.")
    if center_camera[2] <= 1e-6:
        raise ValueError("face_center_camera_mm must lie in front of the camera (+z).")

    z_axis_camera = center_camera / center_distance
    head_x_axis_camera = rotation_camera_from_head[:, 0]  # 头部局部坐标系 H 的 +X 轴，在相机坐标系 C 中的方向。
    y_axis_camera_raw = np.cross(z_axis_camera, head_x_axis_camera)
    if np.linalg.norm(y_axis_camera_raw) <= 1e-7:
        # Only relevant for an extreme/degenerate head pose. Choose the camera
        # basis least parallel to z so the transform stays well-defined.
        candidates = np.eye(3)
        fallback_x = candidates[np.argmin(np.abs(candidates @ z_axis_camera))]
        y_axis_camera_raw = np.cross(z_axis_camera, fallback_x)
    y_axis_camera = _normalize(y_axis_camera_raw, "virtual y axis")
    x_axis_camera = _normalize(
        np.cross(y_axis_camera, z_axis_camera),
        "virtual x axis",
    )
    rotation_normalized_from_camera = np.stack(
        [x_axis_camera, y_axis_camera, z_axis_camera],
        axis=0,
    )

    distance_scale = config.distance_mm / center_distance
    scale = np.diag([1.0, 1.0, distance_scale])
    homography = (
        config.camera_matrix
        @ scale
        @ rotation_normalized_from_camera
        @ np.linalg.inv(calibration.camera_matrix)
    )
    homography /= homography[2, 2]
    head_rotation_normalized_from_head = (
        rotation_normalized_from_camera @ rotation_camera_from_head
    )
    return VirtualCameraTransform(
        rotation_normalized_from_camera=rotation_normalized_from_camera,
        head_rotation_normalized_from_head=head_rotation_normalized_from_head,
        homography_normalized_from_undistorted=homography,
        face_center_camera_mm=center_camera,
        distance_scale=distance_scale,
        config=config,
    )


def crop_normalized_eye(
    normalized_face_image: np.ndarray,
    eye_corners_normalized_xy: Any,
    config: EyeCropConfig | None = None,
) -> EyeCropResult:
    """Crop an eye from the shared normalized face using its two canthi."""

    cv2 = _require_cv2()
    config = config or EyeCropConfig()
    image = np.asarray(normalized_face_image)
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("normalized_face_image must be a non-empty image array.")
    corners = _finite_array(eye_corners_normalized_xy, (2, 2), "eye_corners_normalized_xy")
    center = corners.mean(axis=0)
    corner_distance = float(np.linalg.norm(corners[1] - corners[0]))
    source_width = max(
        corner_distance * config.horizontal_span_scale,
        config.minimum_source_width_px,
    )
    source_height = source_width * config.output_height / config.output_width
    source_left = center[0] - source_width / 2.0
    source_top = center[1] - source_height / 2.0
    scale_x = config.output_width / source_width
    scale_y = config.output_height / source_height
    affine = np.array(
        [
            [scale_x, 0.0, -scale_x * source_left],
            [0.0, scale_y, -scale_y * source_top],
        ],
        dtype=np.float64,
    )
    patch = cv2.warpAffine(
        image,
        affine,
        (config.output_width, config.output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return EyeCropResult(
        image=patch,
        source_center_xy=center,
        source_size_wh=np.array([source_width, source_height], dtype=np.float64),
        affine_output_from_source=affine,
    )


def render_shared_normalized_inputs(
    source_image: np.ndarray,
    calibration: CameraCalibration,
    transform: VirtualCameraTransform,
    left_eye_corners_source_xy: Any,
    right_eye_corners_source_xy: Any,
    eye_crop_config: EyeCropConfig | None = None,
) -> SharedNormalizedInputs:
    """Render one normalized face and derive both eyes in the same N frame.

    ``left_eye_corners_source_xy`` and ``right_eye_corners_source_xy`` are
    expected to use anatomical subject-left/subject-right semantics.  This
    function never swaps sides and never mirrors either eye patch.
    """

    face = transform.render(source_image, calibration)
    left_corners = transform.transform_distorted_pixels(
        left_eye_corners_source_xy,
        calibration,
    )
    right_corners = transform.transform_distorted_pixels(
        right_eye_corners_source_xy,
        calibration,
    )
    return SharedNormalizedInputs(
        face_image=face,
        left_eye=crop_normalized_eye(face, left_corners, eye_crop_config),
        right_eye=crop_normalized_eye(face, right_corners, eye_crop_config),
        left_eye_corners_normalized=left_corners,
        right_eye_corners_normalized=right_corners,
    )
