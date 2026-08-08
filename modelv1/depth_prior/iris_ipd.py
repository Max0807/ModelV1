"""Metric iris-centre reconstruction from calibrated rays and known IPD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class IrisIpdReconstruction:
    """Metric camera coordinates of anatomical iris-centre proxies."""

    left_iris_camera_xyz_mm: Any
    right_iris_camera_xyz_mm: Any
    midpoint_camera_xyz_mm: Any
    left_ray_distance_mm: float
    right_ray_distance_mm: float
    baseline_residual_mm: float
    ray_system_condition: float

    @property
    def midpoint_depth_z_mm(self) -> float:
        return float(self.midpoint_camera_xyz_mm[2])


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Iris/IPD reconstruction requires NumPy.") from error
    return np


def _dot3(left: Any, right: Any) -> float:
    """Return a 3-vector dot product without dispatching to BLAS/MKL."""
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _norm3(vector: Any) -> float:
    return math.sqrt(_dot3(vector, vector))


def _normalize3(vector: Any, *, name: str, np: Any) -> Any:
    norm = _norm3(vector)
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero.")
    return np.asarray([float(vector[index]) / norm for index in range(3)], dtype=np.float64)


def _invert_3x3(matrix: Any, np: Any) -> Any:
    """Invert a 3x3 matrix explicitly, avoiding NumPy's MKL-backed linalg."""
    values = [[float(matrix[row, column]) for column in range(3)] for row in range(3)]
    a, b, c = values[0]
    d, e, f = values[1]
    g, h, i = values[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) <= 1e-12:
        raise ValueError("camera_matrix must be invertible.")
    inverse = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    return np.asarray(inverse, dtype=np.float64) / determinant


def _matvec3(matrix: Any, vector: Any, np: Any) -> Any:
    """Multiply a 3x3 matrix by a 3-vector without dispatching to MKL."""
    return np.asarray(
        [
            sum(float(matrix[row, column]) * float(vector[column]) for column in range(3))
            for row in range(3)
        ],
        dtype=np.float64,
    )


def assign_iris_groups_to_anatomical_sides(
    iris_groups_xy: Mapping[str, Any],
    *,
    left_canthus_projection_xy: Any,
    right_canthus_projection_xy: Any,
) -> dict[str, Any]:
    """Assign two detector iris groups by the minimum canthus-projection cost."""

    np = _require_numpy()
    if len(iris_groups_xy) != 2:
        raise ValueError("Exactly two iris groups are required.")
    names = tuple(iris_groups_xy)
    points = [np.asarray(iris_groups_xy[name], dtype=np.float64).reshape(2) for name in names]
    left = np.asarray(left_canthus_projection_xy, dtype=np.float64).reshape(2)
    right = np.asarray(right_canthus_projection_xy, dtype=np.float64).reshape(2)
    direct_cost = np.linalg.norm(points[0] - left) + np.linalg.norm(points[1] - right)
    swapped_cost = np.linalg.norm(points[1] - left) + np.linalg.norm(points[0] - right)
    if direct_cost <= swapped_cost:
        return {"left": points[0], "right": points[1], "left_group": names[0], "right_group": names[1], "assignment_cost_px": float(direct_cost)}
    return {"left": points[1], "right": points[0], "left_group": names[1], "right_group": names[0], "assignment_cost_px": float(swapped_cost)}


def reconstruct_iris_centres_from_ipd(
    left_iris_xy: Any,
    right_iris_xy: Any,
    *,
    head_left_to_right_direction_flame: Any,
    flame_to_camera_rotation: Any,
    camera_matrix: Any,
    dist_coeffs: Any,
    ipd_mm: float,
    max_condition: float = 250.0,
    max_baseline_residual_mm: float = 3.0,
) -> IrisIpdReconstruction:
    """Intersect two calibrated iris rays with a known metric IPD baseline.

    The FLAME direction specifies only the *orientation* of the interpupillary
    baseline.  Its length is discarded and replaced by the measured ``ipd_mm``.
    This returns iris/pupil-proxy positions, not anatomical eyeball centres.
    """

    np = _require_numpy()
    if ipd_mm <= 0:
        raise ValueError("ipd_mm must be positive.")
    left_pixel = np.asarray(left_iris_xy, dtype=np.float64).reshape(1, 1, 2)
    right_pixel = np.asarray(right_iris_xy, dtype=np.float64).reshape(1, 1, 2)
    camera = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
    pixels = np.concatenate((left_pixel, right_pixel), axis=0)
    if distortion is None or not np.any(distortion):
        homogeneous = np.concatenate(
            (pixels.reshape(2, 2), np.ones((2, 1), dtype=np.float64)), axis=1
        )
        inverse_camera = _invert_3x3(camera, np)
        normalized_homogeneous = np.asarray(
            [_matvec3(inverse_camera, pixel, np) for pixel in homogeneous],
            dtype=np.float64,
        )
        normalized = normalized_homogeneous[:, :2] / normalized_homogeneous[:, 2:]
    else:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "OpenCV is required to undistort iris points when dist_coeffs is non-zero."
            ) from error
        normalized = cv2.undistortPoints(pixels, camera, distortion).reshape(2, 2)
    rays = np.concatenate((normalized, np.ones((2, 1), dtype=np.float64)), axis=1)
    left_ray = _normalize3(rays[0], name="left iris ray", np=np)
    right_ray = _normalize3(rays[1], name="right iris ray", np=np)

    local_direction = np.asarray(head_left_to_right_direction_flame, dtype=np.float64).reshape(3)
    local_direction = _normalize3(local_direction, name="head_left_to_right_direction_flame", np=np)
    rotation = np.asarray(flame_to_camera_rotation, dtype=np.float64).reshape(3, 3)
    baseline = _matvec3(rotation, local_direction * float(ipd_mm), np)

    # lambda_R * r_R - lambda_L * r_L = baseline
    ray_dot = _dot3(left_ray, right_ray)
    denominator = 1.0 - ray_dot * ray_dot
    if denominator <= 1e-12:
        condition = float("inf")
    else:
        # The two singular values of [-left_ray, right_ray] are
        # sqrt(1 +/- abs(ray_dot)); this closed form avoids MKL's SVD.
        condition = math.sqrt((1.0 + abs(ray_dot)) / (1.0 - abs(ray_dot)))
    if not np.isfinite(condition) or condition > max_condition:
        raise ValueError(f"Iris-ray system is ill-conditioned: condition={condition:.1f}")
    # ([ -l, r ]^T [ -l, r ])^-1 [ -l, r ]^T baseline, written explicitly.
    left_rhs = -_dot3(left_ray, baseline)
    right_rhs = _dot3(right_ray, baseline)
    left_distance = (left_rhs + ray_dot * right_rhs) / denominator
    right_distance = (ray_dot * left_rhs + right_rhs) / denominator
    if left_distance <= 0 or right_distance <= 0:
        raise ValueError("Iris rays intersect behind the camera.")
    left_point = left_distance * left_ray
    right_point = right_distance * right_ray
    residual = _norm3((right_point - left_point) - baseline)
    if residual > max_baseline_residual_mm:
        raise ValueError(
            f"Iris/IPD baseline residual {residual:.2f}mm exceeds {max_baseline_residual_mm:.2f}mm."
        )
    return IrisIpdReconstruction(
        left_iris_camera_xyz_mm=left_point,
        right_iris_camera_xyz_mm=right_point,
        midpoint_camera_xyz_mm=0.5 * (left_point + right_point),
        left_ray_distance_mm=left_distance,
        right_ray_distance_mm=right_distance,
        baseline_residual_mm=residual,
        ray_system_condition=condition,
    )
