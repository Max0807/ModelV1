"""Pure geometry used by the final three-stage TableFrame7 preprocessing.

Coordinate convention:

* ``H`` is the head-local FLAME frame with DECA global rotation removed.
* ``C`` is the calibrated real-camera frame.
* ``N`` is the virtual-camera frame.
* ``R_C_H`` maps head-local vectors to the real camera.
* ``R_N_C`` maps real-camera vectors to the virtual camera.

The only absolute metric source in :func:`build_metric_head_pose` is the
Iris/IPD reconstruction.  PnP translation is deliberately not an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _finite(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """把输入复制为 float64 数组，并严格验证 shape 与有限性。"""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _rotation(value: Any, name: str) -> np.ndarray:
    """验证输入是正交且行列式为 +1 的 3×3 旋转矩阵。"""

    matrix = _finite(value, (3, 3), name)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4):
        raise ValueError(f"{name} must be orthonormal.")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-4):
        raise ValueError(f"{name} must have determinant +1.")
    return matrix


def _unit(value: Any, name: str) -> np.ndarray:
    """验证三维向量有效且非零，并返回其单位方向。"""

    vector = _finite(value, (3,), name)
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        raise ValueError(f"{name} must have non-zero length.")
    return vector / length


@dataclass(frozen=True)
class MetricHeadPose:
    """Metric head origin and rotation in the real camera frame.

    ``head_origin_camera_mm`` is inferred from the observed metric iris
    midpoint and the FLAME-local eye-proxy midpoint.  The FLAME proxy is scaled
    by the known IPD, so no PnP depth or PnP translation is used.
    """

    rotation_camera_from_head: np.ndarray
    head_origin_camera_mm: np.ndarray
    iris_midpoint_camera_mm: np.ndarray
    eye_proxy_midpoint_head_mm: np.ndarray
    flame_scale_mm_per_unit: float


def flame_eye_canthus_midpoints(landmarks3d: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return anatomical left/right canthus midpoints in FLAME-local units."""

    landmarks = np.asarray(landmarks3d, dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[1] != 3 or len(landmarks) <= 45:
        raise ValueError("FLAME landmarks must have shape [N,3] with N > 45.")
    if not np.isfinite(landmarks).all():
        raise ValueError("FLAME landmarks must be finite.")
    left = 0.5 * (landmarks[42] + landmarks[45])
    right = 0.5 * (landmarks[36] + landmarks[39])
    return left, right


def build_metric_head_pose(
    iris_midpoint_camera_mm: Any,
    rotation_camera_from_head: Any,
    left_eye_proxy_head_flame: Any,
    right_eye_proxy_head_flame: Any,
    *,
    ipd_mm: float,
) -> MetricHeadPose:
    """Infer ``t_C_H`` from Iris65 and a FLAME canthus-midpoint proxy.

    FLAME 68 has no anatomical iris vertex.  Therefore each local eye proxy is
    the midpoint of its two canthi.  Their baseline gives the local direction
    and scale reference; its metric length is set to ``ipd_mm``.  This is an
    explicit approximation, and the returned scale should be retained for
    auditing.

    Formula::

        s = IPD_mm / ||right_proxy_H - left_proxy_H||
        iris_proxy_mid_H_mm = s * (left_proxy_H + right_proxy_H) / 2
        t_C_H = iris_mid_C_mm - R_C_H @ iris_proxy_mid_H_mm
    """

    if not np.isfinite(ipd_mm) or ipd_mm <= 0:
        raise ValueError("ipd_mm must be finite and positive.")
    midpoint_camera = _finite(
        iris_midpoint_camera_mm, (3,), "iris_midpoint_camera_mm"
    )
    rotation = _rotation(rotation_camera_from_head, "rotation_camera_from_head")
    left_proxy = _finite(left_eye_proxy_head_flame, (3,), "left_eye_proxy_head_flame")
    right_proxy = _finite(
        right_eye_proxy_head_flame, (3,), "right_eye_proxy_head_flame"
    )
    local_baseline = right_proxy - left_proxy
    local_baseline_length = float(np.linalg.norm(local_baseline))
    if local_baseline_length <= 1e-9:
        raise ValueError("FLAME eye-proxy baseline must have non-zero length.")
    scale = float(ipd_mm / local_baseline_length)
    eye_midpoint_head_mm = 0.5 * (left_proxy + right_proxy) * scale
    head_origin_camera_mm = midpoint_camera - rotation @ eye_midpoint_head_mm
    if head_origin_camera_mm[2] <= 0:
        raise ValueError("Inferred head origin must lie in front of the camera.")
    return MetricHeadPose(
        rotation_camera_from_head=rotation,
        head_origin_camera_mm=head_origin_camera_mm,
        iris_midpoint_camera_mm=midpoint_camera,
        eye_proxy_midpoint_head_mm=eye_midpoint_head_mm,
        flame_scale_mm_per_unit=scale,
    )


def build_table_frame7_n(
    table_u_axis_camera: Any,
    table_v_axis_camera: Any,
    table_distance_camera_mm: float,
    rotation_normalized_from_camera: Any,
) -> np.ndarray:
    """Return ``[e_u_N(3), e_v_N(3), d_C/1000]`` as float32.

    A virtual-camera change of axes rotates the two table basis directions.
    The signed camera-origin-to-plane distance is unchanged and is stored in
    metres, matching ModelV1's current ``table_frame7`` contract.
    """

    if not np.isfinite(table_distance_camera_mm):
        raise ValueError("table_distance_camera_mm must be finite.")
    rotation = _rotation(
        rotation_normalized_from_camera, "rotation_normalized_from_camera"
    )
    u_camera = _unit(table_u_axis_camera, "table_u_axis_camera")
    v_camera = _unit(table_v_axis_camera, "table_v_axis_camera")
    if abs(float(np.dot(u_camera, v_camera))) > 1e-4:
        raise ValueError("table U/V axes must be orthogonal.")
    u_normalized = _unit(rotation @ u_camera, "table_u_axis_normalized")
    v_normalized = _unit(rotation @ v_camera, "table_v_axis_normalized")
    result = np.concatenate(
        (u_normalized, v_normalized, [float(table_distance_camera_mm) / 1000.0])
    ).astype(np.float32)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise RuntimeError("Internal error while constructing table_frame7_n.")
    return result
