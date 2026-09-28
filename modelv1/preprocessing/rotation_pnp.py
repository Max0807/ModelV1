"""Scale-free PnP used only to estimate the head rotation ``R_C_H``.

The selected FLAME points are centred and divided by their RMS radius before
OpenCV sees them.  Consequently the internal PnP translation is expressed in
normalised object units and has no metric meaning.  Candidate validation uses
only cheirality and image reprojection error; it contains no absolute-depth
threshold and no FLAME-to-millimetre scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


HEAD_ROTATION_PNP_VERSION = "mediapipe_flame_rotation_only_normalized_v1"

MEDIAPIPE_ROTATION_LANDMARK_INDICES = {
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "nose_tip": 1,
    "mouth_left": 61,
    "mouth_right": 291,
    "chin": 152,
}

FLAME_ROTATION_LANDMARK_INDICES = {
    "left_eye_outer": 36,
    "left_eye_inner": 39,
    "right_eye_inner": 42,
    "right_eye_outer": 45,
    "nose_tip": 30,
    "mouth_left": 48,
    "mouth_right": 54,
    "chin": 8,
}


@dataclass(frozen=True)
class HeadRotationPnpConfig:
    max_reprojection_error_mean_px: float = 8.0
    max_reprojection_error_max_px: float = 20.0
    inlier_reprojection_error_px: float = 10.0
    require_all_points_in_front: bool = True

    def __post_init__(self) -> None:
        """验证 rotation-only PnP 的重投影与内点阈值均为有限正数。"""

        values = (
            self.max_reprojection_error_mean_px,
            self.max_reprojection_error_max_px,
            self.inlier_reprojection_error_px,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Rotation-PnP reprojection thresholds must be positive.")


@dataclass(frozen=True)
class HeadRotationPnpResult:
    rotation_camera_from_head: np.ndarray
    rvec_rad: np.ndarray
    reprojection_error_mean_px: float
    reprojection_error_max_px: float
    point_count: int
    inlier_count: int
    confidence: float
    solver_method: str
    candidate_count: int
    min_normalized_object_depth: float


@dataclass(frozen=True)
class _Candidate:
    method: str
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    errors_px: np.ndarray
    min_depth: float


def _require_cv2() -> Any:
    """延迟导入 OpenCV，并在环境缺失时给出针对 PnP 的错误。"""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Scale-free head-rotation PnP requires OpenCV.") from error
    return cv2


def normalize_flame_object_points(points: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre FLAME points and set their RMS radius to one.

    Returns ``(normalised_points, centroid, rms_radius)``.  Uniformly scaling
    the input points changes only the returned centroid/radius, not the
    normalised geometry used by PnP.
    """

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 4 or array.shape[1] != 3:
        raise ValueError("FLAME PnP points must have shape [N,3] with N >= 4.")
    if not np.isfinite(array).all():
        raise ValueError("FLAME PnP points must be finite.")
    centroid = array.mean(axis=0)
    centred = array - centroid
    rms_radius = float(np.sqrt(np.mean(np.sum(centred * centred, axis=1))))
    if not math.isfinite(rms_radius) or rms_radius <= 1e-12:
        raise ValueError("FLAME PnP points have degenerate spatial extent.")
    return centred / rms_radius, centroid, rms_radius


def _correspondences(
    image_points_by_label: Mapping[str, Sequence[float]],
    landmarks3d: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """按语义标签配对 MediaPipe 2D 与 FLAME 3D 点，并归一化 3D 几何尺度。"""

    landmarks = np.asarray(landmarks3d, dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError("landmarks3d must have shape [N,3].")
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for label, flame_index in FLAME_ROTATION_LANDMARK_INDICES.items():
        pixel = image_points_by_label.get(label)
        if pixel is None or flame_index >= len(landmarks):
            continue
        pixel_array = np.asarray(pixel, dtype=np.float64).reshape(-1)
        point = landmarks[flame_index]
        if pixel_array.shape != (2,) or not np.isfinite(pixel_array).all():
            continue
        if not np.isfinite(point).all():
            continue
        image_points.append(pixel_array)
        object_points.append(point)
    if len(object_points) < 4:
        raise ValueError(
            "At least four finite MediaPipe/FLAME correspondences are required."
        )
    normalised, _, _ = normalize_flame_object_points(np.stack(object_points))
    return normalised, np.stack(image_points)


def _evaluate(
    *,
    method: str,
    success: bool,
    rvec: Any,
    tvec: Any,
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    config: HeadRotationPnpConfig,
) -> _Candidate | None:
    """验证一个 PnP 候选的旋转、正深度和重投影误差，并返回合格候选。"""

    if not success or rvec is None or tvec is None:
        return None
    cv2 = _require_cv2()
    rvec_array = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec_array = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    if not np.isfinite(rvec_array).all() or not np.isfinite(tvec_array).all():
        return None
    rotation, _ = cv2.Rodrigues(rvec_array)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        return None
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        return None
    camera_points = object_points @ rotation.T + tvec_array.reshape(1, 3)
    min_depth = float(camera_points[:, 2].min())
    if config.require_all_points_in_front and min_depth <= 0:
        return None
    projected, _ = cv2.projectPoints(
        object_points,
        rvec_array,
        tvec_array,
        camera_matrix,
        dist_coeffs,
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    if not np.isfinite(errors).all():
        return None
    if float(errors.mean()) > config.max_reprojection_error_mean_px:
        return None
    if float(errors.max()) > config.max_reprojection_error_max_px:
        return None
    return _Candidate(
        method=method,
        rvec=rvec_array.reshape(3),
        tvec=tvec_array.reshape(3),
        rotation=rotation,
        errors_px=errors,
        min_depth=min_depth,
    )


def solve_head_rotation_pnp(
    image_points_by_label: Mapping[str, Sequence[float]],
    landmarks3d: Any,
    camera_matrix: Any,
    dist_coeffs: Any | None = None,
    *,
    config: HeadRotationPnpConfig | None = None,
) -> HeadRotationPnpResult:
    """Estimate only ``R_C_H`` from scale-normalised FLAME geometry."""

    cv2 = _require_cv2()
    active = config or HeadRotationPnpConfig()
    camera = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = (
        None
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    )
    if not np.isfinite(camera).all() or camera[0, 0] <= 0 or camera[1, 1] <= 0:
        raise ValueError("camera_matrix must contain finite positive focal lengths.")
    object_points, image_points = _correspondences(
        image_points_by_label, landmarks3d
    )
    candidates: list[_Candidate] = []
    seeds: list[_Candidate] = []
    for method, flag in (
        ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
        ("EPNP", cv2.SOLVEPNP_EPNP),
        ("SQPNP", cv2.SOLVEPNP_SQPNP),
    ):
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera,
                distortion,
                flags=flag,
            )
        except cv2.error:
            continue
        candidate = _evaluate(
            method=method,
            success=bool(success),
            rvec=rvec,
            tvec=tvec,
            object_points=object_points,
            image_points=image_points,
            camera_matrix=camera,
            dist_coeffs=distortion,
            config=active,
        )
        if candidate is not None:
            candidates.append(candidate)
            if method != "ITERATIVE":
                seeds.append(candidate)
    for seed in seeds:
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera,
                distortion,
                seed.rvec.reshape(3, 1).copy(),
                seed.tvec.reshape(3, 1).copy(),
                True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            continue
        candidate = _evaluate(
            method=f"{seed.method}+ITERATIVE",
            success=bool(success),
            rvec=rvec,
            tvec=tvec,
            object_points=object_points,
            image_points=image_points,
            camera_matrix=camera,
            dist_coeffs=distortion,
            config=active,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError(
            "Rotation-only PnP found no positive-depth solution within the "
            "configured reprojection thresholds."
        )
    selected = min(candidates, key=lambda item: float(item.errors_px.mean()))
    inlier_count = int(
        np.count_nonzero(
            selected.errors_px <= active.inlier_reprojection_error_px
        )
    )
    inlier_ratio = inlier_count / len(selected.errors_px)
    confidence = float(
        np.clip(math.exp(-float(selected.errors_px.mean()) / 4.0) * inlier_ratio, 0, 1)
    )
    return HeadRotationPnpResult(
        rotation_camera_from_head=selected.rotation,
        rvec_rad=selected.rvec,
        reprojection_error_mean_px=float(selected.errors_px.mean()),
        reprojection_error_max_px=float(selected.errors_px.max()),
        point_count=len(image_points),
        inlier_count=inlier_count,
        confidence=confidence,
        solver_method=selected.method,
        candidate_count=len(candidates),
        min_normalized_object_depth=selected.min_depth,
    )
