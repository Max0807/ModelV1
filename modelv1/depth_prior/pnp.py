"""solvePnP geometry for offline face-depth prior generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


# Intrinsics and distortion used by CrossGaze-main/baseline.  They remain
# configurable because a depth prior is only valid for the camera that created it.
CROSSGAZE_CAMERA_MATRIX = (
    (1367.8584, 0.0, 957.9159),
    (0.0, 1369.0087, 543.3381),
    (0.0, 0.0, 1.0),
)
CROSSGAZE_DIST_COEFFS = (
    1.78483784e-01,
    -5.90774600e-01,
    -5.11403240e-04,
    -1.08456025e-03,
    5.95053471e-01,
)
CROSSGAZE_IMAGE_WIDTH_PX = 1920
CROSSGAZE_IMAGE_HEIGHT_PX = 1080

# Bump this identifier whenever the 2D-3D semantics, head-local FLAME frame,
# metric scale, candidate selection, or physical validity contract changes.
# Downstream virtual-camera artifacts must record the exact version they used.
PNP_GEOMETRY_VERSION = "mediapipe_flame_head_local_multistart_v2"


class PnpDependencyError(RuntimeError):
    """Raised when NumPy or OpenCV needed by solvePnP is unavailable."""


def validate_crossgaze_image_size(width_px: int, height_px: int) -> None:
    """Reject images whose pixel coordinates do not match the calibrated camera."""

    actual = (int(width_px), int(height_px))
    expected = (CROSSGAZE_IMAGE_WIDTH_PX, CROSSGAZE_IMAGE_HEIGHT_PX)
    if actual != expected:
        raise ValueError(
            f"CrossGaze camera calibration expects {expected[0]}x{expected[1]} px, "
            f"got {actual[0]}x{actual[1]} px"
        )


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - depends on environment
        raise PnpDependencyError("NumPy is required for the PnP depth module.") from error
    return np


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - depends on environment
        raise PnpDependencyError("OpenCV is required for the PnP depth module.") from error
    return cv2


@dataclass(frozen=True)
class PnpPointMapping:
    """One detected 2D landmark label mapped to one FLAME 68 landmark id."""

    label: str
    flame_landmark_index: int


# MediaPipe labels intentionally preserve the legacy image-side convention:
# 33/133 are image-left and 362/263 are image-right.  Anatomical eye outputs
# swap these two groups explicitly; changing the labels here would silently
# invalidate existing landmark CSVs.
MEDIAPIPE_PNP_LANDMARK_INDICES = {
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "nose_tip": 1,
    "mouth_left": 61,
    "mouth_right": 291,
    "chin": 152,
}

# FLAME 68 landmark indexing follows the mapping used by the CrossGaze baseline.
DEFAULT_PNP_MAPPING = (
    PnpPointMapping("left_eye_outer", 36),
    PnpPointMapping("left_eye_inner", 39),
    PnpPointMapping("right_eye_inner", 42),
    PnpPointMapping("right_eye_outer", 45),
    PnpPointMapping("nose_tip", 30),
    PnpPointMapping("mouth_left", 48),
    PnpPointMapping("mouth_right", 54),
    PnpPointMapping("chin", 8),
)

# FLAME 68 landmark indices returned with anatomical subject-left/right
# semantics. The PnP correspondence labels above intentionally retain the
# legacy image-side convention, but output eye geometry must match ModelV1's
# anatomical left/right image folders and tensors.
FLAME_LEFT_EYE_CANTHUS_INDICES = (42, 45)
FLAME_RIGHT_EYE_CANTHUS_INDICES = (36, 39)


@dataclass(frozen=True)
class PnpCamera:
    """Camera intrinsics and distortion coefficients used by OpenCV."""

    camera_matrix: Any
    dist_coeffs: Any | None = None

    def __post_init__(self) -> None:
        np = _require_numpy()
        camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        if camera_matrix.shape != (3, 3):
            raise ValueError(
                "camera_matrix must have shape [3, 3], got "
                f"{tuple(camera_matrix.shape)}"
            )
        if not np.isfinite(camera_matrix).all() or camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
            raise ValueError("camera_matrix must contain valid positive focal lengths")

        dist_coeffs = None
        if self.dist_coeffs is not None:
            dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)
            if dist_coeffs.size not in {4, 5, 8, 12, 14}:
                raise ValueError("dist_coeffs must contain 4, 5, 8, 12, or 14 values")
            if not np.isfinite(dist_coeffs).all():
                raise ValueError("dist_coeffs contains non-finite values")

        object.__setattr__(self, "camera_matrix", camera_matrix)
        object.__setattr__(self, "dist_coeffs", dist_coeffs)

    @classmethod
    def from_focal_length(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist_coeffs: Sequence[float] | None = None,
    ) -> "PnpCamera":
        return cls(
            camera_matrix=((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)),
            dist_coeffs=dist_coeffs,
        )

    @classmethod
    def crossgaze_default(cls) -> "PnpCamera":
        """Return the calibrated camera constants from the current baseline."""

        return cls(CROSSGAZE_CAMERA_MATRIX, CROSSGAZE_DIST_COEFFS)


@dataclass(frozen=True)
class PnpConfig:
    """Numerical settings and basic validity thresholds for face PnP."""

    use_ransac: bool = False
    ransac_reprojection_error_px: float = 8.0
    ransac_iterations_count: int = 100
    ransac_confidence: float = 0.99
    inlier_reprojection_error_px: float = 10.0
    min_plausible_depth_mm: float = 100.0
    max_plausible_depth_mm: float = 2000.0
    max_reprojection_error_mean_px: float = 8.0
    max_reprojection_error_max_px: float = 20.0
    max_scale_disagreement_ratio: float = 0.25
    require_all_points_in_front: bool = True

    def __post_init__(self) -> None:
        if self.ransac_reprojection_error_px <= 0:
            raise ValueError("ransac_reprojection_error_px must be positive")
        if self.ransac_iterations_count <= 0:
            raise ValueError("ransac_iterations_count must be positive")
        if not 0 < self.ransac_confidence < 1:
            raise ValueError("ransac_confidence must be in (0, 1)")
        if self.inlier_reprojection_error_px <= 0:
            raise ValueError("inlier_reprojection_error_px must be positive")
        if self.min_plausible_depth_mm <= 0:
            raise ValueError("min_plausible_depth_mm must be positive")
        if self.max_plausible_depth_mm <= self.min_plausible_depth_mm:
            raise ValueError("max_plausible_depth_mm must exceed the minimum")
        if self.max_reprojection_error_mean_px <= 0:
            raise ValueError("max_reprojection_error_mean_px must be positive")
        if self.max_reprojection_error_max_px <= 0:
            raise ValueError("max_reprojection_error_max_px must be positive")
        if self.max_scale_disagreement_ratio <= 0:
            raise ValueError("max_scale_disagreement_ratio must be positive")


@dataclass(frozen=True)
class ScaleEstimate:
    """Metric scale recovered from measured inner and outer eye distances."""

    scale_mm_per_flame_unit: float
    outer_scale_mm_per_flame_unit: float
    inner_scale_mm_per_flame_unit: float
    outer_flame_distance: float
    inner_flame_distance: float
    scale_disagreement_ratio: float


@dataclass(frozen=True)
class PnpFaceDepthResult:
    """Camera-space face and eye geometry plus PnP quality indicators."""

    rotation_matrix: Any
    rvec: Any
    tvec_mm: Any
    left_eye_camera_xyz_mm: Any
    right_eye_camera_xyz_mm: Any
    face_depth_z_mm: float
    reprojection_error_mean_px: float
    reprojection_error_max_px: float
    pnp_num_points: int
    pnp_inlier_count: int
    pnp_confidence: float
    pnp_solver_method: str
    pnp_candidate_count: int
    pnp_min_object_depth_mm: float
    depth_is_plausible: bool
    scale: ScaleEstimate

    def as_record(self) -> dict[str, float | int | bool]:
        """Flatten the result into scalar values suitable for CSV or Parquet."""

        return {
            "left_eye_camera_x_mm": float(self.left_eye_camera_xyz_mm[0]),
            "left_eye_camera_y_mm": float(self.left_eye_camera_xyz_mm[1]),
            "left_eye_camera_z_mm": float(self.left_eye_camera_xyz_mm[2]),
            "right_eye_camera_x_mm": float(self.right_eye_camera_xyz_mm[0]),
            "right_eye_camera_y_mm": float(self.right_eye_camera_xyz_mm[1]),
            "right_eye_camera_z_mm": float(self.right_eye_camera_xyz_mm[2]),
            "face_depth_z_mm": self.face_depth_z_mm,
            "reprojection_error_mean_px": self.reprojection_error_mean_px,
            "reprojection_error_max_px": self.reprojection_error_max_px,
            "pnp_num_points": self.pnp_num_points,
            "pnp_inlier_count": self.pnp_inlier_count,
            "pnp_confidence": self.pnp_confidence,
            "pnp_geometry_version": PNP_GEOMETRY_VERSION,
            "pnp_solver_method": self.pnp_solver_method,
            "pnp_candidate_count": self.pnp_candidate_count,
            "pnp_min_object_depth_mm": self.pnp_min_object_depth_mm,
            "depth_is_plausible": self.depth_is_plausible,  # 深度是否合理
            "scale_mm_per_flame_unit": self.scale.scale_mm_per_flame_unit,  # 测量的 FLAME 单位与毫米的比值
            "outer_scale_mm_per_flame_unit": self.scale.outer_scale_mm_per_flame_unit,  # 测量的外眼角距离与毫米的比值
            "inner_scale_mm_per_flame_unit": self.scale.inner_scale_mm_per_flame_unit,  # 测量的内眼角距离与毫米的比值
            "scale_disagreement_ratio": self.scale.scale_disagreement_ratio,  # 测量的内外眼角距离与毫米的比值之间的差异比例
        }


def _as_points(name: str, points: Any, *, minimum_count: int = 1) -> Any:
    np = _require_numpy()
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < minimum_count:
        raise ValueError(
            f"{name} must have shape [N, 3] with N >= {minimum_count}, got "
            f"{tuple(array.shape)}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def compute_scale_estimate(
    landmarks3d: Any,
    *,
    outer_eye_distance_mm: float = 105.0,
    inner_eye_distance_mm: float = 38.0,
) -> ScaleEstimate:
    """Recover FLAME-to-millimetre scale from measured eye-corner distances.

    The defaults are the measured values configured in the current CrossGaze
    baseline.  Reprojection error cannot determine this scale by itself.
    """

    np = _require_numpy()
    if outer_eye_distance_mm <= 0 or inner_eye_distance_mm <= 0:
        raise ValueError("measured eye distances must be positive")

    landmarks = _as_points("landmarks3d", landmarks3d, minimum_count=46)
    outer_flame_distance = float(np.linalg.norm(landmarks[36] - landmarks[45]))
    inner_flame_distance = float(np.linalg.norm(landmarks[39] - landmarks[42]))
    if outer_flame_distance <= 1e-8 or inner_flame_distance <= 1e-8:
        raise ValueError("FLAME eye-corner distances must be non-zero")

    outer_scale = float(outer_eye_distance_mm / outer_flame_distance)
    inner_scale = float(inner_eye_distance_mm / inner_flame_distance)
    scale = 0.5 * (outer_scale + inner_scale)
    disagreement = abs(outer_scale - inner_scale) / max(scale, 1e-8)
    return ScaleEstimate(
        scale_mm_per_flame_unit=scale,  # 测量的 FLAME 单位与毫米的比值
        outer_scale_mm_per_flame_unit=outer_scale,  # 测量的外眼角距离与毫米的比值
        inner_scale_mm_per_flame_unit=inner_scale,  # 测量的内眼角距离与毫米的比值
        outer_flame_distance=outer_flame_distance,  # 测量的外眼角距离
        inner_flame_distance=inner_flame_distance,  # 测量的内眼角距离
        scale_disagreement_ratio=float(disagreement),  # 测量的内外眼角距离与毫米的比值之间的差异比例
    )


def compute_eye_canthus_midpoints(landmarks3d: Any) -> tuple[Any, Any]:
    """Return the two FLAME-local eye-canthus midpoints.

    The pairs match the semantic 68-landmark correspondences in
    :data:`DEFAULT_PNP_MAPPING`.  They are a geometric reference for iris-side
    assignment, not a claim that a canthus midpoint is an eyeball centre.
    """

    landmarks = _as_points("landmarks3d", landmarks3d, minimum_count=1)
    required_index = max(
        *FLAME_LEFT_EYE_CANTHUS_INDICES,
        *FLAME_RIGHT_EYE_CANTHUS_INDICES,
    )
    if required_index >= len(landmarks):
        raise ValueError(
            "landmarks3d does not contain the FLAME eye-canthus indices "
            f"up to {required_index}"
        )
    left_midpoint = 0.5 * (
        landmarks[FLAME_LEFT_EYE_CANTHUS_INDICES[0]]
        + landmarks[FLAME_LEFT_EYE_CANTHUS_INDICES[1]]
    )
    right_midpoint = 0.5 * (
        landmarks[FLAME_RIGHT_EYE_CANTHUS_INDICES[0]]
        + landmarks[FLAME_RIGHT_EYE_CANTHUS_INDICES[1]]
    )
    return left_midpoint, right_midpoint


def _select_pnp_correspondences(
    image_points_by_label: Mapping[str, Sequence[float]],
    landmarks3d: Any,
    mapping: Sequence[PnpPointMapping],
) -> tuple[Any, Any]:
    """Keep finite 2D observations and their corresponding FLAME landmarks.
    选择PnP的3D-2D对应点对：保留有限的2D观测值及其对应的FLAME关键点。
    """

    np = _require_numpy()
    landmarks = _as_points("landmarks3d", landmarks3d, minimum_count=1)
    image_points: list[Any] = []
    object_points: list[Any] = []
    for point in mapping:
        if point.flame_landmark_index < 0 or point.flame_landmark_index >= len(landmarks):
            raise ValueError(
                f"FLAME index {point.flame_landmark_index} for '{point.label}' is invalid"
            )
        value = image_points_by_label.get(point.label)
        if value is None:
            continue
        image_point = np.asarray(value, dtype=np.float64).reshape(-1)
        if image_point.shape != (2,) or not np.isfinite(image_point).all():
            continue
        image_points.append(image_point)
        object_points.append(landmarks[point.flame_landmark_index])

    if len(image_points) < 4:
        raise ValueError(
            "At least four finite 2D-3D correspondences are required for solvePnP; "
            f"received {len(image_points)}"
        )
    return np.asarray(object_points, dtype=np.float64), np.asarray(image_points, dtype=np.float64)


def _compute_pnp_confidence(
    mean_error_px: float,
    inlier_ratio: float,
    scale_disagreement_ratio: float,
) -> float:
    """Return a transparent heuristic quality score, not a calibrated sigma."""

    reprojection_quality = math.exp(-mean_error_px / 4.0)
    scale_quality = math.exp(-2.0 * scale_disagreement_ratio)
    return float(min(1.0, max(0.0, reprojection_quality * inlier_ratio * scale_quality)))


@dataclass(frozen=True)
class _PnpCandidate:
    """One numerically successful and physically checked OpenCV PnP result."""

    method: str
    rvec: Any
    tvec: Any
    reprojection_errors_px: Any
    min_object_depth_mm: float
    ransac_inliers: Any | None = None

    @property
    def mean_reprojection_error_px(self) -> float:
        return float(self.reprojection_errors_px.mean())


def _evaluate_pnp_candidate(
    *,
    method: str,
    success: bool,
    rvec: Any,
    tvec: Any,
    object_points_mm: Any,
    image_points: Any,
    camera: PnpCamera,
    config: PnpConfig,
    ransac_inliers: Any | None = None,
) -> _PnpCandidate | None:
    """Return a candidate only when it satisfies the OpenCV cheirality contract."""

    if not success or rvec is None or tvec is None:
        return None
    np = _require_numpy()
    cv2 = _require_cv2()
    rvec_array = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec_array = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    if not np.isfinite(rvec_array).all() or not np.isfinite(tvec_array).all():
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec_array)
    object_points_camera = (
        object_points_mm @ rotation_matrix.T + tvec_array.reshape(1, 3)
    )
    min_object_depth = float(object_points_camera[:, 2].min())
    face_origin_depth = float(tvec_array[2, 0])
    if not (
        config.min_plausible_depth_mm
        <= face_origin_depth
        <= config.max_plausible_depth_mm
    ):
        return None
    if config.require_all_points_in_front and min_object_depth <= 0.0:
        return None

    projected_points, _ = cv2.projectPoints(
        object_points_mm,
        rvec_array,
        tvec_array,
        camera.camera_matrix,
        camera.dist_coeffs,
    )
    projected_points = projected_points.reshape(-1, 2)
    reprojection_errors = np.linalg.norm(projected_points - image_points, axis=1)
    if not np.isfinite(reprojection_errors).all():
        return None
    if float(reprojection_errors.mean()) > config.max_reprojection_error_mean_px:
        return None
    if float(reprojection_errors.max()) > config.max_reprojection_error_max_px:
        return None
    return _PnpCandidate(
        method=method,
        rvec=rvec_array,
        tvec=tvec_array,
        reprojection_errors_px=reprojection_errors,
        min_object_depth_mm=min_object_depth,
        ransac_inliers=ransac_inliers,
    )


def _solve_physical_pnp_candidates(
    object_points_mm: Any,
    image_points: Any,
    camera: PnpCamera,
    config: PnpConfig,
) -> list[_PnpCandidate]:
    """Solve from multiple initializations and retain only front-facing poses.

    OpenCV's bare ``SOLVEPNP_ITERATIVE`` obtains its own DLT initialization.
    On real ModelV1 frames that initialization can converge to an all-points-
    behind-camera mirror solution with a deceptively small reprojection error.
    EPNP and SQPNP provide independent starts; each positive start is also
    refined by ITERATIVE, then the lowest-error physically valid result wins.
    """

    cv2 = _require_cv2()
    candidates: list[_PnpCandidate] = []
    seed_methods = (
        ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
        ("EPNP", cv2.SOLVEPNP_EPNP),
        ("SQPNP", cv2.SOLVEPNP_SQPNP),
    )
    positive_seeds: list[_PnpCandidate] = []
    for method, flag in seed_methods:
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points_mm,
                image_points,
                camera.camera_matrix,
                camera.dist_coeffs,
                flags=flag,
            )
        except cv2.error:
            continue
        candidate = _evaluate_pnp_candidate(
            method=method,
            success=bool(success),
            rvec=rvec,
            tvec=tvec,
            object_points_mm=object_points_mm,
            image_points=image_points,
            camera=camera,
            config=config,
        )
        if candidate is not None:
            candidates.append(candidate)
            if method != "ITERATIVE":
                positive_seeds.append(candidate)

    for seed in positive_seeds:
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points_mm,
                image_points,
                camera.camera_matrix,
                camera.dist_coeffs,
                seed.rvec.copy(),
                seed.tvec.copy(),
                True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            continue
        candidate = _evaluate_pnp_candidate(
            method=f"{seed.method}+ITERATIVE",
            success=bool(success),
            rvec=rvec,
            tvec=tvec,
            object_points_mm=object_points_mm,
            image_points=image_points,
            camera=camera,
            config=config,
        )
        if candidate is not None:
            candidates.append(candidate)

    if config.use_ransac:
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points_mm,
                image_points,
                camera.camera_matrix,
                camera.dist_coeffs,
                iterationsCount=config.ransac_iterations_count,
                reprojectionError=config.ransac_reprojection_error_px,
                confidence=config.ransac_confidence,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            success, rvec, tvec, inliers = False, None, None, None
        candidate = _evaluate_pnp_candidate(
            method="RANSAC_ITERATIVE",
            success=bool(success),
            rvec=rvec,
            tvec=tvec,
            object_points_mm=object_points_mm,
            image_points=image_points,
            camera=camera,
            config=config,
            ransac_inliers=inliers,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def solve_pnp_face_depth(
    image_points_by_label: Mapping[str, Sequence[float]],
    landmarks3d: Any,
    camera: PnpCamera,
    *,
    scale_mm_per_flame_unit: float | None = None,
    outer_eye_distance_mm: float = 105.0,
    inner_eye_distance_mm: float = 38.0,
    mapping: Sequence[PnpPointMapping] = DEFAULT_PNP_MAPPING,
    config: PnpConfig | None = None,
) -> PnpFaceDepthResult:
    """Estimate camera-space eye-canthus references from FLAME and 2D landmarks.

    The function implements OpenCV's ``s p = K [R|t] P`` model.  It first
    converts the FLAME local geometry into millimetres, estimates ``R,t`` with
    ``solvePnP``, and finally transforms both 68-landmark eye-canthus
    midpoints into camera coordinates.  These points are not anatomical
    eyeball centres; Iris-IPD reconstruction uses them only for direction and
    left/right assignment.
    """

    np = _require_numpy()
    cv2 = _require_cv2()
    active_config = config or PnpConfig()
    landmarks = _as_points("landmarks3d", landmarks3d, minimum_count=1)
    left_eye_midpoint, right_eye_midpoint = compute_eye_canthus_midpoints(landmarks)

    measured_scale = compute_scale_estimate(
        landmarks,
        outer_eye_distance_mm=outer_eye_distance_mm,
        inner_eye_distance_mm=inner_eye_distance_mm,
    )
    if measured_scale.scale_disagreement_ratio > active_config.max_scale_disagreement_ratio:
        raise RuntimeError(
            "FLAME inner/outer eye metric scales disagree: "
            f"ratio={measured_scale.scale_disagreement_ratio:.6g}, "
            f"maximum={active_config.max_scale_disagreement_ratio:.6g}"
        )
    if scale_mm_per_flame_unit is not None:
        if scale_mm_per_flame_unit <= 0:
            raise ValueError("scale_mm_per_flame_unit must be positive")
        scale = replace(
            measured_scale,
            scale_mm_per_flame_unit=float(scale_mm_per_flame_unit),
        )
    else:
        scale = measured_scale

    object_points, image_points = _select_pnp_correspondences(
        image_points_by_label,
        landmarks,
        mapping,
    )
    object_points_mm = object_points * scale.scale_mm_per_flame_unit
    candidates = _solve_physical_pnp_candidates(
        object_points_mm,
        image_points,
        camera,
        active_config,
    )
    if not candidates:
        raise RuntimeError(
            "OpenCV solvePnP found no physically valid front-facing face pose "
            "within the configured depth and reprojection thresholds"
        )
    selected = min(
        candidates,
        key=lambda candidate: candidate.mean_reprojection_error_px,
    )
    rvec = selected.rvec
    tvec = selected.tvec
    reprojection_errors = selected.reprojection_errors_px
    mean_error = float(reprojection_errors.mean())
    max_error = float(reprojection_errors.max())
    if selected.ransac_inliers is not None:
        inlier_count = int(len(selected.ransac_inliers))
    else:
        inlier_count = int(
            np.count_nonzero(reprojection_errors <= active_config.inlier_reprojection_error_px)
        )

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    tvec_mm = tvec.reshape(3)
    left_eye_camera = rotation_matrix @ (
        left_eye_midpoint * scale.scale_mm_per_flame_unit
    ) + tvec_mm
    right_eye_camera = rotation_matrix @ (
        right_eye_midpoint * scale.scale_mm_per_flame_unit
    ) + tvec_mm
    face_depth_z_mm = float(tvec_mm[2])
    inlier_ratio = inlier_count / len(image_points)

    return PnpFaceDepthResult(
        rotation_matrix=rotation_matrix,
        rvec=rvec.reshape(3),
        tvec_mm=tvec_mm,
        left_eye_camera_xyz_mm=left_eye_camera,
        right_eye_camera_xyz_mm=right_eye_camera,
        face_depth_z_mm=face_depth_z_mm,
        reprojection_error_mean_px=mean_error,
        reprojection_error_max_px=max_error,
        pnp_num_points=int(len(image_points)),
        pnp_inlier_count=inlier_count,
        pnp_confidence=_compute_pnp_confidence(
            mean_error,
            inlier_ratio,
            scale.scale_disagreement_ratio,
        ),
        pnp_solver_method=selected.method,
        pnp_candidate_count=len(candidates),
        pnp_min_object_depth_mm=selected.min_object_depth_mm,
        depth_is_plausible=True,
        scale=scale,
    )
