"""Small, testable geometry helpers for the final TableFrame7 pipeline."""

from .table7_geometry import (
    MetricHeadPose,
    build_metric_head_pose,
    build_table_frame7_n,
    flame_eye_canthus_midpoints,
)
from .rotation_pnp import (
    HEAD_ROTATION_PNP_VERSION,
    MEDIAPIPE_ROTATION_LANDMARK_INDICES,
    HeadRotationPnpConfig,
    HeadRotationPnpResult,
    normalize_flame_object_points,
    solve_head_rotation_pnp,
)

__all__ = [
    "MetricHeadPose",
    "build_metric_head_pose",
    "build_table_frame7_n",
    "flame_eye_canthus_midpoints",
    "HEAD_ROTATION_PNP_VERSION",
    "MEDIAPIPE_ROTATION_LANDMARK_INDICES",
    "HeadRotationPnpConfig",
    "HeadRotationPnpResult",
    "normalize_flame_object_points",
    "solve_head_rotation_pnp",
]
