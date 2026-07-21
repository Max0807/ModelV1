"""Offline geometry components for generating gaze depth priors."""

from .deca_flame import (
    DEFAULT_DECA_ROOT,
    FLAME_LEFT_EYE_CENTER_VERTEX,
    FLAME_RIGHT_EYE_CENTER_VERTEX,
    DecaFlameConfig,
    DecaFlameExtractor,
    DecaFlameOutput,
)
from .pnp import (
    CROSSGAZE_CAMERA_MATRIX,
    CROSSGAZE_DIST_COEFFS,
    DEFAULT_PNP_MAPPING,
    PnpCamera,
    PnpConfig,
    PnpFaceDepthResult,
    PnpPointMapping,
    ScaleEstimate,
    compute_scale_estimate,
    solve_pnp_face_depth,
)

__all__ = [
    "CROSSGAZE_CAMERA_MATRIX",
    "CROSSGAZE_DIST_COEFFS",
    "DEFAULT_DECA_ROOT",
    "DEFAULT_PNP_MAPPING",
    "FLAME_LEFT_EYE_CENTER_VERTEX",
    "FLAME_RIGHT_EYE_CENTER_VERTEX",
    "DecaFlameConfig",
    "DecaFlameExtractor",
    "DecaFlameOutput",
    "PnpCamera",
    "PnpConfig",
    "PnpFaceDepthResult",
    "PnpPointMapping",
    "ScaleEstimate",
    "compute_scale_estimate",
    "solve_pnp_face_depth",
]
