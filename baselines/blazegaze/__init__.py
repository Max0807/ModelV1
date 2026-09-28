"""BlazeGaze baseline data contract.

The implementation is intentionally kept outside :mod:`modelv1` so that the
native baseline can evolve without coupling it to the proposed model.
"""

from .data_adapter import (
    BlazeGazeDataAdapter,
    MetricHeadPoseRecord,
    MetricHeadPoseTable,
    PogBoundsMM,
    build_blazegaze_manifest,
)
from .eye_patch import (
    BlazeGazeEyePatchGeometry,
    BlazeGazeEyePatchManifest,
    build_blazegaze_eye_patch,
)
from .losses import (
    NativeBlazeGazeLossConfig,
    NativeBlazeGazeStage1Loss,
    denormalize_pog_mm,
    embedding_consistency_loss,
    mean_pog_error_mm,
    weighted_gaze_l2_loss,
)
from .model import (
    NativeBlazeGaze,
    NativeBlazeGazeConfig,
    NativeBlazeGazeDecoder,
    NativeBlazeGazeEncoder,
    NativeBlazeGazeMLP,
)
from .weighting import OfficialInverseFrequencyGrid

__all__ = [
    "BlazeGazeDataAdapter",
    "MetricHeadPoseRecord",
    "MetricHeadPoseTable",
    "PogBoundsMM",
    "build_blazegaze_manifest",
    "BlazeGazeEyePatchGeometry",
    "BlazeGazeEyePatchManifest",
    "build_blazegaze_eye_patch",
    "NativeBlazeGaze",
    "NativeBlazeGazeConfig",
    "NativeBlazeGazeDecoder",
    "NativeBlazeGazeEncoder",
    "NativeBlazeGazeMLP",
    "NativeBlazeGazeLossConfig",
    "NativeBlazeGazeStage1Loss",
    "weighted_gaze_l2_loss",
    "embedding_consistency_loss",
    "denormalize_pog_mm",
    "mean_pog_error_mm",
    "OfficialInverseFrequencyGrid",
]
