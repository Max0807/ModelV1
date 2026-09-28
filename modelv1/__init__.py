"""ModelV1 package."""

__all__ = [
    "CrossAttentionVisualEncoder",
    "InceptionResnetFaceEncoder",
    "ModelV1",
    "ModelV1Config",
    "ProbabilisticEyeKeypointHead",
    "ProbabilisticEyeLandmarkHead",
    "ProbabilisticLandmarkTokenizer",
    "LandmarkGuidedEyeFusion",
    "PitchToTableVResidual",
    "DirectTableUVLoss",
    "DirectVirtualTableUVLoss",
    "GazeGeometryLoss",
    "GazeGeometryLossConfig",
    "UVLossConfig",
    "UVRegressionLoss",
    "build_modelv1",
    "compute_uv_metrics",
    "probabilistic_eye_keypoint_nll",
    "predict_uv_mm",
]


def __getattr__(name: str):
    """Load the PyTorch model only when a caller requests it.

    Keeping package import lightweight lets data-preparation utilities inspect
    cache files before a training environment has been installed.
    """

    if name in {
        "CrossAttentionVisualEncoder",
        "InceptionResnetFaceEncoder",
        "ModelV1",
        "ModelV1Config",
        "ProbabilisticEyeKeypointHead",
        "build_modelv1",
    }:
        from .model import (
            CrossAttentionVisualEncoder,
            InceptionResnetFaceEncoder,
            ModelV1,
            ModelV1Config,
            ProbabilisticEyeKeypointHead,
            build_modelv1,
        )

        exports = {
            "CrossAttentionVisualEncoder": CrossAttentionVisualEncoder,
            "InceptionResnetFaceEncoder": InceptionResnetFaceEncoder,
            "ModelV1": ModelV1,
            "ModelV1Config": ModelV1Config,
            "ProbabilisticEyeKeypointHead": ProbabilisticEyeKeypointHead,
            "build_modelv1": build_modelv1,
        }
        globals().update(exports)
        return exports[name]
    if name == "ProbabilisticEyeLandmarkHead":
        from .vertical_eye_geometry import ProbabilisticEyeLandmarkHead

        globals()[name] = ProbabilisticEyeLandmarkHead
        return ProbabilisticEyeLandmarkHead
    if name in {"ProbabilisticLandmarkTokenizer", "LandmarkGuidedEyeFusion"}:
        from .vertical_eye_geometry import (
            LandmarkGuidedEyeFusion,
            ProbabilisticLandmarkTokenizer,
        )

        exports = {
            "ProbabilisticLandmarkTokenizer": ProbabilisticLandmarkTokenizer,
            "LandmarkGuidedEyeFusion": LandmarkGuidedEyeFusion,
        }
        globals().update(exports)
        return exports[name]
    if name == "PitchToTableVResidual":
        from .vertical_eye_geometry import PitchToTableVResidual

        globals()[name] = PitchToTableVResidual
        return PitchToTableVResidual
    if name in {
        "DirectTableUVLoss",
        "DirectVirtualTableUVLoss",
        "GazeGeometryLoss",
        "GazeGeometryLossConfig",
        "UVLossConfig",
        "UVRegressionLoss",
        "compute_uv_metrics",
        "probabilistic_eye_keypoint_nll",
    }:
        from .losses import (
            DirectTableUVLoss,
            DirectVirtualTableUVLoss,
            GazeGeometryLoss,
            GazeGeometryLossConfig,
            UVLossConfig,
            UVRegressionLoss,
            compute_uv_metrics,
            probabilistic_eye_keypoint_nll,
        )

        exports = {
            "DirectTableUVLoss": DirectTableUVLoss,
            "DirectVirtualTableUVLoss": DirectVirtualTableUVLoss,
            "UVLossConfig": UVLossConfig,
            "UVRegressionLoss": UVRegressionLoss,
            "compute_uv_metrics": compute_uv_metrics,
            "probabilistic_eye_keypoint_nll": probabilistic_eye_keypoint_nll,
            "GazeGeometryLoss": GazeGeometryLoss,
            "GazeGeometryLossConfig": GazeGeometryLossConfig,
        }
        globals().update(exports)
        return exports[name]
    if name == "predict_uv_mm":
        from .inference import predict_uv_mm

        globals()[name] = predict_uv_mm
        return predict_uv_mm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
