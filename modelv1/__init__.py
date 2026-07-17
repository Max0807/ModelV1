"""ModelV1 package."""

__all__ = [
    "ModelV1",
    "ModelV1Config",
    "UVLossConfig",
    "UVRegressionLoss",
    "build_modelv1",
    "compute_uv_metrics",
    "predict_uv_mm",
]


def __getattr__(name: str):
    """Load the PyTorch model only when a caller requests it.

    Keeping package import lightweight lets data-preparation utilities inspect
    cache files before a training environment has been installed.
    """

    if name in {"ModelV1", "ModelV1Config", "build_modelv1"}:
        from .model import ModelV1, ModelV1Config, build_modelv1

        exports = {
            "ModelV1": ModelV1,
            "ModelV1Config": ModelV1Config,
            "build_modelv1": build_modelv1,
        }
        globals().update(exports)
        return exports[name]
    if name in {"UVLossConfig", "UVRegressionLoss", "compute_uv_metrics"}:
        from .losses import UVLossConfig, UVRegressionLoss, compute_uv_metrics

        exports = {
            "UVLossConfig": UVLossConfig,
            "UVRegressionLoss": UVRegressionLoss,
            "compute_uv_metrics": compute_uv_metrics,
        }
        globals().update(exports)
        return exports[name]
    if name == "predict_uv_mm":
        from .inference import predict_uv_mm

        globals()[name] = predict_uv_mm
        return predict_uv_mm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
