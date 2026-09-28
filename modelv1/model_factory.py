"""Model selection for controlled architecture experiments."""

from __future__ import annotations

from torch import nn

from modelv1.blaze_style_model import BlazeStyleModel, BlazeStyleModelConfig
from modelv1.model import ModelV1, ModelV1Config


MODEL_ARCHITECTURE_MODELV1 = "modelv1"
MODEL_ARCHITECTURE_BLAZE_STYLE = "blaze_style"
MODEL_ARCHITECTURES = (
    MODEL_ARCHITECTURE_MODELV1,
    MODEL_ARCHITECTURE_BLAZE_STYLE,
)


def canonical_model_architecture(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "modelv1": MODEL_ARCHITECTURE_MODELV1,
        "original": MODEL_ARCHITECTURE_MODELV1,
        "cross_attention": MODEL_ARCHITECTURE_MODELV1,
        "blaze": MODEL_ARCHITECTURE_BLAZE_STYLE,
        "blazestyle": MODEL_ARCHITECTURE_BLAZE_STYLE,
        "blaze_style": MODEL_ARCHITECTURE_BLAZE_STYLE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model architecture {value!r}; expected one of "
            f"{MODEL_ARCHITECTURES}."
        ) from exc


def build_experiment_model(
    architecture: object,
    config: ModelV1Config,
) -> nn.Module:
    architecture_name = canonical_model_architecture(architecture)
    if architecture_name == MODEL_ARCHITECTURE_MODELV1:
        return ModelV1(config)
    if not isinstance(config, BlazeStyleModelConfig):
        raise TypeError(
            "The Blaze-style architecture requires BlazeStyleModelConfig."
        )
    return BlazeStyleModel(config)


def build_experiment_model_config(
    architecture: object,
    values: dict[str, object],
) -> ModelV1Config:
    """Build the architecture-specific config without leaking fields."""

    architecture_name = canonical_model_architecture(architecture)
    if architecture_name == MODEL_ARCHITECTURE_BLAZE_STYLE:
        return BlazeStyleModelConfig(**values)
    return ModelV1Config(**values)
