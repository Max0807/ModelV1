"""Shared mode names for the optional binocular-geometry quality gate."""

from __future__ import annotations


EYE_GEOMETRY_GATE_NONE = "none"
EYE_GEOMETRY_GATE_FIXED = "fixed"
EYE_GEOMETRY_GATE_LEARNED_RESIDUAL = "learned_residual"
EYE_GEOMETRY_GATE_MODES = (
    EYE_GEOMETRY_GATE_NONE,
    EYE_GEOMETRY_GATE_FIXED,
    EYE_GEOMETRY_GATE_LEARNED_RESIDUAL,
)


def canonical_eye_geometry_gate_mode(value: str) -> str:
    """Validate and normalize an eye-geometry gate mode."""

    mode = str(value).strip().lower()
    if mode not in EYE_GEOMETRY_GATE_MODES:
        supported = ", ".join(EYE_GEOMETRY_GATE_MODES)
        raise ValueError(
            f"Unknown eye_geometry_gate_mode={value!r}; expected one of: {supported}."
        )
    return mode
