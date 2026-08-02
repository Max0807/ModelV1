"""Scene input representations shared by ModelV1 and its DataLoader."""

from __future__ import annotations


SCENE_REPRESENTATION_FULL25 = "full25"
SCENE_REPRESENTATION_TABLE_FRAME7 = "table_frame7"
SCENE_REPRESENTATION_ORIENTATION6D = "orientation6d"
SCENE_REPRESENTATIONS = (
    SCENE_REPRESENTATION_FULL25,
    SCENE_REPRESENTATION_TABLE_FRAME7,
    SCENE_REPRESENTATION_ORIENTATION6D,
)

FULL_SCENE_DIM = 25
TABLE_FRAME_DIM = 7
ORIENTATION_DIM = 6
ORIENTATION_BASIS_INDICES = (7, 8, 9, 10, 11, 12)
TABLE_FRAME_DISTANCE_INDEX = 3
TABLE_FRAME_DISTANCE_SCALE_MM = 1000.0


def canonical_scene_representation(value: str) -> str:
    """Validate and normalize a scene representation name."""

    representation = str(value).strip().lower()
    if representation not in SCENE_REPRESENTATIONS:
        supported = ", ".join(SCENE_REPRESENTATIONS)
        raise ValueError(
            f"Unknown scene_representation={value!r}; expected one of: "
            f"{supported}."
        )
    return representation


def scene_representation_dim(value: str) -> int:
    """Return the required model input dimension for one representation."""

    representation = canonical_scene_representation(value)
    if representation == SCENE_REPRESENTATION_FULL25:
        return FULL_SCENE_DIM
    if representation == SCENE_REPRESENTATION_TABLE_FRAME7:
        return TABLE_FRAME_DIM
    if representation == SCENE_REPRESENTATION_ORIENTATION6D:
        return ORIENTATION_DIM
    raise AssertionError(f"Unhandled scene representation: {representation}")
