"""DECA input preprocessing with reversible source-image crop geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


FACE_PREPROCESS_DECA = "deca"
FACE_PREPROCESS_LEGACY = "legacy"
FACE_PREPROCESS_CHOICES = (FACE_PREPROCESS_DECA, FACE_PREPROCESS_LEGACY)

DEFAULT_DECA_CROP_SCALE = 1.25
DECA_BBOX_CENTER_Y_OFFSET = 0.12


@dataclass(frozen=True)
class FaceCropTransform:
    """Affine mapping between DECA input pixels and source-image pixels."""

    mode: str
    input_size: int
    source_left_px: float
    source_top_px: float
    source_width_px: float
    source_height_px: float

    def __post_init__(self) -> None:
        if self.mode not in FACE_PREPROCESS_CHOICES:
            raise ValueError(f"Unsupported face preprocessing mode: {self.mode!r}")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.source_width_px <= 0 or self.source_height_px <= 0:
            raise ValueError("source crop dimensions must be positive")

    @property
    def source_right_px(self) -> float:
        return self.source_left_px + self.source_width_px

    @property
    def source_bottom_px(self) -> float:
        return self.source_top_px + self.source_height_px

    def input_pixels_to_source(self, points_xy: Any) -> Any:
        """Map ``[..., 2]`` DECA input coordinates back into source pixels."""

        import numpy as np

        points = np.asarray(points_xy, dtype=np.float64)
        if points.shape[-1] != 2:
            raise ValueError(f"points_xy must end in dimension 2, got {points.shape}")
        scale = np.array(
            [
                self.source_width_px / self.input_size,
                self.source_height_px / self.input_size,
            ],
            dtype=np.float64,
        )
        offset = np.array([self.source_left_px, self.source_top_px], dtype=np.float64)
        return points * scale + offset

    def as_record(self) -> dict[str, float | str]:
        """Return the crop fields stored with generated depth priors."""

        return {
            "deca_face_preprocess": self.mode,
            "deca_crop_left_px": float(self.source_left_px),
            "deca_crop_top_px": float(self.source_top_px),
            "deca_crop_width_px": float(self.source_width_px),
            "deca_crop_height_px": float(self.source_height_px),
        }


def _validate_bbox(face_bbox_xywh: Sequence[float]) -> tuple[float, float, float, float]:
    import numpy as np

    bbox = np.asarray(face_bbox_xywh, dtype=np.float64).reshape(-1)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError("face_bbox_xywh must contain four finite values")
    x, y, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0:
        raise ValueError("face bbox width and height must be positive")
    return x, y, width, height


def build_face_crop_transform(
    face_bbox_xywh: Sequence[float],
    *,
    mode: str = FACE_PREPROCESS_DECA,
    input_size: int = 224,
    deca_crop_scale: float = DEFAULT_DECA_CROP_SCALE,
) -> FaceCropTransform:
    """Build the source-to-input crop used for DECA inference.

    ``deca`` mirrors the official DECA ``bbox`` crop convention: it uses the
    mean bbox size, shifts the crop centre downward by 12 percent of that size,
    then applies the official 1.25 crop scale. ``legacy`` describes the
    existing rectangular detector crop that was directly stretched to 224x224.
    """

    if mode not in FACE_PREPROCESS_CHOICES:
        raise ValueError(
            f"mode must be one of {FACE_PREPROCESS_CHOICES}, got {mode!r}"
        )
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    x, y, width, height = _validate_bbox(face_bbox_xywh)

    if mode == FACE_PREPROCESS_LEGACY:
        return FaceCropTransform(
            mode=mode,
            input_size=input_size,
            source_left_px=x,
            source_top_px=y,
            source_width_px=width,
            source_height_px=height,
        )

    if deca_crop_scale <= 0:
        raise ValueError("deca_crop_scale must be positive")
    old_size = 0.5 * (width + height)
    crop_size = float(int(old_size * deca_crop_scale))
    if crop_size <= 0:
        raise ValueError("DECA crop size must be positive")
    center_x = x + 0.5 * width
    center_y = y + 0.5 * height + old_size * DECA_BBOX_CENTER_Y_OFFSET
    return FaceCropTransform(
        mode=mode,
        input_size=input_size,
        source_left_px=center_x - 0.5 * crop_size,
        source_top_px=center_y - 0.5 * crop_size,
        source_width_px=crop_size,
        source_height_px=crop_size,
    )


def _as_rgb_uint8(image: Any) -> Any:
    import numpy as np

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape [H, W, 3], got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def render_source_crop(source_image_rgb: Any, transform: FaceCropTransform) -> Any:
    """Render ``transform`` from the source image, padding outside pixels with black."""

    try:
        from PIL import Image
    except ModuleNotFoundError as error:  # pragma: no cover - runtime dependency
        raise ModuleNotFoundError("Pillow is required for DECA face preprocessing.") from error

    source = _as_rgb_uint8(source_image_rgb)
    image = Image.fromarray(source, mode="RGB")
    scale_x = transform.source_width_px / transform.input_size
    scale_y = transform.source_height_px / transform.input_size
    affine = (scale_x, 0.0, transform.source_left_px, 0.0, scale_y, transform.source_top_px)
    rendered = image.transform(
        (transform.input_size, transform.input_size),
        Image.Transform.AFFINE,
        affine,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    return _as_rgb_uint8(rendered)


def prepare_deca_face_image(
    source_image_rgb: Any,
    face_bbox_xywh: Sequence[float],
    *,
    mode: str = FACE_PREPROCESS_DECA,
    input_size: int = 224,
    deca_crop_scale: float = DEFAULT_DECA_CROP_SCALE,
    legacy_face_image_rgb: Any | None = None,
) -> tuple[Any, FaceCropTransform]:
    """Return a DECA input image and its reversible source-image transform."""

    transform = build_face_crop_transform(
        face_bbox_xywh,
        mode=mode,
        input_size=input_size,
        deca_crop_scale=deca_crop_scale,
    )
    if mode == FACE_PREPROCESS_DECA:
        return render_source_crop(source_image_rgb, transform), transform

    if legacy_face_image_rgb is None:
        raise ValueError("legacy_face_image_rgb is required for legacy preprocessing")
    legacy = _as_rgb_uint8(legacy_face_image_rgb)
    if legacy.shape[:2] == (input_size, input_size):
        return legacy, transform

    try:
        from PIL import Image
    except ModuleNotFoundError as error:  # pragma: no cover - runtime dependency
        raise ModuleNotFoundError("Pillow is required for DECA face preprocessing.") from error
    resized = Image.fromarray(legacy, mode="RGB").resize(
        (input_size, input_size),
        resample=Image.Resampling.BILINEAR,
    )
    return _as_rgb_uint8(resized), transform
