"""DECA input and fixed-shape calibration for the final TableFrame7 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from modelv1.depth_prior.dataset_assets import resolve_image_asset_path
from modelv1.depth_prior.face_preprocess import (
    FACE_PREPROCESS_DECA,
    prepare_deca_face_image,
)
from modelv1.preprocessing.camera import PinholeCameraCalibration


def load_deca_input(
    row: Mapping[str, str],
    *,
    face_preprocess: str,
    crop_scale: float,
    image_size: int,
    image_root: Path | None,
    camera_calibration: PinholeCameraCalibration,
    torch: Any,
    image_cls: Any,
) -> tuple[np.ndarray, Any]:
    """Load the calibrated source image and build one DECA input tensor."""

    source_path = resolve_image_asset_path(row, "source_image_path", image_root)
    with image_cls.open(source_path) as image:
        source = np.asarray(image.convert("RGB"))
    camera_calibration.validate_image_size(source.shape[1], source.shape[0])
    legacy = None
    if face_preprocess != FACE_PREPROCESS_DECA:
        face_path = resolve_image_asset_path(row, "face_path", image_root)
        with image_cls.open(face_path) as image:
            legacy = np.asarray(image.convert("RGB"))
    bbox = tuple(float(row[f"face_bbox_{axis}"]) for axis in ("x", "y", "w", "h"))
    deca_image, _ = prepare_deca_face_image(
        source,
        bbox,
        mode=face_preprocess,
        input_size=image_size,
        deca_crop_scale=crop_scale,
        legacy_face_image_rgb=legacy,
    )
    tensor = (
        torch.from_numpy(deca_image.copy())
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
    )
    return source, tensor


def calibrate_fixed_shape(
    rows: Sequence[Mapping[str, str]],
    extractor: Any,
    *,
    dataset_name: str,
    calibration_start: int,
    calibration_stop: int,
    calibration_index_mode: str,
    face_preprocess: str,
    crop_scale: float,
    image_root: Path | None,
    camera_calibration: PinholeCameraCalibration,
    torch: Any,
    image_cls: Any,
) -> tuple[np.ndarray, list[Mapping[str, str]], list[int]]:
    """Return the median DECA shape over an inclusive calibration range."""

    candidates = [row for row in rows if row["dataset"] == dataset_name]
    if not candidates:
        raise ValueError(f"No rows found for calibration dataset {dataset_name!r}.")
    missing_indices: list[int] = []
    if calibration_index_mode == "position":
        selected = candidates[calibration_start : calibration_stop + 1]
    elif calibration_index_mode == "frame_idx":
        selected = []
        present: set[int] = set()
        for row in candidates:
            try:
                frame_index = int(row.get("frame_idx", ""))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid frame_idx in sample {row.get('sample_id', '<unknown>')!r}."
                ) from error
            if calibration_start <= frame_index <= calibration_stop:
                selected.append(row)
                present.add(frame_index)
        missing_indices = [
            value
            for value in range(calibration_start, calibration_stop + 1)
            if value not in present
        ]
    else:
        raise ValueError("calibration_index_mode must be frame_idx or position.")
    if not selected:
        raise ValueError("Fixed-shape calibration selected no samples.")
    shapes: list[np.ndarray] = []
    for row in selected:
        _, tensor = load_deca_input(
            row,
            face_preprocess=face_preprocess,
            crop_scale=crop_scale,
            image_size=extractor.config.image_size,
            image_root=image_root,
            camera_calibration=camera_calibration,
            torch=torch,
            image_cls=image_cls,
        )
        output = extractor.extract(tensor)
        shapes.append(output.parameters[0, :100].numpy())
    fixed_shape = np.median(np.stack(shapes, axis=0), axis=0)
    return fixed_shape, selected, missing_indices

