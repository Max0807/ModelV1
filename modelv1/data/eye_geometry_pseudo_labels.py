"""Validated lookup table for 15-point MediaPipe eye pseudo-labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import Tensor


EYE_GEOMETRY_PSEUDO_LABEL_SCHEMA_VERSION = "mediapipe_eye_geometry_v1"
EYE_PSEUDO_POINT_COUNT = 15
EYE_PSEUDO_GEOMETRY_FEATURE_DIM = 10
EYE_PSEUDO_IRIS_CENTER_INDEX = 10
EYE_PSEUDO_POINT_NAMES = (
    "outer_canthus",
    "upper_outer",
    "upper_mid_outer",
    "upper_mid_inner",
    "upper_inner",
    "inner_canthus",
    "lower_inner",
    "lower_mid_inner",
    "lower_mid_outer",
    "lower_outer",
    "iris_center",
    "iris_rim_0",
    "iris_rim_1",
    "iris_rim_2",
    "iris_rim_3",
)
EYE_PSEUDO_GEOMETRY_FEATURE_NAMES = (
    "iris_center_x_local",
    "iris_center_y_local",
    "iris_radius_local",
    "iris_radius_cv",
    "iris_center_fit_offset_local",
    "eyelid_opening_local",
    "point_in_image_ratio",
    "point_in_crop_ratio",
    "canthus_distance_px",
    "quality_score",
)

_REQUIRED_ARRAYS = (
    "sample_id",
    "dataset",
    "image_name",
    "detection_success_mask",
    "valid_mask",
    "left_eye_valid_mask",
    "right_eye_valid_mask",
    "left_eye_points_crop_norm",
    "right_eye_points_crop_norm",
    "left_eye_points_local",
    "right_eye_points_local",
    "left_eye_point_valid_mask",
    "right_eye_point_valid_mask",
    "left_eye_geometry",
    "right_eye_geometry",
    "sample_quality_score",
    "metadata_json",
)


@dataclass(frozen=True)
class EyeGeometryPseudoLabelRecord:
    """One sample in anatomical ``[left, right]`` eye order."""

    sample_id: str
    landmarks_crop_norm: Tensor
    landmarks_local: Tensor
    landmark_valid_mask: Tensor
    geometry_features: Tensor
    eye_valid_mask: Tensor
    detection_success: Tensor
    sample_valid: Tensor
    sample_quality: Tensor


class EyeGeometryPseudoLabelTable:
    """Merge and validate one or more versioned pseudo-label NPZ files."""

    def __init__(
        self,
        paths: tuple[Path, ...],
        records: Mapping[str, EyeGeometryPseudoLabelRecord],
        metadata: tuple[Mapping[str, object], ...],
    ) -> None:
        self.paths = paths
        self.records = dict(records)
        self.metadata = metadata

    @classmethod
    def load(
        cls,
        paths: Iterable[str | Path],
    ) -> "EyeGeometryPseudoLabelTable":
        resolved = tuple(Path(path).resolve() for path in paths)
        if not resolved:
            raise ValueError("At least one eye-geometry pseudo-label NPZ is required.")
        records: dict[str, EyeGeometryPseudoLabelRecord] = {}
        metadata_items: list[Mapping[str, object]] = []
        for path in resolved:
            arrays, metadata = _load_archive(path)
            metadata_items.append(metadata)
            count = len(arrays["sample_id"])
            for index in range(count):
                record = _record_from_arrays(arrays, index=index, path=path)
                if record.sample_id in records:
                    raise ValueError(
                        "Duplicate sample_id across eye-geometry NPZ files: "
                        f"{record.sample_id!r}."
                    )
                records[record.sample_id] = record
        if not records:
            raise ValueError("Eye-geometry pseudo-label archives contain no samples.")
        return cls(resolved, records, tuple(metadata_items))

    def contains(self, sample_id: str) -> bool:
        return sample_id in self.records

    def require_sample_ids(self, sample_ids: Iterable[str]) -> None:
        missing = sorted(set(sample_ids).difference(self.records))
        if missing:
            preview = ", ".join(repr(value) for value in missing[:10])
            raise KeyError(
                f"Missing {len(missing)} eye pseudo-label sample IDs; first: {preview}."
            )

    def lookup(self, sample_id: str) -> EyeGeometryPseudoLabelRecord:
        try:
            record = self.records[sample_id]
        except KeyError as exc:
            raise KeyError(f"No eye pseudo-label record for sample_id={sample_id!r}.") from exc
        # Dataset augmentation mutates masks, so callers receive independent tensors.
        return EyeGeometryPseudoLabelRecord(
            sample_id=record.sample_id,
            landmarks_crop_norm=record.landmarks_crop_norm.clone(),
            landmarks_local=record.landmarks_local.clone(),
            landmark_valid_mask=record.landmark_valid_mask.clone(),
            geometry_features=record.geometry_features.clone(),
            eye_valid_mask=record.eye_valid_mask.clone(),
            detection_success=record.detection_success.clone(),
            sample_valid=record.sample_valid.clone(),
            sample_quality=record.sample_quality.clone(),
        )

    @staticmethod
    def empty(sample_id: str) -> EyeGeometryPseudoLabelRecord:
        return EyeGeometryPseudoLabelRecord(
            sample_id=sample_id,
            landmarks_crop_norm=torch.zeros(
                (2, EYE_PSEUDO_POINT_COUNT, 2), dtype=torch.float32
            ),
            landmarks_local=torch.zeros(
                (2, EYE_PSEUDO_POINT_COUNT, 2), dtype=torch.float32
            ),
            landmark_valid_mask=torch.zeros(
                (2, EYE_PSEUDO_POINT_COUNT), dtype=torch.float32
            ),
            geometry_features=torch.zeros(
                (2, EYE_PSEUDO_GEOMETRY_FEATURE_DIM), dtype=torch.float32
            ),
            eye_valid_mask=torch.zeros(2, dtype=torch.float32),
            detection_success=torch.zeros(1, dtype=torch.float32),
            sample_valid=torch.zeros(1, dtype=torch.float32),
            sample_quality=torch.zeros(1, dtype=torch.float32),
        )


def _load_archive(path: Path) -> tuple[dict[str, np.ndarray], Mapping[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Eye-geometry pseudo-label NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in _REQUIRED_ARRAYS if name not in archive.files]
        if missing:
            raise ValueError(f"Eye-geometry NPZ {path} is missing arrays: {missing}")
        arrays = {name: archive[name].copy() for name in archive.files}
    try:
        metadata = json.loads(str(arrays["metadata_json"].item()))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Eye-geometry NPZ has invalid metadata_json: {path}") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Eye-geometry metadata must be a mapping: {path}")
    if metadata.get("schema_version") != EYE_GEOMETRY_PSEUDO_LABEL_SCHEMA_VERSION:
        raise ValueError(
            "Eye-geometry schema mismatch: expected "
            f"{EYE_GEOMETRY_PSEUDO_LABEL_SCHEMA_VERSION!r}, got "
            f"{metadata.get('schema_version')!r} in {path}."
        )
    if tuple(metadata.get("point_names", ())) != EYE_PSEUDO_POINT_NAMES:
        raise ValueError(f"Eye-geometry point ordering mismatch: {path}")
    if (
        tuple(metadata.get("geometry_feature_names", ()))
        != EYE_PSEUDO_GEOMETRY_FEATURE_NAMES
    ):
        raise ValueError(f"Eye-geometry feature ordering mismatch: {path}")
    _validate_shapes(arrays, path)
    return arrays, metadata


def _validate_shapes(arrays: Mapping[str, np.ndarray], path: Path) -> None:
    count = len(arrays["sample_id"])
    if count == 0:
        raise ValueError(f"Eye-geometry NPZ has no samples: {path}")
    scalar_arrays = (
        "sample_id",
        "dataset",
        "image_name",
        "detection_success_mask",
        "valid_mask",
        "left_eye_valid_mask",
        "right_eye_valid_mask",
        "sample_quality_score",
    )
    for name in scalar_arrays:
        if arrays[name].shape != (count,):
            raise ValueError(f"{name} must have shape ({count},) in {path}.")
    point_shape = (count, EYE_PSEUDO_POINT_COUNT, 2)
    for name in (
        "left_eye_points_crop_norm",
        "right_eye_points_crop_norm",
        "left_eye_points_local",
        "right_eye_points_local",
    ):
        if arrays[name].shape != point_shape:
            raise ValueError(f"{name} must have shape {point_shape} in {path}.")
    mask_shape = (count, EYE_PSEUDO_POINT_COUNT)
    for name in ("left_eye_point_valid_mask", "right_eye_point_valid_mask"):
        if arrays[name].shape != mask_shape:
            raise ValueError(f"{name} must have shape {mask_shape} in {path}.")
    feature_shape = (count, EYE_PSEUDO_GEOMETRY_FEATURE_DIM)
    for name in ("left_eye_geometry", "right_eye_geometry"):
        if arrays[name].shape != feature_shape:
            raise ValueError(f"{name} must have shape {feature_shape} in {path}.")


def _record_from_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    index: int,
    path: Path,
) -> EyeGeometryPseudoLabelRecord:
    sample_id = str(arrays["sample_id"][index]).strip()
    if not sample_id:
        raise ValueError(f"Eye-geometry NPZ contains an empty sample_id: {path}")
    detection = bool(arrays["detection_success_mask"][index])
    eye_valid = np.asarray(
        [
            arrays["left_eye_valid_mask"][index],
            arrays["right_eye_valid_mask"][index],
        ],
        dtype=np.bool_,
    )
    sample_valid = bool(arrays["valid_mask"][index])
    if sample_valid != bool(eye_valid.all()):
        raise ValueError(f"valid_mask disagrees with per-eye validity for {sample_id!r}.")
    if eye_valid.any() and not detection:
        raise ValueError(f"Valid eye without successful detection for {sample_id!r}.")
    quality = float(arrays["sample_quality_score"][index])
    if not math_is_finite_in_unit_interval(quality):
        raise ValueError(f"Invalid sample_quality_score for {sample_id!r}: {quality}.")

    crop = np.stack(
        (
            arrays["left_eye_points_crop_norm"][index],
            arrays["right_eye_points_crop_norm"][index],
        )
    ).astype(np.float32, copy=False)
    local = np.stack(
        (
            arrays["left_eye_points_local"][index],
            arrays["right_eye_points_local"][index],
        )
    ).astype(np.float32, copy=False)
    features = np.stack(
        (
            arrays["left_eye_geometry"][index],
            arrays["right_eye_geometry"][index],
        )
    ).astype(np.float32, copy=False)
    point_mask = np.stack(
        (
            arrays["left_eye_point_valid_mask"][index],
            arrays["right_eye_point_valid_mask"][index],
        )
    ).astype(np.bool_, copy=False)
    if detection and (
        not np.isfinite(crop).all()
        or not np.isfinite(local).all()
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"Detected eye geometry contains non-finite values: {sample_id!r}.")
    # Failed detections deliberately store NaN in the archive. Return zero-filled
    # tensors so downstream masked computation cannot be poisoned by NaN * 0.
    crop = np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)
    local = np.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    if not detection:
        point_mask.fill(False)
        eye_valid.fill(False)
    return EyeGeometryPseudoLabelRecord(
        sample_id=sample_id,
        landmarks_crop_norm=torch.from_numpy(crop.copy()),
        landmarks_local=torch.from_numpy(local.copy()),
        landmark_valid_mask=torch.from_numpy(point_mask.astype(np.float32)),
        geometry_features=torch.from_numpy(features.copy()),
        eye_valid_mask=torch.from_numpy(eye_valid.astype(np.float32)),
        detection_success=torch.tensor([float(detection)], dtype=torch.float32),
        sample_valid=torch.tensor([float(sample_valid)], dtype=torch.float32),
        sample_quality=torch.tensor([quality], dtype=torch.float32),
    )


def math_is_finite_in_unit_interval(value: float) -> bool:
    return bool(np.isfinite(value) and 0.0 <= value <= 1.0)
