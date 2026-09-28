"""Adapt ModelV1 TableFrame7 artefacts to the released BlazeGaze contract.

BlazeGaze describes metric head pose as ``[R|t]`` in the paper, while the
released model consumes two derived tensors:

``head_vector``
    A three-dimensional direction derived from ``R_C_H`` using the exact
    angle convention in ``webeyetrack.model_based.get_head_vector``.

``face_origin_3d``
    The midpoint of the two eye-origin proxies in camera coordinates, in
    centimetres.  This is deliberately distinct from ``t_C_H``: the released
    runtime obtains ``face_origin_3d`` by averaging the reconstructed eye
    origins, rather than by taking the translation column of the face pose.

This module retains both millimetre and centimetre versions of ``[R|t]`` for
auditability and exposes the two tensors actually consumed by the model.  It
does not import the external WebEyeTrack repository.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .eye_patch import (
    EYE_PATCH_HEIGHT,
    EYE_PATCH_WIDTH,
    EXACT_EYE_PATCH_SOURCE,
    BlazeGazeEyePatchManifest,
)


ROTATION_COLUMNS = tuple(
    f"head_rotation_c_from_h_{row}{column}"
    for row in range(3)
    for column in range(3)
)
ORIGIN_COLUMNS_MM = tuple(
    f"head_origin_camera_{axis}_mm" for axis in "xyz"
)
IRIS_MIDPOINT_COLUMNS_MM = tuple(
    f"iris_midpoint_camera_{axis}_mm" for axis in "xyz"
)
POSE_REQUIRED_COLUMNS = (
    "sample_id",
    "dataset",
    "image_name",
    "metric_pose_status",
    *ORIGIN_COLUMNS_MM,
    *IRIS_MIDPOINT_COLUMNS_MM,
    *ROTATION_COLUMNS,
)
STAGE3_REQUIRED_COLUMNS = (
    "sample_id",
    "dataset",
    "image_name",
    "source_image_path",
    "uv_gt_u_mm",
    "uv_gt_v_mm",
)
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 512
HALF_IMAGE_WIDTH = IMAGE_WIDTH // 2


def _require_columns(fieldnames: Sequence[str] | None, required: Sequence[str], path: Path) -> None:
    available = set(fieldnames or ())
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _finite_float(row: Mapping[str, str], column: str, sample_id: str) -> float:
    raw = str(row.get(column, "")).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Sample {sample_id!r} has a non-numeric {column}: {raw!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"Sample {sample_id!r} has a non-finite {column}.")
    return value


def _determinant_3x3(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _validate_rotation(
    rotation: Sequence[Sequence[float]],
    sample_id: str,
    tolerance: float,
) -> None:
    max_orthogonality_error = 0.0
    for row in range(3):
        for column in range(3):
            dot = sum(rotation[k][row] * rotation[k][column] for k in range(3))
            expected = 1.0 if row == column else 0.0
            max_orthogonality_error = max(
                max_orthogonality_error, abs(dot - expected)
            )
    determinant_error = abs(_determinant_3x3(rotation) - 1.0)
    if max_orthogonality_error > tolerance or determinant_error > tolerance:
        raise ValueError(
            f"Sample {sample_id!r} does not contain a valid R_C_H rotation "
            f"(orthogonality error={max_orthogonality_error:.6g}, "
            f"determinant error={determinant_error:.6g})."
        )


def _blazegaze_head_vector(rotation: torch.Tensor) -> torch.Tensor:
    """Port the released BlazeGaze ``get_head_vector`` convention exactly."""

    # WebEyeTrack utilities.rotation_matrix_to_euler_angles:
    # pitch=asin(-R20), yaw=atan2(R21,R22), roll=atan2(R10,R00).
    pitch = torch.asin(torch.clamp(-rotation[2, 0], -1.0, 1.0))
    yaw = torch.atan2(rotation[2, 1], rotation[2, 2])
    roll = torch.atan2(rotation[1, 0], rotation[0, 0])

    # WebEyeTrack model_based.get_head_vector remaps the angles as follows.
    head_pitch = -yaw
    head_yaw = pitch
    x = torch.cos(head_pitch) * torch.sin(head_yaw)
    y = torch.sin(head_pitch)
    z = -torch.cos(head_pitch) * torch.cos(head_yaw)

    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    return torch.stack(
        (
            cos_roll * x - sin_roll * y,
            sin_roll * x + cos_roll * y,
            z,
        )
    ).to(dtype=torch.float32)


@dataclass(frozen=True)
class MetricHeadPoseRecord:
    """One validated metric transform from head coordinates to camera coordinates."""

    sample_id: str
    dataset: str
    image_name: str
    rotation_c_from_h: torch.Tensor
    head_origin_camera_mm: torch.Tensor
    iris_midpoint_camera_mm: torch.Tensor

    @property
    def metric_rt_mm(self) -> torch.Tensor:
        """Return ``[R_C_H | t_C_H]`` with translation in millimetres."""

        return torch.cat(
            (self.rotation_c_from_h, self.head_origin_camera_mm[:, None]), dim=1
        )

    @property
    def metric_rt_cm(self) -> torch.Tensor:
        """Return ``[R_C_H | t_C_H]`` with translation in centimetres."""

        return torch.cat(
            (self.rotation_c_from_h, self.head_origin_camera_cm[:, None]), dim=1
        )

    @property
    def head_origin_camera_cm(self) -> torch.Tensor:
        """Return ``t_C_H`` in centimetres without changing its semantics."""

        return self.head_origin_camera_mm / 10.0

    @property
    def face_origin_3d(self) -> torch.Tensor:
        """Return the released model's eye-midpoint position input in cm."""

        return self.iris_midpoint_camera_mm / 10.0

    @property
    def head_vector(self) -> torch.Tensor:
        """Return the released model's three-dimensional head direction input."""

        return _blazegaze_head_vector(self.rotation_c_from_h)


class MetricHeadPoseTable:
    """Validated lookup table for Stage 2 metric head-pose records."""

    def __init__(self, records: Mapping[str, MetricHeadPoseRecord]) -> None:
        self._records = dict(records)

    @classmethod
    def load(
        cls,
        csv_paths: Iterable[str | Path],
        *,
        rotation_tolerance: float = 5e-3,
    ) -> "MetricHeadPoseTable":
        records: dict[str, MetricHeadPoseRecord] = {}
        for raw_path in csv_paths:
            path = Path(raw_path)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                _require_columns(reader.fieldnames, POSE_REQUIRED_COLUMNS, path)
                for row in reader:
                    if str(row["metric_pose_status"]).strip().lower() != "success":
                        continue
                    sample_id = str(row["sample_id"]).strip()
                    if not sample_id:
                        raise ValueError(f"{path} contains an empty sample_id.")
                    if sample_id in records:
                        raise ValueError(
                            f"Duplicate successful metric pose for sample {sample_id!r}."
                        )
                    flat_rotation = [
                        _finite_float(row, column, sample_id)
                        for column in ROTATION_COLUMNS
                    ]
                    rotation_values = [
                        flat_rotation[start : start + 3]
                        for start in range(0, 9, 3)
                    ]
                    _validate_rotation(
                        rotation_values, sample_id, tolerance=rotation_tolerance
                    )
                    origin_values = [
                        _finite_float(row, column, sample_id)
                        for column in ORIGIN_COLUMNS_MM
                    ]
                    iris_midpoint_values = [
                        _finite_float(row, column, sample_id)
                        for column in IRIS_MIDPOINT_COLUMNS_MM
                    ]
                    if origin_values[2] <= 0.0:
                        raise ValueError(
                            f"Sample {sample_id!r} has non-positive camera depth."
                        )
                    if iris_midpoint_values[2] <= 0.0:
                        raise ValueError(
                            f"Sample {sample_id!r} has a non-positive eye-midpoint depth."
                        )
                    records[sample_id] = MetricHeadPoseRecord(
                        sample_id=sample_id,
                        dataset=str(row["dataset"]).strip(),
                        image_name=str(row["image_name"]).strip(),
                        rotation_c_from_h=torch.tensor(
                            rotation_values, dtype=torch.float32
                        ),
                        head_origin_camera_mm=torch.tensor(
                            origin_values, dtype=torch.float32
                        ),
                        iris_midpoint_camera_mm=torch.tensor(
                            iris_midpoint_values, dtype=torch.float32
                        ),
                    )
        if not records:
            raise ValueError("No successful metric head-pose records were loaded.")
        return cls(records)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self._records

    def lookup(self, sample_id: str) -> MetricHeadPoseRecord:
        try:
            return self._records[sample_id]
        except KeyError as exc:
            raise KeyError(f"No metric head pose for sample {sample_id!r}.") from exc


@dataclass(frozen=True)
class PogBoundsMM:
    """Physical TableFrame bounds used to create BlazeGaze ``pog_norm`` labels.

    The released loader trains on a zero-centred label: it first maps a point
    into ``[0, 1]`` and then subtracts ``[0.5, 0.5]``.  Bounds must come from
    the physical evaluation surface or a fixed protocol, not from test-set
    extrema.
    """

    u_min: float
    u_max: float
    v_min: float
    v_max: float

    def __post_init__(self) -> None:
        values = (self.u_min, self.u_max, self.v_min, self.v_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("PoG bounds must be finite.")
        if self.u_max <= self.u_min or self.v_max <= self.v_min:
            raise ValueError("PoG maximum bounds must be greater than minimum bounds.")

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "PogBoundsMM":
        if len(values) != 4:
            raise ValueError("PoG bounds require u_min u_max v_min v_max.")
        return cls(*(float(value) for value in values))

    @property
    def screen_info_cm(self) -> torch.Tensor:
        """Return ``[height_cm, width_cm]`` as expected by BlazeGaze metrics."""

        return torch.tensor(
            [(self.v_max - self.v_min) / 10.0, (self.u_max - self.u_min) / 10.0],
            dtype=torch.float32,
        )

    def normalize(self, uv_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower = torch.tensor([self.u_min, self.v_min], dtype=torch.float32)
        extent = torch.tensor(
            [self.u_max - self.u_min, self.v_max - self.v_min],
            dtype=torch.float32,
        )
        zero_one = (uv_mm - lower) / extent
        return zero_one, zero_one - 0.5


@dataclass(frozen=True)
class _SampleRecord:
    row: Mapping[str, str]
    pose: MetricHeadPoseRecord
    left_eye_path: Path | None
    right_eye_path: Path | None
    eye_patch_path: Path | None
    eye_strip_source: str


def _resolve_eye_paths(row: Mapping[str, str]) -> tuple[Path, Path, str]:
    virtual_left = str(row.get("virtual_left_eye_path", "")).strip()
    virtual_right = str(row.get("virtual_right_eye_path", "")).strip()
    if bool(virtual_left) != bool(virtual_right):
        raise ValueError(
            f"Sample {row['sample_id']!r} defines only one virtual eye path."
        )
    if virtual_left:
        return Path(virtual_left), Path(virtual_right), "virtual_eye_crops"

    source_path = Path(str(row["source_image_path"]).strip())
    dataset_root = source_path.parent.parent
    image_name = str(row["image_name"]).strip()
    return (
        dataset_root / "insightface_eyes" / "left_eye" / image_name,
        dataset_root / "insightface_eyes" / "right_eye" / image_name,
        "paired_legacy_eye_crops",
    )


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _compose_eye_strip(left_eye_path: Path, right_eye_path: Path) -> Image.Image:
    """Create the released encoder shape from the available paired eye crops."""

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    left = _load_rgb(left_eye_path).resize(
        (HALF_IMAGE_WIDTH, IMAGE_HEIGHT), resampling
    )
    right = _load_rgb(right_eye_path).resize(
        (HALF_IMAGE_WIDTH, IMAGE_HEIGHT), resampling
    )
    strip = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT))
    strip.paste(left, (0, 0))
    strip.paste(right, (HALF_IMAGE_WIDTH, 0))
    return strip


class BlazeGazeDataAdapter(Dataset):
    """Independent Dataset yielding tensors compatible with released BlazeGaze.

    By default ``image`` is HWC because the released TensorFlow encoder has
    input shape ``(128, 512, 3)``.  Set ``image_layout='CHW'`` for a PyTorch
    port.  Pixel values are always float32 in ``[0, 1]``.
    """

    def __init__(
        self,
        stage3_csv_paths: Iterable[str | Path],
        metric_pose_csv_paths: Iterable[str | Path],
        *,
        pog_bounds_mm: PogBoundsMM | Sequence[float] | None = None,
        image_layout: str = "HWC",
        sample_weights: Mapping[str, float] | None = None,
        eye_patch_manifest_paths: Iterable[str | Path] | None = None,
        require_exact_eye_patches: bool = False,
        strict_join: bool = True,
        validate_image_paths: bool = True,
        rotation_tolerance: float = 5e-3,
    ) -> None:
        normalized_layout = image_layout.strip().upper()
        if normalized_layout not in {"HWC", "CHW"}:
            raise ValueError("image_layout must be 'HWC' or 'CHW'.")
        self.image_layout = normalized_layout
        self.pose_table = MetricHeadPoseTable.load(
            metric_pose_csv_paths, rotation_tolerance=rotation_tolerance
        )
        if pog_bounds_mm is not None and not isinstance(pog_bounds_mm, PogBoundsMM):
            pog_bounds_mm = PogBoundsMM.from_sequence(pog_bounds_mm)
        self.pog_bounds_mm = pog_bounds_mm
        self.sample_weights = dict(sample_weights or {})
        self.eye_patch_manifest = (
            BlazeGazeEyePatchManifest.load(eye_patch_manifest_paths)
            if eye_patch_manifest_paths is not None
            else None
        )
        if require_exact_eye_patches and self.eye_patch_manifest is None:
            raise ValueError(
                "require_exact_eye_patches=True requires an eye-patch manifest."
            )
        self.samples: list[_SampleRecord] = []
        seen: set[str] = set()
        missing_pose: list[str] = []
        missing_exact_eye_patch: list[str] = []

        for raw_path in stage3_csv_paths:
            path = Path(raw_path)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                _require_columns(reader.fieldnames, STAGE3_REQUIRED_COLUMNS, path)
                for raw_row in reader:
                    row = dict(raw_row)
                    sample_id = str(row["sample_id"]).strip()
                    if not sample_id:
                        raise ValueError(f"{path} contains an empty sample_id.")
                    if sample_id in seen:
                        raise ValueError(f"Duplicate Stage 3 sample {sample_id!r}.")
                    seen.add(sample_id)
                    if sample_id not in self.pose_table:
                        missing_pose.append(sample_id)
                        continue
                    pose = self.pose_table.lookup(sample_id)
                    dataset_name = str(row["dataset"]).strip()
                    image_name = str(row["image_name"]).strip()
                    if dataset_name != pose.dataset or image_name != pose.image_name:
                        raise ValueError(
                            f"Join mismatch for {sample_id!r}: Stage 2 and Stage 3 "
                            "dataset/image metadata differ."
                        )
                    _finite_float(row, "uv_gt_u_mm", sample_id)
                    _finite_float(row, "uv_gt_v_mm", sample_id)
                    eye_patch_path: Path | None = None
                    left_path: Path | None = None
                    right_path: Path | None = None
                    if (
                        self.eye_patch_manifest is not None
                        and sample_id in self.eye_patch_manifest
                    ):
                        eye_patch_path = self.eye_patch_manifest.lookup(
                            sample_id
                        ).eye_patch_path
                        source = EXACT_EYE_PATCH_SOURCE
                    else:
                        if require_exact_eye_patches:
                            missing_exact_eye_patch.append(sample_id)
                            continue
                        left_path, right_path, source = _resolve_eye_paths(row)
                    if validate_image_paths:
                        candidates = (
                            (eye_patch_path,)
                            if eye_patch_path is not None
                            else (left_path, right_path)
                        )
                        missing_images = [
                            str(candidate)
                            for candidate in candidates
                            if candidate is None or not candidate.is_file()
                        ]
                        if missing_images:
                            raise FileNotFoundError(
                                f"Sample {sample_id!r} is missing eye crops: "
                                f"{missing_images}"
                            )
                    self.samples.append(
                        _SampleRecord(
                            row,
                            pose,
                            left_path,
                            right_path,
                            eye_patch_path,
                            source,
                        )
                    )

        if strict_join and missing_pose:
            preview = ", ".join(repr(value) for value in missing_pose[:5])
            raise ValueError(
                f"{len(missing_pose)} Stage 3 samples have no successful metric pose; "
                f"first samples: {preview}."
            )
        if missing_exact_eye_patch:
            preview = ", ".join(
                repr(value) for value in missing_exact_eye_patch[:5]
            )
            raise ValueError(
                f"{len(missing_exact_eye_patch)} samples lack an exact BlazeGaze "
                f"eye patch; first samples: {preview}."
            )
        if not self.samples:
            raise ValueError("No joined BlazeGaze samples were selected.")
        self.missing_pose_sample_ids = tuple(missing_pose)

    def __len__(self) -> int:
        return len(self.samples)

    def sample_id_at(self, index: int) -> str:
        return str(self.samples[index].row["sample_id"]).strip()

    def target_mm_at(self, index: int) -> torch.Tensor:
        """Return a label without loading its eye-patch image."""

        sample = self.samples[index]
        sample_id = self.sample_id_at(index)
        return torch.tensor(
            [
                _finite_float(sample.row, "uv_gt_u_mm", sample_id),
                _finite_float(sample.row, "uv_gt_v_mm", sample_id),
            ],
            dtype=torch.float32,
        )

    def pog_norm_at(self, index: int) -> torch.Tensor:
        if self.pog_bounds_mm is None:
            raise RuntimeError("pog_norm_at requires fixed physical PoG bounds.")
        _, centered = self.pog_bounds_mm.normalize(self.target_mm_at(index))
        return centered

    def set_sample_weights(self, values: Mapping[str, float]) -> None:
        """Replace per-sample weights after a train-only weight grid is fitted."""

        normalized: dict[str, float] = {}
        known = {self.sample_id_at(index) for index in range(len(self))}
        unknown = set(values).difference(known)
        if unknown:
            preview = ", ".join(repr(value) for value in sorted(unknown)[:5])
            raise KeyError(f"Sample weights contain unknown IDs: {preview}.")
        for sample_id, raw_weight in values.items():
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"Sample {sample_id!r} has an invalid sample weight."
                )
            normalized[str(sample_id)] = weight
        self.sample_weights = normalized

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        if sample.eye_patch_path is not None:
            patch = _load_rgb(sample.eye_patch_path)
            if patch.size != (EYE_PATCH_WIDTH, EYE_PATCH_HEIGHT):
                raise ValueError(
                    f"Exact eye patch {sample.eye_patch_path} has size {patch.size}; "
                    f"expected {(EYE_PATCH_WIDTH, EYE_PATCH_HEIGHT)}."
                )
        else:
            assert sample.left_eye_path is not None
            assert sample.right_eye_path is not None
            patch = _compose_eye_strip(sample.left_eye_path, sample.right_eye_path)
        image_array = np.array(patch, dtype=np.float32, copy=True)
        image = torch.from_numpy(image_array).div_(255.0)
        if self.image_layout == "CHW":
            image = image.permute(2, 0, 1).contiguous()

        sample_id = self.sample_id_at(index)
        uv_mm = self.target_mm_at(index)
        weight = float(self.sample_weights.get(sample_id, 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Sample {sample_id!r} has an invalid sample weight.")

        item: dict[str, object] = {
            "image": image,
            "head_vector": sample.pose.head_vector,
            "face_origin_3d": sample.pose.face_origin_3d,
            "metric_rt_mm": sample.pose.metric_rt_mm,
            "metric_rt_cm": sample.pose.metric_rt_cm,
            "metric_rt_flat_mm": sample.pose.metric_rt_mm.reshape(-1),
            "uv_gt_mm": uv_mm,
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "sample_id": sample_id,
            "participant_id": str(sample.row["dataset"]).strip(),
            "dataset": str(sample.row["dataset"]).strip(),
            "image_name": str(sample.row["image_name"]).strip(),
            "eye_patch_path": (
                str(sample.eye_patch_path) if sample.eye_patch_path is not None else ""
            ),
            "left_eye_path": (
                str(sample.left_eye_path) if sample.left_eye_path is not None else ""
            ),
            "right_eye_path": (
                str(sample.right_eye_path) if sample.right_eye_path is not None else ""
            ),
            "eye_strip_source": sample.eye_strip_source,
        }
        if self.pog_bounds_mm is not None:
            pog_norm_01, pog_norm = self.pog_bounds_mm.normalize(uv_mm)
            item.update(
                {
                    "pog_norm_01": pog_norm_01,
                    "pog_norm": pog_norm,
                    "screen_info": self.pog_bounds_mm.screen_info_cm,
                }
            )
        return item


MANIFEST_FIELDS = (
    "sample_id",
    "participant_id",
    "dataset",
    "image_name",
    "eye_patch_path",
    "left_eye_path",
    "right_eye_path",
    "eye_strip_source",
    "head_vector_x",
    "head_vector_y",
    "head_vector_z",
    "face_origin_3d_x_cm",
    "face_origin_3d_y_cm",
    "face_origin_3d_z_cm",
    *(f"metric_rt_mm_{row}{column}" for row in range(3) for column in range(4)),
    "uv_gt_u_mm",
    "uv_gt_v_mm",
    "pog_norm_01_u",
    "pog_norm_01_v",
    "pog_norm_u",
    "pog_norm_v",
    "screen_height_cm",
    "screen_width_cm",
    "sample_weight",
)


def _manifest_row(adapter: BlazeGazeDataAdapter, sample: _SampleRecord) -> dict[str, object]:
    sample_id = str(sample.row["sample_id"]).strip()
    uv_mm = torch.tensor(
        [
            _finite_float(sample.row, "uv_gt_u_mm", sample_id),
            _finite_float(sample.row, "uv_gt_v_mm", sample_id),
        ],
        dtype=torch.float32,
    )
    row: dict[str, object] = {
        "sample_id": sample_id,
        "participant_id": str(sample.row["dataset"]).strip(),
        "dataset": str(sample.row["dataset"]).strip(),
        "image_name": str(sample.row["image_name"]).strip(),
        "eye_patch_path": (
            str(sample.eye_patch_path) if sample.eye_patch_path is not None else ""
        ),
        "left_eye_path": (
            str(sample.left_eye_path) if sample.left_eye_path is not None else ""
        ),
        "right_eye_path": (
            str(sample.right_eye_path) if sample.right_eye_path is not None else ""
        ),
        "eye_strip_source": sample.eye_strip_source,
        "uv_gt_u_mm": float(uv_mm[0]),
        "uv_gt_v_mm": float(uv_mm[1]),
        "pog_norm_01_u": "",
        "pog_norm_01_v": "",
        "pog_norm_u": "",
        "pog_norm_v": "",
        "screen_height_cm": "",
        "screen_width_cm": "",
        "sample_weight": float(adapter.sample_weights.get(sample_id, 1.0)),
    }
    for axis, value in zip("xyz", sample.pose.head_vector.tolist()):
        row[f"head_vector_{axis}"] = value
    for axis, value in zip("xyz", sample.pose.face_origin_3d.tolist()):
        row[f"face_origin_3d_{axis}_cm"] = value
    for matrix_row in range(3):
        for matrix_column in range(4):
            row[f"metric_rt_mm_{matrix_row}{matrix_column}"] = float(
                sample.pose.metric_rt_mm[matrix_row, matrix_column]
            )
    if adapter.pog_bounds_mm is not None:
        pog_norm_01, pog_norm = adapter.pog_bounds_mm.normalize(uv_mm)
        row.update(
            {
                "pog_norm_01_u": float(pog_norm_01[0]),
                "pog_norm_01_v": float(pog_norm_01[1]),
                "pog_norm_u": float(pog_norm[0]),
                "pog_norm_v": float(pog_norm[1]),
                "screen_height_cm": float(
                    adapter.pog_bounds_mm.screen_info_cm[0]
                ),
                "screen_width_cm": float(
                    adapter.pog_bounds_mm.screen_info_cm[1]
                ),
            }
        )
    return row


def build_blazegaze_manifest(
    adapter: BlazeGazeDataAdapter,
    output_csv: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write a deterministic, flattened audit manifest without duplicating images."""

    path = Path(output_csv)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for sample in adapter.samples:
                writer.writerow(_manifest_row(adapter, sample))
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    source_counts = Counter(sample.eye_strip_source for sample in adapter.samples)
    report: dict[str, object] = {
        "schema_version": "blazegaze_table7_adapter_v2",
        "output_csv": str(path),
        "sample_count": len(adapter),
        "participant_ids": sorted(
            {str(sample.row["dataset"]).strip() for sample in adapter.samples}
        ),
        "missing_metric_pose_count": len(adapter.missing_pose_sample_ids),
        "image_shape_hwc": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "image_range": [0.0, 1.0],
        "image_sources": dict(source_counts),
        "rotation_convention": "R_C_H (head coordinates to camera coordinates)",
        "translation_convention": (
            "metric_rt_mm contains t_C_H in mm; metric_rt_cm contains t_C_H in "
            "cm; face_origin_3d is the iris/eye midpoint proxy in cm and is not t_C_H"
        ),
        "face_origin_source": "iris_midpoint_camera_mm / 10",
        "native_model_inputs": ["image", "head_vector", "face_origin_3d"],
        "has_native_training_labels": adapter.pog_bounds_mm is not None,
        "exact_eye_patch_count": source_counts.get(EXACT_EYE_PATCH_SOURCE, 0),
        "eye_strip_is_exact_official_homography": (
            source_counts.get(EXACT_EYE_PATCH_SOURCE, 0) == len(adapter)
        ),
        "eye_strip_note": (
            "mediapipe478_homography rows use the released obtain_eyepatch "
            "geometry; any paired_legacy_eye_crops rows are explicit fallbacks."
        ),
    }
    return report


def write_report(report: Mapping[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
