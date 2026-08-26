"""Validated lookup table for offline virtual-camera image artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from modelv1.depth_prior.pnp import PNP_GEOMETRY_VERSION


IMAGE_SOURCE_LEGACY = "legacy"
IMAGE_SOURCE_VIRTUAL_CAMERA = "virtual_camera"
IMAGE_SOURCES = (IMAGE_SOURCE_LEGACY, IMAGE_SOURCE_VIRTUAL_CAMERA)
ANATOMICAL_EYE_SEMANTICS = "anatomical_subject_left_right"
LEGACY_EYE_SIDE_CONVERSION = "legacy_image_side_to_anatomical"
ROTATION_N_FROM_C_COLUMNS = tuple(
    f"rotation_n_from_c_{row}{column}"
    for row in range(3)
    for column in range(3)
)


def canonical_image_source(value: str) -> str:
    source = str(value).strip().lower()
    if source not in IMAGE_SOURCES:
        raise ValueError(
            f"Unknown image_source={value!r}; expected one of {IMAGE_SOURCES}."
        )
    return source


@dataclass(frozen=True)
class VirtualCameraRecord:
    sample_id: str
    face_path: Path
    left_eye_path: Path
    right_eye_path: Path
    rotation_n_from_c: Tensor
    virtual_distance_scale: Tensor

    @property
    def rotation_c_from_n(self) -> Tensor:
        return self.rotation_n_from_c.transpose(0, 1)


class VirtualCameraManifest:
    """Per-sample normalized image paths and real-to-virtual rotations."""

    def __init__(
        self,
        path: Path,
        records: dict[str, VirtualCameraRecord],
        invalid_reasons: dict[str, str],
    ) -> None:
        self.path = path
        self.records = records
        self.invalid_reasons = invalid_reasons

    @classmethod
    def load(cls, path: str | Path) -> "VirtualCameraManifest":
        manifest_path = Path(path)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Virtual-camera manifest does not exist: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Virtual-camera manifest is empty: {manifest_path}")

        records: dict[str, VirtualCameraRecord] = {}
        invalid_reasons: dict[str, str] = {}
        seen: set[str] = set()
        for row in rows:
            sample_id = row.get("sample_id", "").strip()
            if not sample_id:
                raise ValueError("Virtual-camera manifest contains an empty sample_id.")
            if sample_id in seen:
                raise ValueError(
                    f"Duplicate sample_id in virtual-camera manifest: {sample_id}"
                )
            seen.add(sample_id)
            try:
                records[sample_id] = cls._parse_record(
                    row,
                    manifest_path.parent,
                )
            except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
                invalid_reasons[sample_id] = str(exc)
        return cls(manifest_path, records, invalid_reasons)

    @staticmethod
    def _parse_record(
        row: dict[str, str],
        manifest_directory: Path,
    ) -> VirtualCameraRecord:
        sample_id = row["sample_id"].strip()
        status = row.get("normalization_status", "").strip().lower()
        if status != "success":
            reason = row.get("normalization_reason", "").strip() or "unknown"
            raise ValueError(f"normalization_status={status or 'missing'}:{reason}")
        semantics = row.get("eye_side_semantics", "").strip()
        if semantics != ANATOMICAL_EYE_SEMANTICS:
            raise ValueError(
                "eye_side_semantics must be "
                f"{ANATOMICAL_EYE_SEMANTICS!r}, got {semantics or 'missing'!r}. "
                "Regenerate with virtual-camera v2; v1 eye directories are swapped."
            )
        conversion = row.get("landmark_eye_side_conversion", "").strip()
        if conversion != LEGACY_EYE_SIDE_CONVERSION:
            raise ValueError(
                "landmark_eye_side_conversion must be "
                f"{LEGACY_EYE_SIDE_CONVERSION!r}, got {conversion or 'missing'!r}."
            )
        pnp_version = row.get("pnp_geometry_version", "").strip()
        if pnp_version != PNP_GEOMETRY_VERSION:
            raise ValueError(
                "pnp_geometry_version must be "
                f"{PNP_GEOMETRY_VERSION!r}, got {pnp_version or 'missing'!r}; "
                "regenerate the normalized images from the verified PnP solver."
            )

        face_path = _resolve_image_path(
            row.get("normalized_face_path", ""),
            manifest_directory,
            "normalized_face_path",
        )
        left_eye_path = _resolve_image_path(
            row.get("normalized_left_eye_path", ""),
            manifest_directory,
            "normalized_left_eye_path",
        )
        right_eye_path = _resolve_image_path(
            row.get("normalized_right_eye_path", ""),
            manifest_directory,
            "normalized_right_eye_path",
        )
        try:
            rotation = torch.tensor(
                [float(row[column]) for column in ROTATION_N_FROM_C_COLUMNS],
                dtype=torch.float32,
            ).reshape(3, 3)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid rotation_n_from_c matrix") from exc
        if not torch.isfinite(rotation).all():
            raise ValueError("rotation_n_from_c must contain finite values")
        identity = torch.eye(3, dtype=rotation.dtype)
        if not torch.allclose(rotation @ rotation.transpose(0, 1), identity, atol=2e-4):
            raise ValueError("rotation_n_from_c must be orthonormal")
        if not torch.isclose(torch.linalg.det(rotation), rotation.new_tensor(1.0), atol=2e-4):
            raise ValueError("rotation_n_from_c must have determinant +1")
        try:
            distance_scale = torch.tensor(
                [float(row["virtual_distance_scale"])],
                dtype=torch.float32,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid virtual_distance_scale") from exc
        if not torch.isfinite(distance_scale).all() or torch.any(distance_scale <= 0):
            raise ValueError("virtual_distance_scale must be finite and positive")
        return VirtualCameraRecord(
            sample_id=sample_id,
            face_path=face_path,
            left_eye_path=left_eye_path,
            right_eye_path=right_eye_path,
            rotation_n_from_c=rotation,
            virtual_distance_scale=distance_scale,
        )

    def __len__(self) -> int:
        return len(self.records)

    def contains(self, sample_id: str) -> bool:
        return sample_id in self.records

    def lookup(self, sample_id: str) -> VirtualCameraRecord:
        try:
            return self.records[sample_id]
        except KeyError as exc:
            reason = self.invalid_reasons.get(sample_id, "sample_id missing")
            raise KeyError(
                f"No valid virtual-camera artifact for {sample_id!r}: {reason}"
            ) from exc

    def require_sample_ids(self, sample_ids: list[str]) -> None:
        missing = [sample_id for sample_id in sample_ids if sample_id not in self.records]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                "Virtual-camera manifest lacks valid artifacts for "
                f"{len(missing)} required samples; first: {preview}."
            )


def merge_virtual_camera_manifests(
    paths: Iterable[str | Path],
    output_path: str | Path,
) -> Path:
    """Write an auditable union of compatible virtual-camera manifests.

    Image files are not copied: every output row keeps its original absolute
    normalized image paths. The merge is therefore inexpensive and suitable
    for a per-run directory. Duplicate ``sample_id`` values are rejected
    instead of silently selecting one camera transform.
    """

    source_paths = [Path(path).resolve() for path in paths]
    if not source_paths:
        raise ValueError("At least one virtual-camera manifest is required.")
    output = Path(output_path).resolve()
    if output in source_paths:
        raise ValueError("Merged virtual-camera manifest must not overwrite an input.")

    fields: list[str] | None = None
    seen_sample_ids: set[str] = set()
    rows: list[dict[str, str]] = []
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Virtual-camera manifest does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Virtual-camera manifest has no header: {path}")
            current_fields = list(reader.fieldnames)
            if fields is None:
                fields = current_fields
            elif current_fields != fields:
                raise ValueError(
                    "Cannot merge virtual-camera manifests with different columns: "
                    f"{source_paths[0]} and {path}"
                )
            for row in reader:
                sample_id = row.get("sample_id", "").strip()
                if not sample_id:
                    raise ValueError(
                        f"Virtual-camera manifest contains an empty sample_id: {path}"
                    )
                if sample_id in seen_sample_ids:
                    raise ValueError(
                        "Duplicate sample_id while merging virtual-camera manifests: "
                        f"{sample_id!r}"
                    )
                seen_sample_ids.add(sample_id)
                rows.append(dict(row))

    assert fields is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _resolve_image_path(
    raw_path: str,
    manifest_directory: Path,
    field_name: str,
) -> Path:
    if not str(raw_path).strip():
        raise ValueError(f"{field_name} is missing")
    path = Path(str(raw_path).strip())
    if not path.is_absolute():
        path = manifest_directory / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path
