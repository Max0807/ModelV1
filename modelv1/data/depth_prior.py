"""Read offline binocular geometry and PnP quality for ModelV1 V2.1/V2.2."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from .normalization import EYE_GEOMETRY_DIM, EYE_GEOMETRY_QUALITY_DIM


EYE_CAMERA_COLUMNS = (
    "left_eye_camera_x_mm",
    "left_eye_camera_y_mm",
    "left_eye_camera_z_mm",
    "right_eye_camera_x_mm",
    "right_eye_camera_y_mm",
    "right_eye_camera_z_mm",
)
EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D = "normalized6d"
EYE_GEOMETRY_REPRESENTATION_RAW_EYE6D = "raw_eye6d"
EYE_GEOMETRY_REPRESENTATIONS = (
    EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
    EYE_GEOMETRY_REPRESENTATION_RAW_EYE6D,
)
EYE_QUALITY_COLUMNS = (
    "reprojection_error_mean_px",
    "reprojection_error_max_px",
    "pnp_inlier_count",
    "pnp_num_points",
    "scale_disagreement_ratio",
    "pnp_confidence",
)
VALID_PRIOR_STATUSES = {"success", "warning"}
DEPTH_LOG_SCALE_STD_COLUMN = "depth_log_scale_std"
DEPTH_UNCERTAINTY_STATUS_COLUMN = "depth_uncertainty_status"


def canonical_eye_geometry_representation(value: str) -> str:
    """Validate and normalize a configured binocular-geometry representation."""

    representation = str(value).strip().lower()
    if representation not in EYE_GEOMETRY_REPRESENTATIONS:
        supported = ", ".join(EYE_GEOMETRY_REPRESENTATIONS)
        raise ValueError(
            f"Unknown eye_geometry_representation={value!r}; expected one of: "
            f"{supported}."
        )
    return representation


def eye_geometry_representation_dim(value: str) -> int:
    """Return the width of one selected binocular-geometry vector."""

    canonical_eye_geometry_representation(value)
    return EYE_GEOMETRY_DIM


def _is_false(value: str) -> bool:
    return value.strip().lower() in {"0", "false", "no"}


def build_eye_geometry_vector(
    left_eye_camera_xyz_mm: Tensor,
    right_eye_camera_xyz_mm: Tensor,
) -> Tensor:
    """Return scale-aware binocular geometry ``[cx/cz, cy/cz, log(cz), b/cz]``."""

    left = torch.as_tensor(left_eye_camera_xyz_mm, dtype=torch.float32).flatten()
    right = torch.as_tensor(right_eye_camera_xyz_mm, dtype=torch.float32).flatten()
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError(
            "left and right eye camera coordinates must each have shape [3]"
        )
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Eye camera coordinates must be finite.")

    center = 0.5 * (left + right)
    center_depth = center[2]
    if float(center_depth) <= 0:
        raise ValueError("Binocular center depth must be positive.")
    baseline = right - left
    geometry = torch.stack(
        (
            center[0] / center_depth,
            center[1] / center_depth,
            torch.log(center_depth),
            baseline[0] / center_depth,
            baseline[1] / center_depth,
            baseline[2] / center_depth,
        )
    )
    if geometry.shape != (EYE_GEOMETRY_DIM,) or not torch.isfinite(geometry).all():
        raise ValueError("Derived eye geometry must be a finite 6D vector.")
    return geometry


def build_raw_eye_geometry_vector(
    left_eye_camera_xyz_mm: Tensor,
    right_eye_camera_xyz_mm: Tensor,
) -> Tensor:
    """Return raw camera coordinates ``[Lx, Ly, Lz, Rx, Ry, Rz]`` in mm."""

    left = torch.as_tensor(left_eye_camera_xyz_mm, dtype=torch.float32).flatten()
    right = torch.as_tensor(right_eye_camera_xyz_mm, dtype=torch.float32).flatten()
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError(
            "left and right eye camera coordinates must each have shape [3]"
        )
    geometry = torch.cat((left, right))
    if geometry.shape != (EYE_GEOMETRY_DIM,) or not torch.isfinite(geometry).all():
        raise ValueError("Raw eye geometry must be a finite 6D vector.")
    return geometry


def build_eye_geometry_representation(
    left_eye_camera_xyz_mm: Tensor,
    right_eye_camera_xyz_mm: Tensor,
    representation: str,
) -> Tensor:
    """Build one configured 6D binocular geometry representation."""

    representation = canonical_eye_geometry_representation(representation)
    if representation == EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D:
        return build_eye_geometry_vector(
            left_eye_camera_xyz_mm,
            right_eye_camera_xyz_mm,
        )
    if representation == EYE_GEOMETRY_REPRESENTATION_RAW_EYE6D:
        return build_raw_eye_geometry_vector(
            left_eye_camera_xyz_mm,
            right_eye_camera_xyz_mm,
        )
    raise AssertionError(f"Unhandled eye geometry representation: {representation}")


def build_eye_geometry_quality_vector(
    reprojection_error_mean_px: float,
    reprojection_error_max_px: float,
    pnp_inlier_count: float,
    pnp_num_points: float,
    scale_disagreement_ratio: float,
) -> Tensor:
    """Return ``[log1p(mean), log1p(max), inlier_ratio, scale disagreement]``."""

    values = (
        reprojection_error_mean_px,
        reprojection_error_max_px,
        pnp_inlier_count,
        pnp_num_points,
        scale_disagreement_ratio,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("PnP quality values must be finite.")
    if reprojection_error_mean_px < 0 or reprojection_error_max_px < 0:
        raise ValueError("Reprojection errors must be non-negative.")
    if pnp_num_points <= 0:
        raise ValueError("pnp_num_points must be positive.")
    if pnp_inlier_count < 0 or pnp_inlier_count > pnp_num_points:
        raise ValueError("pnp_inlier_count must lie in [0, pnp_num_points].")
    if scale_disagreement_ratio < 0:
        raise ValueError("scale_disagreement_ratio must be non-negative.")

    quality = torch.tensor(
        (
            math.log1p(reprojection_error_mean_px),
            math.log1p(reprojection_error_max_px),
            pnp_inlier_count / pnp_num_points,
            scale_disagreement_ratio,
        ),
        dtype=torch.float32,
    )
    if quality.shape != (EYE_GEOMETRY_QUALITY_DIM,) or not torch.isfinite(quality).all():
        raise ValueError("Derived eye geometry quality must be a finite 4D vector.")
    return quality


@dataclass(frozen=True)
class DepthPriorTable:
    """Validated ``sample_id`` lookup for one offline depth-prior CSV."""

    path: Path
    eye_geometry_representation: str
    sample_ids: frozenset[str]
    geometry_by_sample_id: dict[str, Tensor]
    quality_by_sample_id: dict[str, Tensor]
    confidence_by_sample_id: dict[str, float]
    depth_log_scale_std_by_sample_id: dict[str, float]
    invalid_reasons: dict[str, str]
    quality_invalid_reasons: dict[str, str]
    uncertainty_invalid_reasons: dict[str, str]

    @classmethod
    def load(
        cls,
        path: str | Path,
        eye_geometry_representation: str = EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
    ) -> "DepthPriorTable":
        path = Path(path)
        eye_geometry_representation = canonical_eye_geometry_representation(
            eye_geometry_representation
        )
        if not path.is_file():
            raise FileNotFoundError(f"Depth-prior CSV does not exist: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Depth-prior CSV has no rows: {path}")

        required = {"sample_id", *EYE_CAMERA_COLUMNS}
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(
                f"Depth-prior CSV is missing columns {sorted(missing)}: {path}"
            )

        geometry_by_sample_id: dict[str, Tensor] = {}
        quality_by_sample_id: dict[str, Tensor] = {}
        confidence_by_sample_id: dict[str, float] = {}
        depth_log_scale_std_by_sample_id: dict[str, float] = {}
        invalid_reasons: dict[str, str] = {}
        quality_invalid_reasons: dict[str, str] = {}
        uncertainty_invalid_reasons: dict[str, str] = {}
        sample_ids: set[str] = set()
        has_quality_columns = set(EYE_QUALITY_COLUMNS).issubset(rows[0])
        has_uncertainty_columns = {
            DEPTH_LOG_SCALE_STD_COLUMN,
            DEPTH_UNCERTAINTY_STATUS_COLUMN,
        }.issubset(rows[0])
        for row in rows:
            sample_id = row["sample_id"].strip()
            if not sample_id:
                raise ValueError(f"Depth-prior CSV contains an empty sample_id: {path}")
            if sample_id in sample_ids:
                raise ValueError(
                    f"Depth-prior CSV contains duplicate sample_id={sample_id!r}: {path}"
                )
            sample_ids.add(sample_id)

            status = row.get("depth_prior_status", row.get("status", "success"))
            if status.strip().lower() not in VALID_PRIOR_STATUSES:
                invalid_reasons[sample_id] = (
                    row.get("reason", "").strip() or f"status={status!r}"
                )
                continue
            plausible = row.get("depth_is_plausible", "")
            if plausible and _is_false(plausible):
                invalid_reasons[sample_id] = "depth_is_plausible=false"
                continue

            try:
                values = [float(row[column]) for column in EYE_CAMERA_COLUMNS]
            except (TypeError, ValueError) as exc:
                invalid_reasons[sample_id] = f"invalid eye camera coordinates: {exc}"
                continue
            if not all(math.isfinite(value) for value in values):
                invalid_reasons[sample_id] = "non-finite eye camera coordinates"
                continue
            try:
                geometry_by_sample_id[sample_id] = build_eye_geometry_representation(
                    torch.tensor(values[:3], dtype=torch.float32),
                    torch.tensor(values[3:], dtype=torch.float32),
                    eye_geometry_representation,
                )
            except ValueError as exc:
                invalid_reasons[sample_id] = str(exc)
                continue

            if not has_quality_columns:
                quality_invalid_reasons[sample_id] = (
                    "depth-prior CSV does not contain all PnP quality columns"
                )
            else:
                try:
                    quality_values = {
                        column: float(row[column]) for column in EYE_QUALITY_COLUMNS
                    }
                    confidence = quality_values["pnp_confidence"]
                    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                        raise ValueError("pnp_confidence must be finite and lie in [0, 1]")
                    quality_by_sample_id[sample_id] = build_eye_geometry_quality_vector(
                        quality_values["reprojection_error_mean_px"],
                        quality_values["reprojection_error_max_px"],
                        quality_values["pnp_inlier_count"],
                        quality_values["pnp_num_points"],
                        quality_values["scale_disagreement_ratio"],
                    )
                    confidence_by_sample_id[sample_id] = confidence
                except (KeyError, TypeError, ValueError) as exc:
                    quality_invalid_reasons[sample_id] = (
                        f"invalid PnP quality values: {exc}"
                    )

            if not has_uncertainty_columns:
                uncertainty_invalid_reasons[sample_id] = (
                    "depth-prior CSV does not contain PnP uncertainty columns"
                )
                continue
            uncertainty_status = row.get(
                DEPTH_UNCERTAINTY_STATUS_COLUMN,
                "failed",
            ).strip().lower()
            if uncertainty_status not in VALID_PRIOR_STATUSES:
                uncertainty_invalid_reasons[sample_id] = (
                    row.get("depth_uncertainty_reason", "").strip()
                    or f"status={uncertainty_status!r}"
                )
                continue
            try:
                log_scale_std = float(row[DEPTH_LOG_SCALE_STD_COLUMN])
                if not math.isfinite(log_scale_std) or log_scale_std < 0:
                    raise ValueError(
                        "depth_log_scale_std must be finite and non-negative"
                    )
                depth_log_scale_std_by_sample_id[sample_id] = log_scale_std
            except (KeyError, TypeError, ValueError) as exc:
                uncertainty_invalid_reasons[sample_id] = (
                    f"invalid PnP depth uncertainty: {exc}"
                )

        return cls(
            path=path,
            eye_geometry_representation=eye_geometry_representation,
            sample_ids=frozenset(sample_ids),
            geometry_by_sample_id=geometry_by_sample_id,
            quality_by_sample_id=quality_by_sample_id,
            confidence_by_sample_id=confidence_by_sample_id,
            depth_log_scale_std_by_sample_id=depth_log_scale_std_by_sample_id,
            invalid_reasons=invalid_reasons,
            quality_invalid_reasons=quality_invalid_reasons,
            uncertainty_invalid_reasons=uncertainty_invalid_reasons,
        )

    def lookup(self, sample_id: str) -> Tensor:
        try:
            return self.geometry_by_sample_id[sample_id]
        except KeyError as exc:
            reason = self.invalid_reasons.get(sample_id)
            if reason is not None:
                raise KeyError(
                    f"Depth prior for sample_id={sample_id!r} is invalid: {reason}"
                ) from exc
            raise KeyError(
                f"Depth-prior CSV {self.path} has no sample_id={sample_id!r}"
            ) from exc

    def lookup_quality(self, sample_id: str) -> Tensor:
        try:
            return self.quality_by_sample_id[sample_id]
        except KeyError as exc:
            reason = self.quality_invalid_reasons.get(sample_id)
            if reason is not None:
                raise KeyError(
                    f"PnP quality for sample_id={sample_id!r} is invalid: {reason}"
                ) from exc
            raise KeyError(
                f"Depth-prior CSV {self.path} has no valid PnP quality for "
                f"sample_id={sample_id!r}"
            ) from exc

    def lookup_confidence(self, sample_id: str) -> float:
        try:
            return self.confidence_by_sample_id[sample_id]
        except KeyError as exc:
            reason = self.quality_invalid_reasons.get(sample_id)
            if reason is not None:
                raise KeyError(
                    f"PnP confidence for sample_id={sample_id!r} is invalid: {reason}"
                ) from exc
            raise KeyError(
                f"Depth-prior CSV {self.path} has no valid PnP confidence for "
                f"sample_id={sample_id!r}"
            ) from exc

    def lookup_depth_log_scale_std(self, sample_id: str) -> float:
        try:
            return self.depth_log_scale_std_by_sample_id[sample_id]
        except KeyError as exc:
            reason = self.uncertainty_invalid_reasons.get(sample_id)
            if reason is not None:
                raise KeyError(
                    f"PnP depth uncertainty for sample_id={sample_id!r} is "
                    f"invalid: {reason}"
                ) from exc
            raise KeyError(
                f"Depth-prior CSV {self.path} has no PnP depth uncertainty for "
                f"sample_id={sample_id!r}"
            ) from exc

    def is_gate_valid(self, sample_id: str, *, require_quality: bool) -> bool:
        if sample_id not in self.geometry_by_sample_id:
            return False
        if sample_id not in self.confidence_by_sample_id:
            return False
        return not require_quality or sample_id in self.quality_by_sample_id

    def require_sample_ids(self, sample_ids: Iterable[str]) -> None:
        missing: list[str] = []
        for sample_id in sample_ids:
            if sample_id not in self.geometry_by_sample_id:
                missing.append(sample_id)
        if missing:
            preview = ", ".join(repr(sample_id) for sample_id in missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                f"Depth-prior CSV {self.path} has no valid geometry for "
                f"{len(missing)} required samples: {preview}{suffix}"
            )

    def require_present_sample_ids(self, sample_ids: Iterable[str]) -> None:
        """Require rows to exist while allowing a quality gate to mask invalid priors."""

        missing = [sample_id for sample_id in sample_ids if sample_id not in self.sample_ids]
        if missing:
            preview = ", ".join(repr(sample_id) for sample_id in missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                f"Depth-prior CSV {self.path} has no row for {len(missing)} "
                f"required samples: {preview}{suffix}"
            )

    def require_uncertainty_sample_ids(self, sample_ids: Iterable[str]) -> None:
        missing = [
            sample_id
            for sample_id in sample_ids
            if sample_id not in self.depth_log_scale_std_by_sample_id
        ]
        if missing:
            preview = ", ".join(repr(sample_id) for sample_id in missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                f"Depth-prior CSV {self.path} has no valid PnP uncertainty for "
                f"{len(missing)} required samples: {preview}{suffix}"
            )
