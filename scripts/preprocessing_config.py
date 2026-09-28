"""Typed YAML configuration for the ModelV1 preprocessing controller."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "preprocessing" / "modelv1_preprocessing_v2.yaml"
)


def canonical_dataset_id(value: str) -> str:
    text = str(value).strip()
    prefix = "dataset_dual_rigid_body_"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if not text.isdigit() or int(text) < 0:
        raise argparse.ArgumentTypeError(
            "dataset ID must be a non-negative integer or "
            "dataset_dual_rigid_body_<ID>."
        )
    return str(int(text))


def depth_method(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "pnp": "pnp1010",
        "pnp1010": "pnp1010",
        "iris": "iris65",
        "iris65": "iris65",
        "iris_ipd_65mm": "iris65",
        "both": "both",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "depth method must be pnp1010, iris65, or both."
        ) from exc


def _config_section(
    payload: Mapping[str, Any], name: str, *, config_path: Path
) -> Mapping[str, Any]:
    section = payload.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"{config_path}: '{name}' must be a YAML mapping.")
    return section


def _config_value(
    section: Mapping[str, Any],
    name: str,
    expected_type: type,
    *,
    config_path: Path,
    optional: bool = False,
) -> Any:
    value = section.get(name)
    if value is None and optional:
        return None
    if value is None:
        raise ValueError(f"{config_path}: missing required setting '{name}'.")
    if expected_type is bool:
        valid = isinstance(value, bool)
    elif expected_type is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected_type is float:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        valid = isinstance(value, expected_type)
    if not valid:
        suffix = " or null" if optional else ""
        raise ValueError(
            f"{config_path}: '{name}' must be {expected_type.__name__}{suffix}."
        )
    if expected_type is str and not value.strip():
        raise ValueError(f"{config_path}: '{name}' cannot be empty.")
    return float(value) if expected_type is float else value


def _resolve_config_path(
    value: str, *, dataset_id: str, config_path: Path, name: str
) -> Path:
    expanded = os.path.expandvars(value.strip()).replace("{dataset_id}", dataset_id)
    resolved = Path(expanded).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


@dataclass(frozen=True)
class PreprocessingConfig:
    config_path: Path
    dataset_id: str
    depth_method: str
    source_root: Path
    output_root: Path
    face_image_dir: str
    insightface_coordinates_file: str
    handeye_translation_scale: float
    table_z_source: str
    pnp_scale_mm_per_flame_unit: float
    ipd_mm: float
    max_ipd_residual_mm: float
    max_ray_condition: float
    calibration_start: int | None
    calibration_stop: int | None
    calibration_index_mode: str
    estimate_depth_uncertainty: bool
    uncertainty_bootstrap_samples: int
    uncertainty_iris_std_px: float
    uncertainty_ipd_std_mm: float
    uncertainty_min_success_count: int
    uncertainty_seed: int
    virtual_face_width: int
    virtual_face_height: int
    virtual_eye_width: int
    virtual_eye_height: int
    virtual_eye_span_scale: float
    virtual_focal_length_px: float
    virtual_distance_mm: float
    virtual_limit_per_dataset: int | None
    reuse_base_dataset: bool
    overwrite: bool
    dry_run: bool

    def __post_init__(self) -> None:
        if Path(self.face_image_dir).name != self.face_image_dir:
            raise ValueError("input.face_image_dir must be one directory name, not a path.")
        if Path(self.insightface_coordinates_file).name != self.insightface_coordinates_file:
            raise ValueError(
                "input.insightface_coordinates_file must be one filename, not a path."
            )
        if self.handeye_translation_scale <= 0:
            raise ValueError("dataset_build.handeye_translation_scale must be positive.")
        if self.table_z_source not in {"per-sample-target", "per-dataset-mean"}:
            raise ValueError(
                "dataset_build.table_z_source must be per-sample-target or per-dataset-mean."
            )
        if self.pnp_scale_mm_per_flame_unit <= 0:
            raise ValueError("pnp1010.fixed_scale_mm_per_flame_unit must be positive.")
        image_sizes = (
            self.virtual_face_width,
            self.virtual_face_height,
            self.virtual_eye_width,
            self.virtual_eye_height,
        )
        if min(image_sizes) <= 0:
            raise ValueError("virtual_camera image widths and heights must be positive.")
        virtual_scales = (
            self.virtual_eye_span_scale,
            self.virtual_focal_length_px,
            self.virtual_distance_mm,
        )
        if min(virtual_scales) <= 0:
            raise ValueError("virtual_camera scale, focal length and distance must be positive.")
        if self.virtual_limit_per_dataset is not None and self.virtual_limit_per_dataset <= 0:
            raise ValueError("virtual_camera.limit_per_dataset must be positive or null.")
        if self.calibration_index_mode not in {"frame_idx", "position"}:
            raise ValueError(
                "iris65.calibration_index_mode must be frame_idx or position."
            )

        needs_iris = self.depth_method in {"iris65", "both"}
        if needs_iris:
            if self.ipd_mm <= 0:
                raise ValueError("iris65.ipd_mm must be positive.")
            if self.max_ipd_residual_mm <= 0 or self.max_ray_condition <= 0:
                raise ValueError(
                    "iris65.max_ipd_residual_mm and max_ray_condition must be positive."
                )
            if self.calibration_start is None or self.calibration_stop is None:
                raise ValueError(
                    "iris65.calibration_start and calibration_stop are required."
                )
            if self.calibration_stop < self.calibration_start:
                raise ValueError("iris65.calibration_stop must be >= calibration_start.")
        elif self.estimate_depth_uncertainty:
            raise ValueError(
                "iris65.estimate_depth_uncertainty requires depth_method iris65 or both."
            )

        if self.uncertainty_bootstrap_samples <= 0:
            raise ValueError("iris65.uncertainty_bootstrap_samples must be positive.")
        if self.uncertainty_iris_std_px < 0 or self.uncertainty_ipd_std_mm < 0:
            raise ValueError("iris65 uncertainty standard deviations cannot be negative.")
        if self.uncertainty_min_success_count <= 0:
            raise ValueError("iris65.uncertainty_min_success_count must be positive.")

    def manifest_parameters(self) -> dict[str, Any]:
        return {
            "dataset": {"id": self.dataset_id, "depth_method": self.depth_method},
            "paths": {
                "source_root": str(self.source_root),
                "output_root": str(self.output_root),
            },
            "input": {
                "face_image_dir": self.face_image_dir,
                "insightface_coordinates_file": self.insightface_coordinates_file,
            },
            "dataset_build": {
                "handeye_translation_scale": self.handeye_translation_scale,
                "table_z_source": self.table_z_source,
            },
            "pnp1010": {
                "fixed_scale_mm_per_flame_unit": self.pnp_scale_mm_per_flame_unit,
            },
            "iris65": {
                "ipd_mm": self.ipd_mm,
                "max_ipd_residual_mm": self.max_ipd_residual_mm,
                "max_ray_condition": self.max_ray_condition,
                "calibration_start": self.calibration_start,
                "calibration_stop": self.calibration_stop,
                "calibration_index_mode": self.calibration_index_mode,
                "estimate_depth_uncertainty": self.estimate_depth_uncertainty,
                "uncertainty_bootstrap_samples": self.uncertainty_bootstrap_samples,
                "uncertainty_iris_std_px": self.uncertainty_iris_std_px,
                "uncertainty_ipd_std_mm": self.uncertainty_ipd_std_mm,
                "uncertainty_min_success_count": self.uncertainty_min_success_count,
                "uncertainty_seed": self.uncertainty_seed,
            },
            "virtual_camera": {
                "face_width": self.virtual_face_width,
                "face_height": self.virtual_face_height,
                "eye_width": self.virtual_eye_width,
                "eye_height": self.virtual_eye_height,
                "eye_span_scale": self.virtual_eye_span_scale,
                "focal_length_px": self.virtual_focal_length_px,
                "distance_mm": self.virtual_distance_mm,
                "limit_per_dataset": self.virtual_limit_per_dataset,
            },
            "execution": {
                "reuse_base_dataset": self.reuse_base_dataset,
                "overwrite": self.overwrite,
                "dry_run": self.dry_run,
            },
        }


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    config_path = path.expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Preprocessing config does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{config_path}: YAML root must be a mapping.")

    sections = {
        name: _config_section(payload, name, config_path=config_path)
        for name in (
            "dataset",
            "paths",
            "input",
            "dataset_build",
            "pnp1010",
            "iris65",
            "virtual_camera",
            "execution",
        )
    }
    dataset = sections["dataset"]
    paths = sections["paths"]
    input_config = sections["input"]
    dataset_build = sections["dataset_build"]
    pnp = sections["pnp1010"]
    iris = sections["iris65"]
    virtual = sections["virtual_camera"]
    execution = sections["execution"]

    raw_dataset_id = dataset.get("id")
    if (
        isinstance(raw_dataset_id, bool)
        or not isinstance(raw_dataset_id, (str, int))
    ):
        raise ValueError(f"{config_path}: 'dataset.id' must be a string or integer.")
    dataset_id = canonical_dataset_id(str(raw_dataset_id))
    selected_depth_method = depth_method(
        _config_value(dataset, "depth_method", str, config_path=config_path)
    )
    source_root_text = _config_value(
        paths, "source_root", str, config_path=config_path
    )
    output_root_text = _config_value(
        paths, "output_root", str, config_path=config_path
    )

    return PreprocessingConfig(
        config_path=config_path,
        dataset_id=dataset_id,
        depth_method=selected_depth_method,
        source_root=_resolve_config_path(
            source_root_text,
            dataset_id=dataset_id,
            config_path=config_path,
            name="paths.source_root",
        ),
        output_root=_resolve_config_path(
            output_root_text,
            dataset_id=dataset_id,
            config_path=config_path,
            name="paths.output_root",
        ),
        face_image_dir=_config_value(
            input_config, "face_image_dir", str, config_path=config_path
        ),
        insightface_coordinates_file=_config_value(
            input_config,
            "insightface_coordinates_file",
            str,
            config_path=config_path,
        ),
        handeye_translation_scale=_config_value(
            dataset_build,
            "handeye_translation_scale",
            float,
            config_path=config_path,
        ),
        table_z_source=_config_value(
            dataset_build, "table_z_source", str, config_path=config_path
        ),
        pnp_scale_mm_per_flame_unit=_config_value(
            pnp,
            "fixed_scale_mm_per_flame_unit",
            float,
            config_path=config_path,
        ),
        ipd_mm=_config_value(iris, "ipd_mm", float, config_path=config_path),
        max_ipd_residual_mm=_config_value(
            iris, "max_ipd_residual_mm", float, config_path=config_path
        ),
        max_ray_condition=_config_value(
            iris, "max_ray_condition", float, config_path=config_path
        ),
        calibration_start=_config_value(
            iris, "calibration_start", int, config_path=config_path, optional=True
        ),
        calibration_stop=_config_value(
            iris, "calibration_stop", int, config_path=config_path, optional=True
        ),
        calibration_index_mode=_config_value(
            iris, "calibration_index_mode", str, config_path=config_path
        ),
        estimate_depth_uncertainty=_config_value(
            iris, "estimate_depth_uncertainty", bool, config_path=config_path
        ),
        uncertainty_bootstrap_samples=_config_value(
            iris, "uncertainty_bootstrap_samples", int, config_path=config_path
        ),
        uncertainty_iris_std_px=_config_value(
            iris, "uncertainty_iris_std_px", float, config_path=config_path
        ),
        uncertainty_ipd_std_mm=_config_value(
            iris, "uncertainty_ipd_std_mm", float, config_path=config_path
        ),
        uncertainty_min_success_count=_config_value(
            iris, "uncertainty_min_success_count", int, config_path=config_path
        ),
        uncertainty_seed=_config_value(
            iris, "uncertainty_seed", int, config_path=config_path
        ),
        virtual_face_width=_config_value(
            virtual, "face_width", int, config_path=config_path
        ),
        virtual_face_height=_config_value(
            virtual, "face_height", int, config_path=config_path
        ),
        virtual_eye_width=_config_value(
            virtual, "eye_width", int, config_path=config_path
        ),
        virtual_eye_height=_config_value(
            virtual, "eye_height", int, config_path=config_path
        ),
        virtual_eye_span_scale=_config_value(
            virtual, "eye_span_scale", float, config_path=config_path
        ),
        virtual_focal_length_px=_config_value(
            virtual, "focal_length_px", float, config_path=config_path
        ),
        virtual_distance_mm=_config_value(
            virtual, "distance_mm", float, config_path=config_path
        ),
        virtual_limit_per_dataset=_config_value(
            virtual,
            "limit_per_dataset",
            int,
            config_path=config_path,
            optional=True,
        ),
        reuse_base_dataset=_config_value(
            execution, "reuse_base_dataset", bool, config_path=config_path
        ),
        overwrite=_config_value(
            execution, "overwrite", bool, config_path=config_path
        ),
        dry_run=_config_value(execution, "dry_run", bool, config_path=config_path),
    )
