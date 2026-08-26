"""Read versioned preprocessing hand-off contracts for ModelV1 training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from modelv1.processed_artifacts import canonical_dataset_id


PREPROCESSING_PIPELINE_VERSION = "modelv1_metric_depth_virtual_camera_v2"
PREPROCESSED_V2_DEPTH_METHODS = ("pnp1010", "iris65")


def canonical_preprocessed_v2_depth_method(value: str) -> str:
    """Normalize the two supported metric-depth hand-off names."""

    normalized = str(value).strip().lower()
    aliases = {
        "pnp": "pnp1010",
        "pnp1010": "pnp1010",
        "iris": "iris65",
        "iris65": "iris65",
        "iris_ipd_65mm": "iris65",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(PREPROCESSED_V2_DEPTH_METHODS)
        raise ValueError(
            f"Unknown preprocessed-v2 depth_method={value!r}; expected {supported}."
        ) from exc


def _required_text(payload: dict[str, object], key: str, manifest_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Preprocessed-v2 training manifest requires non-empty {key!r}: "
            f"{manifest_path}"
        )
    return value.strip()


def _referenced_path(payload: dict[str, object], key: str, manifest_path: Path) -> Path:
    path = Path(_required_text(payload, key, manifest_path))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Preprocessed-v2 manifest references missing {key}: {path}"
        )
    return path


@dataclass(frozen=True)
class PreprocessedV2TrainingInput:
    """Validated contents of one ``training_inputs_<method>_dataset<ID>.json``."""

    manifest_path: Path
    dataset_id: str
    dataset_name: str
    depth_method: str
    dataset_csv: Path
    depth_prior_csv: Path
    virtual_camera_manifest: Path
    virtual_camera_report: Path
    virtual_camera_pose_source: Path
    pnp_geometry_version: str
    deca_cache: Path | None
    sample_count: int

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_dataset_id: str | int | None = None,
        expected_depth_method: str | None = None,
    ) -> "PreprocessedV2TrainingInput":
        manifest_path = Path(path).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Preprocessed-v2 training manifest does not exist: {manifest_path}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                f"Preprocessed-v2 training manifest must contain a JSON object: "
                f"{manifest_path}"
            )
        pipeline_version = _required_text(
            payload, "pipeline_version", manifest_path
        )
        if pipeline_version != PREPROCESSING_PIPELINE_VERSION:
            raise ValueError(
                "Preprocessed-v2 pipeline version mismatch: expected "
                f"{PREPROCESSING_PIPELINE_VERSION!r}, got {pipeline_version!r} in "
                f"{manifest_path}"
            )

        dataset_id = canonical_dataset_id(
            _required_text(payload, "dataset_id", manifest_path)
        )
        dataset_name = _required_text(payload, "dataset_name", manifest_path)
        expected_name = f"dataset_dual_rigid_body_{dataset_id}"
        if dataset_name != expected_name:
            raise ValueError(
                f"dataset_name mismatch in {manifest_path}: expected "
                f"{expected_name!r}, got {dataset_name!r}."
            )
        if expected_dataset_id is not None:
            requested_id = canonical_dataset_id(expected_dataset_id)
            if dataset_id != requested_id:
                raise ValueError(
                    f"Training manifest dataset mismatch: requested {requested_id}, "
                    f"got {dataset_id} in {manifest_path}."
                )

        depth_method = canonical_preprocessed_v2_depth_method(
            _required_text(payload, "depth_method", manifest_path)
        )
        if expected_depth_method is not None:
            requested_method = canonical_preprocessed_v2_depth_method(
                expected_depth_method
            )
            if depth_method != requested_method:
                raise ValueError(
                    f"Training manifest depth method mismatch: requested "
                    f"{requested_method!r}, got {depth_method!r} in {manifest_path}."
                )

        sample_count_value = payload.get("sample_count")
        if isinstance(sample_count_value, bool):
            raise ValueError(f"sample_count must be a positive integer: {manifest_path}")
        try:
            sample_count = int(sample_count_value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sample_count must be a positive integer: {manifest_path}"
            ) from exc
        if sample_count <= 0:
            raise ValueError(f"sample_count must be positive: {manifest_path}")

        deca_value = payload.get("deca_cache")
        deca_cache: Path | None
        if deca_value is None:
            deca_cache = None
        elif isinstance(deca_value, str) and deca_value.strip():
            deca_path = Path(deca_value.strip())
            if not deca_path.is_absolute():
                deca_path = manifest_path.parent / deca_path
            deca_cache = deca_path.resolve()
            if not deca_cache.is_file():
                raise FileNotFoundError(
                    f"Preprocessed-v2 manifest references missing deca_cache: "
                    f"{deca_cache}"
                )
        else:
            raise ValueError(
                f"deca_cache must be a path string or null: {manifest_path}"
            )

        return cls(
            manifest_path=manifest_path,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            depth_method=depth_method,
            dataset_csv=_referenced_path(payload, "dataset_csv", manifest_path),
            depth_prior_csv=_referenced_path(
                payload, "depth_prior_csv", manifest_path
            ),
            virtual_camera_manifest=_referenced_path(
                payload, "virtual_camera_manifest", manifest_path
            ),
            virtual_camera_report=_referenced_path(
                payload, "virtual_camera_report", manifest_path
            ),
            virtual_camera_pose_source=_referenced_path(
                payload, "virtual_camera_pose_source", manifest_path
            ),
            pnp_geometry_version=_required_text(
                payload, "pnp_geometry_version", manifest_path
            ),
            deca_cache=deca_cache,
            sample_count=sample_count,
        )


def load_preprocessed_v2_training_inputs(
    root: str | Path,
    dataset_ids: Iterable[str | int],
    depth_method: str,
) -> list[PreprocessedV2TrainingInput]:
    """Load one authoritative training hand-off per requested Dataset."""

    root = Path(root).resolve()
    method = canonical_preprocessed_v2_depth_method(depth_method)
    normalized_ids = [canonical_dataset_id(value) for value in dataset_ids]
    if not normalized_ids:
        raise ValueError("preprocessed-v2 dataset_ids must not be empty.")
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError(
            f"preprocessed-v2 dataset_ids contain duplicates: {normalized_ids}"
        )
    return [
        PreprocessedV2TrainingInput.load(
            root
            / f"dataset{dataset_id}"
            / "manifests"
            / f"training_inputs_{method}_dataset{dataset_id}.json",
            expected_dataset_id=dataset_id,
            expected_depth_method=method,
        )
        for dataset_id in normalized_ids
    ]
