"""Canonical names for one independently preprocessed collection dataset.

Keeping this convention in one small dependency-free module lets preprocessing
and training select a dataset by its numeric suffix instead of repeatedly
editing unrelated file paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def canonical_dataset_id(value: str | int) -> str:
    """Normalize ``11`` / ``dataset_dual_rigid_body_11`` to ``"11"``."""

    text = str(value).strip()
    prefix = "dataset_dual_rigid_body_"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if not text.isdigit() or int(text) < 0:
        raise ValueError(
            "dataset_id must be a non-negative integer or "
            "dataset_dual_rigid_body_<integer>, got "
            f"{value!r}."
        )
    return str(int(text))


@dataclass(frozen=True)
class ProcessedDatasetArtifacts:
    """All derived files that belong to one collection dataset."""

    dataset_id: str
    processed_dir: Path

    @property
    def suffix(self) -> str:
        return f"_{self.dataset_id}"

    @property
    def dataset_name(self) -> str:
        return f"dataset_dual_rigid_body_{self.dataset_id}"

    @property
    def dataset_csv(self) -> Path:
        # The established dataset-table convention predates the other artifacts
        # and intentionally has no separator before the numeric ID.
        return self.processed_dir / f"modelv1_dataset{self.dataset_id}.csv"

    @property
    def dataset_report(self) -> Path:
        return self.processed_dir / f"modelv1_dataset{self.dataset_id}_report.json"

    @property
    def deca_cache(self) -> Path:
        return self.processed_dir / f"deca_features_deca_crop_v1{self.suffix}.npz"

    @property
    def mediapipe_iris_centres(self) -> Path:
        return self.processed_dir / f"mediapipe_iris_centres{self.suffix}.csv"

    @property
    def mediapipe_pnp_landmarks(self) -> Path:
        return self.processed_dir / f"mediapipe_pnp_landmarks{self.suffix}.csv"

    @property
    def iris_ipd_depth_prior(self) -> Path:
        return self.processed_dir / f"depth_priors_iris_ipd_65mm_v1{self.suffix}.csv"

    @property
    def pnp_depth_prior(self) -> Path:
        return self.processed_dir / f"depth_priors_deca_crop_v1{self.suffix}.csv"

    @property
    def common_depth_prior_dataset_csv(self) -> Path:
        return self.processed_dir / f"modelv1_dataset_depth_prior_common{self.suffix}.csv"


def processed_dataset_artifacts(
    dataset_id: str | int,
    processed_dir: str | Path = "data/processed",
) -> ProcessedDatasetArtifacts:
    return ProcessedDatasetArtifacts(
        dataset_id=canonical_dataset_id(dataset_id),
        processed_dir=Path(processed_dir),
    )
