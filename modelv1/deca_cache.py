"""Schema and reader for frozen DECA face features.

The cache deliberately stores only DECA's coarse ``E_flame`` output.  This is
the 236-D parameter vector expected by :class:`modelv1.ModelV1` and does not
require the FLAME decoder or DECA renderer at training time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    np = None  # type: ignore[assignment]


DECA_CACHE_FORMAT_VERSION = 1
DECA_FEATURE_DIM = 236
DECA_PARAMETER_LAYOUT = {
    "shape": [0, 100],
    "tex": [100, 150],
    "exp": [150, 200],
    "pose": [200, 206],
    "cam": [206, 209],
    "light": [209, 236],
}


def require_numpy() -> Any:
    if np is None:
        raise ModuleNotFoundError(
            "NumPy is required to read DECA feature caches. Install numpy first."
        )
    return np


@dataclass
class DecaFeatureCache:
    """In-memory lookup table loaded from one ``.npz`` DECA cache file."""

    path: Path
    features: np.ndarray
    sample_ids: tuple[str, ...]
    image_sha256: tuple[str, ...]
    metadata: dict[str, object]
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        numpy = require_numpy()
        if self.features.ndim != 2:
            raise ValueError(
                f"DECA features must be 2D, got shape {self.features.shape} in {self.path}"
            )
        if self.features.shape[1] != DECA_FEATURE_DIM:
            raise ValueError(
                f"Expected {DECA_FEATURE_DIM} DECA features, got {self.features.shape[1]} in {self.path}"
            )
        if not numpy.isfinite(self.features).all():
            raise ValueError(f"DECA cache contains non-finite features: {self.path}")
        if len(self.sample_ids) != self.features.shape[0]:
            raise ValueError(f"sample_id count does not match feature count in {self.path}")
        if len(self.image_sha256) != self.features.shape[0]:
            raise ValueError(f"image_sha256 count does not match feature count in {self.path}")

        self._index = {sample_id: idx for idx, sample_id in enumerate(self.sample_ids)}
        if len(self._index) != len(self.sample_ids):
            raise ValueError(f"Duplicate sample_id values in DECA cache: {self.path}")

    @classmethod
    def load(cls, path: str | Path) -> "DecaFeatureCache":
        """Load and validate an object-free ``.npz`` cache."""

        numpy = require_numpy()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DECA feature cache does not exist: {path}")

        try:
            with numpy.load(path, allow_pickle=False) as data:
                required = {"deca_feat", "sample_id", "image_sha256", "metadata_json"}
                missing = required.difference(data.files)
                if missing:
                    raise ValueError(
                        f"DECA cache {path} is missing required arrays: {sorted(missing)}"
                    )
                features = numpy.asarray(data["deca_feat"], dtype=numpy.float32)
                sample_ids = tuple(str(value) for value in data["sample_id"].tolist())
                image_sha256 = tuple(
                    str(value) for value in data["image_sha256"].tolist()
                )
                metadata_json = str(data["metadata_json"].item())
        except ValueError as exc:
            if "Object arrays" in str(exc):
                raise ValueError(
                    f"DECA cache must not use pickle/object arrays: {path}"
                ) from exc
            raise

        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid metadata_json in DECA cache: {path}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata_json must decode to an object in DECA cache: {path}")
        if metadata.get("cache_format_version") != DECA_CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported DECA cache format in {path}: "
                f"{metadata.get('cache_format_version')!r}"
            )

        return cls(path, features, sample_ids, image_sha256, metadata)

    def lookup(self, sample_id: str) -> np.ndarray:
        """Return one immutable 236-D float32 vector by dataset sample id."""

        try:
            return self.features[self._index[sample_id]]
        except KeyError as exc:
            raise KeyError(f"DECA cache has no feature for sample_id={sample_id!r}") from exc

    def has_sample_id(self, sample_id: str) -> bool:
        return sample_id in self._index

    def image_digest(self, sample_id: str) -> str:
        try:
            return self.image_sha256[self._index[sample_id]]
        except KeyError as exc:
            raise KeyError(f"DECA cache has no feature for sample_id={sample_id!r}") from exc

    def require_sample_ids(self, sample_ids: Iterable[str]) -> None:
        """Fail early when a CSV and cache do not describe the same samples."""

        missing = [sample_id for sample_id in sample_ids if sample_id not in self._index]
        if missing:
            preview = ", ".join(repr(sample_id) for sample_id in missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(
                f"DECA cache {self.path} misses {len(missing)} dataset samples: {preview}{suffix}"
            )
