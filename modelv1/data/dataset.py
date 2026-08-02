"""PyTorch Dataset and DataLoader helpers for the ModelV1 gaze dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable, Literal

from modelv1.deca_cache import (
    DECA_FEATURE_REPRESENTATION_FULL236,
    DecaFeatureCache,
    canonical_deca_feature_representation,
    select_deca_feature_representation,
)
from modelv1.data.depth_prior import (
    EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
    DepthPriorTable,
    canonical_eye_geometry_representation,
)
from modelv1.data.normalization import (
    EYE_GEOMETRY_DIM,
    EYE_GEOMETRY_QUALITY_DIM,
    EyeGeometryNormalizer,
    EyeGeometryQualityNormalizer,
    UVTargetNormalizer,
    fit_uv_target_normalizer,
)
from modelv1.geometry_gate import (
    EYE_GEOMETRY_GATE_LEARNED_RESIDUAL,
    EYE_GEOMETRY_GATE_NONE,
    canonical_eye_geometry_gate_mode,
)
from modelv1.scene import (
    ORIENTATION_BASIS_INDICES,
    SCENE_REPRESENTATION_FULL25,
    SCENE_REPRESENTATION_ORIENTATION6D,
    SCENE_REPRESENTATION_TABLE_FRAME7,
    TABLE_FRAME_DISTANCE_INDEX,
    TABLE_FRAME_DISTANCE_SCALE_MM,
    canonical_scene_representation,
)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    raise ModuleNotFoundError(
        "PyTorch is required for modelv1.data. Install torch before using the Dataset."
    ) from exc

try:
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    raise ModuleNotFoundError(
        "Pillow is required for image loading. Install Pillow before using the Dataset."
    ) from exc


CROP_CAM_COLUMNS = [f"crop_cam_{idx:02d}" for idx in range(36)]
SCENE_COLUMNS = [f"scene_{idx:02d}" for idx in range(25)]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECA_CACHE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "deca_features_deca_crop_v1.npz"
)
DEFAULT_DEPTH_PRIOR_PATH = (
    PROJECT_ROOT / "data" / "processed" / "depth_priors_deca_crop_v1.csv"
)

FACE_SIZE = (224, 224)
EYE_SIZE = (60, 36)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageTransform = Callable[[torch.Tensor], torch.Tensor]
SplitMode = Literal["random_80_20", "dataset_5"]


class ModelV1Dataset(Dataset):
    """Dataset backed by `data/processed/modelv1_dataset.csv`.

    Each item returns a dictionary with:

    - `face` (optional), `left_eye`, `right_eye`: image tensors in CHW format.
    - `crop_cam_vec`: 36D crop/camera metadata tensor.
    - `scene_vec`: selected full25, TableFrame 7D, or Orientation 6D tensor.
    - `deca_feat`: cached 236D frozen DECA face feature.
      Its representation is selected as `full236` or `geometry156`.
    - `eye_geometry_vec` (optional): selected, training-z-score-normalized 6D
      binocular camera geometry.
    - gated modes also return confidence/mask and, for learned mode, 4D quality.
    - `uv_gt`: 2D table-local gaze target in millimeters.
    - `uv_target`: z-score-normalized target used by the UV head.
    - extra metadata for validation and debugging.
    """

    def __init__(
        self,
        csv_path: str | Path,
        datasets: Iterable[str] | None = None,
        normalize_images: bool = True,
        face_transform: ImageTransform | None = None,
        eye_transform: ImageTransform | None = None,
        return_paths: bool = True,
        load_face_image: bool = True,
        deca_cache_path: str | Path | None = DEFAULT_DECA_CACHE_PATH,
        require_deca_features: bool = True,
        target_normalizer: UVTargetNormalizer | None = None,
        fit_target_normalizer: bool = True,
        use_eye_geometry: bool = False,
        depth_prior_csv_path: str | Path | None = None,
        depth_prior_table: DepthPriorTable | None = None,
        eye_geometry_normalizer: EyeGeometryNormalizer | None = None,
        eye_geometry_gate_mode: str = EYE_GEOMETRY_GATE_NONE,
        eye_geometry_quality_normalizer: EyeGeometryQualityNormalizer | None = None,
        eye_geometry_representation: str = EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
        scene_representation: str = SCENE_REPRESENTATION_FULL25,
        deca_feature_representation: str = DECA_FEATURE_REPRESENTATION_FULL236,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.normalize_images = normalize_images
        self.face_transform = face_transform
        self.eye_transform = eye_transform
        self.return_paths = return_paths
        self.load_face_image = load_face_image
        self.use_eye_geometry = use_eye_geometry
        self.eye_geometry_normalizer = eye_geometry_normalizer
        self.eye_geometry_representation = canonical_eye_geometry_representation(
            eye_geometry_representation
        )
        self.eye_geometry_gate_mode = canonical_eye_geometry_gate_mode(
            eye_geometry_gate_mode
        )
        if not self.use_eye_geometry and self.eye_geometry_gate_mode != EYE_GEOMETRY_GATE_NONE:
            raise ValueError(
                "eye_geometry_gate_mode must be 'none' when use_eye_geometry=False."
            )
        self.use_eye_geometry_quality_gate = (
            self.use_eye_geometry
            and self.eye_geometry_gate_mode != EYE_GEOMETRY_GATE_NONE
        )
        self.eye_geometry_quality_normalizer = eye_geometry_quality_normalizer
        self.scene_representation = canonical_scene_representation(
            scene_representation
        )
        self.deca_feature_representation = canonical_deca_feature_representation(
            deca_feature_representation
        )
        if not load_face_image and face_transform is not None:
            raise ValueError("face_transform requires load_face_image=True.")
        if depth_prior_csv_path is not None and depth_prior_table is not None:
            raise ValueError(
                "Provide either depth_prior_csv_path or depth_prior_table, not both."
            )

        requested = normalize_dataset_names(datasets)
        self.rows = read_rows(self.csv_path)
        if requested is not None:
            self.rows = [row for row in self.rows if row["dataset"] in requested]

        if not self.rows:
            raise ValueError(f"No samples found in {self.csv_path}")

        validate_required_columns(self.rows[0])
        self.target_normalizer = target_normalizer
        if self.target_normalizer is None and fit_target_normalizer:
            self.target_normalizer = fit_uv_target_normalizer(self.rows)
        if require_deca_features and deca_cache_path is None:
            raise ValueError("require_deca_features=True requires deca_cache_path.")
        self.deca_cache = (
            DecaFeatureCache.load(deca_cache_path) if deca_cache_path is not None else None
        )
        if self.deca_cache is not None:
            self.deca_cache.require_sample_ids(row["sample_id"] for row in self.rows)
        if self.use_eye_geometry:
            if depth_prior_table is None:
                if depth_prior_csv_path is None:
                    raise ValueError(
                        "use_eye_geometry=True requires a depth-prior CSV."
                    )
                depth_prior_table = DepthPriorTable.load(
                    depth_prior_csv_path,
                    self.eye_geometry_representation,
                )
            if (
                depth_prior_table.eye_geometry_representation
                != self.eye_geometry_representation
            ):
                raise ValueError(
                    "depth_prior_table representation does not match "
                    "eye_geometry_representation."
                )
            sample_ids = [row["sample_id"] for row in self.rows]
            if self.use_eye_geometry_quality_gate:
                depth_prior_table.require_present_sample_ids(sample_ids)
            else:
                depth_prior_table.require_sample_ids(sample_ids)
        self.depth_prior_table = depth_prior_table if self.use_eye_geometry else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]

        face = (
            self._load_image(row["face_path"], FACE_SIZE)
            if self.load_face_image
            else None
        )
        left_eye = self._load_image(row["left_eye_path"], EYE_SIZE)
        right_eye = self._load_image(row["right_eye_path"], EYE_SIZE)

        if face is not None and self.face_transform is not None:
            face = self.face_transform(face)
        if self.eye_transform is not None:
            left_eye = self.eye_transform(left_eye)
            right_eye = self.eye_transform(right_eye)

        uv_gt = float_tensor(row, ["uv_gt_u_mm", "uv_gt_v_mm"])
        item: dict[str, object] = {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "crop_cam_vec": float_tensor(row, CROP_CAM_COLUMNS),
            "scene_vec": build_scene_input_vector(
                float_tensor(row, SCENE_COLUMNS),
                self.scene_representation,
            ),
            "uv_gt": uv_gt,
            "uv_target": (
                self.target_normalizer.normalize(uv_gt)
                if self.target_normalizer is not None
                else uv_gt.clone()
            ),
            "gaze_target_w": float_tensor(
                row,
                ["gaze_target_w_x_mm", "gaze_target_w_y_mm", "gaze_target_w_z_mm"],
            ),
            "table_origin_w": float_tensor(
                row,
                ["table_origin_w_x_mm", "table_origin_w_y_mm", "table_origin_w_z_mm"],
            ),
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "image_name": row["image_name"],
        }
        if face is not None:
            item["face"] = face
        if self.deca_cache is not None:
            item["deca_feat"] = torch.from_numpy(
                select_deca_feature_representation(
                    self.deca_cache.lookup(row["sample_id"]),
                    self.deca_feature_representation,
                ).copy()
            )
        if self.depth_prior_table is not None:
            sample_id = row["sample_id"]
            require_quality = (
                self.eye_geometry_gate_mode == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
            )
            gate_valid = (
                self.depth_prior_table.is_gate_valid(
                    sample_id,
                    require_quality=require_quality,
                )
                if self.use_eye_geometry_quality_gate
                else True
            )
            if gate_valid:
                geometry = self.depth_prior_table.lookup(sample_id).clone()
                if self.eye_geometry_normalizer is not None:
                    geometry = self.eye_geometry_normalizer.normalize(geometry)
            else:
                geometry = torch.zeros(EYE_GEOMETRY_DIM, dtype=torch.float32)
            item["eye_geometry_vec"] = geometry

            if self.use_eye_geometry_quality_gate:
                item["eye_geometry_confidence"] = torch.tensor(
                    [self.depth_prior_table.lookup_confidence(sample_id)]
                    if gate_valid
                    else [0.0],
                    dtype=torch.float32,
                )
                item["eye_geometry_valid_mask"] = torch.tensor(
                    [1.0 if gate_valid else 0.0],
                    dtype=torch.float32,
                )
            if (
                self.eye_geometry_gate_mode
                == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
            ):
                if gate_valid:
                    quality = self.depth_prior_table.lookup_quality(sample_id).clone()
                    if self.eye_geometry_quality_normalizer is not None:
                        quality = self.eye_geometry_quality_normalizer.normalize(quality)
                else:
                    quality = torch.zeros(
                        EYE_GEOMETRY_QUALITY_DIM,
                        dtype=torch.float32,
                    )
                item["eye_geometry_quality_vec"] = quality
        if self.return_paths:
            item["paths"] = {
                "face": row["face_path"],
                "left_eye": row["left_eye_path"],
                "right_eye": row["right_eye_path"],
                "source": row["source_image_path"],
            }
        return item

    def _load_image(self, path_text: str, size: tuple[int, int]) -> torch.Tensor:
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")

        image = Image.open(path).convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.BILINEAR)

        width, height = image.size
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(height, width, 3).permute(2, 0, 1).float().div(255.0)

        if self.normalize_images:
            mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor

    def raw_eye_geometry(self, index: int) -> torch.Tensor:
        if self.depth_prior_table is None:
            raise RuntimeError("This dataset was created without eye geometry.")
        return self.depth_prior_table.lookup(self.rows[index]["sample_id"]).clone()

    def raw_eye_geometry_quality(self, index: int) -> torch.Tensor:
        if self.depth_prior_table is None:
            raise RuntimeError("This dataset was created without eye geometry.")
        return self.depth_prior_table.lookup_quality(
            self.rows[index]["sample_id"]
        ).clone()

    def eye_geometry_is_valid(self, index: int) -> bool:
        if self.depth_prior_table is None:
            return False
        if not self.use_eye_geometry_quality_gate:
            return self.rows[index]["sample_id"] in self.depth_prior_table.geometry_by_sample_id
        return self.depth_prior_table.is_gate_valid(
            self.rows[index]["sample_id"],
            require_quality=(
                self.eye_geometry_gate_mode
                == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
            ),
        )


def build_modelv1_dataloaders(
    csv_path: str | Path = "data/processed/modelv1_dataset.csv",
    train_datasets: Iterable[str] = ("3", "4"),
    val_datasets: Iterable[str] = ("5",),
    split_mode: SplitMode = "dataset_5",
    all_datasets: Iterable[str] = ("3", "4", "5"),
    val_ratio: float = 0.2,
    split_seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    normalize_images: bool = True,
    load_face_image: bool = True,
    deca_cache_path: str | Path | None = DEFAULT_DECA_CACHE_PATH,
    require_deca_features: bool = True,
    normalize_uv_targets: bool = True,
    target_normalizer: UVTargetNormalizer | None = None,
    use_eye_geometry: bool = False,
    depth_prior_csv_path: str | Path | None = DEFAULT_DEPTH_PRIOR_PATH,
    eye_geometry_normalizer: EyeGeometryNormalizer | None = None,
    eye_geometry_gate_mode: str = EYE_GEOMETRY_GATE_NONE,
    eye_geometry_quality_normalizer: EyeGeometryQualityNormalizer | None = None,
    eye_geometry_representation: str = EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
    scene_representation: str = SCENE_REPRESENTATION_FULL25,
    deca_feature_representation: str = DECA_FEATURE_REPRESENTATION_FULL236,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation loaders.

    ``split_mode="random_80_20"`` merges datasets 3, 4, and 5, then performs
    a deterministic random split using ``val_ratio`` and ``split_seed``.
    ``split_mode="dataset_5"`` keeps datasets 3 and 4 for training and
    dataset 5 for validation/test, which separates collection sessions.
    """

    if split_mode not in {"random_80_20", "dataset_5"}:
        raise ValueError(
            f"Unknown split_mode={split_mode!r}; "
            "expected 'random_80_20' or 'dataset_5'."
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    eye_geometry_gate_mode = canonical_eye_geometry_gate_mode(
        eye_geometry_gate_mode
    )
    eye_geometry_representation = canonical_eye_geometry_representation(
        eye_geometry_representation
    )
    scene_representation = canonical_scene_representation(scene_representation)
    deca_feature_representation = canonical_deca_feature_representation(
        deca_feature_representation
    )
    if not use_eye_geometry and eye_geometry_gate_mode != EYE_GEOMETRY_GATE_NONE:
        raise ValueError(
            "eye_geometry_gate_mode must be 'none' when use_eye_geometry=False."
        )
    depth_prior_table = None
    if use_eye_geometry:
        if depth_prior_csv_path is None:
            raise ValueError(
                "use_eye_geometry=True requires depth_prior_csv_path."
            )
        depth_prior_table = DepthPriorTable.load(
            depth_prior_csv_path,
            eye_geometry_representation,
        )

    if split_mode == "random_80_20":
        all_set = ModelV1Dataset(
            csv_path,
            datasets=all_datasets,
            normalize_images=normalize_images,
            load_face_image=load_face_image,
            deca_cache_path=deca_cache_path,
            require_deca_features=require_deca_features,
            fit_target_normalizer=False,
            use_eye_geometry=use_eye_geometry,
            depth_prior_table=depth_prior_table,
            eye_geometry_gate_mode=eye_geometry_gate_mode,
            eye_geometry_representation=eye_geometry_representation,
            scene_representation=scene_representation,
            deca_feature_representation=deca_feature_representation,
        )
        val_count = int(round(len(all_set) * val_ratio))
        val_count = max(1, min(len(all_set) - 1, val_count))
        generator = torch.Generator().manual_seed(split_seed)
        indices = torch.randperm(len(all_set), generator=generator).tolist()
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]
        normalizer = target_normalizer
        if normalize_uv_targets and normalizer is None:
            normalizer = fit_uv_target_normalizer(
                [all_set.rows[index] for index in train_indices]
            )
        all_set.target_normalizer = normalizer
        valid_train_indices = (
            [
                index
                for index in train_indices
                if all_set.eye_geometry_is_valid(index)
            ]
            if use_eye_geometry
            else []
        )
        if use_eye_geometry and not valid_train_indices:
            raise ValueError("Training split has no valid eye geometry.")
        geometry_normalizer = eye_geometry_normalizer
        if use_eye_geometry and geometry_normalizer is None:
            geometry_normalizer = EyeGeometryNormalizer.fit(
                torch.stack(
                    [
                        all_set.raw_eye_geometry(index)
                        for index in valid_train_indices
                    ],
                    dim=0,
                )
            )
        all_set.eye_geometry_normalizer = geometry_normalizer
        quality_normalizer = eye_geometry_quality_normalizer
        if (
            eye_geometry_gate_mode == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
            and quality_normalizer is None
        ):
            quality_normalizer = EyeGeometryQualityNormalizer.fit(
                torch.stack(
                    [
                        all_set.raw_eye_geometry_quality(index)
                        for index in valid_train_indices
                    ],
                    dim=0,
                )
            )
        all_set.eye_geometry_quality_normalizer = quality_normalizer
        train_set = Subset(all_set, train_indices)
        val_set = Subset(all_set, val_indices)
    else:
        train_set = ModelV1Dataset(
            csv_path,
            datasets=train_datasets,
            normalize_images=normalize_images,
            load_face_image=load_face_image,
            deca_cache_path=deca_cache_path,
            require_deca_features=require_deca_features,
            fit_target_normalizer=False,
            use_eye_geometry=use_eye_geometry,
            depth_prior_table=depth_prior_table,
            eye_geometry_gate_mode=eye_geometry_gate_mode,
            eye_geometry_representation=eye_geometry_representation,
            scene_representation=scene_representation,
            deca_feature_representation=deca_feature_representation,
        )
        val_set = ModelV1Dataset(
            csv_path,
            datasets=val_datasets,
            normalize_images=normalize_images,
            load_face_image=load_face_image,
            deca_cache_path=deca_cache_path,
            require_deca_features=require_deca_features,
            fit_target_normalizer=False,
            use_eye_geometry=use_eye_geometry,
            depth_prior_table=depth_prior_table,
            eye_geometry_gate_mode=eye_geometry_gate_mode,
            eye_geometry_representation=eye_geometry_representation,
            scene_representation=scene_representation,
            deca_feature_representation=deca_feature_representation,
        )
        normalizer = target_normalizer
        if normalize_uv_targets and normalizer is None:
            normalizer = fit_uv_target_normalizer(train_set.rows)
        train_set.target_normalizer = normalizer
        val_set.target_normalizer = normalizer
        valid_train_indices = (
            [
                index
                for index in range(len(train_set))
                if train_set.eye_geometry_is_valid(index)
            ]
            if use_eye_geometry
            else []
        )
        if use_eye_geometry and not valid_train_indices:
            raise ValueError("Training split has no valid eye geometry.")
        geometry_normalizer = eye_geometry_normalizer
        if use_eye_geometry and geometry_normalizer is None:
            geometry_normalizer = EyeGeometryNormalizer.fit(
                torch.stack(
                    [
                        train_set.raw_eye_geometry(index)
                        for index in valid_train_indices
                    ],
                    dim=0,
                )
            )
        train_set.eye_geometry_normalizer = geometry_normalizer
        val_set.eye_geometry_normalizer = geometry_normalizer
        quality_normalizer = eye_geometry_quality_normalizer
        if (
            eye_geometry_gate_mode == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
            and quality_normalizer is None
        ):
            quality_normalizer = EyeGeometryQualityNormalizer.fit(
                torch.stack(
                    [
                        train_set.raw_eye_geometry_quality(index)
                        for index in valid_train_indices
                    ],
                    dim=0,
                )
            )
        train_set.eye_geometry_quality_normalizer = quality_normalizer
        val_set.eye_geometry_quality_normalizer = quality_normalizer

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def get_uv_target_normalizer(dataset: Dataset) -> UVTargetNormalizer | None:
    """Return the normalizer fitted by :func:`build_modelv1_dataloaders`."""

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if not isinstance(base_dataset, ModelV1Dataset):
        raise TypeError("Expected a ModelV1Dataset or a torch.utils.data.Subset of one.")
    return base_dataset.target_normalizer


def build_scene_input_vector(
    full_scene_vec: torch.Tensor,
    representation: str,
) -> torch.Tensor:
    """Build the selected scene representation from the full 25D vector."""

    representation = canonical_scene_representation(representation)
    full_scene_vec = torch.as_tensor(full_scene_vec, dtype=torch.float32).flatten()
    if full_scene_vec.shape != (len(SCENE_COLUMNS),):
        raise ValueError(
            f"full_scene_vec must have shape ({len(SCENE_COLUMNS)},), "
            f"got {tuple(full_scene_vec.shape)}"
    )
    if representation == SCENE_REPRESENTATION_FULL25:
        return full_scene_vec
    if representation not in (
        SCENE_REPRESENTATION_TABLE_FRAME7,
        SCENE_REPRESENTATION_ORIENTATION6D,
    ):
        raise AssertionError(f"Unhandled scene representation: {representation}")

    # e1_c and e2_c are the first two columns of R_CW: a continuous 6D
    # orientation representation. The third axis is recoverable as e1_c x e2_c.
    basis = full_scene_vec[list(ORIENTATION_BASIS_INDICES)]
    e1_c = basis[:3]
    e2_c = basis[3:]
    n_c = full_scene_vec[:3]
    if (
        not torch.isfinite(basis).all()
        or abs(float(torch.linalg.vector_norm(e1_c)) - 1.0) > 1e-3
        or abs(float(torch.linalg.vector_norm(e2_c)) - 1.0) > 1e-3
        or abs(float(torch.dot(e1_c, e2_c))) > 1e-3
        or not torch.allclose(
            torch.cross(e1_c, e2_c, dim=0),
            n_c,
            atol=1e-3,
            rtol=1e-3,
        )
    ):
        raise ValueError(
            "Scene orientation must be a finite, orthonormal, right-handed basis."
        )
    if representation == SCENE_REPRESENTATION_ORIENTATION6D:
        return basis

    signed_distance_m = (
        full_scene_vec[TABLE_FRAME_DISTANCE_INDEX]
        / TABLE_FRAME_DISTANCE_SCALE_MM
    ).reshape(1)
    table_frame = torch.cat((basis, signed_distance_m), dim=0)
    if table_frame.shape != (7,) or not torch.isfinite(table_frame).all():
        raise ValueError("Derived TableFrame scene vector must be finite 7D.")
    return table_frame


def get_eye_geometry_normalizer(
    dataset: Dataset,
) -> EyeGeometryNormalizer | None:
    """Return the train-fitted eye geometry normalizer, if enabled."""

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if not isinstance(base_dataset, ModelV1Dataset):
        raise TypeError("Expected a ModelV1Dataset or a torch.utils.data.Subset of one.")
    return base_dataset.eye_geometry_normalizer


def get_eye_geometry_quality_normalizer(
    dataset: Dataset,
) -> EyeGeometryQualityNormalizer | None:
    """Return the train-fitted PnP quality normalizer, if enabled."""

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if not isinstance(base_dataset, ModelV1Dataset):
        raise TypeError("Expected a ModelV1Dataset or a torch.utils.data.Subset of one.")
    return base_dataset.eye_geometry_quality_normalizer


def normalize_dataset_names(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    result = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text.isdigit():
            result.add(f"dataset_dual_rigid_body_{text}")
        else:
            result.add(text)
    return result


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def validate_required_columns(row: dict[str, str]) -> None:
    required = [
        "sample_id",
        "dataset",
        "image_name",
        "face_path",
        "left_eye_path",
        "right_eye_path",
        "source_image_path",
        "uv_gt_u_mm",
        "uv_gt_v_mm",
        "gaze_target_w_x_mm",
        "gaze_target_w_y_mm",
        "gaze_target_w_z_mm",
        "table_origin_w_x_mm",
        "table_origin_w_y_mm",
        "table_origin_w_z_mm",
    ]
    missing = [
        column
        for column in required + CROP_CAM_COLUMNS + SCENE_COLUMNS
        if column not in row
    ]
    if missing:
        raise ValueError(f"Dataset CSV is missing required columns: {missing}")


def float_tensor(row: dict[str, str], columns: list[str]) -> torch.Tensor:
    values = []
    for column in columns:
        value = row.get(column, "")
        if value == "":
            raise ValueError(f"Missing numeric value for column: {column}")
        values.append(float(value))
    return torch.tensor(values, dtype=torch.float32)
