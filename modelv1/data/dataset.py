"""PyTorch Dataset and DataLoader helpers for the ModelV1 gaze dataset."""

from __future__ import annotations

import csv
import copy
from pathlib import Path
from typing import Callable, Iterable, Literal, Tuple

from modelv1.deca_cache import (
    DECA_FEATURE_REPRESENTATION_FULL236,
    DECA_FEATURE_REPRESENTATION_NONE,
    DecaFeatureCache,
    canonical_deca_feature_representation,
    select_deca_feature_representation,
)
from modelv1.data.depth_prior import (
    EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
    EYE_GEOMETRY_REPRESENTATION_RAW_EYE6D,
    DepthPriorTable,
    build_eye_geometry_vector,
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
from modelv1.data.virtual_camera_manifest import (
    IMAGE_SOURCE_LEGACY,
    IMAGE_SOURCE_VIRTUAL_CAMERA,
    VirtualCameraManifest,
    canonical_image_source,
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

FACE_SIZE = (160, 160)
EYE_SIZE = (60, 36)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FACENET_MEAN = (0.5, 0.5, 0.5)
FACENET_STD = (0.5, 0.5, 0.5)

ImageTransform = Callable[[torch.Tensor], torch.Tensor]
PairedImageTransform = Callable[
    [torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]
SplitMode = Literal["random_80_20", "dataset_5", "explicit_datasets"]


class ModelV1Dataset(Dataset):
    """Dataset backed by `data/processed/modelv1_dataset.csv`.

    Each item returns a dictionary with:

    - `face` (optional), `left_eye`, `right_eye`: image tensors in CHW format.
    - `crop_cam_vec`: 36D crop/camera metadata tensor.
    - `scene_vec`: selected full25, TableFrame 7D, or Orientation 6D tensor.
    - `deca_feat` (optional): cached frozen DECA face feature, selected as
      `full236` or `geometry156`; omitted when the representation is `none`.
    - `eye_geometry_vec` (optional): selected, training-z-score-normalized 6D
      binocular camera geometry.
    - gated modes also return confidence/mask and, for learned mode, 4D quality.
    - V4 returns raw binocular PnP coordinates in mm, label-free log-depth
      uncertainty, TableFrame7, and optional train-normalized PnP quality.
    - `uv_gt`: 2D table-local gaze target in millimeters.
    - `uv_target`: z-score-normalized target used by the UV head.
    - virtual-camera mode additionally returns `rotation_n_from_c` and its
      transpose `rotation_c_from_n`, both as 3x3 tensors.
    - extra metadata for validation and debugging.
    """

    def __init__(
        self,
        csv_path: str | Path,
        datasets: Iterable[str] | None = None,
        normalize_images: bool = True,
        face_transform: ImageTransform | None = None,
        eye_transform: ImageTransform | None = None,
        paired_eye_transform: PairedImageTransform | None = None,
        return_paths: bool = True,
        load_face_image: bool = True,
        deca_cache_path: str | Path | None = DEFAULT_DECA_CACHE_PATH,
        require_deca_features: bool = True,
        target_normalizer: UVTargetNormalizer | None = None,
        fit_target_normalizer: bool = True,
        use_eye_geometry: bool = False,
        use_gaze_geometry: bool = False,
        require_depth_uncertainty: bool = False,
        require_pnp_quality: bool = False,
        use_depth_correction: bool = False,
        filter_invalid_depth_prior_samples: bool = False,
        depth_prior_csv_path: str | Path | None = None,
        depth_prior_table: DepthPriorTable | None = None,
        eye_geometry_normalizer: EyeGeometryNormalizer | None = None,
        eye_geometry_gate_mode: str = EYE_GEOMETRY_GATE_NONE,
        eye_geometry_quality_normalizer: EyeGeometryQualityNormalizer | None = None,
        depth_correction_geometry_normalizer: EyeGeometryNormalizer | None = None,
        eye_geometry_representation: str = EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
        scene_representation: str = SCENE_REPRESENTATION_FULL25,
        deca_feature_representation: str = DECA_FEATURE_REPRESENTATION_FULL236,
        image_source: str = IMAGE_SOURCE_LEGACY,
        virtual_camera_manifest_path: str | Path | None = None,
        virtual_camera_manifest: VirtualCameraManifest | None = None,
        filter_invalid_virtual_camera_samples: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.normalize_images = normalize_images
        self.face_transform = face_transform
        self.eye_transform = eye_transform
        self.paired_eye_transform = paired_eye_transform
        self.return_paths = return_paths
        self.load_face_image = load_face_image
        self.image_source = canonical_image_source(image_source)
        self.filter_invalid_virtual_camera_samples = (
            filter_invalid_virtual_camera_samples
        )
        self.use_eye_geometry = use_eye_geometry
        self.use_gaze_geometry = use_gaze_geometry
        self.require_depth_uncertainty = require_depth_uncertainty
        self.require_pnp_quality = require_pnp_quality
        self.use_depth_correction = use_depth_correction
        self.filter_invalid_depth_prior_samples = filter_invalid_depth_prior_samples
        if self.use_depth_correction and not self.use_gaze_geometry:
            raise ValueError(
                "use_depth_correction=True requires use_gaze_geometry=True."
            )
        if self.use_depth_correction and not self.require_pnp_quality:
            raise ValueError(
                "use_depth_correction=True requires normalized PnP quality."
            )
        if self.use_eye_geometry and self.use_gaze_geometry:
            raise ValueError(
                "Legacy learned eye geometry and V4 geometric eye hypotheses "
                "cannot be enabled together."
            )
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
        self.depth_correction_geometry_normalizer = (
            depth_correction_geometry_normalizer
        )
        self.scene_representation = canonical_scene_representation(
            scene_representation
        )
        self.deca_feature_representation = canonical_deca_feature_representation(
            deca_feature_representation
        )
        self.uses_deca_features = (
            self.deca_feature_representation
            != DECA_FEATURE_REPRESENTATION_NONE
        )
        if not load_face_image and face_transform is not None:
            raise ValueError("face_transform requires load_face_image=True.")
        if eye_transform is not None and paired_eye_transform is not None:
            raise ValueError(
                "Provide either eye_transform or paired_eye_transform, not both."
            )
        if depth_prior_csv_path is not None and depth_prior_table is not None:
            raise ValueError(
                "Provide either depth_prior_csv_path or depth_prior_table, not both."
            )
        if (
            virtual_camera_manifest_path is not None
            and virtual_camera_manifest is not None
        ):
            raise ValueError(
                "Provide either virtual_camera_manifest_path or "
                "virtual_camera_manifest, not both."
            )
        if self.image_source == IMAGE_SOURCE_VIRTUAL_CAMERA:
            if virtual_camera_manifest is None:
                if virtual_camera_manifest_path is None:
                    raise ValueError(
                        "image_source='virtual_camera' requires "
                        "virtual_camera_manifest_path."
                    )
                virtual_camera_manifest = VirtualCameraManifest.load(
                    virtual_camera_manifest_path
                )
        elif (
            virtual_camera_manifest_path is not None
            or virtual_camera_manifest is not None
            or self.filter_invalid_virtual_camera_samples
        ):
            raise ValueError(
                "Virtual-camera manifest options require "
                "image_source='virtual_camera'."
            )
        self.virtual_camera_manifest = virtual_camera_manifest

        requested = normalize_dataset_names(datasets)
        self.rows = read_rows(self.csv_path)
        if requested is not None:
            self.rows = [row for row in self.rows if row["dataset"] in requested]

        if self.image_source == IMAGE_SOURCE_VIRTUAL_CAMERA:
            assert self.virtual_camera_manifest is not None
            if self.filter_invalid_virtual_camera_samples:
                before_count = len(self.rows)
                self.rows = [
                    row
                    for row in self.rows
                    if self.virtual_camera_manifest.contains(row["sample_id"])
                ]
                skipped_count = before_count - len(self.rows)
                if skipped_count:
                    print(
                        "Filtered "
                        f"{skipped_count} samples without valid virtual-camera "
                        f"artifacts from {self.csv_path}.",
                        flush=True,
                    )
            else:
                self.virtual_camera_manifest.require_sample_ids(
                    [row["sample_id"] for row in self.rows]
                )

        if not self.rows:
            raise ValueError(f"No samples found in {self.csv_path}")

        validate_required_columns(
            self.rows[0],
            require_gaze_geometry=self.use_gaze_geometry,
        )
        if self.use_eye_geometry or self.use_gaze_geometry:
            depth_representation = (
                EYE_GEOMETRY_REPRESENTATION_RAW_EYE6D
                if self.use_gaze_geometry
                else self.eye_geometry_representation
            )
            if depth_prior_table is None:
                if depth_prior_csv_path is None:
                    raise ValueError(
                        "use_eye_geometry=True requires a depth-prior CSV."
                    )
                depth_prior_table = DepthPriorTable.load(
                    depth_prior_csv_path,
                    depth_representation,
                )
            if (
                depth_prior_table.eye_geometry_representation
                != depth_representation
            ):
                raise ValueError(
                    "depth_prior_table representation does not match "
                    "eye_geometry_representation."
                )
            if self.filter_invalid_depth_prior_samples:
                before_count = len(self.rows)

                def has_required_depth_prior(row: dict[str, str]) -> bool:
                    sample_id = row["sample_id"]
                    if sample_id not in depth_prior_table.geometry_by_sample_id:
                        return False
                    if (
                        self.require_depth_uncertainty
                        and sample_id
                        not in depth_prior_table.depth_log_scale_std_by_sample_id
                    ):
                        return False
                    return not (
                        self.require_pnp_quality
                        and sample_id not in depth_prior_table.quality_by_sample_id
                    )

                self.rows = [row for row in self.rows if has_required_depth_prior(row)]
                skipped_count = before_count - len(self.rows)
                if skipped_count:
                    print(
                        "Filtered "
                        f"{skipped_count} samples without the required valid depth prior "
                        f"from {self.csv_path}.",
                        flush=True,
                    )
                if not self.rows:
                    raise ValueError(
                        "No samples remain after filtering invalid depth-prior geometry "
                        f"from {self.csv_path}."
                    )
            sample_ids = [row["sample_id"] for row in self.rows]
            if self.use_eye_geometry_quality_gate:
                depth_prior_table.require_present_sample_ids(sample_ids)
            else:
                depth_prior_table.require_sample_ids(sample_ids)
            if self.require_depth_uncertainty:
                depth_prior_table.require_uncertainty_sample_ids(sample_ids)
            if self.require_pnp_quality:
                missing_quality = [
                    sample_id
                    for sample_id in sample_ids
                    if sample_id not in depth_prior_table.quality_by_sample_id
                ]
                if missing_quality:
                    raise ValueError(
                        "Depth-prior CSV has no valid PnP quality for "
                        f"{len(missing_quality)} required samples."
                    )
        self.target_normalizer = target_normalizer
        if self.target_normalizer is None and fit_target_normalizer:
            self.target_normalizer = fit_uv_target_normalizer(self.rows)
        if require_deca_features and not self.uses_deca_features:
            raise ValueError(
                "require_deca_features must be False when "
                "deca_feature_representation='none'."
            )
        if require_deca_features and deca_cache_path is None:
            raise ValueError("require_deca_features=True requires deca_cache_path.")
        self.deca_cache = (
            DecaFeatureCache.load(deca_cache_path)
            if self.uses_deca_features and deca_cache_path is not None
            else None
        )
        if self.deca_cache is not None:
            self.deca_cache.require_sample_ids(row["sample_id"] for row in self.rows)
        self.depth_prior_table = (
            depth_prior_table
            if self.use_eye_geometry or self.use_gaze_geometry
            else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        virtual_record = (
            self.virtual_camera_manifest.lookup(row["sample_id"])
            if self.virtual_camera_manifest is not None
            else None
        )
        face_path = (
            str(virtual_record.face_path)
            if virtual_record is not None
            else row["face_path"]
        )
        left_eye_path = (
            str(virtual_record.left_eye_path)
            if virtual_record is not None
            else row["left_eye_path"]
        )
        right_eye_path = (
            str(virtual_record.right_eye_path)
            if virtual_record is not None
            else row["right_eye_path"]
        )

        face = (
            self._load_image(face_path, FACE_SIZE)
            if self.load_face_image
            else None
        )
        left_eye = self._load_image(left_eye_path, EYE_SIZE)
        right_eye = self._load_image(right_eye_path, EYE_SIZE)

        if face is not None and self.face_transform is not None:
            face = self.face_transform(face)
        if self.paired_eye_transform is not None:
            left_eye, right_eye = self.paired_eye_transform(left_eye, right_eye)
        elif self.eye_transform is not None:
            left_eye = self.eye_transform(left_eye)
            right_eye = self.eye_transform(right_eye)

        # The VGGFace2-pretrained Inception-ResNet uses fixed image
        # standardization (pixel values mapped from [0, 1] to [-1, 1]).
        if face is not None:
            face = self._normalize_image(face, FACENET_MEAN, FACENET_STD)
        if self.normalize_images:
            left_eye = self._normalize_image(left_eye)
            right_eye = self._normalize_image(right_eye)

        uv_gt = float_tensor(row, ["uv_gt_u_mm", "uv_gt_v_mm"])
        scene_vec = build_scene_input_vector(
            float_tensor(row, SCENE_COLUMNS),
            self.scene_representation,
        )
        item: dict[str, object] = {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "crop_cam_vec": float_tensor(row, CROP_CAM_COLUMNS),
            "scene_vec": scene_vec,
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
        if virtual_record is not None:
            item["rotation_n_from_c"] = virtual_record.rotation_n_from_c.clone()
            item["rotation_c_from_n"] = virtual_record.rotation_c_from_n.clone()
        if self.deca_cache is not None:
            item["deca_feat"] = torch.from_numpy(
                select_deca_feature_representation(
                    self.deca_cache.lookup(row["sample_id"]),
                    self.deca_feature_representation,
                ).copy()
            )
        if self.use_gaze_geometry:
            assert self.depth_prior_table is not None
            sample_id = row["sample_id"]
            item["table_frame7"] = scene_vec
            item["raw_eye_geometry_mm"] = self.depth_prior_table.lookup(
                sample_id
            ).clone()
            if self.use_depth_correction:
                raw_eyes = item["raw_eye_geometry_mm"]
                assert torch.is_tensor(raw_eyes)
                pnp_geometry = build_eye_geometry_vector(
                    raw_eyes[:3],
                    raw_eyes[3:],
                )
                if self.depth_correction_geometry_normalizer is None:
                    raise RuntimeError(
                        "Depth correction requires a training-fitted PnP "
                        "geometry normalizer."
                    )
                item["pnp_geometry_vec"] = (
                    self.depth_correction_geometry_normalizer.normalize(
                        pnp_geometry
                    )
                )
            item["gaze_target_camera_mm"] = float_tensor(
                row,
                [
                    "gaze_cam_recomputed_x_mm",
                    "gaze_cam_recomputed_y_mm",
                    "gaze_cam_recomputed_z_mm",
                ],
            )
            uncertainty = (
                self.depth_prior_table.lookup_depth_log_scale_std(sample_id)
                if self.require_depth_uncertainty
                else self.depth_prior_table.depth_log_scale_std_by_sample_id.get(
                    sample_id,
                    0.0,
                )
            )
            item["depth_log_scale_sigma"] = torch.tensor(
                [uncertainty],
                dtype=torch.float32,
            )
            if self.require_pnp_quality:
                quality = self.depth_prior_table.lookup_quality(sample_id).clone()
                if self.eye_geometry_quality_normalizer is not None:
                    quality = self.eye_geometry_quality_normalizer.normalize(quality)
                item["pnp_quality_vec"] = quality
                item["pnp_confidence"] = torch.tensor(
                    [self.depth_prior_table.lookup_confidence(sample_id)],
                    dtype=torch.float32,
                )
        elif self.depth_prior_table is not None:
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
                "face": face_path,
                "left_eye": left_eye_path,
                "right_eye": right_eye_path,
                "source": row["source_image_path"],
            }
        return item

    def _load_image(self, path_text: str, size: tuple[int, int]) -> torch.Tensor:
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")

        image = Image.open(path).convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.BILINEAR)

        width, height = image.size
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(height, width, 3).permute(2, 0, 1).float().div(255.0)

        return tensor

    @staticmethod
    def _normalize_image(
        tensor: torch.Tensor,
        mean_values: tuple[float, float, float] = IMAGENET_MEAN,
        std_values: tuple[float, float, float] = IMAGENET_STD,
    ) -> torch.Tensor:
        mean = tensor.new_tensor(mean_values).view(3, 1, 1)
        std = tensor.new_tensor(std_values).view(3, 1, 1)
        return (tensor - mean) / std

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

    def raw_depth_correction_geometry(self, index: int) -> torch.Tensor:
        """Return the label-free normalized6d PnP representation before z-score."""

        raw_eyes = self.raw_eye_geometry(index)
        return build_eye_geometry_vector(raw_eyes[:3], raw_eyes[3:])

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
    train_paired_eye_transform: PairedImageTransform | None = None,
    deca_cache_path: str | Path | None = DEFAULT_DECA_CACHE_PATH,
    require_deca_features: bool = True,
    normalize_uv_targets: bool = True,
    target_normalizer: UVTargetNormalizer | None = None,
    use_eye_geometry: bool = False,
    use_gaze_geometry: bool = False,
    require_depth_uncertainty: bool = False,
    require_pnp_quality: bool = False,
    use_depth_correction: bool = False,
    filter_invalid_depth_prior_samples: bool = False,
    depth_prior_csv_path: str | Path | None = DEFAULT_DEPTH_PRIOR_PATH,
    eye_geometry_normalizer: EyeGeometryNormalizer | None = None,
    eye_geometry_gate_mode: str = EYE_GEOMETRY_GATE_NONE,
    eye_geometry_quality_normalizer: EyeGeometryQualityNormalizer | None = None,
    depth_correction_geometry_normalizer: EyeGeometryNormalizer | None = None,
    eye_geometry_representation: str = EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
    scene_representation: str = SCENE_REPRESENTATION_FULL25,
    deca_feature_representation: str = DECA_FEATURE_REPRESENTATION_FULL236,
    image_source: str = IMAGE_SOURCE_LEGACY,
    virtual_camera_manifest_path: str | Path | None = None,
    filter_invalid_virtual_camera_samples: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation loaders.

    ``split_mode="random_80_20"`` merges datasets 3, 4, and 5, then performs
    a deterministic random split using ``val_ratio`` and ``split_seed``.
    ``split_mode="dataset_5"`` keeps datasets 3 and 4 for training and
    dataset 5 for validation/test, which separates collection sessions.
    ``split_mode="explicit_datasets"`` uses ``train_datasets`` and
    ``val_datasets`` exactly as supplied, enabling arbitrary session holdouts.
    """

    if split_mode not in {"random_80_20", "dataset_5", "explicit_datasets"}:
        raise ValueError(
            f"Unknown split_mode={split_mode!r}; "
            "expected 'random_80_20', 'dataset_5', or 'explicit_datasets'."
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
    image_source = canonical_image_source(image_source)
    uses_deca_features = (
        deca_feature_representation != DECA_FEATURE_REPRESENTATION_NONE
    )
    if require_deca_features and not uses_deca_features:
        raise ValueError(
            "require_deca_features must be False when "
            "deca_feature_representation='none'."
        )
    if not uses_deca_features:
        deca_cache_path = None
    if not use_eye_geometry and eye_geometry_gate_mode != EYE_GEOMETRY_GATE_NONE:
        raise ValueError(
            "eye_geometry_gate_mode must be 'none' when use_eye_geometry=False."
        )
    if use_eye_geometry and use_gaze_geometry:
        raise ValueError(
            "use_eye_geometry and use_gaze_geometry cannot both be true."
        )
    if use_depth_correction and not use_gaze_geometry:
        raise ValueError(
            "use_depth_correction=True requires use_gaze_geometry=True."
        )
    if use_depth_correction and not require_pnp_quality:
        raise ValueError(
            "use_depth_correction=True requires require_pnp_quality=True."
        )
    depth_prior_table = None
    if use_eye_geometry or use_gaze_geometry:
        if depth_prior_csv_path is None:
            raise ValueError(
                "use_eye_geometry=True requires depth_prior_csv_path."
            )
        depth_prior_table = DepthPriorTable.load(
            depth_prior_csv_path,
            (
                EYE_GEOMETRY_REPRESENTATION_RAW_EYE6D
                if use_gaze_geometry
                else eye_geometry_representation
            ),
        )
    virtual_camera_manifest = None
    if image_source == IMAGE_SOURCE_VIRTUAL_CAMERA:
        if virtual_camera_manifest_path is None:
            raise ValueError(
                "image_source='virtual_camera' requires "
                "virtual_camera_manifest_path."
            )
        virtual_camera_manifest = VirtualCameraManifest.load(
            virtual_camera_manifest_path
        )
    elif virtual_camera_manifest_path is not None:
        raise ValueError(
            "virtual_camera_manifest_path requires image_source='virtual_camera'."
        )

    if split_mode == "random_80_20":
        all_set = ModelV1Dataset(
            csv_path,
            datasets=all_datasets,
            normalize_images=normalize_images,
            paired_eye_transform=train_paired_eye_transform,
            load_face_image=load_face_image,
            deca_cache_path=deca_cache_path,
            require_deca_features=require_deca_features,
            fit_target_normalizer=False,
            use_eye_geometry=use_eye_geometry,
            use_gaze_geometry=use_gaze_geometry,
            require_depth_uncertainty=require_depth_uncertainty,
            require_pnp_quality=require_pnp_quality,
            use_depth_correction=use_depth_correction,
            filter_invalid_depth_prior_samples=filter_invalid_depth_prior_samples,
            depth_prior_table=depth_prior_table,
            eye_geometry_gate_mode=eye_geometry_gate_mode,
            eye_geometry_representation=eye_geometry_representation,
            scene_representation=scene_representation,
            deca_feature_representation=deca_feature_representation,
            image_source=image_source,
            virtual_camera_manifest=virtual_camera_manifest,
            filter_invalid_virtual_camera_samples=(
                filter_invalid_virtual_camera_samples
            ),
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
            if use_eye_geometry or require_pnp_quality or use_depth_correction
            else []
        )
        if (
            use_eye_geometry or require_pnp_quality or use_depth_correction
        ) and not valid_train_indices:
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
        correction_normalizer = depth_correction_geometry_normalizer
        if use_depth_correction and correction_normalizer is None:
            correction_normalizer = EyeGeometryNormalizer.fit(
                torch.stack(
                    [
                        all_set.raw_depth_correction_geometry(index)
                        for index in valid_train_indices
                    ],
                    dim=0,
                )
            )
        all_set.depth_correction_geometry_normalizer = correction_normalizer
        quality_normalizer = eye_geometry_quality_normalizer
        if (
            (
                eye_geometry_gate_mode == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
                or require_pnp_quality
            )
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
        # A shallow validation view shares immutable rows/caches and fitted
        # normalizers, but deliberately disables train-only augmentation.
        val_base_set = copy.copy(all_set)
        val_base_set.paired_eye_transform = None
        train_set = Subset(all_set, train_indices)
        val_set = Subset(val_base_set, val_indices)
    else:
        normalized_train_datasets = normalize_dataset_names(train_datasets)
        normalized_val_datasets = normalize_dataset_names(val_datasets)
        if not normalized_train_datasets or not normalized_val_datasets:
            raise ValueError(
                "Explicit/session holdout splits require non-empty "
                "train_datasets and val_datasets."
            )
        overlap = normalized_train_datasets.intersection(normalized_val_datasets)
        if overlap:
            raise ValueError(
                "Training and validation datasets must be disjoint; overlap: "
                f"{sorted(overlap)}"
            )
        train_set = ModelV1Dataset(
            csv_path,
            datasets=normalized_train_datasets,
            normalize_images=normalize_images,
            paired_eye_transform=train_paired_eye_transform,
            load_face_image=load_face_image,
            deca_cache_path=deca_cache_path,
            require_deca_features=require_deca_features,
            fit_target_normalizer=False,
            use_eye_geometry=use_eye_geometry,
            use_gaze_geometry=use_gaze_geometry,
            require_depth_uncertainty=require_depth_uncertainty,
            require_pnp_quality=require_pnp_quality,
            use_depth_correction=use_depth_correction,
            filter_invalid_depth_prior_samples=filter_invalid_depth_prior_samples,
            depth_prior_table=depth_prior_table,
            eye_geometry_gate_mode=eye_geometry_gate_mode,
            eye_geometry_representation=eye_geometry_representation,
            scene_representation=scene_representation,
            deca_feature_representation=deca_feature_representation,
            image_source=image_source,
            virtual_camera_manifest=virtual_camera_manifest,
            filter_invalid_virtual_camera_samples=(
                filter_invalid_virtual_camera_samples
            ),
        )
        val_set = ModelV1Dataset(
            csv_path,
            datasets=normalized_val_datasets,
            normalize_images=normalize_images,
            load_face_image=load_face_image,
            deca_cache_path=deca_cache_path,
            require_deca_features=require_deca_features,
            fit_target_normalizer=False,
            use_eye_geometry=use_eye_geometry,
            use_gaze_geometry=use_gaze_geometry,
            require_depth_uncertainty=require_depth_uncertainty,
            require_pnp_quality=require_pnp_quality,
            use_depth_correction=use_depth_correction,
            filter_invalid_depth_prior_samples=filter_invalid_depth_prior_samples,
            depth_prior_table=depth_prior_table,
            eye_geometry_gate_mode=eye_geometry_gate_mode,
            eye_geometry_representation=eye_geometry_representation,
            scene_representation=scene_representation,
            deca_feature_representation=deca_feature_representation,
            image_source=image_source,
            virtual_camera_manifest=virtual_camera_manifest,
            filter_invalid_virtual_camera_samples=(
                filter_invalid_virtual_camera_samples
            ),
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
            if use_eye_geometry or require_pnp_quality or use_depth_correction
            else []
        )
        if (
            use_eye_geometry or require_pnp_quality or use_depth_correction
        ) and not valid_train_indices:
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
        correction_normalizer = depth_correction_geometry_normalizer
        if use_depth_correction and correction_normalizer is None:
            correction_normalizer = EyeGeometryNormalizer.fit(
                torch.stack(
                    [
                        train_set.raw_depth_correction_geometry(index)
                        for index in valid_train_indices
                    ],
                    dim=0,
                )
            )
        train_set.depth_correction_geometry_normalizer = correction_normalizer
        val_set.depth_correction_geometry_normalizer = correction_normalizer
        quality_normalizer = eye_geometry_quality_normalizer
        if (
            (
                eye_geometry_gate_mode == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
                or require_pnp_quality
            )
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


def get_depth_correction_geometry_normalizer(
    dataset: Dataset,
) -> EyeGeometryNormalizer | None:
    """Return the train-fitted 6D PnP normalizer for DepthCorrectionHead."""

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if not isinstance(base_dataset, ModelV1Dataset):
        raise TypeError("Expected a ModelV1Dataset or a torch.utils.data.Subset of one.")
    return base_dataset.depth_correction_geometry_normalizer


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


def validate_required_columns(
    row: dict[str, str],
    *,
    require_gaze_geometry: bool = False,
) -> None:
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
    if require_gaze_geometry:
        required.extend(
            [
                "gaze_cam_recomputed_x_mm",
                "gaze_cam_recomputed_y_mm",
                "gaze_cam_recomputed_z_mm",
            ]
        )
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
