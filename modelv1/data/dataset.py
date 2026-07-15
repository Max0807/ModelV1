"""PyTorch Dataset and DataLoader helpers for the ModelV1 gaze dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
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

FACE_SIZE = (224, 224)
EYE_SIZE = (60, 36)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageTransform = Callable[[torch.Tensor], torch.Tensor]


class ModelV1Dataset(Dataset):
    """Dataset backed by `data/processed/modelv1_dataset.csv`.

    Each item returns a dictionary with:

    - `face`, `left_eye`, `right_eye`: image tensors in CHW format.
    - `crop_cam_vec`: 36D crop/camera metadata tensor.
    - `scene_vec`: 25D table/camera geometry tensor.
    - `uv_gt`: 2D table-local gaze target in millimeters.
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
    ) -> None:
        self.csv_path = Path(csv_path)
        self.normalize_images = normalize_images
        self.face_transform = face_transform
        self.eye_transform = eye_transform
        self.return_paths = return_paths

        requested = normalize_dataset_names(datasets)
        self.rows = read_rows(self.csv_path)
        if requested is not None:
            self.rows = [row for row in self.rows if row["dataset"] in requested]

        if not self.rows:
            raise ValueError(f"No samples found in {self.csv_path}")

        validate_required_columns(self.rows[0])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]

        face = self._load_image(row["face_path"], FACE_SIZE)
        left_eye = self._load_image(row["left_eye_path"], EYE_SIZE)
        right_eye = self._load_image(row["right_eye_path"], EYE_SIZE)

        if self.face_transform is not None:
            face = self.face_transform(face)
        if self.eye_transform is not None:
            left_eye = self.eye_transform(left_eye)
            right_eye = self.eye_transform(right_eye)

        item: dict[str, object] = {
            "face": face,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "crop_cam_vec": float_tensor(row, CROP_CAM_COLUMNS),
            "scene_vec": float_tensor(row, SCENE_COLUMNS),
            "uv_gt": float_tensor(row, ["uv_gt_u_mm", "uv_gt_v_mm"]),
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


def build_modelv1_dataloaders(
    csv_path: str | Path = "data/processed/modelv1_dataset.csv",
    train_datasets: Iterable[str] = ("3", "4"),
    val_datasets: Iterable[str] = ("5",),
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    normalize_images: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation loaders.

    The default split uses datasets 3 and 4 for training and dataset 5 for
    validation, which keeps validation separated by collection session.
    """

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_set = ModelV1Dataset(
        csv_path,
        datasets=train_datasets,
        normalize_images=normalize_images,
    )
    val_set = ModelV1Dataset(
        csv_path,
        datasets=val_datasets,
        normalize_images=normalize_images,
    )

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
