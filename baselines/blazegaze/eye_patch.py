"""BlazeGaze's released MediaPipe-homography eye-patch preprocessing.

The implementation is a dependency-light port of
``webeyetrack.model_based.obtain_eyepatch``. It keeps the released landmark
indices, padding convention, integer conversion and resize order, while adding
explicit validation and audit metadata for offline dataset generation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


FACE_CENTER_INDEX = 4
HOMOGRAPHY_CORNER_INDICES = (103, 150, 379, 332)
EYE_PATCH_TOP_INDEX = 151
EYE_PATCH_BOTTOM_INDEX = 195
REQUIRED_LANDMARK_INDICES = tuple(
    sorted(
        {
            FACE_CENTER_INDEX,
            *HOMOGRAPHY_CORNER_INDICES,
            EYE_PATCH_TOP_INDEX,
            EYE_PATCH_BOTTOM_INDEX,
        }
    )
)
FACE_CROP_SIZE = 512
EYE_PATCH_WIDTH = 512
EYE_PATCH_HEIGHT = 128
DEFAULT_FACE_PADDING_COEFS = (0.4, 0.2)
EXACT_EYE_PATCH_SOURCE = "mediapipe478_homography"


@dataclass(frozen=True)
class BlazeGazeEyePatchGeometry:
    """Generated patch and the geometry needed to audit it."""

    patch_rgb: np.ndarray
    homography: np.ndarray
    source_quad_px: np.ndarray
    crop_top_y: int
    crop_bottom_y: int
    source_quad_area_px2: float
    homography_condition: float
    black_pixel_fraction: float


def landmarks_to_pixel_array(
    landmarks: Sequence[object], image_width: int, image_height: int
) -> np.ndarray:
    """Convert MediaPipe normalized landmarks to an ``[N,2]`` pixel array."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    points = np.asarray(
        [
            [float(landmark.x) * image_width, float(landmark.y) * image_height]
            for landmark in landmarks
        ],
        dtype=np.float32,
    )
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected MediaPipe landmarks [N,2], got {points.shape}.")
    return points


def build_blazegaze_eye_patch(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
    *,
    face_padding_coefs: Sequence[float] = DEFAULT_FACE_PADDING_COEFS,
    face_crop_size: int = FACE_CROP_SIZE,
    output_size: tuple[int, int] = (EYE_PATCH_WIDTH, EYE_PATCH_HEIGHT),
    max_homography_condition: float = 1e12,
) -> BlazeGazeEyePatchGeometry:
    """Generate one released-style binocular eye patch.

    ``output_size`` follows OpenCV's ``(width, height)`` order. Input and
    output are RGB; OpenCV's geometric operations do not change channel order.
    Invalid geometry is rejected rather than silently clamped or reordered.
    """

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image [H,W,3], got {image.shape}.")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB input, got {image.dtype}.")
    if face_crop_size <= 0 or output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("Crop and output dimensions must be positive.")

    points = np.asarray(landmarks_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected landmarks [N,2], got {points.shape}.")
    if points.shape[0] <= max(REQUIRED_LANDMARK_INDICES):
        raise ValueError(
            f"At least {max(REQUIRED_LANDMARK_INDICES) + 1} landmarks are required; "
            f"got {points.shape[0]}."
        )
    selected = points[list(REQUIRED_LANDMARK_INDICES)]
    if not np.isfinite(selected).all():
        raise ValueError("Required MediaPipe landmarks contain NaN or infinity.")

    padding = np.asarray(face_padding_coefs, dtype=np.float64)
    if padding.shape != (2,) or not np.isfinite(padding).all():
        raise ValueError("face_padding_coefs must contain two finite values.")

    source_quad = points[list(HOMOGRAPHY_CORNER_INDICES)].copy()
    center = points[FACE_CENTER_INDEX]
    source_quad = source_quad + padding * (source_quad - center)
    source_quad = source_quad.astype(np.float32)
    source_quad_area = abs(float(cv2.contourArea(source_quad)))
    if source_quad_area <= 1.0:
        raise ValueError(
            f"Degenerate homography quadrilateral: area={source_quad_area:.6g}px^2."
        )

    size = float(face_crop_size)
    destination_quad = np.asarray(
        [[0.0, 0.0], [0.0, size], [size, size], [size, 0.0]],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(source_quad, destination_quad)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("OpenCV failed to compute a finite homography.")
    condition = float(np.linalg.cond(homography))
    if not math.isfinite(condition) or condition > max_homography_condition:
        raise ValueError(
            f"Ill-conditioned homography: condition={condition:.6g}."
        )

    warped_face = cv2.warpPerspective(
        np.ascontiguousarray(image),
        homography,
        (face_crop_size, face_crop_size),
    )

    # Keep the released implementation's homogeneous transform and int32 cast.
    homogeneous_points = np.vstack(
        (points.T, np.ones((1, points.shape[0]), dtype=np.float64))
    )
    warped_homogeneous = homography @ homogeneous_points
    denominators = warped_homogeneous[2]
    if np.any(np.abs(denominators) < 1e-12):
        raise ValueError("A warped landmark has a zero homogeneous denominator.")
    warped_points = (
        warped_homogeneous[:2] / denominators[None, :]
    ).T.astype(np.int32)

    crop_top_y = int(warped_points[EYE_PATCH_TOP_INDEX, 1])
    crop_bottom_y = int(warped_points[EYE_PATCH_BOTTOM_INDEX, 1])
    if not (0 <= crop_top_y < crop_bottom_y <= face_crop_size):
        raise ValueError(
            "Invalid warped eye bounds: "
            f"top={crop_top_y}, bottom={crop_bottom_y}, size={face_crop_size}."
        )

    eye_band = warped_face[crop_top_y:crop_bottom_y, :]
    if eye_band.size == 0:
        raise ValueError("The homography produced an empty eye band.")
    patch = cv2.resize(eye_band, output_size)
    if patch.shape != (output_size[1], output_size[0], 3):
        raise RuntimeError(f"Unexpected eye-patch shape: {patch.shape}.")
    black_fraction = float(np.mean(np.all(patch == 0, axis=2)))

    return BlazeGazeEyePatchGeometry(
        patch_rgb=patch,
        homography=homography.astype(np.float64),
        source_quad_px=source_quad,
        crop_top_y=crop_top_y,
        crop_bottom_y=crop_bottom_y,
        source_quad_area_px2=source_quad_area,
        homography_condition=condition,
        black_pixel_fraction=black_fraction,
    )


@dataclass(frozen=True)
class BlazeGazeEyePatchRecord:
    sample_id: str
    eye_patch_path: Path
    landmark_count: int


class BlazeGazeEyePatchManifest:
    """Successful exact eye patches indexed by Stage 3 ``sample_id``."""

    REQUIRED_COLUMNS = (
        "sample_id",
        "status",
        "eye_patch_path",
        "landmark_count",
    )

    def __init__(self, records: dict[str, BlazeGazeEyePatchRecord]) -> None:
        self._records = dict(records)

    @classmethod
    def load(
        cls, manifest_paths: Iterable[str | Path]
    ) -> "BlazeGazeEyePatchManifest":
        records: dict[str, BlazeGazeEyePatchRecord] = {}
        for raw_path in manifest_paths:
            path = Path(raw_path)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or ())
                missing = [name for name in cls.REQUIRED_COLUMNS if name not in fields]
                if missing:
                    raise ValueError(f"{path} is missing columns: {missing}")
                for row in reader:
                    if str(row["status"]).strip().lower() != "success":
                        continue
                    sample_id = str(row["sample_id"]).strip()
                    if not sample_id:
                        raise ValueError(f"{path} contains an empty sample_id.")
                    if sample_id in records:
                        raise ValueError(f"Duplicate eye patch for {sample_id!r}.")
                    patch_path = Path(str(row["eye_patch_path"]).strip())
                    if not patch_path.is_absolute():
                        patch_path = (path.parent / patch_path).resolve()
                    landmark_count = int(str(row["landmark_count"]).strip())
                    records[sample_id] = BlazeGazeEyePatchRecord(
                        sample_id=sample_id,
                        eye_patch_path=patch_path,
                        landmark_count=landmark_count,
                    )
        if not records:
            raise ValueError("No successful BlazeGaze eye patches were loaded.")
        return cls(records)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self._records

    def lookup(self, sample_id: str) -> BlazeGazeEyePatchRecord:
        try:
            return self._records[sample_id]
        except KeyError as exc:
            raise KeyError(f"No exact BlazeGaze eye patch for {sample_id!r}.") from exc
