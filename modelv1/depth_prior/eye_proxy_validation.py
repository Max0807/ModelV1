"""Utilities for validating FLAME eye-point proxies against iris observations.

The helpers deliberately evaluate *image alignment* only.  A small 2D error
to the iris does not prove that a mesh vertex is the 3D eyeball centre; it is
useful for rejecting bad proxies and side-convention mistakes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


IRIS_GROUP_468 = "iris_group_468"
IRIS_GROUP_473 = "iris_group_473"
IRIS_GROUPS = (IRIS_GROUP_468, IRIS_GROUP_473)


@dataclass(frozen=True)
class IrisObservation:
    """Two image-space iris centres, retained without anatomical side labels."""

    group_468_xy: tuple[float, float]
    group_473_xy: tuple[float, float]

    def as_mapping(self) -> dict[str, tuple[float, float]]:
        return {
            IRIS_GROUP_468: self.group_468_xy,
            IRIS_GROUP_473: self.group_473_xy,
        }


def detect_iris_centres_rgb(image_rgb: Any, *, face_mesh: Any | None = None) -> IrisObservation:
    """Detect the two MediaPipe iris centres from one RGB source image.

    MediaPipe's group numbers are intentionally retained rather than calling
    either group anatomical-left/right.  The latter depends on mirroring and
    must be established from the calibrated projection, not assumed.
    """

    try:
        import mediapipe as mp
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Iris detection requires the legacy 'mediapipe' Python package. "
            "Alternatively provide a CSV with saved iris centres."
        ) from error

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape [H, W, 3].")
    height, width = image.shape[:2]
    if face_mesh is None:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
        ) as detector:
            result = detector.process(image)
    else:
        result = face_mesh.process(image)
    if not result.multi_face_landmarks:
        raise RuntimeError("MediaPipe Face Mesh did not detect one face.")
    landmarks = result.multi_face_landmarks[0].landmark
    if len(landmarks) < 478:
        raise RuntimeError(
            "MediaPipe did not return refined iris landmarks; "
            "ensure refine_landmarks=True is supported."
        )

    def centre(indices: Sequence[int]) -> tuple[float, float]:
        xy = np.asarray(
            [(landmarks[index].x * width, landmarks[index].y * height) for index in indices],
            dtype=np.float64,
        )
        return (float(xy[:, 0].mean()), float(xy[:, 1].mean()))

    return IrisObservation(
        group_468_xy=centre(range(468, 473)),
        group_473_xy=centre(range(473, 478)),
    )


def load_iris_centres_csv(
    path: str | Path,
    image_name: str,
    *,
    sample_id: str | None = None,
) -> IrisObservation:
    """Load one saved observation from a transparent, portable CSV schema.

    Required columns are ``image_name``, ``iris_group_468_x/y`` and
    ``iris_group_473_x/y``.  These group labels avoid silently assuming the
    camera/mirror convention in a third-party detector.
    """

    csv_path = Path(path)
    required = {
        "image_name",
        "iris_group_468_x",
        "iris_group_468_y",
        "iris_group_473_x",
        "iris_group_473_y",
    }
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Iris CSV must contain " + ", ".join(sorted(required))
            )
        fieldnames = set(reader.fieldnames)
        rows = list(reader)
    if sample_id is not None and "sample_id" in fieldnames:
        matches = [row for row in rows if row.get("sample_id") == sample_id]
    else:
        matches = [row for row in rows if row["image_name"] == image_name]
    if len(matches) != 1:
        identifier = f"sample_id {sample_id!r}" if sample_id is not None else f"image_name {image_name!r}"
        raise KeyError(f"Expected exactly one iris CSV row for {identifier}, found {len(matches)}.")
    row = matches[0]
    if "status" in row and row["status"] != "success":
        raise RuntimeError(f"Iris CSV marks {image_name!r} as {row['status']}: {row.get('reason', '')}")
    try:
        return IrisObservation(
            group_468_xy=(
                float(row["iris_group_468_x"]),
                float(row["iris_group_468_y"]),
            ),
            group_473_xy=(
                float(row["iris_group_473_x"]),
                float(row["iris_group_473_y"]),
            ),
        )
    except ValueError as error:
        raise ValueError(f"Invalid iris centre values for {image_name!r}.") from error


def project_flame_local_points(
    points_flame: Any,
    *,
    rotation_matrix: Any,
    tvec_mm: Any,
    scale_mm_per_flame_unit: float,
    camera_matrix: Any,
    dist_coeffs: Any,
) -> Any:
    """Project FLAME-local points using the metric PnP solution."""

    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Projection requires OpenCV and NumPy.") from error
    points = np.asarray(points_flame, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_flame must have shape [N, 3].")
    if scale_mm_per_flame_unit <= 0:
        raise ValueError("scale_mm_per_flame_unit must be positive.")
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(tvec_mm, dtype=np.float64).reshape(3)
    camera_points = np.asarray(
        [
            [
                sum(float(rotation[row, column]) * float(point[column] * scale_mm_per_flame_unit) for column in range(3))
                + float(translation[row])
                for row in range(3)
            ]
            for point in points
        ],
        dtype=np.float64,
    )
    pixels, _ = cv2.projectPoints(
        camera_points,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64),
    )
    return pixels.reshape(-1, 2)


def evaluate_iris_proxy_candidates(
    candidates_flame: Mapping[str, Any],
    *,
    iris: IrisObservation,
    rotation_matrix: Any,
    tvec_mm: Any,
    scale_mm_per_flame_unit: float,
    camera_matrix: Any,
    dist_coeffs: Any,
) -> list[dict[str, float | str]]:
    """Return each candidate's 2D error to both iris landmark groups."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Candidate evaluation requires NumPy.") from error
    if not candidates_flame:
        raise ValueError("candidates_flame must not be empty.")
    names = list(candidates_flame)
    points = np.asarray([candidates_flame[name] for name in names], dtype=np.float64)
    pixels = project_flame_local_points(
        points,
        rotation_matrix=rotation_matrix,
        tvec_mm=tvec_mm,
        scale_mm_per_flame_unit=scale_mm_per_flame_unit,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    iris_points = iris.as_mapping()
    records: list[dict[str, float | str]] = []
    for name, pixel in zip(names, pixels):
        error_468 = float(np.linalg.norm(pixel - iris_points[IRIS_GROUP_468]))
        error_473 = float(np.linalg.norm(pixel - iris_points[IRIS_GROUP_473]))
        best_group = IRIS_GROUP_468 if error_468 <= error_473 else IRIS_GROUP_473
        records.append(
            {
                "candidate": name,
                "projected_x_px": float(pixel[0]),
                "projected_y_px": float(pixel[1]),
                "error_to_iris_group_468_px": error_468,
                "error_to_iris_group_473_px": error_473,
                "best_iris_group": best_group,
                "best_iris_error_px": min(error_468, error_473),
            }
        )
    return records
