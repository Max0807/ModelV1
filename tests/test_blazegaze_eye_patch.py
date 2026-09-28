from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from baselines.blazegaze.eye_patch import (
    BlazeGazeEyePatchManifest,
    build_blazegaze_eye_patch,
)


def _landmarks() -> np.ndarray:
    points = np.full((478, 2), [100.0, 50.0], dtype=np.float32)
    points[4] = [100.0, 50.0]
    points[103] = [10.0, 10.0]
    points[150] = [10.0, 90.0]
    points[379] = [190.0, 90.0]
    points[332] = [190.0, 10.0]
    points[151] = [100.0, 30.0]
    points[195] = [100.0, 70.0]
    return points


def test_eye_patch_matches_released_transform_order_and_shape() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(200, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(100, dtype=np.uint8)[:, None]

    result = build_blazegaze_eye_patch(
        image,
        _landmarks(),
        face_padding_coefs=(0.0, 0.0),
    )

    assert result.patch_rgb.shape == (128, 512, 3)
    assert result.patch_rgb.dtype == np.uint8
    # The released implementation casts the homogeneous result to int32,
    # therefore a floating value infinitesimally below 128 truncates to 127.
    assert result.crop_top_y == 127
    assert result.crop_bottom_y == 384
    assert result.source_quad_area_px2 == pytest.approx(14400.0)
    source_h = np.array([10.0, 10.0, 1.0])
    mapped = result.homography @ source_h
    mapped = mapped[:2] / mapped[2]
    assert mapped == pytest.approx([0.0, 0.0], abs=1e-5)


def test_eye_patch_rejects_inverted_vertical_bounds() -> None:
    points = _landmarks()
    points[151], points[195] = points[195].copy(), points[151].copy()

    with pytest.raises(ValueError, match="Invalid warped eye bounds"):
        build_blazegaze_eye_patch(
            np.zeros((100, 200, 3), dtype=np.uint8),
            points,
            face_padding_coefs=(0.0, 0.0),
        )


def test_eye_patch_manifest_resolves_relative_success_paths(tmp_path: Path) -> None:
    patch_path = tmp_path / "patches" / "sample.png"
    patch_path.parent.mkdir()
    patch_path.touch()
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "status", "eye_patch_path", "landmark_count"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "person/sample",
                "status": "success",
                "eye_patch_path": "patches/sample.png",
                "landmark_count": 478,
            }
        )
        writer.writerow(
            {
                "sample_id": "person/failed",
                "status": "failed",
                "eye_patch_path": "",
                "landmark_count": 0,
            }
        )

    manifest = BlazeGazeEyePatchManifest.load([manifest_path])

    assert len(manifest) == 1
    assert manifest.lookup("person/sample").eye_patch_path == patch_path.resolve()
    assert manifest.lookup("person/sample").landmark_count == 478
