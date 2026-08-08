from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from modelv1.data.virtual_camera_manifest import (
    ANATOMICAL_EYE_SEMANTICS,
    IMAGE_SOURCE_VIRTUAL_CAMERA,
    LEGACY_EYE_SIDE_CONVERSION,
    VirtualCameraManifest,
    canonical_image_source,
    merge_virtual_camera_manifests,
)


class VirtualCameraManifestTests(unittest.TestCase):
    def _write_manifest(
        self,
        root: Path,
        *,
        sample_id: str = "dataset_3/frame_0001",
        eye_semantics: str = ANATOMICAL_EYE_SEMANTICS,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, size in (
            ("face", (224, 224)),
            ("left_eye", (60, 36)),
            ("right_eye", (60, 36)),
        ):
            path = root / f"{name}.png"
            Image.new("RGB", size, color=(10, 20, 30)).save(path)
            paths[name] = path
        row = {
            "sample_id": sample_id,
            "normalization_status": "success",
            "normalization_reason": "",
            "eye_side_semantics": eye_semantics,
            "landmark_eye_side_conversion": LEGACY_EYE_SIDE_CONVERSION,
            "normalized_face_path": str(paths["face"]),
            "normalized_left_eye_path": str(paths["left_eye"]),
            "normalized_right_eye_path": str(paths["right_eye"]),
        }
        identity = torch.eye(3).reshape(-1).tolist()
        for index, value in enumerate(identity):
            row[f"rotation_n_from_c_{index // 3}{index % 3}"] = value
        path = root / "manifest.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_loads_anatomical_v2_paths_and_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(Path(temporary_directory))
            manifest = VirtualCameraManifest.load(manifest_path)
            record = manifest.lookup("dataset_3/frame_0001")
            self.assertEqual(canonical_image_source("VIRTUAL_CAMERA"), IMAGE_SOURCE_VIRTUAL_CAMERA)
            self.assertTrue(record.face_path.is_file())
            torch.testing.assert_close(record.rotation_n_from_c, torch.eye(3))
            torch.testing.assert_close(record.rotation_c_from_n, torch.eye(3))

    def test_rejects_legacy_manifest_without_anatomical_eye_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = self._write_manifest(
                Path(temporary_directory),
                eye_semantics="",
            )
            manifest = VirtualCameraManifest.load(manifest_path)
            self.assertFalse(manifest.contains("dataset_3/frame_0001"))
            with self.assertRaisesRegex(KeyError, "Regenerate with virtual-camera v2"):
                manifest.lookup("dataset_3/frame_0001")

    def test_merges_disjoint_manifests_without_copying_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = self._write_manifest(
                root / "dataset3",
                sample_id="dataset_3/frame_0001",
            )
            second = self._write_manifest(
                root / "dataset13",
                sample_id="dataset_13/frame_0001",
            )
            merged_path = merge_virtual_camera_manifests(
                [first, second],
                root / "run" / "manifest.csv",
            )
            merged = VirtualCameraManifest.load(merged_path)
            self.assertEqual(len(merged), 2)
            self.assertTrue(merged.lookup("dataset_3/frame_0001").face_path.is_file())
            self.assertTrue(merged.lookup("dataset_13/frame_0001").face_path.is_file())


if __name__ == "__main__":
    unittest.main()
