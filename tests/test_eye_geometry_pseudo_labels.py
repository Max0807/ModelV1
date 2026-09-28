from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from modelv1.data.augmentation import (
    EyeAppearanceAugmentationConfig,
    PairedEyeAppearanceAugmentation,
)
from modelv1.data.dataset import CROP_CAM_COLUMNS, SCENE_COLUMNS, ModelV1Dataset
from modelv1.data.eye_geometry_pseudo_labels import (
    EYE_GEOMETRY_PSEUDO_LABEL_SCHEMA_VERSION,
    EYE_PSEUDO_GEOMETRY_FEATURE_DIM,
    EYE_PSEUDO_GEOMETRY_FEATURE_NAMES,
    EYE_PSEUDO_POINT_COUNT,
    EYE_PSEUDO_POINT_NAMES,
    EyeGeometryPseudoLabelTable,
)


def write_archive(path: Path, sample_ids: tuple[str, ...]) -> None:
    count = len(sample_ids)
    point_shape = (count, EYE_PSEUDO_POINT_COUNT, 2)
    mask_shape = (count, EYE_PSEUDO_POINT_COUNT)
    feature_shape = (count, EYE_PSEUDO_GEOMETRY_FEATURE_DIM)
    points = np.full(point_shape, 0.5, dtype=np.float32)
    local = np.full(point_shape, 0.25, dtype=np.float32)
    point_mask = np.ones(mask_shape, dtype=np.bool_)
    features = np.full(feature_shape, 0.4, dtype=np.float32)
    features[:, -1] = 0.8
    detection = np.ones(count, dtype=np.bool_)
    eye_valid = np.ones(count, dtype=np.bool_)
    sample_valid = np.ones(count, dtype=np.bool_)
    quality = np.full(count, 0.8, dtype=np.float32)
    if count > 1:
        detection[-1] = False
        eye_valid[-1] = False
        sample_valid[-1] = False
        quality[-1] = 0.0
        points[-1] = np.nan
        local[-1] = np.nan
        features[-1] = np.nan
        point_mask[-1] = False
    metadata = {
        "schema_version": EYE_GEOMETRY_PSEUDO_LABEL_SCHEMA_VERSION,
        "point_names": list(EYE_PSEUDO_POINT_NAMES),
        "geometry_feature_names": list(EYE_PSEUDO_GEOMETRY_FEATURE_NAMES),
    }
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids),
        dataset=np.asarray(["13"] * count),
        image_name=np.asarray([f"{sample_id}.jpg" for sample_id in sample_ids]),
        detection_success_mask=detection,
        valid_mask=sample_valid,
        left_eye_valid_mask=eye_valid,
        right_eye_valid_mask=eye_valid,
        left_eye_points_crop_norm=points,
        right_eye_points_crop_norm=points,
        left_eye_points_local=local,
        right_eye_points_local=local,
        left_eye_point_valid_mask=point_mask,
        right_eye_point_valid_mask=point_mask,
        left_eye_geometry=features,
        right_eye_geometry=features,
        sample_quality_score=quality,
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def minimal_dataset_row(sample_id: str, image_path: Path) -> dict[str, str]:
    row = {
        "sample_id": sample_id,
        "dataset": "dataset_dual_rigid_body_13",
        "image_name": f"{sample_id}.png",
        "face_path": str(image_path),
        "left_eye_path": str(image_path),
        "right_eye_path": str(image_path),
        "source_image_path": str(image_path),
        "uv_gt_u_mm": "10.0",
        "uv_gt_v_mm": "20.0",
        "gaze_target_w_x_mm": "10.0",
        "gaze_target_w_y_mm": "20.0",
        "gaze_target_w_z_mm": "0.0",
        "table_origin_w_x_mm": "0.0",
        "table_origin_w_y_mm": "0.0",
        "table_origin_w_z_mm": "0.0",
    }
    row.update({column: "0.0" for column in CROP_CAM_COLUMNS})
    row.update({column: "0.0" for column in SCENE_COLUMNS})
    return row


def full_occlusion_transform() -> PairedEyeAppearanceAugmentation:
    return PairedEyeAppearanceAugmentation(
        EyeAppearanceAugmentationConfig(
            photometric_probability=0.0,
            brightness_min=1.0,
            brightness_max=1.0,
            contrast_min=1.0,
            contrast_max=1.0,
            gamma_min=1.0,
            gamma_max=1.0,
            blur_probability=0.0,
            blur_kernel_size=3,
            blur_sigma_min=0.1,
            blur_sigma_max=0.1,
            noise_probability=0.0,
            noise_std_max=0.0,
            occlusion_probability=1.0,
            occlusion_area_min=1.0,
            occlusion_area_max=1.0,
            occlusion_aspect_min=60.0 / 36.0,
            occlusion_aspect_max=60.0 / 36.0,
        )
    )


class EyeGeometryPseudoLabelTableTests(unittest.TestCase):
    def test_loads_anatomical_pair_and_zero_fills_failed_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.npz"
            write_archive(path, ("ok", "failed"))
            table = EyeGeometryPseudoLabelTable.load((path,))

            valid = table.lookup("ok")
            self.assertEqual(
                tuple(valid.landmarks_crop_norm.shape),
                (2, EYE_PSEUDO_POINT_COUNT, 2),
            )
            self.assertEqual(tuple(valid.geometry_features.shape), (2, 10))
            self.assertEqual(float(valid.sample_valid.item()), 1.0)

            failed = table.lookup("failed")
            self.assertTrue(torch.isfinite(failed.landmarks_crop_norm).all())
            self.assertEqual(int(failed.landmark_valid_mask.count_nonzero()), 0)
            self.assertEqual(int(failed.eye_valid_mask.count_nonzero()), 0)
            self.assertEqual(float(failed.sample_valid.item()), 0.0)

    def test_duplicate_sample_ids_across_archives_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            write_archive(first, ("duplicate",))
            write_archive(second, ("duplicate",))
            with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
                EyeGeometryPseudoLabelTable.load((first, second))

    def test_missing_sample_ids_fail_only_in_strict_lookup_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.npz"
            write_archive(path, ("present",))
            table = EyeGeometryPseudoLabelTable.load((path,))
            with self.assertRaisesRegex(KeyError, "Missing 1"):
                table.require_sample_ids(("present", "missing"))
            empty = table.empty("missing")
            self.assertEqual(empty.sample_id, "missing")
            self.assertEqual(int(empty.landmark_valid_mask.count_nonzero()), 0)

    def test_dataset_emits_15_points_and_updates_occlusion_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "eye.png"
            Image.new("RGB", (60, 36), color=(20, 40, 60)).save(image_path)
            csv_path = root / "dataset.csv"
            row = minimal_dataset_row("sample", image_path)
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            archive_path = root / "labels.npz"
            write_archive(archive_path, ("sample",))

            dataset = ModelV1Dataset(
                csv_path,
                datasets=("13",),
                paired_eye_transform=full_occlusion_transform(),
                eye_geometry_pseudo_label_paths=(archive_path,),
                require_eye_geometry_pseudo_labels=True,
            )
            item = dataset[0]

        self.assertEqual(tuple(item["eye_pseudo_landmarks_xy"].shape), (2, 15, 2))
        self.assertEqual(
            int(item["eye_pseudo_landmark_base_valid_mask"].count_nonzero()),
            30,
        )
        self.assertEqual(int(item["eye_pseudo_landmark_valid_mask"].count_nonzero()), 0)
        self.assertEqual(int(item["eye_pseudo_occluded_mask"].count_nonzero()), 30)
        torch.testing.assert_close(
            item["eye_pseudo_landmarks_xy"],
            torch.full((2, 15, 2), 0.5),
        )

if __name__ == "__main__":
    unittest.main()
