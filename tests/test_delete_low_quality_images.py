import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.delete_low_quality_images import (
    apply_data_log_update,
    frame_index_from_image_filename,
    iter_matching_files,
    plan_data_log_updates,
    quarantine_changes,
)


class DeleteLowQualityImagesTests(unittest.TestCase):
    def test_frame_index_from_supported_image_names(self) -> None:
        self.assertEqual(frame_index_from_image_filename("img_20260422_151156_00021.jpg"), 21)
        self.assertEqual(frame_index_from_image_filename("frame_000123_1776841916195.jpg"), 123)
        self.assertIsNone(frame_index_from_image_filename("not_a_crossgaze_image.jpg"))

    def test_plan_and_apply_data_log_update(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_log = Path(temporary_dir) / "data_log_123.csv"
            data_log.write_bytes(
                b"\xef\xbb\xbfframe_idx,image_filename,value\r\n"
                b"20,img_20260422_151155_00020.jpg,keep\r\n"
                b"21,img_20260422_151156_00021.jpg,remove\r\n"
                b"22,img_20260422_151157_00022.jpg,keep\r\n"
            )

            updates = plan_data_log_updates(Path(temporary_dir), [21])

            self.assertEqual(len(updates), 1)
            self.assertEqual([row[2] for row in updates[0].removed_rows], ["remove"])
            apply_data_log_update(updates[0])

            rewritten = data_log.read_bytes()
            self.assertTrue(rewritten.startswith(b"\xef\xbb\xbf"))
            self.assertIn(b"\r\n", rewritten)
            with data_log.open("r", encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual([row["frame_idx"] for row in rows], ["20", "22"])

    def test_unrecognized_filename_falls_back_to_frame_idx(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_log = Path(temporary_dir) / "data_log_456.csv"
            data_log.write_text(
                "frame_idx,image_filename\n"
                "7,custom_name.jpg\n"
                "8,img_20260422_151156_00009.jpg\n",
                encoding="utf-8",
            )

            update = plan_data_log_updates(Path(temporary_dir), [7])[0]

            self.assertEqual(update.removed_rows, [["7", "custom_name.jpg"]])
            self.assertEqual(
                update.retained_rows,
                [["8", "img_20260422_151156_00009.jpg"]],
            )

    def test_quarantine_moves_images_and_backs_up_original_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset_dir = Path(temporary_dir)
            image_dir = dataset_dir / "insightface_img"
            image_dir.mkdir()
            image_path = image_dir / "img_20260422_151156_00021.jpg"
            image_path.write_bytes(b"image data")

            data_log = dataset_dir / "data_log_123.csv"
            original_csv = (
                b"frame_idx,image_filename,value\n"
                b"20,img_20260422_151155_00020.jpg,keep\n"
                b"21,img_20260422_151156_00021.jpg,remove\n"
            )
            data_log.write_bytes(original_csv)

            matches = list(iter_matching_files(dataset_dir, [21]))
            updates = plan_data_log_updates(dataset_dir, [21])
            batch_dir = quarantine_changes(dataset_dir, matches, updates, [21])

            quarantined_image = batch_dir / "images" / "insightface_img" / image_path.name
            self.assertFalse(image_path.exists())
            self.assertEqual(quarantined_image.read_bytes(), b"image data")
            self.assertEqual(
                (batch_dir / "data_logs" / data_log.name).read_bytes(),
                original_csv,
            )
            with data_log.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual([row["frame_idx"] for row in rows], ["20"])

            manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["frame_indices"], [21])
            self.assertEqual(manifest["data_logs"][0]["removed_rows"], 1)


if __name__ == "__main__":
    unittest.main()
