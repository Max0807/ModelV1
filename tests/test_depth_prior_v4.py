from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from modelv1.data.depth_prior import DepthPriorTable
from modelv1.depth_prior.pnp import PNP_GEOMETRY_VERSION


class DepthPriorV4Tests(unittest.TestCase):
    def test_loads_raw_mm_geometry_quality_and_log_scale_uncertainty(self) -> None:
        row = {
            "sample_id": "dataset/frame",
            "depth_prior_status": "success",
            "pnp_geometry_version": PNP_GEOMETRY_VERSION,
            "pnp_min_object_depth_mm": "580",
            "pnp_tvec_z_mm": "600",
            "left_eye_camera_x_mm": "-30",
            "left_eye_camera_y_mm": "1",
            "left_eye_camera_z_mm": "600",
            "right_eye_camera_x_mm": "30",
            "right_eye_camera_y_mm": "1",
            "right_eye_camera_z_mm": "602",
            "reprojection_error_mean_px": "2",
            "reprojection_error_max_px": "4",
            "pnp_inlier_count": "7",
            "pnp_num_points": "8",
            "scale_disagreement_ratio": "0.05",
            "pnp_confidence": "0.8",
            "depth_uncertainty_status": "success",
            "depth_uncertainty_reason": "",
            "depth_log_scale_std": "0.04",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            table = DepthPriorTable.load(path, "raw_eye6d")
        torch.testing.assert_close(
            table.lookup("dataset/frame"),
            torch.tensor([-30.0, 1.0, 600.0, 30.0, 1.0, 602.0]),
        )
        self.assertAlmostEqual(
            table.lookup_depth_log_scale_std("dataset/frame"),
            0.04,
        )
        self.assertEqual(tuple(table.lookup_quality("dataset/frame").shape), (4,))


if __name__ == "__main__":
    unittest.main()
