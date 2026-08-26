from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from modelv1.depth_prior.pnp import PNP_GEOMETRY_VERSION
from scripts.prepare_modelv1_preprocessing import PipelineArtifacts, depth_method
from scripts.validate_pnp_geometry import audit


class PreprocessingControllerTests(unittest.TestCase):
    def test_depth_method_aliases_are_canonical(self) -> None:
        self.assertEqual(depth_method("PnP1010"), "pnp1010")
        self.assertEqual(depth_method("iris65"), "iris65")
        self.assertEqual(depth_method("both"), "both")

    def test_artifacts_keep_depth_methods_separate(self) -> None:
        artifacts = PipelineArtifacts("11", Path("preprocessed"))
        self.assertIn("pnp1010", str(artifacts.pnp_prior))
        self.assertIn("iris65", str(artifacts.iris_prior))
        self.assertNotEqual(artifacts.pnp_dataset, artifacts.iris_dataset)
        self.assertIn("common_pnp1010_iris65", str(artifacts.virtual_root("both")))

    def test_geometry_audit_can_exclude_failed_reconstructions(self) -> None:
        identity = {
            f"rotation_{row}{column}": str(float(row == column))
            for row in range(3)
            for column in range(3)
        }
        success = {
            "sample_id": "dataset11/frame_1",
            "dataset": "dataset_dual_rigid_body_11",
            "depth_prior_status": "success",
            "pnp_geometry_version": PNP_GEOMETRY_VERSION,
            "tvec_z_mm": "700",
            "left_eye_camera_z_mm": "695",
            "right_eye_camera_z_mm": "705",
            "pnp_min_object_depth_mm": "650",
            "reprojection_error_mean_px": "2",
            "reprojection_error_max_px": "5",
            "scale_disagreement_ratio": "0.1",
            **identity,
        }
        failed = {
            **success,
            "sample_id": "dataset11/frame_2",
            "depth_prior_status": "failed",
            "pnp_geometry_version": "",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pnp.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(success))
                writer.writeheader()
                writer.writerows([success, failed])

            permissive = audit(path, allow_failed_rows=True)
            strict = audit(path, allow_failed_rows=False)

        self.assertTrue(permissive["passed"])
        self.assertEqual(permissive["success_count"], 1)
        self.assertEqual(permissive["failed_count"], 1)
        self.assertFalse(strict["passed"])
        self.assertEqual(strict["violations"], {"non_success_row": 1})


if __name__ == "__main__":
    unittest.main()
