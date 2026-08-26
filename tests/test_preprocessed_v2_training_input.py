from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from modelv1.preprocessed_v2 import (
    PREPROCESSING_PIPELINE_VERSION,
    PreprocessedV2TrainingInput,
    canonical_preprocessed_v2_depth_method,
    load_preprocessed_v2_training_inputs,
)


class PreprocessedV2TrainingInputTests(unittest.TestCase):
    def _write_contract(
        self,
        root: Path,
        *,
        dataset_id: str,
        method: str,
    ) -> Path:
        dataset_root = root / f"dataset{dataset_id}"
        referenced = dataset_root / "referenced"
        referenced.mkdir(parents=True)
        paths = {
            "dataset_csv": referenced / "dataset.csv",
            "depth_prior_csv": referenced / "depth.csv",
            "virtual_camera_manifest": referenced / "virtual.csv",
            "virtual_camera_report": referenced / "virtual.json",
            "virtual_camera_pose_source": referenced / "pnp.csv",
            "deca_cache": referenced / "deca.npz",
        }
        for path in paths.values():
            path.write_text("placeholder", encoding="utf-8")
        manifest = (
            dataset_root
            / "manifests"
            / f"training_inputs_{method}_dataset{dataset_id}.json"
        )
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "pipeline_version": PREPROCESSING_PIPELINE_VERSION,
                    "dataset_id": dataset_id,
                    "dataset_name": f"dataset_dual_rigid_body_{dataset_id}",
                    "depth_method": method,
                    **{key: str(path) for key, path in paths.items()},
                    "pnp_geometry_version": "test_pnp_v2",
                    "sample_count": 2,
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_depth_method_aliases(self) -> None:
        self.assertEqual(canonical_preprocessed_v2_depth_method("pnp"), "pnp1010")
        self.assertEqual(
            canonical_preprocessed_v2_depth_method("iris_ipd_65mm"),
            "iris65",
        )

    def test_loads_requested_dataset_and_depth_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_contract(
                root,
                dataset_id="11",
                method="iris65",
            )
            loaded = PreprocessedV2TrainingInput.load(
                manifest,
                expected_dataset_id="11",
                expected_depth_method="iris65",
            )
        self.assertEqual(loaded.dataset_id, "11")
        self.assertEqual(loaded.depth_method, "iris65")
        self.assertEqual(loaded.sample_count, 2)

    def test_root_resolver_switches_dataset_and_method(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_contract(root, dataset_id="11", method="pnp1010")
            self._write_contract(root, dataset_id="3", method="pnp1010")
            loaded = load_preprocessed_v2_training_inputs(
                root,
                ["11", "3"],
                "pnp1010",
            )
        self.assertEqual([item.dataset_id for item in loaded], ["11", "3"])
        self.assertTrue(all(item.depth_method == "pnp1010" for item in loaded))

    def test_rejects_manifest_dataset_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_contract(
                root,
                dataset_id="11",
                method="iris65",
            )
            with self.assertRaisesRegex(ValueError, "dataset mismatch"):
                PreprocessedV2TrainingInput.load(
                    manifest,
                    expected_dataset_id="3",
                    expected_depth_method="iris65",
                )


if __name__ == "__main__":
    unittest.main()
