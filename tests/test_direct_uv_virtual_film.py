from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from PIL import Image

from modelv1.data.dataset import (
    CROP_CAM_COLUMNS,
    DIRECT_UV_TARGET_FRAME_TABLE_LOCAL,
    DIRECT_UV_TARGET_FRAME_VIRTUAL_CAMERA,
    ModelV1Dataset,
    SCENE_COLUMNS,
    build_virtual_camera_pose_table_feature,
    build_modelv1_dataloaders,
    transform_gaze_target_to_virtual_camera_uv,
    transform_table_frame7_to_virtual_camera,
)
from modelv1.data.normalization import (
    UVTargetNormalizer,
    VirtualDistanceScaleNormalizer,
)
from modelv1.geometry import virtual_camera_xy_to_table_uv
from modelv1.inference import predict_uv_mm
from modelv1.losses import (
    DirectTableUVLoss,
    DirectVirtualTableUVLoss,
    UVLossConfig,
)
from modelv1.model import ModelV1, ModelV1Config
from modelv1.data.virtual_camera_manifest import (
    ANATOMICAL_EYE_SEMANTICS,
    LEGACY_EYE_SIDE_CONVERSION,
)
from modelv1.depth_prior.pnp import PNP_GEOMETRY_VERSION
from scripts.train_modelv1 import run_epoch


class DirectUVVirtualFiLMTests(unittest.TestCase):
    def _config(self, *, use_virtual_pose_film: bool = False) -> ModelV1Config:
        return ModelV1Config(
            prediction_mode="direct_uv",
            gaze_prediction_frame="camera",
            deca_feature_representation="none",
            deca_feature_dim=0,
            use_face_image=False,
            use_crop_cam=False,
            scene_representation="table_frame7",
            scene_dim=7,
            use_eye_geometry=False,
            eye_geometry_representation="raw_eye6d",
            eye_backbone="cnn",
            eye_embedding_dim=16,
            per_eye_embedding_dim=8,
            scene_embedding_dim=8,
            scene_hidden_dims=(8,),
            fusion_hidden_dims=(16, 8),
            use_virtual_distance_film=not use_virtual_pose_film,
            use_virtual_pose_film=use_virtual_pose_film,
            virtual_pose_dim=9,
            virtual_distance_film_hidden_dims=(4,),
            virtual_distance_film_embedding_dim=6,
            virtual_distance_film_delta_max=0.1,
            branch_dropout=0.0,
            fusion_dropout=0.0,
        )

    def test_log_scale_normalizer_fits_training_values(self) -> None:
        scales = torch.tensor([[0.5], [1.0], [2.0]])
        normalizer = VirtualDistanceScaleNormalizer.fit(scales)
        normalized = normalizer.normalize(scales)
        torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(1))
        torch.testing.assert_close(
            normalized.std(dim=0, unbiased=False),
            torch.ones(1),
        )
        restored = VirtualDistanceScaleNormalizer.from_state_dict(
            normalizer.state_dict()
        )
        torch.testing.assert_close(
            restored.normalize(scales),
            normalized,
        )

    def test_table_frame_rotates_to_virtual_camera_without_changing_distance(self) -> None:
        frame_c = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.7])
        rotation_n_from_c = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        frame_n = transform_table_frame7_to_virtual_camera(
            frame_c,
            rotation_n_from_c,
        )
        torch.testing.assert_close(
            frame_n,
            torch.tensor([0.0, 0.0, -1.0, 0.0, 1.0, 0.0, -0.7]),
        )

    def test_virtual_camera_pose_table_feature_is_r6d_and_scaled_translation(self) -> None:
        table_frame7_n = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.7])

        pose = build_virtual_camera_pose_table_feature(table_frame7_n)

        torch.testing.assert_close(
            pose,
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.7]),
        )

    def test_gaze_target_rotates_to_virtual_camera_xy(self) -> None:
        target_c = torch.tensor([100.0, 200.0, 300.0])
        rotation_n_from_c = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        uv_n = transform_gaze_target_to_virtual_camera_uv(
            target_c,
            rotation_n_from_c,
        )
        torch.testing.assert_close(uv_n, torch.tensor([300.0, 200.0]))

    def test_raw_uv_target_selects_requested_coordinate_frame(self) -> None:
        row = {
            "sample_id": "sample/1",
            "uv_gt_u_mm": "10.0",
            "uv_gt_v_mm": "20.0",
            "gaze_cam_recomputed_x_mm": "100.0",
            "gaze_cam_recomputed_y_mm": "200.0",
            "gaze_cam_recomputed_z_mm": "300.0",
        }
        rotation_n_from_c = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        record = SimpleNamespace(rotation_n_from_c=rotation_n_from_c)
        manifest = SimpleNamespace(lookup=lambda sample_id: record)
        dataset = object.__new__(ModelV1Dataset)
        dataset.rows = [row]
        dataset.virtual_camera_manifest = manifest

        dataset.direct_uv_target_frame = DIRECT_UV_TARGET_FRAME_TABLE_LOCAL
        torch.testing.assert_close(
            dataset.raw_uv_target(0),
            torch.tensor([10.0, 20.0]),
        )
        dataset.direct_uv_target_frame = DIRECT_UV_TARGET_FRAME_VIRTUAL_CAMERA
        torch.testing.assert_close(
            dataset.raw_uv_target(0),
            torch.tensor([300.0, 200.0]),
        )

    def test_direct_uv_uses_identity_initialized_film_and_no_geometry(self) -> None:
        model = ModelV1(self._config()).eval()
        self.assertIsNotNone(model.direct_uv_film_fusion)
        self.assertIsNone(model.fusion_mlp)
        self.assertIsNone(model.gaze_head)
        self.assertIsNone(model.ray_table_geometry)

        batch = {
            "left_eye": torch.zeros((2, 3, 36, 60)),
            "right_eye": torch.zeros((2, 3, 36, 60)),
            "table_frame7_n": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.7]]
            ).expand(2, -1),
            "virtual_log_scale_normalized": torch.tensor([[-1.0], [1.0]]),
        }
        with torch.no_grad():
            features = model(batch, return_features=True)
        self.assertEqual(tuple(features["uv"].shape), (2, 2))
        self.assertEqual(tuple(features["virtual_scale_embedding"].shape), (2, 6))
        self.assertEqual(tuple(features["virtual_film_gamma"].shape), (2, 16))
        self.assertEqual(tuple(features["virtual_film_beta"].shape), (2, 16))
        torch.testing.assert_close(
            features["virtual_film_gamma"],
            torch.zeros((2, 16)),
        )
        torch.testing.assert_close(
            features["virtual_film_beta"],
            torch.zeros((2, 16)),
        )

    def test_direct_uv_pose_film_uses_pose9_condition(self) -> None:
        model = ModelV1(self._config(use_virtual_pose_film=True)).eval()
        table_frame7_n = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.7]]
        ).expand(2, -1)
        pose = torch.stack(
            [
                build_virtual_camera_pose_table_feature(table_frame7_n[0]),
                build_virtual_camera_pose_table_feature(table_frame7_n[1]),
            ],
            dim=0,
        )
        batch = {
            "left_eye": torch.zeros((2, 3, 36, 60)),
            "right_eye": torch.zeros((2, 3, 36, 60)),
            "table_frame7_n": table_frame7_n,
            "virtual_camera_pose_table": pose,
        }

        with torch.no_grad():
            features = model(batch, return_features=True)

        self.assertEqual(tuple(features["uv"].shape), (2, 2))
        self.assertEqual(tuple(features["virtual_film_condition"].shape), (2, 9))
        self.assertEqual(tuple(features["virtual_pose_embedding"].shape), (2, 6))
        torch.testing.assert_close(features["virtual_film_condition"], pose)
        torch.testing.assert_close(
            features["virtual_film_gamma"],
            torch.zeros((2, 16)),
        )
        torch.testing.assert_close(
            features["virtual_film_beta"],
            torch.zeros((2, 16)),
        )

    def test_direct_uv_film_requires_scale_input(self) -> None:
        model = ModelV1(self._config())
        batch = {
            "left_eye": torch.zeros((1, 3, 36, 60)),
            "right_eye": torch.zeros((1, 3, 36, 60)),
            "table_frame7_n": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.7]]
            ),
        }
        with self.assertRaisesRegex(KeyError, "virtual_log_scale_normalized"):
            model(batch)

    def test_direct_uv_pose_film_requires_pose_input(self) -> None:
        model = ModelV1(self._config(use_virtual_pose_film=True))
        batch = {
            "left_eye": torch.zeros((1, 3, 36, 60)),
            "right_eye": torch.zeros((1, 3, 36, 60)),
            "table_frame7_n": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.7]]
            ),
        }
        with self.assertRaisesRegex(KeyError, "virtual_camera_pose_table"):
            model(batch)

    def test_explicit_split_val_batch_includes_virtual_pose_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "image.png"
            Image.new("RGB", (64, 64), color=(10, 20, 30)).save(image_path)
            csv_path = root / "dataset.csv"
            manifest_path = root / "manifest.csv"
            rows = [
                self._minimal_dataset_row(
                    "train/1",
                    "train",
                    image_path,
                    uv=(0.0, 0.0),
                ),
                self._minimal_dataset_row(
                    "val/1",
                    "val",
                    image_path,
                    uv=(10.0, 20.0),
                ),
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest_rows = [
                self._minimal_manifest_row("train/1", image_path),
                self._minimal_manifest_row("val/1", image_path),
            ]
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
                writer.writeheader()
                writer.writerows(manifest_rows)

            _, val_loader = build_modelv1_dataloaders(
                csv_path=csv_path,
                train_datasets=("train",),
                val_datasets=("val",),
                split_mode="explicit_datasets",
                all_datasets=("train", "val"),
                val_ratio=0.5,
                split_seed=42,
                batch_size=1,
                num_workers=0,
                pin_memory=False,
                load_face_image=False,
                deca_cache_path=None,
                require_deca_features=False,
                image_source="virtual_camera",
                virtual_camera_manifest_path=manifest_path,
                direct_uv_target_frame=DIRECT_UV_TARGET_FRAME_TABLE_LOCAL,
                scene_representation="table_frame7",
                use_virtual_distance_film=False,
                use_virtual_pose_film=True,
            )

            batch = next(iter(val_loader))

        self.assertIn("virtual_camera_pose_table", batch)
        self.assertEqual(tuple(batch["virtual_camera_pose_table"].shape), (1, 9))
        self.assertIn("table_frame7_n", batch)
        self.assertNotIn("virtual_log_scale_normalized", batch)

    def test_virtual_face_can_use_legacy_eye_paths(self) -> None:
        """The hybrid ablation must not load the virtual 60x36 eye crops."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy_face = root / "legacy_face.png"
            legacy_left_eye = root / "legacy_left_eye.png"
            legacy_right_eye = root / "legacy_right_eye.png"
            virtual_face = root / "virtual_face.png"
            virtual_left_eye = root / "virtual_left_eye.png"
            virtual_right_eye = root / "virtual_right_eye.png"
            for path in (
                legacy_face,
                legacy_left_eye,
                legacy_right_eye,
                virtual_face,
                virtual_left_eye,
                virtual_right_eye,
            ):
                Image.new("RGB", (64, 64), color=(10, 20, 30)).save(path)

            row = self._minimal_dataset_row(
                "hybrid/1", "hybrid", legacy_face, uv=(0.0, 0.0)
            )
            row["left_eye_path"] = str(legacy_left_eye)
            row["right_eye_path"] = str(legacy_right_eye)
            csv_path = root / "dataset.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            manifest_row = self._minimal_manifest_row("hybrid/1", virtual_face)
            manifest_row["normalized_left_eye_path"] = str(virtual_left_eye)
            manifest_row["normalized_right_eye_path"] = str(virtual_right_eye)
            manifest_path = root / "manifest.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest_row))
                writer.writeheader()
                writer.writerow(manifest_row)

            dataset = ModelV1Dataset(
                csv_path,
                datasets=("hybrid",),
                load_face_image=False,
                deca_cache_path=None,
                require_deca_features=False,
                image_source="virtual_camera",
                eye_image_source="legacy",
                virtual_camera_manifest_path=manifest_path,
                scene_representation="table_frame7",
            )
            paths = dataset[0]["paths"]

        self.assertEqual(paths["face"], str(virtual_face))
        self.assertEqual(paths["left_eye"], str(legacy_left_eye))
        self.assertEqual(paths["right_eye"], str(legacy_right_eye))

    @staticmethod
    def _minimal_dataset_row(
        sample_id: str,
        dataset: str,
        image_path: Path,
        *,
        uv: tuple[float, float],
    ) -> dict[str, str]:
        row = {
            "sample_id": sample_id,
            "dataset": dataset,
            "image_name": f"{sample_id.replace('/', '_')}.png",
            "face_path": str(image_path),
            "left_eye_path": str(image_path),
            "right_eye_path": str(image_path),
            "source_image_path": str(image_path),
            "uv_gt_u_mm": str(uv[0]),
            "uv_gt_v_mm": str(uv[1]),
            "gaze_target_w_x_mm": str(uv[0]),
            "gaze_target_w_y_mm": str(uv[1]),
            "gaze_target_w_z_mm": "0.0",
            "table_origin_w_x_mm": "0.0",
            "table_origin_w_y_mm": "0.0",
            "table_origin_w_z_mm": "0.0",
        }
        row.update({column: "0.0" for column in CROP_CAM_COLUMNS})
        scene = [0.0] * len(SCENE_COLUMNS)
        scene[0:4] = [0.0, 0.0, 1.0, -700.0]
        scene[7:13] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        scene[13:22] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        row.update(
            {
                column: str(value)
                for column, value in zip(SCENE_COLUMNS, scene)
            }
        )
        return row

    @staticmethod
    def _minimal_manifest_row(sample_id: str, image_path: Path) -> dict[str, str]:
        row = {
            "sample_id": sample_id,
            "normalization_status": "success",
            "normalization_reason": "",
            "eye_side_semantics": ANATOMICAL_EYE_SEMANTICS,
            "landmark_eye_side_conversion": LEGACY_EYE_SIDE_CONVERSION,
            "pnp_geometry_version": PNP_GEOMETRY_VERSION,
            "virtual_distance_scale": "1.0",
            "normalized_face_path": str(image_path),
            "normalized_left_eye_path": str(image_path),
            "normalized_right_eye_path": str(image_path),
        }
        for index, value in enumerate(torch.eye(3).reshape(-1).tolist()):
            row[f"rotation_n_from_c_{index // 3}{index % 3}"] = str(value)
        return row

    def test_virtual_camera_xy_recovers_inclined_table_uv(self) -> None:
        angle = torch.deg2rad(torch.tensor(30.0))
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        e1_n = torch.tensor([cosine, 0.0, -sine])
        e2_n = torch.tensor([0.0, 1.0, 0.0])
        normal_n = torch.cross(e1_n, e2_n, dim=0)
        distance_mm = -600.0
        table_frame7_n = torch.cat(
            (e1_n, e2_n, torch.tensor([distance_mm / 1000.0]))
        ).unsqueeze(0)
        uv_expected = torch.tensor([[120.0, -35.0]])
        point_n = (
            distance_mm * normal_n
            + uv_expected[0, 0] * e1_n
            + uv_expected[0, 1] * e2_n
        )
        point_xy_n = point_n[:2].unsqueeze(0).requires_grad_(True)

        uv_actual = virtual_camera_xy_to_table_uv(
            point_xy_n,
            table_frame7_n,
            distance_scale_mm=1000.0,
        )

        torch.testing.assert_close(uv_actual, uv_expected, atol=1e-4, rtol=1e-5)
        uv_actual.sum().backward()
        self.assertIsNotNone(point_xy_n.grad)
        self.assertTrue(torch.isfinite(point_xy_n.grad).all())

    def test_direct_virtual_loss_uses_physical_table_uv(self) -> None:
        normalizer = UVTargetNormalizer(
            mean_mm=torch.tensor([10.0, 20.0]),
            std_mm=torch.tensor([2.0, 4.0]),
        )
        criterion = DirectVirtualTableUVLoss(
            normalizer,
            UVLossConfig(beta_mm=30.0),
            table_distance_scale_mm=1000.0,
        )
        point_xy_n_pred = torch.tensor([[1.5, 2.0]], requires_grad=True)
        batch = {
            "table_frame7_n": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.7]]
            ),
            "uv_gt_table_mm": torch.tensor([[10.0, 20.0]]),
        }

        outputs = criterion(point_xy_n_pred, batch)

        torch.testing.assert_close(
            outputs["point_xy_n_mm"],
            torch.tensor([[13.0, 28.0]]),
        )
        torch.testing.assert_close(
            outputs["uv_pred_table_mm"],
            torch.tensor([[13.0, 28.0]]),
        )
        # mean(0.5 * [3^2, 8^2] / beta=30), in physical millimetres.
        torch.testing.assert_close(outputs["loss"], torch.tensor(73.0 / 120.0))
        outputs["loss"].backward()
        self.assertTrue(torch.isfinite(point_xy_n_pred.grad).all())

    def test_direct_table_loss_needs_no_output_side_geometry(self) -> None:
        normalizer = UVTargetNormalizer(
            mean_mm=torch.tensor([10.0, 20.0]),
            std_mm=torch.tensor([2.0, 4.0]),
        )
        criterion = DirectTableUVLoss(
            normalizer,
            UVLossConfig(beta_mm=30.0),
        )
        uv_table_pred_normalized = torch.tensor(
            [[1.5, 2.0]],
            requires_grad=True,
        )
        # No TableFrame7, plane normal, distance, P_N or z_N is supplied to
        # the loss after the model has emitted its direct table-UV prediction.
        batch = {"uv_gt_table_mm": torch.tensor([[10.0, 20.0]])}

        outputs = criterion(uv_table_pred_normalized, batch)

        torch.testing.assert_close(
            outputs["uv_pred_table_mm"],
            torch.tensor([[13.0, 28.0]]),
        )
        torch.testing.assert_close(outputs["loss"], torch.tensor(73.0 / 120.0))
        outputs["loss"].backward()
        self.assertTrue(torch.isfinite(uv_table_pred_normalized.grad).all())

    def test_vertical_residual_changes_only_physical_table_v(self) -> None:
        normalizer = UVTargetNormalizer(
            mean_mm=torch.tensor([10.0, 20.0]),
            std_mm=torch.tensor([2.0, 4.0]),
        )
        base_prediction = torch.tensor([[1.5, 2.0]], requires_grad=True)
        delta_v_mm = torch.tensor([[5.0]], requires_grad=True)
        outputs = DirectTableUVLoss(normalizer)(
            {
                "uv": base_prediction,
                "vertical_delta_v_mm": delta_v_mm,
            },
            {"uv_gt_table_mm": torch.tensor([[13.0, 33.0]])},
        )

        torch.testing.assert_close(
            outputs["uv_base_table_mm"], torch.tensor([[13.0, 28.0]])
        )
        torch.testing.assert_close(
            outputs["uv_pred_table_mm"], torch.tensor([[13.0, 33.0]])
        )
        torch.testing.assert_close(
            outputs["uv_pred_table_mm"][:, 0],
            outputs["uv_base_table_mm"][:, 0],
        )
        outputs["loss"].backward()
        self.assertTrue(torch.isfinite(base_prediction.grad).all())
        self.assertTrue(torch.isfinite(delta_v_mm.grad).all())

    def test_pitch_to_v_inference_returns_corrected_table_coordinates(self) -> None:
        class ResidualInferenceModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(
                    pitch_to_v_target_frame="table_local"
                )

            def forward(self, batch):
                batch_size = int(batch["left_eye"].shape[0])
                return {
                    "uv": torch.tensor([[1.5, 2.0]]).expand(batch_size, -1),
                    "vertical_delta_v_mm": torch.tensor([[5.0]]).expand(
                        batch_size, -1
                    ),
                }

        normalizer = UVTargetNormalizer(
            mean_mm=torch.tensor([10.0, 20.0]),
            std_mm=torch.tensor([2.0, 4.0]),
        )
        prediction = predict_uv_mm(
            ResidualInferenceModel(),
            {"left_eye": torch.zeros((1, 3, 56, 90))},
            normalizer,
        )

        torch.testing.assert_close(prediction, torch.tensor([[13.0, 33.0]]))

    def test_run_epoch_reports_table_uv_epe_for_direct_virtual_loss(self) -> None:
        class ConstantDirectModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output = nn.Parameter(torch.tensor([[1.5, 2.0]]))

            def forward(self, batch, *, return_features=False):
                return self.output.expand(batch["uv_gt"].shape[0], -1)

        normalizer = UVTargetNormalizer(
            mean_mm=torch.tensor([10.0, 20.0]),
            std_mm=torch.tensor([2.0, 4.0]),
        )
        criterion = DirectVirtualTableUVLoss(normalizer)
        batch = {
            # Deliberately unrelated virtual-camera target: EPE must use the
            # table-local target below after converting the prediction.
            "uv_gt": torch.tensor([[1000.0, 2000.0]]),
            "uv_gt_table_mm": torch.tensor([[10.0, 20.0]]),
            "table_frame7_n": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.7]]
            ),
        }

        metrics, sample_count = run_epoch(
            model=ConstantDirectModel(),
            loader=[batch],
            criterion=criterion,
            device=torch.device("cpu"),
            optimizer=None,
            scaler=torch.cuda.amp.GradScaler(enabled=False),
            amp_enabled=False,
            grad_clip_norm=None,
            gate_regularization_weight=0.0,
        )

        self.assertEqual(sample_count, 1)
        self.assertAlmostEqual(metrics["loss"], 73.0 / 120.0, places=6)
        self.assertAlmostEqual(metrics["epe_mm"], 73.0**0.5, places=6)

    def test_run_epoch_direct_table_path_uses_no_output_geometry(self) -> None:
        class ConstantDirectModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output = nn.Parameter(torch.tensor([[1.5, 2.0]]))

            def forward(self, batch, *, return_features=False):
                return self.output.expand(batch["uv_gt"].shape[0], -1)

        normalizer = UVTargetNormalizer(
            mean_mm=torch.tensor([10.0, 20.0]),
            std_mm=torch.tensor([2.0, 4.0]),
        )
        batch = {
            "uv_gt": torch.tensor([[10.0, 20.0]]),
            "uv_gt_table_mm": torch.tensor([[10.0, 20.0]]),
        }

        metrics, sample_count = run_epoch(
            model=ConstantDirectModel(),
            loader=[batch],
            criterion=DirectTableUVLoss(normalizer),
            device=torch.device("cpu"),
            optimizer=None,
            scaler=torch.cuda.amp.GradScaler(enabled=False),
            amp_enabled=False,
            grad_clip_norm=None,
            gate_regularization_weight=0.0,
        )

        self.assertEqual(sample_count, 1)
        self.assertAlmostEqual(metrics["loss"], 73.0 / 120.0, places=6)
        self.assertAlmostEqual(metrics["epe_mm"], 73.0**0.5, places=6)

    def test_run_epoch_reports_vertical_residual_safety_metrics(self) -> None:
        class ConstantResidualModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output = nn.Parameter(torch.tensor([[1.5, 2.0]]))
                self.delta_v_mm = nn.Parameter(torch.tensor([[5.0]]))
                self.config = SimpleNamespace(
                    use_depth_correction=False,
                    use_eye_iris_auxiliary=False,
                    use_eye_keypoint_auxiliary=False,
                    use_low_dof_eye_template=False,
                    use_pitch_to_v_residual=True,
                    pitch_to_v_max_abs_delta_mm=100.0,
                )

            def forward(self, batch, *, return_features=False):
                batch_size = int(batch["uv_gt"].shape[0])
                return {
                    "uv": self.output.expand(batch_size, -1),
                    "vertical_delta_v_mm": self.delta_v_mm.expand(batch_size, -1),
                    "vertical_geometry_gate": torch.ones((batch_size, 1)),
                    "vertical_camera_table_vertical_sensitivity_mm_per_rad": (
                        torch.full((batch_size, 1), 700.0)
                    ),
                    "vertical_camera_table_geometry_valid_mask": torch.ones(
                        (batch_size, 1)
                    ),
                    "vertical_geometric_pitch_delta_v_mm": torch.full(
                        (batch_size, 1), 70.0
                    ),
                    "vertical_fallback_mask": torch.zeros((batch_size, 1)),
                }

        normalizer = UVTargetNormalizer(
            mean_mm=torch.tensor([10.0, 20.0]),
            std_mm=torch.tensor([2.0, 4.0]),
        )
        batch = {
            "uv_gt": torch.tensor([[13.0, 33.0]]),
            "uv_gt_table_mm": torch.tensor([[13.0, 33.0]]),
        }

        metrics, sample_count = run_epoch(
            model=ConstantResidualModel(),
            loader=[batch],
            criterion=DirectTableUVLoss(normalizer),
            device=torch.device("cpu"),
            optimizer=None,
            scaler=torch.cuda.amp.GradScaler(enabled=False),
            amp_enabled=False,
            grad_clip_norm=None,
            gate_regularization_weight=0.0,
            vertical_residual_regularization_weight=0.01,
        )

        self.assertEqual(sample_count, 1)
        self.assertAlmostEqual(metrics["epe_mm"], 0.0)
        self.assertAlmostEqual(metrics["vertical_base_mae_v_mm"], 5.0)
        self.assertAlmostEqual(metrics["vertical_v_improvement_mm"], 5.0)
        self.assertAlmostEqual(metrics["vertical_delta_abs_mean_mm"], 5.0)
        self.assertAlmostEqual(metrics["vertical_geometry_gate_mean"], 1.0)
        self.assertAlmostEqual(
            metrics["vertical_camera_table_sensitivity_abs_mean_mm_per_rad"],
            700.0,
        )
        self.assertAlmostEqual(
            metrics["vertical_camera_table_geometry_valid_rate"], 1.0
        )
        self.assertAlmostEqual(
            metrics["vertical_geometric_pitch_delta_abs_mean_mm"], 70.0
        )
        self.assertAlmostEqual(metrics["vertical_fallback_rate"], 0.0)
        self.assertAlmostEqual(metrics["vertical_u_invariance_max_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
