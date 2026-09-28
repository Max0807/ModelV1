from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from baselines.blazegaze.data_adapter import (
    BlazeGazeDataAdapter,
    MetricHeadPoseTable,
    PogBoundsMM,
    build_blazegaze_manifest,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pose_row(sample_id: str = "person/sample") -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": sample_id,
        "dataset": "person",
        "image_name": "sample.png",
        "metric_pose_status": "success",
        "head_origin_camera_x_mm": 100.0,
        "head_origin_camera_y_mm": -20.0,
        "head_origin_camera_z_mm": 600.0,
        "iris_midpoint_camera_x_mm": 105.0,
        "iris_midpoint_camera_y_mm": -15.0,
        "iris_midpoint_camera_z_mm": 565.0,
    }
    for matrix_row in range(3):
        for matrix_column in range(3):
            row[f"head_rotation_c_from_h_{matrix_row}{matrix_column}"] = float(
                matrix_row == matrix_column
            )
    return row


def _fixture_csvs(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "person"
    source_dir = dataset_root / "insightface_img"
    left_dir = dataset_root / "insightface_eyes" / "left_eye"
    right_dir = dataset_root / "insightface_eyes" / "right_eye"
    source_dir.mkdir(parents=True)
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    Image.new("RGB", (20, 10), (255, 0, 0)).save(left_dir / "sample.png")
    Image.new("RGB", (20, 10), (0, 255, 0)).save(right_dir / "sample.png")

    stage3_path = tmp_path / "stage3.csv"
    pose_path = tmp_path / "pose.csv"
    _write_csv(
        stage3_path,
        [
            {
                "sample_id": "person/sample",
                "dataset": "person",
                "image_name": "sample.png",
                "source_image_path": str(source_dir / "sample.png"),
                "virtual_left_eye_path": "",
                "virtual_right_eye_path": "",
                "uv_gt_u_mm": 0.0,
                "uv_gt_v_mm": 0.0,
            }
        ],
    )
    _write_csv(pose_path, [_pose_row()])
    return stage3_path, pose_path


def test_metric_rt_and_released_inputs_have_correct_units(tmp_path: Path) -> None:
    pose_path = tmp_path / "pose.csv"
    _write_csv(pose_path, [_pose_row()])

    record = MetricHeadPoseTable.load([pose_path]).lookup("person/sample")

    assert record.metric_rt_mm.shape == (3, 4)
    assert record.metric_rt_cm.shape == (3, 4)
    assert torch.allclose(record.metric_rt_mm[:, 3], torch.tensor([100.0, -20.0, 600.0]))
    assert torch.allclose(record.metric_rt_cm[:, 3], torch.tensor([10.0, -2.0, 60.0]))
    assert torch.allclose(record.face_origin_3d, torch.tensor([10.5, -1.5, 56.5]))
    assert not torch.allclose(record.metric_rt_cm[:, 3], record.face_origin_3d)
    assert torch.allclose(record.head_vector, torch.tensor([0.0, 0.0, -1.0]))


def test_head_vector_matches_released_webeyetrack_angle_convention(
    tmp_path: Path,
) -> None:
    pose_path = tmp_path / "pose.csv"
    row = _pose_row()
    rotation = (
        (0.9664366447363417, 0.11779292475702668, 0.22830908564476438),
        (0.07257911292257238, -0.9776751346354441, 0.19718925803131016),
        (0.2464396154804162, -0.17400045400156744, -0.9534103827466497),
    )
    for matrix_row in range(3):
        for matrix_column in range(3):
            row[f"head_rotation_c_from_h_{matrix_row}{matrix_column}"] = rotation[
                matrix_row
            ][matrix_column]
    _write_csv(pose_path, [row])

    head_vector = MetricHeadPoseTable.load([pose_path]).lookup(
        "person/sample"
    ).head_vector

    assert torch.allclose(
        head_vector,
        torch.tensor([0.22830909, 0.19718926, 0.95341038]),
        atol=1e-6,
    )
    assert torch.linalg.vector_norm(head_vector).item() == pytest.approx(1.0)


def test_adapter_yields_native_blazegaze_shapes_and_labels(tmp_path: Path) -> None:
    stage3_path, pose_path = _fixture_csvs(tmp_path)
    adapter = BlazeGazeDataAdapter(
        [stage3_path],
        [pose_path],
        pog_bounds_mm=PogBoundsMM(-100.0, 100.0, -50.0, 50.0),
    )

    sample = adapter[0]

    assert sample["image"].shape == (128, 512, 3)
    assert sample["image"].dtype == torch.float32
    assert torch.allclose(sample["image"][64, 64], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(sample["image"][64, 448], torch.tensor([0.0, 1.0, 0.0]))
    assert sample["head_vector"].shape == (3,)
    assert sample["face_origin_3d"].shape == (3,)
    assert sample["metric_rt_flat_mm"].shape == (12,)
    assert torch.allclose(sample["pog_norm_01"], torch.tensor([0.5, 0.5]))
    assert torch.allclose(sample["pog_norm"], torch.zeros(2))
    assert torch.allclose(sample["screen_info"], torch.tensor([10.0, 20.0]))
    assert sample["eye_strip_source"] == "paired_legacy_eye_crops"


def test_adapter_can_return_chw_without_changing_values(tmp_path: Path) -> None:
    stage3_path, pose_path = _fixture_csvs(tmp_path)
    adapter = BlazeGazeDataAdapter(
        [stage3_path], [pose_path], image_layout="CHW"
    )

    assert adapter[0]["image"].shape == (3, 128, 512)


def test_adapter_prefers_exact_homography_patch(tmp_path: Path) -> None:
    stage3_path, pose_path = _fixture_csvs(tmp_path)
    exact_path = tmp_path / "exact.png"
    Image.new("RGB", (512, 128), (0, 0, 255)).save(exact_path)
    manifest_path = tmp_path / "eye_patch_manifest.csv"
    _write_csv(
        manifest_path,
        [
            {
                "sample_id": "person/sample",
                "status": "success",
                "eye_patch_path": str(exact_path),
                "landmark_count": 478,
            }
        ],
    )
    (tmp_path / "person" / "insightface_eyes" / "left_eye" / "sample.png").unlink()
    (tmp_path / "person" / "insightface_eyes" / "right_eye" / "sample.png").unlink()

    adapter = BlazeGazeDataAdapter(
        [stage3_path],
        [pose_path],
        eye_patch_manifest_paths=[manifest_path],
        require_exact_eye_patches=True,
    )
    sample = adapter[0]

    assert torch.allclose(sample["image"][64, 256], torch.tensor([0.0, 0.0, 1.0]))
    assert sample["eye_strip_source"] == "mediapipe478_homography"
    assert sample["eye_patch_path"] == str(exact_path)
    assert sample["left_eye_path"] == ""


def test_manifest_keeps_pose_auditable(tmp_path: Path) -> None:
    stage3_path, pose_path = _fixture_csvs(tmp_path)
    adapter = BlazeGazeDataAdapter([stage3_path], [pose_path])
    output_path = tmp_path / "manifest.csv"

    report = build_blazegaze_manifest(adapter, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert report["sample_count"] == 1
    assert report["has_native_training_labels"] is False
    assert report["eye_strip_is_exact_official_homography"] is False
    assert float(row["metric_rt_mm_03"]) == pytest.approx(100.0)
    assert float(row["face_origin_3d_x_cm"]) == pytest.approx(10.5)
    assert row["pog_norm_u"] == ""


def test_invalid_rotation_is_rejected(tmp_path: Path) -> None:
    pose_path = tmp_path / "pose.csv"
    row = _pose_row()
    row["head_rotation_c_from_h_00"] = 2.0
    _write_csv(pose_path, [row])

    with pytest.raises(ValueError, match="valid R_C_H rotation"):
        MetricHeadPoseTable.load([pose_path])
