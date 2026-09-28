"""Validate Stage 2/3 table7 preprocessing math invariants.

This script is intentionally read-only for generated preprocessing artifacts.
It checks the final table7 pipeline against the new definition:

* Stage 2: PnP provides only scale-free head rotation; Iris65/IPD provides
  metric iris points and the metric head translation.
* Stage 3: virtual-camera normalization is rebuilt from R_C_H + t_head_C, then
  table_frame7_n is recomputed from real-camera table axes and depth.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_csv(path: Path) -> list[dict[str, str]]:
    """读取一个预处理 CSV，并保留原始字符串字段用于独立重算。"""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    """读取阶段报告 JSON 并返回字典。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _load_module(name: str, path: Path) -> Any:
    """从指定源码路径动态加载模块，避免依赖包级初始化副作用。"""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_camera_calibration(repo_root: Path, calibration_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """加载相机 K/畸变；缺少 PyYAML 时使用最小解析逻辑作为只读校验后备。"""

    camera_module = _load_module(
        "table7_camera_calibration",
        repo_root / "modelv1" / "preprocessing" / "camera.py",
    )
    try:
        calibration = camera_module.load_camera_calibration(calibration_path)
        return (
            np.asarray(calibration.camera_matrix, dtype=np.float64),
            np.asarray(calibration.distortion_coefficients, dtype=np.float64),
        )
    except RuntimeError as error:
        if "PyYAML" not in str(error):
            raise

    text = calibration_path.read_text(encoding="utf-8")
    rows: list[list[float]] = []
    distortion: list[float] = []
    mode: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("camera_matrix:"):
            mode = "camera_matrix"
            if "[" in line:
                row_text = line.split("[", 1)[1].split("]", 1)[0]
                rows.append([float(item.strip()) for item in row_text.split(",")])
            continue
        if line.startswith("distortion_coefficients:"):
            mode = "distortion_coefficients"
            if "[" in line:
                coeff_text = line.split("[", 1)[1].split("]", 1)[0]
                distortion.extend(float(item.strip()) for item in coeff_text.split(","))
            continue
        if line and not line.startswith("-") and not raw_line.startswith(" "):
            mode = None
        if mode == "camera_matrix" and line.startswith("- ["):
            row_text = line.split("[", 1)[1].split("]", 1)[0]
            rows.append([float(item.strip()) for item in row_text.split(",")])
            continue
        if mode == "distortion_coefficients" and line.startswith("-"):
            value_text = line[1:].strip()
            if value_text.startswith("["):
                coeff_text = value_text.split("[", 1)[1].split("]", 1)[0]
                distortion.extend(float(item.strip()) for item in coeff_text.split(","))
            else:
                distortion.append(float(value_text))
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"Cannot parse camera_matrix from {calibration_path}")
    if len(distortion) not in {4, 5, 8, 12, 14}:
        raise ValueError(f"Cannot parse distortion_coefficients from {calibration_path}")
    return np.asarray(rows, dtype=np.float64), np.asarray(distortion, dtype=np.float64)


def _float(row: dict[str, str], key: str) -> float:
    """把校验行中的指定字段转换为浮点数。"""

    return float(row[key])


def _vec_mm(row: dict[str, str], prefix: str) -> np.ndarray:
    """从带 _mm 后缀的 x/y/z 字段恢复毫米三维向量。"""

    return np.array(
        [
            _float(row, f"{prefix}_x_mm"),
            _float(row, f"{prefix}_y_mm"),
            _float(row, f"{prefix}_z_mm"),
        ],
        dtype=np.float64,
    )


def _vec(row: dict[str, str], prefix: str) -> np.ndarray:
    """从普通 x/y/z 字段恢复 float64 三维向量。"""

    return np.array(
        [
            _float(row, f"{prefix}_x"),
            _float(row, f"{prefix}_y"),
            _float(row, f"{prefix}_z"),
        ],
        dtype=np.float64,
    )


def _rotation(row: dict[str, str], prefix: str) -> np.ndarray:
    """从九个带前缀的 CSV 字段恢复 3×3 旋转矩阵。"""

    return np.array(
        [[_float(row, f"{prefix}_{i}{j}") for j in range(3)] for i in range(3)],
        dtype=np.float64,
    )


def _table7(row: dict[str, str]) -> np.ndarray:
    """从最终 CSV 恢复 table_frame7_n_0..6 七维向量。"""

    return np.array([_float(row, f"table_frame7_n_{idx}") for idx in range(7)], dtype=np.float64)


def _image_is_160(path: Path) -> bool:
    """检查虚拟人脸图是否可读取且为 160×160；无 Pillow 时跳过尺寸读取。"""

    try:
        from PIL import Image  # type: ignore
    except Exception:
        return True
    try:
        with Image.open(path) as image:
            return image.size == (160, 160)
    except Exception:
        return False


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """重算 Stage 2/3 几何不变量，并汇总逐项通过状态和失败样例。"""

    repo_root = Path(args.repo_root).resolve()
    dataset_id = str(args.dataset_id)
    root = repo_root / "data" / "preprocessed_table7" / f"dataset{dataset_id}"

    base_path = root / "stage1_base" / f"base_dataset{dataset_id}.csv"
    pose_path = root / "stage2_metric_head_pose" / f"iris65_metric_head_pose_dataset{dataset_id}.csv"
    pose_report_path = (
        root / "stage2_metric_head_pose" / f"iris65_metric_head_pose_dataset{dataset_id}_report.json"
    )
    final_path = root / "stage3_virtual_table7" / f"training_table7_dataset{dataset_id}.csv"
    final_report_path = (
        root / "stage3_virtual_table7" / f"training_table7_dataset{dataset_id}_report.json"
    )

    virtual_camera = _load_module("table7_virtual_camera", repo_root / "modelv1" / "data" / "virtual_camera.py")
    table7_geometry = _load_module(
        "table7_geometry", repo_root / "modelv1" / "preprocessing" / "table7_geometry.py"
    )

    base_rows = _load_csv(base_path)
    pose_rows = _load_csv(pose_path)
    final_rows = _load_csv(final_path)
    pose_report = _load_json(pose_report_path)
    final_report = _load_json(final_report_path)

    base_by_id = {row["sample_id"]: row for row in base_rows}
    pose_by_id = {row["sample_id"]: row for row in pose_rows}
    stage2_success = [row for row in pose_rows if row["metric_pose_status"] == "success"]
    stage2_failed = [row for row in pose_rows if row["metric_pose_status"] != "success"]

    camera_matrix, distortion = _load_camera_calibration(repo_root, Path(args.camera_calibration_file))
    camera = virtual_camera.CameraCalibration(camera_matrix, distortion)
    virtual_config = virtual_camera.VirtualCameraConfig(
        output_width=args.virtual_width,
        output_height=args.virtual_height,
        focal_length_px=args.virtual_focal_px,
        distance_mm=args.virtual_distance_mm,
    )

    checks: list[dict[str, Any]] = []
    failed_checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        """记录一项校验结果，并把失败项同步加入最终失败列表。"""

        record = {"name": name, "passed": bool(passed), "detail": detail}
        checks.append(record)
        if not passed:
            failed_checks.append(record)

    add_check(
        "stage2 row count equals stage1",
        len(pose_rows) == len(base_rows),
        f"{len(pose_rows)} vs {len(base_rows)}",
    )
    add_check(
        "stage2 report success count matches CSV",
        pose_report.get("counts", {}).get("success") == len(stage2_success),
        str(pose_report.get("counts")),
    )
    add_check(
        "stage2 report failed count matches CSV",
        pose_report.get("counts", {}).get("failed") == len(stage2_failed),
        str(pose_report.get("counts")),
    )

    stage2_stats = {
        "rows_checked": len(stage2_success),
        "rotation_valid": 0,
        "positive_depth": 0,
        "iris_midpoint_identity": 0,
        "iris_ipd_residual": 0,
        "head_origin_formula": 0,
        "pnp_quality_gate": 0,
        "confidence_range": 0,
    }
    stage2_failure_examples: list[dict[str, Any]] = []

    for row in stage2_success:
        sample_id = row["sample_id"]
        row_errors: list[str] = []
        try:
            rotation = _rotation(row, "head_rotation_c_from_h")
            left = _vec_mm(row, "left_iris_proxy_camera")
            right = _vec_mm(row, "right_iris_proxy_camera")
            midpoint = _vec_mm(row, "iris_midpoint_camera")
            head_origin = _vec_mm(row, "head_origin_camera")
            eye_proxy_mid = _vec_mm(row, "eye_proxy_midpoint_head")

            finite_all = all(
                np.isfinite(value).all()
                for value in (rotation, left, right, midpoint, head_origin, eye_proxy_mid)
            )
            rotation_orth_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
            rotation_det = float(np.linalg.det(rotation))
            if finite_all and rotation_orth_error <= args.rotation_tolerance and abs(rotation_det - 1.0) <= args.rotation_tolerance:
                stage2_stats["rotation_valid"] += 1
            else:
                row_errors.append(
                    f"invalid R_C_H: orth={rotation_orth_error:.3g}, det={rotation_det:.8g}"
                )

            if min(float(left[2]), float(right[2]), float(midpoint[2]), float(head_origin[2])) > 0.0:
                stage2_stats["positive_depth"] += 1
            else:
                row_errors.append("left/right/midpoint/head-origin z must be positive")

            if float(np.max(np.abs(midpoint - 0.5 * (left + right)))) <= args.mm_tolerance:
                stage2_stats["iris_midpoint_identity"] += 1
            else:
                row_errors.append("iris_midpoint_C != 0.5 * (left_iris_C + right_iris_C)")

            baseline_length = float(np.linalg.norm(right - left))
            length_error = abs(baseline_length - args.ipd_mm)
            residual_csv = _float(row, "iris_baseline_residual_mm")
            condition = _float(row, "iris_ray_system_condition")
            if (
                residual_csv <= args.max_ipd_residual_mm + args.mm_tolerance
                and condition <= args.max_ray_condition + args.mm_tolerance
                and length_error <= args.max_ipd_residual_mm + args.mm_tolerance
            ):
                stage2_stats["iris_ipd_residual"] += 1
            else:
                row_errors.append(
                    f"bad Iris65 gate: residual={residual_csv:.6g}, length_error={length_error:.6g}, cond={condition:.6g}"
                )

            formula_error = float(np.max(np.abs((head_origin + rotation @ eye_proxy_mid) - midpoint)))
            if formula_error <= args.mm_tolerance:
                stage2_stats["head_origin_formula"] += 1
            else:
                row_errors.append(f"head_origin_C + R_C_H @ eye_mid_H != iris_mid_C; err={formula_error:.6g}mm")

            reproj_mean = _float(row, "rotation_pnp_reprojection_error_mean_px")
            reproj_max = _float(row, "rotation_pnp_reprojection_error_max_px")
            candidate_count = int(float(row["rotation_pnp_candidate_count"]))
            min_depth = _float(row, "rotation_pnp_min_normalized_object_depth")
            if reproj_mean <= 8.0 and reproj_max <= 20.0 and candidate_count > 0 and min_depth > 0.0:
                stage2_stats["pnp_quality_gate"] += 1
            else:
                row_errors.append(
                    f"bad PnP gate: mean={reproj_mean}, max={reproj_max}, candidates={candidate_count}, min_depth={min_depth}"
                )

            confidence = _float(row, "rotation_pnp_confidence")
            if 0.0 <= confidence <= 1.0:
                stage2_stats["confidence_range"] += 1
            else:
                row_errors.append(f"rotation confidence out of range: {confidence}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            row_errors.append(f"{type(exc).__name__}: {exc}")
        if row_errors and len(stage2_failure_examples) < args.max_examples:
            stage2_failure_examples.append({"sample_id": sample_id, "errors": row_errors})

    for key, count in stage2_stats.items():
        if key != "rows_checked":
            add_check(
                f"stage2 {key}",
                count == stage2_stats["rows_checked"],
                f"{count}/{stage2_stats['rows_checked']}; examples={stage2_failure_examples[:2]}",
            )

    stage3_ids = {row["sample_id"] for row in final_rows}
    stage2_success_ids = {row["sample_id"] for row in stage2_success}
    add_check(
        "stage3 rows equal stage2 successful rows",
        len(final_rows) == len(stage2_success),
        f"{len(final_rows)} vs {len(stage2_success)}",
    )
    add_check(
        "stage3 sample ids equal stage2 successful sample ids",
        stage3_ids == stage2_success_ids,
        f"missing={len(stage2_success_ids - stage3_ids)}, extra={len(stage3_ids - stage2_success_ids)}",
    )
    add_check(
        "stage3 report success count matches CSV",
        final_report.get("counts", {}).get("success") == len(final_rows),
        str(final_report.get("counts")),
    )
    add_check(
        "stage3 report failed count matches stage2 failed count",
        final_report.get("counts", {}).get("failed") == len(stage2_failed),
        str(final_report.get("counts")),
    )

    stage3_stats = {
        "rows_checked": len(final_rows),
        "paths_valid": 0,
        "virtual_rotation_valid": 0,
        "virtual_z_axis_aligned_to_head_origin": 0,
        "table_axes_unit_orthogonal": 0,
        "table_distance": 0,
        "table7_recomputed": 0,
    }
    stage3_failure_examples: list[dict[str, Any]] = []

    for row in final_rows:
        sample_id = row["sample_id"]
        row_errors = []
        try:
            if sample_id not in base_by_id or sample_id not in pose_by_id:
                raise RuntimeError("sample_id missing from linked Stage1/Stage2 CSV")
            base = base_by_id[sample_id]
            pose = pose_by_id[sample_id]

            source_image = repo_root / row["source_image_path"]
            virtual_face = repo_root / row["virtual_face_path"]
            if source_image.exists() and virtual_face.exists() and _image_is_160(virtual_face):
                stage3_stats["paths_valid"] += 1
            else:
                row_errors.append("source image or 160x160 virtual face image is missing/invalid")

            rotation_c_h = _rotation(pose, "head_rotation_c_from_h")
            head_origin = _vec_mm(pose, "head_origin_camera")
            transform = virtual_camera.build_virtual_camera_transform(
                camera,
                rotation_c_h,
                head_origin,
                virtual_config,
            )
            rotation_n_c = transform.rotation_normalized_from_camera
            rotation_orth_error = float(np.max(np.abs(rotation_n_c @ rotation_n_c.T - np.eye(3))))
            rotation_det = float(np.linalg.det(rotation_n_c))
            if (
                np.isfinite(rotation_n_c).all()
                and rotation_orth_error <= args.strict_rotation_tolerance
                and abs(rotation_det - 1.0) <= args.strict_rotation_tolerance
            ):
                stage3_stats["virtual_rotation_valid"] += 1
            else:
                row_errors.append(f"invalid R_N_C: orth={rotation_orth_error:.3g}, det={rotation_det:.8g}")

            center_direction = head_origin / np.linalg.norm(head_origin)
            z_axis = rotation_n_c[2]
            if abs(float(np.dot(z_axis, center_direction)) - 1.0) <= args.strict_rotation_tolerance:
                stage3_stats["virtual_z_axis_aligned_to_head_origin"] += 1
            else:
                row_errors.append("virtual camera z-axis is not aligned to head origin direction")

            actual_table7 = _table7(row)
            u_n = actual_table7[:3]
            v_n = actual_table7[3:6]
            if (
                abs(float(np.linalg.norm(u_n)) - 1.0) <= args.vector_tolerance
                and abs(float(np.linalg.norm(v_n)) - 1.0) <= args.vector_tolerance
                and abs(float(np.dot(u_n, v_n))) <= args.vector_tolerance
            ):
                stage3_stats["table_axes_unit_orthogonal"] += 1
            else:
                row_errors.append("table axes in table_frame7_n are not unit/orthogonal")

            expected_distance_m = _float(base, "table_distance_camera_mm") / 1000.0
            if abs(float(actual_table7[6]) - expected_distance_m) <= args.table_distance_tolerance_m:
                stage3_stats["table_distance"] += 1
            else:
                row_errors.append(
                    f"table_frame7_n[6] mismatch: actual={actual_table7[6]}, expected={expected_distance_m}"
                )

            recomputed_table7 = table7_geometry.build_table_frame7_n(
                _vec(base, "table_u_axis_camera"),
                _vec(base, "table_v_axis_camera"),
                _float(base, "table_distance_camera_mm"),
                rotation_n_c,
            ).astype(np.float64)
            if float(np.max(np.abs(actual_table7 - recomputed_table7))) <= args.table7_tolerance:
                stage3_stats["table7_recomputed"] += 1
            else:
                row_errors.append(
                    "table_frame7_n cannot be exactly recomputed from Stage1 table axes/depth and Stage2 head pose"
                )
        except Exception as exc:  # pragma: no cover - diagnostic path
            if not row_errors:
                row_errors.append(f"{type(exc).__name__}: {exc}")
        if row_errors and len(stage3_failure_examples) < args.max_examples:
            stage3_failure_examples.append({"sample_id": sample_id, "errors": row_errors})

    for key, count in stage3_stats.items():
        if key != "rows_checked":
            add_check(
                f"stage3 {key}",
                count == stage3_stats["rows_checked"],
                f"{count}/{stage3_stats['rows_checked']}; examples={stage3_failure_examples[:2]}",
            )

    return {
        "dataset": dataset_id,
        "artifact_root": str(root),
        "rows": {
            "stage1_base": len(base_rows),
            "stage2_pose": len(pose_rows),
            "stage2_success": len(stage2_success),
            "stage2_failed": len(stage2_failed),
            "stage3_final": len(final_rows),
        },
        "stage2_stats": stage2_stats,
        "stage2_failure_examples": stage2_failure_examples,
        "stage3_stats": stage3_stats,
        "stage3_failure_examples": stage3_failure_examples,
        "pose_report_counts": pose_report.get("counts"),
        "final_report_counts": final_report.get("counts"),
        "checks_total": len(checks),
        "checks_failed": len(failed_checks),
        "failed_checks": failed_checks,
    }


def main() -> None:
    """解析校验阈值，打印 JSON 摘要，并在任一检查失败时返回非零状态。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="11")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--camera-calibration-file", default="configs/camera/crossgaze_camera.yaml")
    parser.add_argument("--ipd-mm", type=float, default=65.0)
    parser.add_argument("--max-ipd-residual-mm", type=float, default=4.0)
    parser.add_argument("--max-ray-condition", type=float, default=250.0)
    parser.add_argument("--virtual-width", type=int, default=160)
    parser.add_argument("--virtual-height", type=int, default=160)
    parser.add_argument("--virtual-focal-px", type=float, default=480.0)
    parser.add_argument("--virtual-distance-mm", type=float, default=600.0)
    parser.add_argument("--rotation-tolerance", type=float, default=1e-5)
    parser.add_argument("--strict-rotation-tolerance", type=float, default=1e-8)
    parser.add_argument("--mm-tolerance", type=float, default=1e-5)
    parser.add_argument("--vector-tolerance", type=float, default=1e-5)
    parser.add_argument("--table-distance-tolerance-m", type=float, default=1e-7)
    parser.add_argument("--table7-tolerance", type=float, default=5e-7)
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    summary = validate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["checks_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
