"""Stage 2B: build metric ``R_C_H, t_C_H`` using PnP rotation and Iris65.

PnP translation is an internal solver variable only.  Absolute position comes
from the two calibrated iris rays constrained by the configured IPD, followed
by the FLAME-local eye-proxy offset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.depth_prior.deca_flame import DecaFlameConfig, DecaFlameExtractor
from modelv1.depth_prior.iris_ipd import reconstruct_iris_centres_from_ipd
from modelv1.preprocessing import (
    HEAD_ROTATION_PNP_VERSION,
    build_metric_head_pose,
    flame_eye_canthus_midpoints,
    solve_head_rotation_pnp,
)
from modelv1.preprocessing.camera import load_camera_calibration
from modelv1.preprocessing.deca_inputs import (
    calibrate_fixed_shape,
    load_deca_input,
)
from scripts.table7_preprocessing_config import (
    DEFAULT_CONFIG_PATH,
    load_table7_preprocessing_config,
)


FIELDS = (
    "sample_id", "dataset", "image_name", "metric_pose_status", "reason",
    "left_iris_proxy_camera_x_mm", "left_iris_proxy_camera_y_mm",
    "left_iris_proxy_camera_z_mm", "right_iris_proxy_camera_x_mm",
    "right_iris_proxy_camera_y_mm", "right_iris_proxy_camera_z_mm",
    "iris_midpoint_camera_x_mm", "iris_midpoint_camera_y_mm",
    "iris_midpoint_camera_z_mm", "head_origin_camera_x_mm",
    "head_origin_camera_y_mm", "head_origin_camera_z_mm",
    "eye_proxy_midpoint_head_x_mm", "eye_proxy_midpoint_head_y_mm",
    "eye_proxy_midpoint_head_z_mm",
    *tuple(f"head_rotation_c_from_h_{r}{c}" for r in range(3) for c in range(3)),
    "ipd_mm", "flame_scale_from_ipd_mm_per_unit",
    "iris_baseline_residual_mm", "iris_ray_system_condition",
    "rotation_pnp_reprojection_error_mean_px",
    "rotation_pnp_reprojection_error_max_px",
    "rotation_pnp_confidence", "rotation_pnp_solver_method",
    "rotation_pnp_geometry_version", "rotation_pnp_candidate_count",
    "rotation_pnp_min_normalized_object_depth",
    "fixed_shape_calibration_count", "eye15_quality_score",
)


def parse_args() -> argparse.Namespace:
    """解析 Stage 2B 使用的统一 Table7 配置路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取非空 CSV，并以字符串字典列表保留全部样本字段。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _empty(row: dict[str, str], reason: str) -> dict[str, Any]:
    """为几何重建失败样本建立保留主键和失败原因的空 pose 记录。"""

    record = {field: "" for field in FIELDS}
    record.update(
        {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "image_name": row["image_name"],
            "metric_pose_status": "failed",
            "reason": reason,
        }
    )
    return record


def _atomic_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    """原子写入成功与失败 pose 行，避免生成不完整的 Stage 2B CSV。"""

    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Set execution.overwrite: true.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    """原子写入 Stage 2B 的相机、标定、公式和成功率审计报告。"""

    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Set execution.overwrite: true.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    """融合 DECA/FLAME、rotation-only PnP 与 Iris65，生成毫米级 R_C_H/t_C_H。"""

    cli = parse_args()
    config = load_table7_preprocessing_config(cli.config)
    existing = [
        path
        for path in (config.metric_pose_csv, config.metric_pose_report)
        if path.exists()
    ]
    if existing and not config.overwrite and not config.dry_run:
        raise FileExistsError(
            f"Output exists: {existing}. Set execution.overwrite: true."
        )
    for path in (config.base_csv, config.eye_geometry_npz):
        if not path.is_file():
            raise FileNotFoundError(f"Required stage input is missing: {path}")
    all_rows = _read_csv(config.base_csv)
    rows = all_rows
    if config.limit is not None:
        rows = rows[: config.limit]
    if config.dry_run:
        print(
            f"[dry-run] stage 2B would reconstruct {len(rows)} poses and write "
            f"{config.metric_pose_csv}"
        )
        return 0
    try:
        import torch
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Stage 2B requires PyTorch and Pillow.") from error

    with np.load(config.eye_geometry_npz, allow_pickle=False) as archive:  # 加载 Stage 2A 生成的 NPZ 文件
        required = {
            "sample_id", "detection_success_mask", "valid_mask",
            "left_eye_points_px", "right_eye_points_px", "pnp_points_px",
            "pnp_point_valid_mask", "sample_quality_score", "metadata_json",
        }  # 定义 Stage 2B 必须使用的数组
        missing = required.difference(archive.files)  # 计算 NPZ 中缺少哪些必要数组
        if missing:
            raise ValueError(f"Observation NPZ is missing arrays: {sorted(missing)}")
        observations = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(observations["metadata_json"].item()))  # 从 NumPy 标量中取出 JSON 字符串，然后解析成 Python 字典
    pnp_names = tuple(metadata.get("pnp_point_names", ()))  # 读取 PnP 点名称及顺序(8个点)
    if not pnp_names:
        raise ValueError("Observation NPZ lacks pnp_point_names metadata.")
    observation_index = {
        str(sample_id): index
        for index, sample_id in enumerate(observations["sample_id"])
    }  # 把 NPZ 中所有数组复制到普通字典中
    missing_ids = [row["sample_id"] for row in rows if row["sample_id"] not in observation_index]  # 检查本次要处理的 Stage 1 样本是否都能在 Stage 2A NPZ 中找到
    if missing_ids:  # 如果有样本缺失，终止整个 Stage 2B，避免 CSV 与 NPZ 错位
        raise KeyError(f"Observation NPZ lacks {len(missing_ids)} sample IDs.")

    extractor = DecaFlameExtractor(DecaFlameConfig(device=None))  # 创建 DECA/FLAME 提取器
    camera = load_camera_calibration(config.camera_calibration_file)  # 读取真实相机标定文件
    fixed_shape, calibration_rows, missing_calibration = calibrate_fixed_shape(
        all_rows,  # 表示标定样本从完整 CSV 中选择，不受当前 limit 限制
        extractor,
        dataset_name=config.dataset_name,
        calibration_start=config.calibration_start,  # 指定标定范围
        calibration_stop=config.calibration_stop,
        calibration_index_mode=config.calibration_index_mode,  # 指定索引方式
        face_preprocess=config.face_preprocess,  # 配置 DECA 人脸裁剪和预处理方式
        crop_scale=config.deca_crop_scale,  # 配置 DECA 人脸裁剪比例 1.5
        image_root=config.source_root,  # 指定图片根目录
        camera_calibration=camera,  # 配置相机标定参数
        torch=torch,  # 配置 PyTorch
        image_cls=Image,  # 配置 Pillow
    )  # 从一个指定帧区间中提取 DECA shape 参数，并计算中位数，得到固定人物脸型
    camera_matrix = np.asarray(camera.camera_matrix, dtype=np.float64)  # 将相机内参矩阵转换为 float64 NumPy 数组
    dist_coeffs = np.asarray(camera.distortion_coefficients, dtype=np.float64)  # 读取镜头畸变系数
    records: list[dict[str, Any]] = []  # 创建最终 CSV 记录列表
    for row_index, row in enumerate(rows, start=1):  # 遍历所有待处理样本
        try:
            index = observation_index[row["sample_id"]]  # 根据 sample_id 找到当前样本在 Stage 2A NPZ 中的位置
            if not bool(observations["detection_success_mask"][index]):  # 如果 Stage 2A 的 MediaPipe 检测失败，则不能继续进行 PnP 和虹膜重建
                raise ValueError("mediapipe_detection_failed")
            if not bool(observations["valid_mask"][index]):  # 要求左右眼 Eye15 都通过 Stage 2A 的几何质量检查
                raise ValueError("eye15_quality_invalid")
            pnp_valid = np.asarray(observations["pnp_point_valid_mask"][index], dtype=bool)  # 取得当前样本每个 PnP 点的有效性
            pnp_pixels = np.asarray(observations["pnp_points_px"][index], dtype=np.float64)  # 取得当前样本的 PnP 像素坐标(8, 2)
            pnp_points = {
                name: tuple(float(value) for value in pnp_pixels[point_index])
                for point_index, name in enumerate(pnp_names)
                if pnp_valid[point_index]
            }
            _, tensor = load_deca_input(
                row,
                face_preprocess=config.face_preprocess,
                crop_scale=config.deca_crop_scale,
                image_size=extractor.config.image_size,
                image_root=config.source_root,
                camera_calibration=camera,
                torch=torch,
                image_cls=Image,
            )  # 读取当前原始图像并生成人脸模型输入 tensor[1, 3, 224, 224]
            flame = extractor.extract(tensor, fixed_shape_params=fixed_shape)  # 运行 DECA/FLAME;当前帧使用前面标定得到的固定人物脸型fixed_shape
            landmarks = flame.head_local_landmarks3d[0].numpy()  # 取得头部局部坐标系中的 FLAME 三维关键点
            pnp = solve_head_rotation_pnp(
                pnp_points,  # MediaPipe 的二维像素点
                landmarks,  # FLAME 的三维局部点
                camera_matrix,  # 相机内参
                dist_coeffs,  # 镜头畸变
            )  # 估计头部旋转
            rotation_c_h = pnp.rotation_camera_from_head  # 只取 PnP 求出的旋转矩阵
            # 因为 FLAME 68 点中没有真正的虹膜中心点，所以代码使用一只眼两个眼角的中点近似该眼的局部代理中心。
            left_proxy_h, right_proxy_h = flame_eye_canthus_midpoints(landmarks)  # 从 FLAME 68 点中构造左右眼代理位置；都位于头部局部坐标系 H，单位仍然是 FLAME 单位
            left_iris_xy = np.asarray(observations["left_eye_points_px"][index, 10], dtype=np.float64)  # 读取左眼 Eye15 的虹膜中心像素坐标(Stage 2A 定义)
            right_iris_xy = np.asarray(observations["right_eye_points_px"][index, 10], dtype=np.float64)  # 读取右眼 Eye15 的虹膜中心像素坐标(Stage 2A 定义)
            iris = reconstruct_iris_centres_from_ipd(
                left_iris_xy,  # 输入左虹膜二维像素坐标
                right_iris_xy,  # 输入右虹膜二维像素坐标
                head_left_to_right_direction_flame=right_proxy_h - left_proxy_h,  # 使用 FLAME 左眼代理点到右眼代理点的方向，作为头部局部坐标系中的双眼基线方向
                flame_to_camera_rotation=rotation_c_h,  # 使用 PnP 求得的 R_C_H，将双眼基线方向旋转到真实相机坐标系
                camera_matrix=camera_matrix,  # 使用相机内参和畸变参数，把两个虹膜像素点反投影成相机射线
                dist_coeffs=dist_coeffs,
                ipd_mm=config.ipd_mm,
                max_condition=config.max_ray_condition,
                max_baseline_residual_mm=config.max_ipd_residual_mm,
            )  # 使用两个虹膜像素点和已知 IPD，重建左右虹膜在相机坐标系中的毫米三维位置
            pose = build_metric_head_pose(
                iris.midpoint_camera_xyz_mm,
                rotation_c_h,
                left_proxy_h,
                right_proxy_h,
                ipd_mm=config.ipd_mm,
            )  # 根据虹膜中点、头部旋转和 FLAME 眼部代理位置计算 t_C_H。
            left = iris.left_iris_camera_xyz_mm  # 左右虹膜代理点在真实相机坐标系中的毫米坐标
            right = iris.right_iris_camera_xyz_mm
            midpoint = iris.midpoint_camera_xyz_mm  # 左右虹膜三维位置中点，单位毫米
            origin = pose.head_origin_camera_mm  # 头部原点在真实相机坐标系中的毫米位置，也就是 t_C_H
            local_midpoint = pose.eye_proxy_midpoint_head_mm  # 双眼代理中点在头部局部坐标系中的毫米坐标
            record = {field: "" for field in FIELDS}  # 先为所有 CSV 字段填充空字符串，确保每条记录拥有相同字段
            record.update(
                {
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "image_name": row["image_name"],
                    "metric_pose_status": "success",
                    "reason": "",
                    **{
                        f"left_iris_proxy_camera_{axis}_mm": float(left[i])
                        for i, axis in enumerate("xyz")
                    },  # 虹膜坐标
                    **{
                        f"right_iris_proxy_camera_{axis}_mm": float(right[i])
                        for i, axis in enumerate("xyz")
                    },
                    **{
                        f"iris_midpoint_camera_{axis}_mm": float(midpoint[i])
                        for i, axis in enumerate("xyz")
                    },  # 虹膜中点
                    **{
                        f"head_origin_camera_{axis}_mm": float(origin[i])
                        for i, axis in enumerate("xyz")
                    },  # 头部原点
                    **{
                        f"eye_proxy_midpoint_head_{axis}_mm": float(local_midpoint[i])
                        for i, axis in enumerate("xyz")
                    },  # 头部局部眼部代理中点
                    **{
                        f"head_rotation_c_from_h_{r}{c}": float(rotation_c_h[r, c])
                        for r in range(3) for c in range(3)
                    },  # 头部旋转矩阵
                    "ipd_mm": config.ipd_mm,  # 记录本次使用的真实瞳距
                    "flame_scale_from_ipd_mm_per_unit": pose.flame_scale_mm_per_unit,  # 记录 FLAME 单位到毫米的缩放系数
                    "iris_baseline_residual_mm": iris.baseline_residual_mm,  # 记录重建后的左右虹膜基线与目标 IPD 基线之间的残差
                    "iris_ray_system_condition": iris.ray_system_condition,  # 记录双眼射线方程的条件数
                    "rotation_pnp_reprojection_error_mean_px": (
                        pnp.reprojection_error_mean_px
                    ),  # PnP 所有对应点的平均重投影误差，单位像素
                    "rotation_pnp_reprojection_error_max_px": (
                        pnp.reprojection_error_max_px
                    ),  # PnP 所有对应点的最大重投影误差，单位像素
                    "rotation_pnp_confidence": pnp.confidence,  # 基于平均重投影误差和内点比例计算的置信度 [0,1]
                    "rotation_pnp_solver_method": pnp.solver_method,  # 记录最终采用的 PnP 求解方式
                    "rotation_pnp_geometry_version": HEAD_ROTATION_PNP_VERSION,  # 记录 PnP 点定义和归一化策略的版本
                    "rotation_pnp_candidate_count": pnp.candidate_count,  # 记录通过正深度与重投影误差检查的 PnP 候选数量。
                    "rotation_pnp_min_normalized_object_depth": (
                        pnp.min_normalized_object_depth
                    ),  # 记录 PnP 候选中最小的三维点深度
                    "fixed_shape_calibration_count": len(calibration_rows),  # 记录固定人物脸型使用了多少张标定图片
                    "eye15_quality_score": float(observations["sample_quality_score"][index]),  # 保存 Stage 2A 计算的 Eye15 综合质量分数
                }
            )
            records.append(record)
        except Exception as error:
            records.append(_empty(row, f"{type(error).__name__}: {error}"))
        if row_index % config.log_every == 0 or row_index == len(rows):
            print(f"Stage 2B processed {row_index}/{len(rows)}")

    _atomic_csv(config.metric_pose_csv, records, config.overwrite)
    counts = Counter(record["metric_pose_status"] for record in records)
    report = {
        "schema_version": "iris65_metric_head_pose_v2_rotation_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config.config_path),
        "input_base_csv": str(config.base_csv),
        "input_observations_npz": str(config.eye_geometry_npz),
        "camera_calibration_file": str(config.camera_calibration_file),
        "camera_calibration": {
            "name": camera.name,
            "image_size_px": [camera.image_width_px, camera.image_height_px],
            "camera_matrix": camera.camera_matrix,
            "distortion_coefficients": camera.distortion_coefficients,
        },
        "output_csv": str(config.metric_pose_csv),
        "counts": dict(counts),
        "absolute_depth_source": "Iris rays + configured IPD only",
        "pnp_role": (
            "R_C_H only; centred/RMS-normalised FLAME object points; "
            "internal translation is non-metric and discarded"
        ),
        "pnp_geometry_version": HEAD_ROTATION_PNP_VERSION,
        "pnp_candidate_validation": (
            "positive normalised object depth plus reprojection thresholds; "
            "no absolute-depth threshold"
        ),
        "head_origin_formula": "t_C_H = iris_mid_C - R_C_H @ eye_proxy_mid_H_mm",
        "eye_proxy_definition": "midpoint of each FLAME eye canthus pair",
        "fixed_shape_calibration": {
            "start": config.calibration_start,
            "stop": config.calibration_stop,
            "index_mode": config.calibration_index_mode,
            "count": len(calibration_rows),
            "missing_frame_indices": missing_calibration,
        },
    }
    _atomic_json(config.metric_pose_report, report, config.overwrite)
    print(f"Stage 2B wrote: {config.metric_pose_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

