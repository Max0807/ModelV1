"""Stage 3: render virtual-camera images and write the final TableFrame7 CSV."""

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

from modelv1.data.virtual_camera import (
    CameraCalibration,
    VirtualCameraConfig,
    build_virtual_camera_transform,
)
from modelv1.preprocessing.camera import load_camera_calibration
from modelv1.depth_prior.dataset_assets import resolve_image_asset_path
from modelv1.preprocessing import build_table_frame7_n
from scripts.table7_preprocessing_config import (
    DEFAULT_CONFIG_PATH,
    load_table7_preprocessing_config,
)


TABLE_FIELDS = tuple(f"table_frame7_n_{index}" for index in range(7))
OUTPUT_FIELDS = (
    "sample_id", "dataset", "image_name", "source_image_path",
    "virtual_face_path",
    *TABLE_FIELDS,
    "uv_gt_u_mm", "uv_gt_v_mm",
)


def parse_args() -> argparse.Namespace:
    """解析 Stage 3 使用的统一 Table7 配置路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取非空阶段 CSV，并返回按列名访问的字符串行。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _index(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    """按唯一且非空的 sample_id 建立阶段间连接索引。"""

    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id or sample_id in result:
            raise ValueError(f"{label} has empty or duplicate sample_id={sample_id!r}.")
        result[sample_id] = row
    return result


def _vector(
    row: dict[str, str], prefix: str, *, suffix: str = ""
) -> np.ndarray:
    """从 CSV 的 x/y/z 三列恢复一个 float64 三维向量。"""

    return np.asarray(
        [float(row[f"{prefix}_{axis}{suffix}"]) for axis in "xyz"],
        dtype=np.float64,
    )


def _rotation(row: dict[str, str]) -> np.ndarray:
    """从九个扁平字段恢复 head_rotation_c_from_h 的 3×3 矩阵。"""

    return np.asarray(
        [float(row[f"head_rotation_c_from_h_{r}{c}"]) for r in range(3) for c in range(3)],
        dtype=np.float64,
    ).reshape(3, 3)


def _output_path(root: Path, row: dict[str, str]) -> Path:
    """根据数据集和图像 stem 生成虚拟人脸 PNG 路径。"""

    dataset = row["dataset"].strip()
    stem = Path(row["image_name"]).stem
    return root / "images" / "face" / dataset / f"{stem}.png"


def _read_image(path: Path) -> np.ndarray:
    """用支持 Windows Unicode 路径的 fromfile/imdecode 方式读取 BGR 原图。"""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Stage 3 requires OpenCV.") from error
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"image_decode_failed:{path}")
    return image


def _write_png(path: Path, image: np.ndarray, overwrite: bool) -> None:
    """编码并写入 PNG，创建父目录且遵守覆盖开关。"""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Stage 3 requires OpenCV.") from error
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output image exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG encoding failed: {path}")
    encoded.tofile(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    """原子写入最终虚拟图像路径、TableFrame7 和 UV 标签 CSV。"""

    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Set execution.overwrite: true.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    """原子写入 Stage 3 成功/失败统计与最终字段契约报告。"""

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


def main() -> int:  # 定义 Stage 3 主入口并返回进程退出码
    """连接 Stage 1/2，渲染虚拟相机图像并生成 table_frame7_n[7]。"""

    cli = parse_args()  # 解析命令行中的统一配置文件路径
    config = load_table7_preprocessing_config(cli.config)  # 加载并校验 Table7 预处理配置
    existing = [  # 收集已经存在的最终 CSV 和审计报告
        path for path in (config.final_csv, config.final_report) if path.exists()  # 只保留实际存在的输出路径
    ]  # 完成已有输出路径列表
    if existing and not config.overwrite and not config.dry_run:  # 非试运行且禁止覆盖时保护已有输出
        raise FileExistsError(  # 阻止意外覆盖最终结果
            f"Output exists: {existing}. Set execution.overwrite: true."  # 提示开启覆盖配置
        )  # 完成文件已存在异常
    required_inputs = [config.base_csv, config.metric_pose_csv]  # Stage 1 CSV 和 Stage 2B 位姿 CSV 必须存在
    for path in required_inputs:  # 逐个检查本阶段所需输入
        if not path.is_file():  # 输入路径必须是实际文件
            raise FileNotFoundError(f"Required stage input is missing: {path}")  # 缺失输入时立即终止
    base_rows = _read_csv(config.base_csv)  # 读取 Stage 1 的全部基础样本行
    if config.limit is not None:  # Debug 时可限制处理样本数
        base_rows = base_rows[: config.limit]  # 仅保留前 limit 个样本
    pose_by_id = _index(_read_csv(config.metric_pose_csv), "metric pose CSV")  # 按 sample_id 索引 Stage 2B 位姿
    if config.dry_run:  # 试运行只检查输入并报告计划
        print(  # 输出预计处理数量和目标 CSV
            f"[dry-run] stage 3 would normalize {len(base_rows)} samples and write "  # 描述预计归一化样本数
            f"{config.final_csv}"  # 显示最终 CSV 路径
        )  # 完成试运行信息输出
        return 0  # 不渲染图像也不写文件

    camera = load_camera_calibration(config.camera_calibration_file)  # 读取真实相机内参与畸变参数
    calibration = CameraCalibration(  # 转换为虚拟相机模块使用的标定对象
        camera_matrix=np.asarray(camera.camera_matrix, dtype=np.float64),  # 保存 3×3 相机内参矩阵
        distortion_coefficients=np.asarray(  # 转换镜头畸变系数的数据类型
            camera.distortion_coefficients, dtype=np.float64  # 使用 float64 保持几何计算精度
        ),  # 完成畸变系数数组转换
    )  # 完成真实相机标定对象
    virtual_config = VirtualCameraConfig(  # 配置归一化人脸虚拟相机
        output_width=config.virtual_face_width,  # 设置虚拟人脸图宽度
        output_height=config.virtual_face_height,  # 设置虚拟人脸图高度
        focal_length_px=config.virtual_focal_length_px,  # 设置虚拟相机像素焦距
        distance_mm=config.virtual_distance_mm,  # 设置归一化后的标准观察距离
    )  # 完成虚拟相机配置
    successes: list[dict[str, Any]] = []  # 收集可写入训练 CSV 的成功样本
    failures: list[dict[str, str]] = []  # 收集失败样本及失败原因
    for index, base in enumerate(base_rows, start=1):  # 从 1 开始逐个处理 Stage 1 样本
        sample_id = base["sample_id"]  # 读取当前样本唯一标识
        try:  # 隔离单样本异常以继续处理后续样本
            pose = pose_by_id.get(sample_id)  # 查找当前样本的 Stage 2B 位姿记录
            if pose is None:  # 位姿 CSV 中必须存在当前样本
                raise KeyError("metric_pose_missing")  # 标记位姿记录缺失
            if pose.get("metric_pose_status") != "success":  # 只接受 Stage 2B 重建成功的位姿
                raise ValueError(f"metric_pose_{pose.get('metric_pose_status', 'missing')}")  # 保留原位姿状态作为失败原因
            source_path = resolve_image_asset_path(  # 解析当前样本的原始图像路径
                base, "source_image_path", config.source_root  # 使用 Stage 1 字段和数据根目录定位图片
            )  # 完成源图路径解析
            image = _read_image(source_path)  # 以 BGR 数组读取原始图像
            camera.validate_image_size(image.shape[1], image.shape[0])  # 确认图片宽高与相机标定一致
            rotation_c_h = _rotation(pose)  # 从 CSV 恢复头部到相机的 3×3 旋转矩阵
            head_origin_c = _vector(pose, "head_origin_camera", suffix="_mm")  # 恢复相机系中的毫米级头部原点
            transform = build_virtual_camera_transform(  # 根据真实头部位姿建立虚拟相机变换
                calibration,  # 传入真实相机内参与畸变
                rotation_c_h,  # 传入头部到相机的旋转 R_C_H
                head_origin_c,  # 传入头部原点 t_C_H，单位毫米
                virtual_config,  # 传入虚拟相机输出规格
            )  # 得到真实相机到归一化相机的映射
            face_path = _output_path(config.stage3_dir, base)  # 生成虚拟人脸图输出路径
            face_image = transform.render(image, calibration)  # 将原图重映射到归一化虚拟相机
            _write_png(face_path, face_image, config.overwrite)  # 写出归一化人脸 PNG
            table7 = build_table_frame7_n(  # 构造模型使用的七维桌面几何参数
                _vector(base, "table_u_axis_camera"),  # 读取真实相机系中的桌面 U 轴
                _vector(base, "table_v_axis_camera"),  # 读取真实相机系中的桌面 V 轴
                float(base["table_distance_camera_mm"]),  # 读取相机原点到桌面的有符号毫米距离
                transform.rotation_normalized_from_camera,  # 将桌面方向旋转到归一化相机坐标系
            )  # 得到 [U_N(3), V_N(3), distance_m] 共七个值
            record: dict[str, Any] = {  # 建立一个成功训练样本的最终 CSV 行
                "sample_id": sample_id,  # 保存样本唯一标识
                "dataset": base["dataset"],  # 保存数据集名称
                "image_name": base["image_name"],  # 保存原始图像名称
                "source_image_path": str(source_path.resolve()),  # 保存原图绝对路径
                "virtual_face_path": str(face_path.resolve()),  # 保存虚拟人脸图绝对路径
                "uv_gt_u_mm": float(base["uv_gt_u_mm"]),  # 保存桌面真值 U 坐标，单位毫米
                "uv_gt_v_mm": float(base["uv_gt_v_mm"]),  # 保存桌面真值 V 坐标，单位毫米
            }  # 完成基础训练字段
            record.update(  # 将七维桌面参数展开为七个 CSV 字段
                {f"table_frame7_n_{i}": float(value) for i, value in enumerate(table7)}  # 按索引写入 table_frame7_n_0 至 _6
            )  # 完成 table_frame7_n 字段写入
            successes.append(record)  # 收集当前成功样本
        except Exception as error:  # 捕获当前样本任一步骤的异常
            failures.append(  # 保存失败样本而不中断整个数据集
                {  # 建立简洁的失败审计记录
                    "sample_id": sample_id,  # 记录失败样本标识
                    "reason": f"{type(error).__name__}: {error}",  # 记录异常类型与具体信息
                }  # 完成失败记录
            )  # 加入失败样本列表
        if index % config.log_every == 0 or index == len(base_rows):  # 按间隔或在末尾打印进度
            print(f"Stage 3 processed {index}/{len(base_rows)}")  # 显示已处理数和总数

    if not successes:  # 至少需要一个成功样本才能形成训练集
        raise RuntimeError("Stage 3 produced no successful training samples.")  # 全部失败时拒绝写空训练 CSV
    _atomic_csv(config.final_csv, successes, config.overwrite)  # 原子写出最终成功样本 CSV
    failure_types = Counter(item["reason"].split(":", 1)[0] for item in failures)  # 按异常类型统计失败数量
    report = {  # 构造 Stage 3 审计报告
        "schema_version": "modelv1_virtual_table7_training_v1",  # 记录最终训练数据格式版本
        "created_at_utc": datetime.now(timezone.utc).isoformat(),  # 记录 UTC 创建时间
        "config": str(config.config_path),  # 记录本次使用的配置文件
        "output_csv": str(config.final_csv),  # 记录最终训练 CSV 路径
        "camera_calibration_file": str(config.camera_calibration_file),  # 记录相机标定文件路径
        "camera_calibration": {  # 固化运行时实际使用的相机参数
            "name": camera.name,  # 相机标定名称
            "image_size_px": [camera.image_width_px, camera.image_height_px],  # 标定对应的图像宽高
            "camera_matrix": camera.camera_matrix,  # 真实相机 3×3 内参矩阵
            "distortion_coefficients": camera.distortion_coefficients,  # 镜头畸变系数
        },  # 完成相机参数记录
        "counts": {  # 汇总输入、成功和失败数量
            "input": len(base_rows),  # 本次参与处理的样本数
            "success": len(successes),  # 成功进入最终训练 CSV 的样本数
            "failed": len(failures),  # 处理失败的样本数
        },  # 完成数量统计
        "failure_types": dict(failure_types),  # 按异常类型保存失败计数
        "failures": failures,  # 保存每个失败样本的标识与原因
        "table_frame7_n_contract": [  # 描述七维桌面参数的固定排列顺序
            "table_u_axis_normalized_x", "table_u_axis_normalized_y",  # 第 0、1 维为归一化相机系 U 轴 x、y
            "table_u_axis_normalized_z", "table_v_axis_normalized_x",  # 第 2、3 维为 U 轴 z 和 V 轴 x
            "table_v_axis_normalized_y", "table_v_axis_normalized_z",  # 第 4、5 维为 V 轴 y、z
            "signed_camera_to_table_distance_m",  # 第 6 维为有符号相机到桌面距离，单位米
        ],  # 完成 table_frame7_n 字段契约
        "stored_model_metadata": "table_frame7_n[7] only; pose9 is not stored",  # 明确最终只保存七维场景参数
        "virtual_camera_center_source": "Iris65-derived metric head origin",  # 记录虚拟相机中心来自 Iris65 头部原点
        "virtual_camera": {  # 记录虚拟相机配置
            "face_size": [config.virtual_face_width, config.virtual_face_height],  # 虚拟人脸图宽高
            "focal_length_px": config.virtual_focal_length_px,  # 记录虚拟相机像素焦距
            "distance_mm": config.virtual_distance_mm,  # 记录标准化相机观察距离
        },  # 完成虚拟相机配置记录
    }  # 完成 Stage 3 审计报告
    _atomic_json(config.final_report, report, config.overwrite)  # 原子写出最终 JSON 审计报告
    print(f"Stage 3 wrote {len(successes)} samples: {config.final_csv}")  # 打印最终成功数和输出位置
    return 0  # 返回成功退出码


if __name__ == "__main__":
    raise SystemExit(main())
