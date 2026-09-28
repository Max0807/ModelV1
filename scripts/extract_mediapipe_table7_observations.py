"""Stage 2A: one MediaPipe pass for eye15, iris centres, and PnP 2D points."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.depth_prior.dataset_assets import resolve_image_asset_path
from modelv1.preprocessing import MEDIAPIPE_ROTATION_LANDMARK_INDICES
from scripts.extract_mediapipe_eye_geometry import (
    GEOMETRY_FEATURE_NAMES,
    LEFT_EYE_MEDIAPIPE_INDICES,
    RIGHT_EYE_MEDIAPIPE_INDICES,
    _empty_arrays,
    build_metadata,
    eye_bbox,
    extract_selected_points,
    fit_eye_geometry,
    read_rows,
    sha256_file,
    write_json_atomic,
    write_npz_atomic,
)
from scripts.table7_preprocessing_config import (
    DEFAULT_CONFIG_PATH,
    load_table7_preprocessing_config,
)


PNP_POINT_NAMES = tuple(MEDIAPIPE_ROTATION_LANDMARK_INDICES)
PNP_INDICES = tuple(
    MEDIAPIPE_ROTATION_LANDMARK_INDICES[name] for name in PNP_POINT_NAMES
)


def parse_args() -> argparse.Namespace:
    """解析 Stage 2A 使用的统一 Table7 配置路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _points(landmarks, indices, width: int, height: int) -> np.ndarray:
    """把指定 MediaPipe 归一化 landmark 转换为原图像素坐标数组。"""

    return np.asarray(
        [[landmarks[i].x * width, landmarks[i].y * height] for i in indices],
        dtype=np.float32,
    )


def main() -> int:
    """逐图运行一次 FaceMesh，生成 Eye15、虹膜中心、PnP 2D 点及质量报告。"""

    cli = parse_args()
    config = load_table7_preprocessing_config(cli.config)
    if not config.base_csv.is_file():
        raise FileNotFoundError(f"Stage 1 CSV is missing: {config.base_csv}")
    existing = [
        path
        for path in (config.eye_geometry_npz, config.eye_geometry_report)
        if path.exists()
    ]
    if existing and not config.overwrite:
        raise FileExistsError(
            f"Output exists: {existing}. Set execution.overwrite: true."
        )
    try:
        import mediapipe as mp
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Stage 2A requires MediaPipe and Pillow.") from error

    args = Namespace(
        dataset_id=config.dataset_numeric_id,  # 数据集的数字编号
        dataset_name=config.dataset_name,  # 数据集名称，例如 table_frame7_n
        csv=config.base_csv,  # Stage 1 生成的基础 CSV
        output=config.eye_geometry_npz,  # Stage 2A 要生成的 NPZ
        report=config.eye_geometry_report,  # Stage 2A 要生成的 JSON 报告
        image_root=config.source_root,  # 原始图像根目录
        min_points=config.eye_fit_min_points,  # 眼部拟合所需的最少有效点数
        max_iris_radius_cv=config.eye_fit_max_iris_radius_cv,  # 虹膜半径变异系数上限
        max_center_offset=config.eye_fit_max_center_offset,  # 虹膜拟合中心偏移上限
        min_quality=config.eye_fit_min_quality,  # 单眼最低质量分数
        limit=config.max_samples,  # 本次最多处理多少个样本
    )  # 手动构造一个类似命令行参数的对象
    rows = read_rows(
        config.base_csv, dataset_name=config.dataset_name, limit=config.limit
    )  # 读取 Stage 1 CSV，并筛选本阶段需要处理的样本
    if config.dry_run:
        print(
            f"[dry-run] stage 2A would detect {len(rows)} images and write "
            f"{config.eye_geometry_npz}"
        )
        return 0
    arrays = _empty_arrays(len(rows))  # 为全部样本预分配 Eye15、质量和有效性数组
    arrays["pnp_points_px"] = np.full(
        (len(rows), len(PNP_POINT_NAMES), 2), np.nan, dtype=np.float32
    )  # [N, 8, 2] 创建 PnP 二维关键点数组
    arrays["pnp_point_valid_mask"] = np.zeros(
        (len(rows), len(PNP_POINT_NAMES)), dtype=np.bool_
    )  # [N, 8] PnP 关键点是否有效
    reasons = [""] * len(rows)  # 记录每个样本的整体处理异常原因
    left_reasons = [""] * len(rows)  # 记录每个样本左眼没有通过几何或质量检查的原因
    right_reasons = [""] * len(rows)  # 记录每个样本右眼没有通过几何或质量检查的原因
    resolved_paths = [""] * len(rows)  # 记录每个样本最终解析得到的绝对图像路径

    # 创建 MediaPipe FaceMesh 实例；FaceMesh 只创建一次，然后供全部图像复用
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,  # 输入是彼此独立的静态图片
        max_num_faces=1,  # 每张图片最多检测一张人脸
        refine_landmarks=True,  # 启用精细关键点(FaceMesh 会返回额外的眼睛和虹膜点，关键点数量通常从 468 增加到 478。)
        min_detection_confidence=config.mediapipe_min_detection_confidence,  # 设置 MediaPipe 人脸检测的最低置信度
    ) as detector:  # 完成 FaceMesh 初始化，并将实例保存到detector
        for index, row in enumerate(rows):  # 遍历 Stage 1 选中的全部样本
            try:
                image_path = resolve_image_asset_path(
                    row, "source_image_path", config.source_root
                )  # 解析当前图片的真实路径
                resolved_paths[index] = str(image_path.resolve())  # 保存解析后的图像路径
                with Image.open(image_path) as image:  # 使用 Pillow 打开图片
                    rgb = np.asarray(image.convert("RGB"))  # 先将图片统一转换为 RGB 三通道，再转换为 NumPy 数组
                height, width = rgb.shape[:2]  # 从 NumPy 数组中读取图片尺寸
                result = detector.process(rgb)  # 将 RGB 图片输入 MediaPipe FaceMesh，执行一次人脸关键点推理
                if not result.multi_face_landmarks:  # 检查是否检测到人脸
                    raise RuntimeError("MediaPipe Face Mesh found no face")
                landmarks = result.multi_face_landmarks[0].landmark  # 取得检测到的第一张人脸的关键点列表
                left = extract_selected_points(
                    landmarks, LEFT_EYE_MEDIAPIPE_INDICES, width=width, height=height
                )  # 从全部 FaceMesh 点中提取“人体解剖学左眼”的 15 个点，并转换成像素坐标，(15, 2)
                right = extract_selected_points(
                    landmarks, RIGHT_EYE_MEDIAPIPE_INDICES, width=width, height=height
                )  # 从全部 FaceMesh 点中提取“人体解剖学右眼”的 15 个点，并转换成像素坐标，(15, 2)
                pnp = _points(landmarks, PNP_INDICES, width, height)  # 从 FaceMesh 点中提取用于 PnP 头部姿态计算的 8 个二维点
                left_fit = fit_eye_geometry(
                    left,  # 形状 [15, 2]，左眼像素坐标
                    bbox=eye_bbox(row, "left"),
                    image_width=width,  # 整张图的尺寸
                    image_height=height,
                    min_canthus_distance_px=config.min_canthus_distance_px,
                    min_crop_inside_ratio=config.min_crop_inside_ratio,
                    min_quality_score=config.min_eye_quality_score,
                )  # 对左眼 Eye15 点进行坐标转换、几何计算和质量检查
                right_fit = fit_eye_geometry(
                    right,
                    bbox=eye_bbox(row, "right"),
                    image_width=width,
                    image_height=height,
                    min_canthus_distance_px=config.min_canthus_distance_px,
                    min_crop_inside_ratio=config.min_crop_inside_ratio,
                    min_quality_score=config.min_eye_quality_score,
                )  # 对右眼 Eye15 点进行坐标转换、几何计算和质量检查
                arrays["image_size_px"][index] = (width, height)
                arrays["detection_success_mask"][index] = True
                arrays["left_eye_points_px"][index] = left
                arrays["right_eye_points_px"][index] = right
                arrays["left_eye_points_crop_norm"][index] = left_fit.crop_normalized  # 相对眼框归一化坐标，[15, 2]
                arrays["right_eye_points_crop_norm"][index] = right_fit.crop_normalized
                arrays["left_eye_points_local"][index] = left_fit.local  # 眼睛局部坐标，[15, 2]
                arrays["right_eye_points_local"][index] = right_fit.local
                arrays["left_eye_point_valid_mask"][index] = left_fit.point_valid_mask  # 15 个点各自是否有效，[15]
                arrays["right_eye_point_valid_mask"][index] = right_fit.point_valid_mask
                arrays["left_eye_geometry"][index] = left_fit.features  # 10 个眼部几何特征，[10]
                arrays["right_eye_geometry"][index] = right_fit.features
                arrays["left_eye_valid_mask"][index] = left_fit.valid  # 左眼整体是否有效
                arrays["right_eye_valid_mask"][index] = right_fit.valid
                arrays["valid_mask"][index] = left_fit.valid and right_fit.valid  # 整个样本是否有效
                arrays["sample_quality_score"][index] = min(
                    float(left_fit.features[-1]), float(right_fit.features[-1])
                )  # 样本质量得分
                arrays["pnp_points_px"][index] = pnp
                arrays["pnp_point_valid_mask"][index] = np.isfinite(pnp).all(axis=1)
                left_reasons[index] = left_fit.invalid_reason  # 描述无效原因
                right_reasons[index] = right_fit.invalid_reason
            except Exception as error:
                reasons[index] = f"{type(error).__name__}: {error}"
            processed = index + 1
            if processed % config.log_every == 0 or processed == len(rows):
                print(f"Stage 2A processed {processed}/{len(rows)}")

    metadata = build_metadata(args, len(rows))  # 调用已有辅助函数，生成通用元数据
    metadata.update(
        {
            "schema_version": "mediapipe_eye_geometry_v1",  # 眼部几何数组的结构版本
            "table7_observation_schema": "mediapipe_table7_observations_v1",  # 
            "pnp_point_names": list(PNP_POINT_NAMES),  # 保存 8 个 PnP 点的名称和排列顺序
            "pnp_mediapipe_indices": list(PNP_INDICES),  # 保存每个 PnP 点对应的 MediaPipe landmark 索引
            "iris_centres": "eye point index 10 in anatomical left/right arrays",
        }
    )
    payload = {
        "sample_id": np.asarray([row["sample_id"] for row in rows]),
        "dataset": np.asarray([row["dataset"] for row in rows]),
        "image_name": np.asarray([row["image_name"] for row in rows]),
        "source_image_path": np.asarray(resolved_paths),
        "detection_reason": np.asarray(reasons), # 保存图像读取、FaceMesh 检测或处理异常原因
        "left_eye_invalid_reason": np.asarray(left_reasons),  # 保存左眼质量检查失败原因
        "right_eye_invalid_reason": np.asarray(right_reasons),  # 保存右眼质量检查失败原因
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),  # 将元数据字典转换成 JSON 字符串后保存进 NPZ
        **arrays,  # 将前面构造的所有数值数组和掩码加入 payload
    }
    write_npz_atomic(config.eye_geometry_npz, payload)  # 将 payload 写入 Stage 2A NPZ 文件
    report = {
        **metadata,
        "output_npz": str(config.eye_geometry_npz),  # 记录 NPZ 输出路径
        "output_npz_sha256": sha256_file(config.eye_geometry_npz),  # 计算 NPZ 文件的 SHA-256 摘要
        "counts": {
            "total": len(rows),
            "detection_success": int(arrays["detection_success_mask"].sum()),
            "both_eyes_valid": int(arrays["valid_mask"].sum()),
        },
        "failure_types": dict(
            Counter(reason.split(":", 1)[0] for reason in reasons if reason)
        ),
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
    }
    write_json_atomic(config.eye_geometry_report, report)  # 原子方式写出质量报告 JSON
    print(f"Stage 2A wrote: {config.eye_geometry_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
