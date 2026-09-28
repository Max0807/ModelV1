"""Stage 1: build the minimal raw/camera/table CSV for TableFrame7.

This implementation reads only the collection log and InsightFace detection
table.  It does not discover, read, or merge any historical PnP/depth file and
does not construct the retired scene25/crop36 vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.table7_preprocessing_config import (
    DEFAULT_CONFIG_PATH,
    Table7PreprocessingConfig,
    load_table7_preprocessing_config,
)


IMAGE_WIDTH = 1920.0
IMAGE_HEIGHT = 1080.0
HAND_EYE_RIGID_TO_OPTICAL = (
    (0.00146143, 0.50358286, 0.86394569, 0.06705835),
    (-0.99981876, -0.01566358, 0.01082137, 0.03084992),
    (0.01898194, -0.86380493, 0.50346870, -0.03494481),
    (0.0, 0.0, 0.0, 1.0),
)
TABLE_U_WORLD = (1.0, 0.0, 0.0)
TABLE_V_WORLD = (0.0, 1.0, 0.0)
TABLE_NORMAL_WORLD = (0.0, 0.0, 1.0)

IDENTITY_FIELDS = (
    "sample_id", "dataset", "image_name", "frame_idx", "source_dataset_dir",
    "source_image_path", "face_path",
)
BBOX_FIELDS = tuple(
    f"{name}_{axis}"
    for name in ("face_bbox", "left_eye_bbox", "right_eye_bbox")
    for axis in ("x", "y", "w", "h")
)
LABEL_FIELDS = (
    "z_table_mm", "uv_gt_u_mm", "uv_gt_v_mm",
    "gaze_target_w_x_mm", "gaze_target_w_y_mm", "gaze_target_w_z_mm",
    "gaze_cam_recomputed_x_mm", "gaze_cam_recomputed_y_mm",
    "gaze_cam_recomputed_z_mm",
)
TRANSLATION_FIELDS = (
    "t_wc_x_mm", "t_wc_y_mm", "t_wc_z_mm",
    "t_cw_x_mm", "t_cw_y_mm", "t_cw_z_mm",
)
ROTATION_C_W_FIELDS = tuple(
    f"camera_rotation_c_from_w_{row}{column}"
    for row in range(3) for column in range(3)
)
ROTATION_W_C_FIELDS = tuple(
    f"camera_rotation_w_from_c_{row}{column}"
    for row in range(3) for column in range(3)
)
TABLE_FIELDS = (
    "table_u_axis_camera_x", "table_u_axis_camera_y", "table_u_axis_camera_z",
    "table_v_axis_camera_x", "table_v_axis_camera_y", "table_v_axis_camera_z",
    "table_distance_camera_mm",
)
OUTPUT_FIELDS = (
    IDENTITY_FIELDS + BBOX_FIELDS + LABEL_FIELDS + TRANSLATION_FIELDS
    + ROTATION_C_W_FIELDS + ROTATION_W_C_FIELDS + TABLE_FIELDS
)


def parse_args() -> argparse.Namespace:
    """解析 Stage 1 使用的统一 Table7 配置路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取非空 CSV，并将每一行返回为按列名索引的字符串字典。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _index_unique(
    rows: list[dict[str, str]], key: str, *, label: str
) -> dict[str, dict[str, str]]:
    """按指定字段建立唯一索引，忽略空键并拒绝重复键。"""

    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "").strip()
        if not value:
            continue
        if value in result:
            raise ValueError(f"Duplicate {key}={value!r} in {label}.")
        result[value] = row
    return result


def _float(row: Mapping[str, str], key: str) -> float:
    """从 CSV 行读取有限浮点数，并为缺失或非法值提供字段级错误。"""

    raw = row.get(key, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid numeric field {key}={raw!r}.") from error
    if not math.isfinite(value):
        raise ValueError(f"Field {key} must be finite.")
    return value


def _status_success(row: Mapping[str, str]) -> bool:
    """判断检测行的 status 是否为 success；缺失 status 时按成功处理。"""

    return row.get("status", "success").strip().lower() == "success"


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """执行小型二维矩阵乘法，用于组合 Vicon 与 hand-eye 旋转。"""

    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def _matvec(a: list[list[float]], vector: list[float]) -> list[float]:
    """将三维向量左乘矩阵，用于旋转或刚体变换。"""

    return [sum(row[k] * vector[k] for k in range(3)) for row in a]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    """转置小型矩阵；正交旋转矩阵借此计算逆旋转。"""

    return [list(row) for row in zip(*matrix)]


def _add(left: list[float], right: list[float]) -> list[float]:
    """逐元素相加两个等长向量。"""

    return [a + b for a, b in zip(left, right)]


def _subtract(left: list[float], right: list[float]) -> list[float]:
    """逐元素计算 left-right，用于构造相对位置向量。"""

    return [a - b for a, b in zip(left, right)]


def _dot(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
    """计算两个等长向量的点积，用于轴投影和平面距离。"""

    return sum(a * b for a, b in zip(left, right))


def _camera_pose(
    row: Mapping[str, str], translation_scale: float
) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
    """组合 Vicon 刚体位姿和 hand-eye 外参，返回 W↔C 的旋转与毫米平移。"""

    rotation_world_from_rigid = [
        [_float(row, f"cam_r{r}{c}") for c in range(1, 4)]
        for r in range(1, 4)
    ]
    translation_world_from_rigid = [
        _float(row, key) for key in ("cam_tx", "cam_ty", "cam_tz")
    ]
    rotation_rigid_from_camera = [
        list(source[:3]) for source in HAND_EYE_RIGID_TO_OPTICAL[:3]
    ]
    translation_rigid_from_camera = [
        source[3] * translation_scale for source in HAND_EYE_RIGID_TO_OPTICAL[:3]
    ]
    rotation_world_from_camera = _matmul(
        rotation_world_from_rigid, rotation_rigid_from_camera
    )
    translation_world_from_camera = _add(
        _matvec(rotation_world_from_rigid, translation_rigid_from_camera),
        translation_world_from_rigid,
    )
    rotation_camera_from_world = _transpose(rotation_world_from_camera)
    translation_camera_from_world = [
        -value
        for value in _matvec(
            rotation_camera_from_world, translation_world_from_camera
        )
    ]
    return (
        rotation_world_from_camera,
        translation_world_from_camera,
        rotation_camera_from_world,
        translation_camera_from_world,
    )


def _bbox(
    row: Mapping[str, str], prefix: str
) -> tuple[float, float, float, float]:
    """读取 InsightFace xywh 框，将其裁到 1920×1080 内并保证宽高为正。"""

    x = _float(row, f"{prefix}_x")
    y = _float(row, f"{prefix}_y")
    width = _float(row, f"{prefix}_width")
    height = _float(row, f"{prefix}_height")

    x = max(0.0, min(IMAGE_WIDTH - 1.0, x))  # 确保框在图像内
    y = max(0.0, min(IMAGE_HEIGHT - 1.0, y))  # 确保框在图像内
    x2 = max(x + 1.0, min(IMAGE_WIDTH, x + width))  # 右下角坐标
    y2 = max(y + 1.0, min(IMAGE_HEIGHT, y + height))  # 右下角坐标
    return x, y, x2 - x, y2 - y


def _format(value: Any) -> str:
    """把输出字段稳定格式化为 CSV 字符串，并拒绝非有限浮点数。"""

    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Refusing to write non-finite value: {value}")
        return f"{value:.10g}"
    return str(value)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    """先写临时 CSV,写入成功后再原子替换目标，避免中断时留下半成品。"""

    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Set execution.overwrite: true.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _format(row.get(field)) for field in OUTPUT_FIELDS})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    """以 UTF-8 写入临时 JSON 后原子替换目标，并遵守覆盖开关。"""

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


def _build_rows(
    config: Table7PreprocessingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """连接采集日志和 InsightFace 表，计算相机/桌面几何、UV 标签及跳过统计。
    主要完成:
        定位 data_log_*.csv。
        读取 insightface_coordinates.csv。
        按 image_name 连接两个 CSV。
        读取注视目标世界坐标。
        读取 face、left-eye、right-eye bbox。
        组合 Vicon 相机刚体位姿和 hand-eye 外参。
        计算：
            R_W_C、t_W_C
            R_C_W、t_C_W
            table_u_axis_camera
            table_v_axis_camera
            table_distance_camera_mm
            uv_gt_u_mm
            uv_gt_v_mm
        检查原图路径。
        统计被跳过样本的原因。
    """

    dataset_dir = config.source_root / config.dataset_name
    logs = sorted(dataset_dir.glob("data_log_*.csv"))
    insight_path = dataset_dir / config.insightface_coordinates_file
    if not insight_path.is_file():  # 检查 InsightFace CSV 存在
        raise FileNotFoundError(f"InsightFace CSV is missing: {insight_path}")
    data_rows = _read_csv(logs[0])  # 读取唯一的采集csv
    insight_rows = _read_csv(insight_path)  # 读取 InsightFace 检测结果
    insight_by_image = _index_unique(
        insight_rows, "image_name", label=str(insight_path)
    )  # 将 InsightFace 行转换为以图像名为键的字典
    target_z_values = []  # 创建列表，准备收集所有有效的 gaze_target_tz
    for row in data_rows:
        try:
            target_z_values.append(_float(row, "gaze_target_tz"))
        except ValueError:
            pass
    dataset_mean_z = mean(target_z_values) if target_z_values else None  # 这个值只在table_z_source: per-dataset-mean使用
    output: list[dict[str, Any]] = []  # 创建最终有效样本列表
    skipped: Counter[str] = Counter()  # 创建按原因计数的跳过统计器
    for data in data_rows:
        image_name = data.get("image_filename", "").strip()  # 读取原图文件名
        if not image_name:
            skipped["missing_image_name"] += 1
            continue
        insight = insight_by_image.get(image_name)  # 使用当前日志的 image_name 查找同名 InsightFace 行
        if insight is None:
            skipped["missing_insightface_row"] += 1
            continue
        if not _status_success(insight):
            skipped["failed_insightface"] += 1
            continue
        try:
            gaze_world = [
                _float(data, key)
                for key in ("gaze_target_tx", "gaze_target_ty", "gaze_target_tz")
            ]  # 注视点在世界坐标系的坐标
            face_bbox = _bbox(insight, "face")  # 确保人脸框坐标在图像内
            left_bbox = _bbox(insight, "left_eye")
            right_bbox = _bbox(insight, "right_eye")
            r_wc, t_wc, r_cw, t_cw = _camera_pose(
                data, config.handeye_translation_scale
            )  # 读取采集日志中的相机 Vicon 刚体位姿，再与固定 hand-eye 外参组合
            if config.table_z_source == "per-sample-target":
                table_z = gaze_world[2]  # 使用每个样本的 gaze_target_tz 作为桌面高度
            elif dataset_mean_z is not None:
                table_z = dataset_mean_z
            else:
                raise ValueError("Dataset has no valid gaze_target_tz for table height.")
                        
            # 把桌面坐标轴变换到相机系
            table_origin_world = [t_wc[0], t_wc[1], table_z]  # 定义桌面原点
            table_u_camera = _matvec(r_cw, list(TABLE_U_WORLD))  # 世界系桌面 U 轴旋转到相机系:e_u_C = R_C_W @ e_u_W
            table_v_camera = _matvec(r_cw, list(TABLE_V_WORLD))  # 世界系桌面 V 轴旋转到相机系:e_v_C = R_C_W @ e_v_W
            table_normal_camera = _matvec(r_cw, list(TABLE_NORMAL_WORLD))  # 世界系桌面法向量旋转到相机系:n_C = R_C_W @ n_W
            table_origin_camera = _add(
                _matvec(r_cw, table_origin_world), t_cw
            )  # 把桌面原点变换到相机系:O_table_C  = R_C_W @ O_table_W  + t_C_W
            table_distance_camera = _dot(
                table_normal_camera, table_origin_camera
            )  # 桌面法向量在桌面原点处的投影：相机到桌面的有符号距离
            expected_distance = table_z - t_wc[2]  # 期望距离 = 桌面世界系 z - 相机世界系 z
            if abs(table_distance_camera - expected_distance) > 1e-3:
                raise ValueError("Camera/table signed distance check failed.")
            
            # 计算 UV 注视标签
            target_from_origin = _subtract(gaze_world, table_origin_world)  # 计算桌面原点指向注视目标的向量
            uv = [
                _dot(list(TABLE_U_WORLD), target_from_origin),
                _dot(list(TABLE_V_WORLD), target_from_origin),
            ]  # 分别投影到桌面 U/V 轴，得到桌面坐标

            # 将世界系注视目标变换到真实相机系：gaze_target_C = R_C_W @ gaze_target_W + t_C_W
            gaze_camera = _add(_matvec(r_cw, gaze_world), t_cw)  
        except ValueError:
            skipped["invalid_geometry_or_bbox"] += 1
            continue
        source_path = dataset_dir / "insightface_img" / image_name  # 构造当前样本的源图路径
        if not source_path.is_file():
            skipped["missing_source_image"] += 1
            continue
        face_path = dataset_dir / config.face_image_dir / image_name
        row: dict[str, Any] = {
            "sample_id": f"{config.dataset_name}/{Path(image_name).stem}",  # # 跨 Stage 连接使用的唯一样本 ID
            "dataset": config.dataset_name,  # # 完整数据集名称
            "image_name": image_name,  # 源图文件名
            "frame_idx": data.get("frame_idx", "").strip(),  # 原始帧序号
            "source_dataset_dir": str(dataset_dir),  # 源数据集目录
            "source_image_path": str(source_path),  # 原始图像路径
            "face_path": str(face_path),  # 已有人脸裁剪路径
            "z_table_mm": table_z,  # 世界系桌面高度
            "uv_gt_u_mm": uv[0],  # 桌面坐标 U 轴
            "uv_gt_v_mm": uv[1],  # 桌面坐标 V 轴
            **{
                f"gaze_target_w_{axis}_mm": gaze_world[i]
                for i, axis in enumerate("xyz")
            },  # 桌面坐标 X/Y/Z;动态生成 世界系 注视目标字段
            **{
                f"gaze_cam_recomputed_{axis}_mm": gaze_camera[i]
                for i, axis in enumerate("xyz")
            },  # 桌面坐标 X/Y/Z;动态生成 相机系 注视目标字段
            **{f"t_wc_{axis}_mm": t_wc[i] for i, axis in enumerate("xyz")},  # 动态生成世界系平移字段
            **{f"t_cw_{axis}_mm": t_cw[i] for i, axis in enumerate("xyz")},  # 动态生成相机平移字段
        }
        for name, bbox in (
            ("face_bbox", face_bbox),
            ("left_eye_bbox", left_bbox),
            ("right_eye_bbox", right_bbox),
        ):  # 展开三个 bbox
            for axis, value in zip(("x", "y", "w", "h"), bbox):
                row[f"{name}_{axis}"] = value  # 将 [x,y,w,h] 与字段后缀逐项对应
        for r in range(3):  # 展开 R_C_W 和 R_W_C 的 3×3 旋转矩阵字段
            for c in range(3):
                row[f"camera_rotation_c_from_w_{r}{c}"] = r_cw[r][c]
                row[f"camera_rotation_w_from_c_{r}{c}"] = r_wc[r][c]
        for index, axis in enumerate("xyz"):  # 展开桌面 U/V 轴，生成table_u/v_axis_camera_x/y/z
            row[f"table_u_axis_camera_{axis}"] = table_u_camera[index]
            row[f"table_v_axis_camera_{axis}"] = table_v_camera[index]
        row["table_distance_camera_mm"] = table_distance_camera  # 保存相机原点到桌面平面的有符号距离
        output.append(row)
    report = {
        "schema_version": "modelv1_table7_base_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config.dataset_name,
        "data_log_csv": str(logs[0]),
        "insightface_csv": str(insight_path),
        "input_rows": len(data_rows),
        "written_rows": len(output),
        "skipped": dict(skipped),
        "table_z_source": config.table_z_source,
        "historical_pnp_files_read": False,
        "legacy_scene25_built": False,
        "legacy_crop36_built": False,
    }
    return output, report


def main() -> int:  # -> int 表示函数返回进程退出码：0
    """
    执行 Stage 1，检查目标文件并写出基础 CSV 与审计报告。
    负责：读取配置、检查输出、调用 _build_rows()、处理 dry-run、保存 CSV 和报告
    """

    cli = parse_args()  # 解析命令行参数
    config = load_table7_preprocessing_config(cli.config)  # 读取 YAML 配置，并转换为 Table7PreprocessingConfig 对象
    existing = [
        path for path in (config.base_csv, config.base_report) if path.exists()
    ]  # 检查 Stage 1 的两个目标文件是否已经存在:base_dataset<ID>.csv; base_dataset<ID>_report.json
    if existing and not config.overwrite and not config.dry_run:
        raise FileExistsError(
            f"Output exists: {existing}. Set execution.overwrite: true."
        )  # 决定是否因为输出文件已存在而拒绝运行
    rows, report = _build_rows(config)  # 调用 Stage 1 的核心处理函数。
    if not rows:
        raise RuntimeError("Stage 1 selected no valid samples.")
    if config.dry_run:  # 打印预期行为，但不真正写文件
        print(f"[dry-run] stage 1 would write {len(rows)} rows to {config.base_csv}")
        return 0
    _atomic_csv(config.base_csv, rows, config.overwrite)  # 将 rows 写入 Stage 1 基础 CSV
    report.update(
        {
            "config": str(config.config_path),  # Path 被转换为字符串，方便写入 JSON
            "output_csv": str(config.base_csv),  # 记录本次 Stage 1 生成的 CSV 路径
            "field_count": len(OUTPUT_FIELDS),  # 记录基础 CSV 的字段数量。
            "fields": list(OUTPUT_FIELDS),  # 把所有 CSV 字段名保存到报告中
        }
    )
    _atomic_json(config.base_report, report, config.overwrite)  # 把报告写入 Stage 1 JSON 文件
    print(f"Stage 1 wrote {len(rows)} rows: {config.base_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
