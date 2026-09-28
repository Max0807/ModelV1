"""Typed configuration for the final three-stage TableFrame7 preprocessing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "preprocessing" / "modelv1_table7_final.yaml"
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """确认 YAML 节点是映射类型，并返回带类型约束的配置节。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping.")
    return value


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    """从配置节读取必填键，缺失时给出包含节名的明确错误。"""

    if key not in mapping:
        raise ValueError(f"Missing config value: {section}.{key}")
    return mapping[key]


def _path(value: Any, *, dataset_id: str) -> Path:
    """展开环境变量和 dataset_id 占位符，并生成规范化绝对路径。"""

    text = os.path.expandvars(str(value)).replace("{dataset_id}", dataset_id)
    path = Path(text)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def canonical_dataset_id(value: Any) -> str:
    """把数字或完整数据集目录名统一为纯数字数据集后缀。"""

    text = str(value).strip()
    prefix = "dataset_dual_rigid_body_"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if not text or not text.isdigit():
        raise ValueError(
            "dataset.id must be a numeric suffix or dataset_dual_rigid_body_<ID>."
        )
    return text


@dataclass(frozen=True)
class Table7PreprocessingConfig:
    config_path: Path
    dataset_id: str
    source_root: Path
    output_root: Path
    camera_calibration_file: Path
    face_image_dir: str
    insightface_coordinates_file: str
    handeye_translation_scale: float
    table_z_source: str
    mediapipe_min_detection_confidence: float
    min_canthus_distance_px: float
    min_crop_inside_ratio: float
    min_eye_quality_score: float
    ipd_mm: float
    max_ipd_residual_mm: float
    max_ray_condition: float
    calibration_start: int
    calibration_stop: int
    calibration_index_mode: str
    face_preprocess: str
    deca_crop_scale: float
    virtual_face_width: int
    virtual_face_height: int
    virtual_focal_length_px: float
    virtual_distance_mm: float
    overwrite: bool
    dry_run: bool
    limit: int | None
    log_every: int
    stages: tuple[int, ...]

    @property
    def dataset_name(self) -> str:
        """返回源数据目录使用的完整数据集名称。"""

        return f"dataset_dual_rigid_body_{self.dataset_id}"

    @property
    def stage1_dir(self) -> Path:
        """返回 Stage 1 基础 CSV 和报告的输出目录。"""

        return self.output_root / "stage1_base"

    @property
    def stage2_dir(self) -> Path:
        """返回 Stage 2A/2B 观测和 metric pose 的共享输出目录。"""

        return self.output_root / "stage2_metric_head_pose"

    @property
    def stage3_dir(self) -> Path:
        """返回 Stage 3 虚拟图像和最终 TableFrame7 CSV 的输出目录。"""

        return self.output_root / "stage3_virtual_table7"

    @property
    def base_csv(self) -> Path:
        """返回 Stage 1 基础样本表的标准文件路径。"""

        return self.stage1_dir / f"base_dataset{self.dataset_id}.csv"

    @property
    def base_report(self) -> Path:
        """返回 Stage 1 统计与字段审计报告路径。"""

        return self.stage1_dir / f"base_dataset{self.dataset_id}_report.json"

    @property
    def eye_geometry_npz(self) -> Path:
        """返回 Stage 2A MediaPipe Eye15/PnP 观测归档路径。"""

        return self.stage2_dir / f"mediapipe_eye15_dataset{self.dataset_id}.npz"

    @property
    def eye_geometry_report(self) -> Path:
        """返回 Stage 2A 检测覆盖率与质量统计报告路径。"""

        return self.stage2_dir / f"mediapipe_eye15_dataset{self.dataset_id}_report.json"

    @property
    def metric_pose_csv(self) -> Path:
        """返回 Stage 2B Iris65 metric head pose CSV 路径。"""

        return self.stage2_dir / f"iris65_metric_head_pose_dataset{self.dataset_id}.csv"

    @property
    def metric_pose_report(self) -> Path:
        """返回 Stage 2B 成功率、标定和几何约定报告路径。"""

        return self.stage2_dir / f"iris65_metric_head_pose_dataset{self.dataset_id}_report.json"

    @property
    def final_csv(self) -> Path:
        """返回 Stage 3 最终训练 CSV 的标准路径。"""

        return self.stage3_dir / f"training_table7_dataset{self.dataset_id}.csv"

    @property
    def final_report(self) -> Path:
        """返回 Stage 3 虚拟图像与 TableFrame7 生成报告路径。"""

        return self.stage3_dir / f"training_table7_dataset{self.dataset_id}_report.json"


def load_table7_preprocessing_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> Table7PreprocessingConfig:
    """读取 YAML，解析各阶段参数和路径，并返回不可变的类型化配置对象。
    这个过程会：
    - 读取 dataset ID。
    - 展开 {dataset_id}。
    - 把相对路径转换为绝对路径。
    - 读取 Stage 1–3 参数。
    - 检查参数是否合法。
    - 生成各阶段标准输出路径。
    """

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read preprocessing config files.") from error

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Preprocessing config does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(payload, "config")
    dataset = _mapping(_required(root, "dataset", "config"), "dataset")
    paths = _mapping(_required(root, "paths", "config"), "paths")
    camera = _mapping(_required(root, "camera", "config"), "camera")
    base = _mapping(_required(root, "base_dataset", "config"), "base_dataset")
    mediapipe = _mapping(_required(root, "mediapipe", "config"), "mediapipe")
    metric = _mapping(_required(root, "metric_head_pose", "config"), "metric_head_pose")
    calibration = _mapping(
        _required(metric, "fixed_shape_calibration", "metric_head_pose"),
        "metric_head_pose.fixed_shape_calibration",
    )
    virtual = _mapping(_required(root, "virtual_camera", "config"), "virtual_camera")
    execution = _mapping(_required(root, "execution", "config"), "execution")
    dataset_id = canonical_dataset_id(_required(dataset, "id", "dataset"))
    limit_raw = execution.get("limit")
    stages = tuple(int(value) for value in execution.get("stages", (1, 2, 3)))

    config = Table7PreprocessingConfig(
        config_path=config_path,
        dataset_id=dataset_id,
        source_root=_path(_required(paths, "source_root", "paths"), dataset_id=dataset_id),
        output_root=_path(_required(paths, "output_root", "paths"), dataset_id=dataset_id),
        camera_calibration_file=_path(
            _required(camera, "calibration_file", "camera"),
            dataset_id=dataset_id,
        ),
        face_image_dir=str(base.get("face_image_dir", "insightface_face")),
        insightface_coordinates_file=str(
            base.get("insightface_coordinates_file", "insightface_coordinates.csv")
        ),
        handeye_translation_scale=float(base.get("handeye_translation_scale", 1000.0)),
        table_z_source=str(base.get("table_z_source", "per-sample-target")),
        mediapipe_min_detection_confidence=float(
            mediapipe.get("min_detection_confidence", 0.5)
        ),
        min_canthus_distance_px=float(mediapipe.get("min_canthus_distance_px", 8.0)),
        min_crop_inside_ratio=float(mediapipe.get("min_crop_inside_ratio", 0.8)),
        min_eye_quality_score=float(mediapipe.get("min_eye_quality_score", 0.25)),
        ipd_mm=float(metric.get("ipd_mm", 65.0)),
        max_ipd_residual_mm=float(metric.get("max_ipd_residual_mm", 4.0)),
        max_ray_condition=float(metric.get("max_ray_condition", 250.0)),
        calibration_start=int(_required(calibration, "start", "fixed_shape_calibration")),
        calibration_stop=int(_required(calibration, "stop", "fixed_shape_calibration")),
        calibration_index_mode=str(calibration.get("index_mode", "frame_idx")),
        face_preprocess=str(metric.get("face_preprocess", "deca")),
        deca_crop_scale=float(metric.get("deca_crop_scale", 1.25)),
        virtual_face_width=int(virtual.get("face_width", 160)),
        virtual_face_height=int(virtual.get("face_height", 160)),
        virtual_focal_length_px=float(virtual.get("focal_length_px", 480.0)),
        virtual_distance_mm=float(virtual.get("distance_mm", 600.0)),
        overwrite=bool(execution.get("overwrite", False)),
        dry_run=bool(execution.get("dry_run", False)),
        limit=None if limit_raw is None else int(limit_raw),
        log_every=int(execution.get("log_every", 20)),
        stages=stages,
    )
    _validate(config)  # 校验标定文件、枚举值、数值阈值、样本限制和阶段选择
    return config


def _validate(config: Table7PreprocessingConfig) -> None:
    """校验标定文件、枚举值、数值阈值、样本限制和阶段选择。"""

    if not config.camera_calibration_file.is_file():
        raise FileNotFoundError(
            f"Camera calibration file does not exist: {config.camera_calibration_file}"
        )
    if config.table_z_source not in {"per-sample-target", "per-dataset-mean"}:
        raise ValueError("base_dataset.table_z_source is invalid.")
    if config.calibration_index_mode not in {"frame_idx", "position"}:
        raise ValueError("fixed_shape_calibration.index_mode is invalid.")
    if config.calibration_start < 0 or config.calibration_stop < config.calibration_start:
        raise ValueError("Fixed-shape calibration range must be non-negative and inclusive.")
    positive = {
        "handeye_translation_scale": config.handeye_translation_scale,
        "ipd_mm": config.ipd_mm,
        "max_ipd_residual_mm": config.max_ipd_residual_mm,
        "max_ray_condition": config.max_ray_condition,
        "deca_crop_scale": config.deca_crop_scale,
        "virtual_focal_length_px": config.virtual_focal_length_px,
        "virtual_distance_mm": config.virtual_distance_mm,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"These config values must be positive: {positive}")
    if not 0 <= config.mediapipe_min_detection_confidence <= 1:
        raise ValueError("mediapipe.min_detection_confidence must lie in [0, 1].")
    if config.limit is not None and config.limit <= 0:
        raise ValueError("execution.limit must be null or positive.")
    if config.log_every <= 0:
        raise ValueError("execution.log_every must be positive.")
    if not config.stages or any(stage not in {1, 2, 3} for stage in config.stages):
        raise ValueError("execution.stages must contain only 1, 2, and/or 3.")

