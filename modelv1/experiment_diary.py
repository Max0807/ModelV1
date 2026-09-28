"""Append completed ModelV1 training runs to a human-readable experiment diary."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


INDEX_END = "<!-- AUTO-INDEX-END -->"
DETAILS_END = "<!-- AUTO-DETAILS-END -->"
RECORD_PATTERN = re.compile(r"<!-- experiment-record: (\{.*?\}) -->")

DEFAULT_TEMPLATE = """# ModelV1 实验日记

从启用自动记录后的下一次正式训练开始更新。`--dry-run` 不写入本日记。

## 实验索引

| 完成时间 | Run | 实验说明 | 最佳 Val EPE | 相比上一实验 | 最佳 Epoch |
|---|---|---|---:|---:|---:|
<!-- AUTO-INDEX-END -->

## 实验详情

<!-- AUTO-DETAILS-END -->
"""


def append_completed_experiment(
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    diary_path: Path,
    parameter_count: int,
    train_sample_count: int,
    val_sample_count: int,
) -> bool:
    """Append one completed run; return ``False`` if it was already recorded."""

    metrics_path = run_dir / "metrics.csv"
    rows = _read_metric_rows(metrics_path)
    best = min(rows, key=lambda row: _float_metric(row, "val_epe_mm"))
    final = rows[-1]

    diary_path.parent.mkdir(parents=True, exist_ok=True)
    if diary_path.exists():
        diary_text = diary_path.read_text(encoding="utf-8")
    else:
        diary_text = DEFAULT_TEMPLATE
    _validate_diary_template(diary_text, diary_path)

    run_key = _project_relative(run_dir)
    records = _read_records(diary_text)
    if any(record.get("run_dir") == run_key for record in records):
        return False

    best_epe = _float_metric(best, "val_epe_mm")
    previous_best = (
        float(records[-1]["best_val_epe_mm"])
        if records and "best_val_epe_mm" in records[-1]
        else None
    )
    delta = best_epe - previous_best if previous_best is not None else None
    completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    experiment = config["experiment"]
    data = config["data"]
    model = config["model"]
    loss = config["loss"]
    training = config["training"]
    optimizer = training["optimizer"]
    scheduler = training["scheduler"]
    wandb = config["logging"]["wandb"]
    depth_prior_description = _describe_depth_prior_artifact(data)
    run_name = str(experiment["run_name"])
    model_change = _describe_model_change(model)
    description = model_change

    record = {
        "run_dir": run_key,
        "completed_at": completed_at,
        "run_name": run_name,
        "best_val_epe_mm": best_epe,
        "best_epoch": int(best["epoch"]),
    }
    record_marker = (
        "<!-- experiment-record: "
        + json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        + " -->"
    )
    index_row = (
        f"| {_escape_table(completed_at)} | {_escape_table(run_name)} | "
        f"{_escape_table(description)} | {best_epe:.3f} mm | "
        f"{_format_delta(delta)} | {int(best['epoch'])} |"
    )

    loss_description = _describe_loss(loss)
    details = f"""## {completed_at} · {run_name}

{record_marker}

- 实验目的：{description}
- 模型变化：{model_change}
- W&B：project=`{wandb["project"]}`，mode=`{wandb["mode"]}`，tags={wandb["tags"]}
- 产物目录：`{run_key}`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `{data["split_mode"]}`；datasets={data.get("train_datasets")} → {data.get("val_datasets")}；train={train_sample_count}，val={val_sample_count} |
| 深度先验 | `{depth_prior_description}` |
| 人脸/双眼视觉融合 | use={model.get("use_face_image", False)}，face=`inception_resnet_v1(vggface2)`，eye=`{model.get("eye_backbone", "resnet18")}`，cross_attention={model.get("visual_attention_heads", 8)}x{model.get("visual_attention_dim", 128)}，face_frozen={model.get("freeze_face_image_backbone", False)} |
| 眼部骨干 | `{model["eye_backbone"]}`，input_size=`(90, 56)` (W,H)，weights=`{model.get("eye_backbone_weights")}` |
| Crop分支 | use={model.get("use_crop_cam", True)}，input_dim={model.get("crop_cam_dim", 36)} |
| Scene分支 | representation=`{model.get("scene_representation", "full25")}`，input_dim={model.get("scene_dim", 25)} |
| 几何分支 | use={model.get("use_eye_geometry", False)}，gate=`{model.get("eye_geometry_gate_mode", "none")}` |
| 融合层 | `{model["fusion_hidden_dims"]}`；总参数={parameter_count:,} |
| 训练 | epochs={training["epochs"]}，batch={data["batch_size"]}，optimizer=`{optimizer["name"]}`，lr={optimizer["lr"]}，weight_decay={optimizer["weight_decay"]} |
| 调度器 | `{scheduler["name"]}`，eta_min={scheduler["eta_min"]} |
| 损失 | {loss_description} |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch {int(best["epoch"])}） | {_metric(best, "val_loss")} | {_metric(best, "val_epe_mm", " mm")} | {_metric(best, "val_median_epe_mm", " mm")} | {_metric(best, "val_mae_u_mm", " mm")} | {_metric(best, "val_mae_v_mm", " mm")} |
| 最终训练（epoch {int(final["epoch"])}） | {_metric(final, "train_loss")} | {_metric(final, "train_epe_mm", " mm")} | {_metric(final, "train_median_epe_mm", " mm")} | {_metric(final, "train_mae_u_mm", " mm")} | {_metric(final, "train_mae_v_mm", " mm")} |
| 最终验证（epoch {int(final["epoch"])}） | {_metric(final, "val_loss")} | {_metric(final, "val_epe_mm", " mm")} | {_metric(final, "val_median_epe_mm", " mm")} | {_metric(final, "val_mae_u_mm", " mm")} | {_metric(final, "val_mae_v_mm", " mm")} |

相比上一实验：{_format_delta_sentence(delta)}

"""

    diary_text = diary_text.replace(
        INDEX_END,
        f"{index_row}\n{INDEX_END}",
        1,
    )
    diary_text = diary_text.replace(
        DETAILS_END,
        f"{details}{DETAILS_END}",
        1,
    )
    with diary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(diary_text)
    return True


def _describe_depth_prior_artifact(data: Mapping[str, Any]) -> str:
    """Return a diary-safe depth-prior description."""

    dataset_ids = data.get("dataset_ids")
    if isinstance(dataset_ids, (list, tuple)) and dataset_ids:
        ids = ", ".join(str(value) for value in dataset_ids)
        processed_dir = str(data.get("processed_data_dir", "data/processed"))
        prior_kind = str(data.get("depth_prior_kind", "iris_ipd_65mm"))
        if prior_kind == "iris_ipd_65mm":
            prior_pattern = "depth_priors_iris_ipd_65mm_v1_<id>.csv"
        elif prior_kind == "pnp1010":
            prior_pattern = "depth_priors_deca_crop_v1_<id>.csv"
        else:
            prior_pattern = f"<unknown prior kind: {prior_kind}>"
        depth_prior = f"numbered datasets [{ids}]; {processed_dir}/{prior_pattern}"
        return depth_prior

    return str(data.get("depth_prior_csv_path", "<not configured>"))


def _read_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment metrics do not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Experiment metrics contain no epochs: {path}")
    return rows


def _read_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for match in RECORD_PATTERN.finditer(text):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _validate_diary_template(text: str, path: Path) -> None:
    missing = [marker for marker in (INDEX_END, DETAILS_END) if marker not in text]
    if missing:
        raise ValueError(
            f"Experiment diary {path} is missing automatic markers: {missing}"
        )


def _float_metric(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid metric {key!r} in experiment metrics.") from exc


def _metric(row: Mapping[str, str], key: str, suffix: str = "") -> str:
    return f"{_float_metric(row, key):.3f}{suffix}"


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "—"
    return f"{delta:+.3f} mm"


def _format_delta_sentence(delta: float | None) -> str:
    if delta is None:
        return "这是自动日记记录的首个实验，暂无上一实验可比较。"
    if delta < 0:
        return f"最佳 Val EPE 下降 {abs(delta):.3f} mm，性能提升。"
    if delta > 0:
        return f"最佳 Val EPE 上升 {delta:.3f} mm，性能下降。"
    return "最佳 Val EPE 与上一实验相同。"


def _escape_table(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _describe_model_change(model: Mapping[str, Any]) -> str:
    changes = []
    if bool(model.get("use_face_image", False)):
        changes.append(
            "增加 RGB 人脸编码器及 face-eye 视觉融合"
        )
    if not bool(model.get("use_crop_cam", True)):
        changes.append("移除 crop_cam 36D 分支")
    scene_representation = str(model.get("scene_representation", "full25"))
    if scene_representation == "table_frame7":
        changes.append("Scene 从 full25 替换为 TableFrame 7D")
    elif scene_representation == "orientation6d":
        changes.append("Scene 替换为桌面 Orientation 6D")
    if not bool(model.get("use_eye_geometry", False)):
        changes.append("不使用双眼 6D 几何先验")
        return "；".join(changes) + "。"
    if str(model.get("eye_geometry_representation", "normalized6d")) == "raw_eye6d":
        changes.append("双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D")
    gate_mode = str(model.get("eye_geometry_gate_mode", "none"))
    if gate_mode == "none":
        changes.append("V2.1 标准化双眼 6D 几何直接拼接")
    elif gate_mode == "fixed":
        changes.append("双眼 6D 几何使用固定 PnP 质量门控")
    elif gate_mode == "learned_residual":
        changes.append("V2.2 双眼 6D 几何使用可学习残差质量门控")
    else:
        changes.append(f"双眼 6D 几何门控模式 `{gate_mode}`")
    return "；".join(changes) + "。"


def _describe_loss(loss: Mapping[str, Any]) -> str:
    """Describe both the legacy direct-UV and V4 gaze-geometry losses."""

    if "uv_huber_beta_mm" in loss:
        fields = (
            "uv_huber_beta_mm",
            "uv_huber_weight",
            "gaze_angular_weight",
            "ray_validity_weight",
        )
        return "，".join(
            f"{key}={loss[key]}" for key in fields if key in loss
        )

    if "beta_mm" in loss:
        return (
            f'beta_mm={loss["beta_mm"]}，'
            f'gate_reg={loss.get("gate_regularization_weight", 0.0)}'
        )

    if not loss:
        return "未配置"
    return "，".join(f"{key}={value}" for key, value in loss.items())


def _project_relative(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()
