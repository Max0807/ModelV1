"""Train ModelV1 with offline DECA features and normalized UV targets.

The script owns the complete training lifecycle: deterministic data split,
model/optimizer/scheduler setup, AMP, validation, W&B and local logging,
checkpointing, and resume validation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1 import ModelV1, ModelV1Config, UVLossConfig, UVRegressionLoss
from modelv1.deca_cache import DecaFeatureCache
from modelv1.data import (
    build_modelv1_dataloaders,
    get_eye_geometry_normalizer,
    get_eye_geometry_quality_normalizer,
    get_uv_target_normalizer,
)
from modelv1.data.normalization import (
    EyeGeometryNormalizer,
    EyeGeometryQualityNormalizer,
    UVTargetNormalizer,
)
from modelv1.depth_prior.face_preprocess import (
    DEFAULT_DECA_CROP_SCALE,
    FACE_PREPROCESS_CHOICES,
    FACE_PREPROCESS_DECA,
    FACE_PREPROCESS_LEGACY,
)
from modelv1.experiment_diary import append_completed_experiment
from modelv1.geometry_gate import (
    EYE_GEOMETRY_GATE_LEARNED_RESIDUAL,
    EYE_GEOMETRY_GATE_NONE,
    canonical_eye_geometry_gate_mode,
)
from modelv1.deca_cache import (
    DECA_FEATURE_REPRESENTATION_FULL236,
    canonical_deca_feature_representation,
    deca_feature_representation_dim,
)
from modelv1.model import (
    DECA_BRANCH_MODE_FACTORIZED_GEOMETRY,
    canonical_deca_branch_mode,
)
from modelv1.data.depth_prior import (
    EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
    canonical_eye_geometry_representation,
    eye_geometry_representation_dim,
)
from modelv1.scene import (
    SCENE_REPRESENTATION_FULL25,
    canonical_scene_representation,
    scene_representation_dim,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "modelv1" / "train_random_80_20_100.yaml"
METRIC_NAMES = ("epe_mm", "median_epe_mm", "mae_u_mm", "mae_v_mm")
METRIC_FIELDS = [
    "epoch",
    "global_step",
    "lr",
    "train_loss",
    *(f"train_{name}" for name in METRIC_NAMES),
    "train_seconds",
    "train_samples_per_second",
    "val_loss",
    *(f"val_{name}" for name in METRIC_NAMES),
    "val_seconds",
    "val_samples_per_second",
    "epoch_seconds",
    "elapsed_seconds",
    "best_val_epe_mm",
]
TUPLE_CONFIG_KEYS = (
    "deca_factor_hidden_dims",
    "face_hidden_dims",
    "visual_fusion_hidden_dims",
    "crop_cam_hidden_dims",
    "scene_hidden_dims",
    "fusion_hidden_dims",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--resume", type=Path, default=None, help="Path to last.pt or best.pt.")
    parser.add_argument("--device", default=None, help="Override training.device.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one epoch with W&B disabled to verify the complete training path.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Top-level training config must be a mapping.")
    data_section = config.get("data")
    if isinstance(data_section, dict):
        data_section["deca_face_preprocess"] = str(
            data_section.get("deca_face_preprocess", FACE_PREPROCESS_DECA)
        ).strip().lower()
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    for key in ("experiment", "data", "model", "loss", "training", "logging"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Config requires a {key!r} mapping.")
    if int(config["training"].get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive.")
    if int(config["data"].get("batch_size", 0)) <= 0:
        raise ValueError("data.batch_size must be positive.")
    if float(config["data"].get("val_ratio", 0.0)) <= 0:
        raise ValueError("data.val_ratio must be positive.")
    if float(config["training"]["optimizer"].get("lr", 0.0)) <= 0:
        raise ValueError("training.optimizer.lr must be positive.")
    experiment_config = config["experiment"]
    if not isinstance(experiment_config.get("description"), str):
        raise ValueError("experiment.description must be a string.")
    if not isinstance(experiment_config.get("diary_enabled"), bool):
        raise ValueError("experiment.diary_enabled must be true or false.")
    if (
        experiment_config["diary_enabled"]
        and not str(experiment_config.get("diary_path", "")).strip()
    ):
        raise ValueError(
            "experiment.diary_path is required when diary_enabled=true."
        )
    wandb_config = config["logging"].get("wandb")
    if not isinstance(wandb_config, dict):
        raise ValueError("logging.wandb must be a mapping.")
    for key in ("enabled", "project", "entity", "mode", "tags"):
        if key not in wandb_config:
            raise ValueError(f"logging.wandb requires {key!r} in the config file.")
    if not isinstance(wandb_config["enabled"], bool):
        raise ValueError("logging.wandb.enabled must be true or false.")
    if not str(wandb_config["project"]).strip():
        raise ValueError("logging.wandb.project must be a non-empty name.")
    if not str(wandb_config["mode"]).strip():
        raise ValueError("logging.wandb.mode must be non-empty.")
    if not isinstance(wandb_config["tags"], list):
        raise ValueError("logging.wandb.tags must be a list.")
    face_preprocess = str(
        config["data"].get("deca_face_preprocess", FACE_PREPROCESS_DECA)
    ).strip().lower()
    if face_preprocess not in FACE_PREPROCESS_CHOICES:
        raise ValueError(
            "data.deca_face_preprocess must be one of "
            f"{FACE_PREPROCESS_CHOICES}, got {face_preprocess!r}."
        )
    if float(config["data"].get("deca_crop_scale", DEFAULT_DECA_CROP_SCALE)) <= 0:
        raise ValueError("data.deca_crop_scale must be positive.")
    use_eye_geometry = config["model"].get("use_eye_geometry", False)
    if not isinstance(use_eye_geometry, bool):
        raise ValueError("model.use_eye_geometry must be true or false.")
    use_crop_cam = config["model"].get("use_crop_cam", True)
    if not isinstance(use_crop_cam, bool):
        raise ValueError("model.use_crop_cam must be true or false.")
    use_face_image = config["model"].get("use_face_image", False)
    if not isinstance(use_face_image, bool):
        raise ValueError("model.use_face_image must be true or false.")
    load_face_image = config["data"].get("load_face_image", False)
    if not isinstance(load_face_image, bool):
        raise ValueError("data.load_face_image must be true or false.")
    if use_face_image and not load_face_image:
        raise ValueError(
            "data.load_face_image must be true when model.use_face_image=true."
        )
    deca_feature_representation = canonical_deca_feature_representation(
        config["model"].get(
            "deca_feature_representation",
            DECA_FEATURE_REPRESENTATION_FULL236,
        )
    )
    expected_deca_feature_dim = deca_feature_representation_dim(
        deca_feature_representation
    )
    if int(config["model"].get("deca_feature_dim", 236)) != expected_deca_feature_dim:
        raise ValueError(
            f"model.deca_feature_representation={deca_feature_representation!r} "
            f"requires model.deca_feature_dim={expected_deca_feature_dim}."
        )
    deca_branch_mode = canonical_deca_branch_mode(
        config["model"].get("deca_branch_mode", "flat")
    )
    if (
        deca_branch_mode == DECA_BRANCH_MODE_FACTORIZED_GEOMETRY
        and deca_feature_representation != "geometry156"
    ):
        raise ValueError(
            "model.deca_branch_mode='factorized_geometry' requires "
            "model.deca_feature_representation='geometry156'."
        )
    scene_representation = canonical_scene_representation(
        config["model"].get(
            "scene_representation",
            SCENE_REPRESENTATION_FULL25,
        )
    )
    expected_scene_dim = scene_representation_dim(scene_representation)
    if int(config["model"].get("scene_dim", 25)) != expected_scene_dim:
        raise ValueError(
            f"model.scene_representation={scene_representation!r} requires "
            f"model.scene_dim={expected_scene_dim}."
        )
    eye_geometry_representation = canonical_eye_geometry_representation(
        config["model"].get(
            "eye_geometry_representation",
            EYE_GEOMETRY_REPRESENTATION_NORMALIZED6D,
        )
    )
    expected_eye_geometry_dim = eye_geometry_representation_dim(
        eye_geometry_representation
    )
    if int(config["model"].get("eye_geometry_dim", 6)) != expected_eye_geometry_dim:
        raise ValueError(
            f"model.eye_geometry_representation={eye_geometry_representation!r} "
            f"requires model.eye_geometry_dim={expected_eye_geometry_dim}."
        )
    gate_mode = canonical_eye_geometry_gate_mode(
        config["model"].get("eye_geometry_gate_mode", EYE_GEOMETRY_GATE_NONE)
    )
    if not use_eye_geometry and gate_mode != EYE_GEOMETRY_GATE_NONE:
        raise ValueError(
            "model.eye_geometry_gate_mode must be 'none' when "
            "model.use_eye_geometry=false."
        )
    if int(config["model"].get("eye_geometry_quality_dim", 4)) != 4:
        raise ValueError("model.eye_geometry_quality_dim must be 4 for V2.2.")
    if use_eye_geometry and not config["data"].get("depth_prior_csv_path"):
        raise ValueError(
            "data.depth_prior_csv_path is required when model.use_eye_geometry=true."
        )
    gate_regularization_weight = float(
        config["loss"].get("gate_regularization_weight", 0.0)
    )
    if gate_regularization_weight < 0:
        raise ValueError("loss.gate_regularization_weight must be non-negative.")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_deca_cache_preprocess(
    cache_path: Path,
    data_config: Mapping[str, Any],
) -> None:
    """Reject a feature cache rendered with different DECA image preprocessing."""

    expected_mode = str(
        data_config.get("deca_face_preprocess", FACE_PREPROCESS_DECA)
    ).strip().lower()
    expected_scale = float(
        data_config.get("deca_crop_scale", DEFAULT_DECA_CROP_SCALE)
    )
    cache = DecaFeatureCache.load(cache_path)
    # Caches produced before this field existed were necessarily legacy crops.
    cache_mode = str(
        cache.metadata.get("face_preprocess", FACE_PREPROCESS_LEGACY)
    ).strip().lower()
    if cache_mode != expected_mode:
        raise ValueError(
            "DECA cache preprocessing mismatch: "
            f"config requests {expected_mode!r}, but {cache_path} was built with "
            f"{cache_mode!r}. Rebuild the cache with scripts/cache_deca_features.py."
        )
    if expected_mode == FACE_PREPROCESS_DECA:
        cache_scale = cache.metadata.get("deca_crop_scale")
        if cache_scale is None or abs(float(cache_scale) - expected_scale) > 1e-8:
            raise ValueError(
                "DECA cache crop-scale mismatch: "
                f"config requests {expected_scale}, but {cache_path} stores "
                f"{cache_scale!r}. Rebuild the cache with the same scale."
            )


def make_run_dir(config: dict[str, Any], resume: Path | None) -> Path:
    if resume is not None:
        checkpoint = resolve_project_path(resume)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        if checkpoint.parent.name != "checkpoints":
            raise ValueError("Resume checkpoint must live in a run's checkpoints directory.")
        run_dir = checkpoint.parent.parent
        config["experiment"]["run_name"] = run_dir.name
        return run_dir

    project = str(config["logging"]["wandb"]["project"])
    run_name = config["experiment"].get("run_name")
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["experiment"]["run_name"] = str(run_name)
    output_dir = resolve_project_path(config["experiment"]["output_dir"])
    run_dir = output_dir / project / str(run_name)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    return run_dir


def setup_file_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("modelv1.train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def save_config(config: Mapping[str, Any], run_dir: Path) -> None:
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, allow_unicode=False, sort_keys=False)


class MetricsCsvWriter:
    def __init__(self, path: Path) -> None:
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open("r", encoding="utf-8", newline="") as existing_handle:
                existing_fields = next(csv.reader(existing_handle), [])
            if existing_fields != METRIC_FIELDS:
                raise ValueError(
                    f"Existing metrics file uses a different schema: {path}. "
                    "Start a new run instead of resuming this run."
                )
        self.handle = path.open("a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=METRIC_FIELDS)
        if not exists:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, metrics: Mapping[str, float | int]) -> None:
        self.writer.writerow({key: metrics[key] for key in METRIC_FIELDS})
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class NoOpTracker:
    def log(self, metrics: Mapping[str, float | int], step: int) -> None:
        del metrics, step

    def finish(self) -> None:
        pass


def start_tracker(
    config: Mapping[str, Any],
    run_dir: Path,
    static_metadata: Mapping[str, Any] | None = None,
) -> Any:
    wandb_config = config["logging"]["wandb"]
    if not bool(wandb_config["enabled"]):
        return NoOpTracker()
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "W&B logging is enabled, but wandb is not installed. Run pip install -r requirements.txt."
        ) from exc

    run_config = to_jsonable(config)
    if static_metadata is not None:
        run_config["model_statistics"] = to_jsonable(static_metadata)

    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        mode=wandb_config["mode"],
        name=config["experiment"]["run_name"],
        tags=wandb_config["tags"],
        dir=str(run_dir),
        config=run_config,
    )
    run.define_metric("epoch")
    for split in ("train", "val"):
        run.define_metric(f"{split}/loss", step_metric="epoch", summary="min")
        run.define_metric(f"{split}/epe_mm", step_metric="epoch", summary="min")
        run.define_metric(f"{split}/median_epe_mm", step_metric="epoch", summary="min")
        run.define_metric(f"{split}/mae_u_mm", step_metric="epoch", summary="min")
        run.define_metric(f"{split}/mae_v_mm", step_metric="epoch", summary="min")
    run.define_metric("checkpoint/best_val_epe_mm", step_metric="epoch", summary="min")
    return run


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is unavailable.")
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def make_model_config(section: Mapping[str, Any]) -> ModelV1Config:
    values = dict(section)
    for key in TUPLE_CONFIG_KEYS:
        if key in values:
            values[key] = tuple(values[key])
    return ModelV1Config(**values)


def move_batch_to_device(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def run_epoch(
    *,
    model: nn.Module,
    loader: Any,
    criterion: UVRegressionLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    grad_clip_norm: float | None,
    gate_regularization_weight: float,
) -> tuple[dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    start = time.perf_counter()
    total_loss = 0.0
    sample_count = 0
    predictions: list[Tensor] = []
    targets_mm: list[Tensor] = []

    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        batch_size = int(device_batch["uv_gt"].shape[0])
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if is_train else torch.no_grad()
        with grad_context:
            with autocast_context(amp_enabled):
                use_gate_regularization = (
                    is_train and gate_regularization_weight > 0
                )
                model_output = model(
                    device_batch,
                    return_features=use_gate_regularization,
                )
                if isinstance(model_output, dict):
                    uv_pred = model_output["uv"]
                else:
                    uv_pred = model_output
                loss = criterion(uv_pred, device_batch["uv_target"])
                if use_gate_regularization:
                    gate_delta = model_output.get("eye_geometry_gate_delta")
                    valid_mask = device_batch.get("eye_geometry_valid_mask")
                    if gate_delta is None or not torch.is_tensor(valid_mask):
                        raise RuntimeError(
                            "Gate regularization requires learned gate delta "
                            "and eye_geometry_valid_mask."
                        )
                    valid_mask = valid_mask.to(
                        device=gate_delta.device,
                        dtype=gate_delta.dtype,
                    )
                    gate_regularization = (
                        gate_delta.square() * valid_mask
                    ).sum() / valid_mask.sum().clamp_min(1.0)
                    loss = (
                        loss
                        + gate_regularization_weight * gate_regularization
                    )
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
        predictions.append(uv_pred.detach().cpu())
        targets_mm.append(device_batch["uv_gt"].detach().cpu())

    if sample_count == 0:
        raise RuntimeError("DataLoader yielded no samples.")
    metric_tensors = criterion.metrics(torch.cat(predictions), torch.cat(targets_mm))
    elapsed = time.perf_counter() - start
    metrics = {"loss": total_loss / sample_count}
    metrics.update({name: float(value) for name, value in metric_tensors.items()})
    metrics["seconds"] = elapsed
    metrics["samples_per_second"] = sample_count / elapsed if elapsed > 0 else 0.0
    return metrics, sample_count


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    best_val_epe_mm: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    normalizer: UVTargetNormalizer,
    eye_geometry_normalizer: EyeGeometryNormalizer | None,
    eye_geometry_quality_normalizer: EyeGeometryQualityNormalizer | None,
    config: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_epe_mm": best_val_epe_mm,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "normalizer": normalizer.state_dict(),
            "eye_geometry_normalizer": (
                eye_geometry_normalizer.state_dict()
                if eye_geometry_normalizer is not None
                else None
            ),
            "eye_geometry_quality_normalizer": (
                eye_geometry_quality_normalizer.state_dict()
                if eye_geometry_quality_normalizer is not None
                else None
            ),
            "config": to_jsonable(config),
            "rng_state": capture_rng_state(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    normalizer: UVTargetNormalizer,
    eye_geometry_normalizer: EyeGeometryNormalizer | None,
    eye_geometry_quality_normalizer: EyeGeometryQualityNormalizer | None,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device)
    required = {"model", "optimizer", "scheduler", "scaler", "normalizer", "epoch", "global_step"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Resume checkpoint is missing keys: {sorted(missing)}")
    saved_normalizer = UVTargetNormalizer.from_state_dict(checkpoint["normalizer"])
    if not torch.allclose(saved_normalizer.mean_mm, normalizer.mean_mm) or not torch.allclose(
        saved_normalizer.std_mm, normalizer.std_mm
    ):
        raise ValueError(
            "Resume checkpoint uses different UV normalization statistics. "
            "Use the original split/config for this checkpoint."
        )
    saved_eye_geometry_state = checkpoint.get("eye_geometry_normalizer")
    if eye_geometry_normalizer is None:
        if saved_eye_geometry_state is not None:
            raise ValueError(
                "Resume checkpoint uses eye geometry, but the current config disables it."
            )
    else:
        if saved_eye_geometry_state is None:
            raise ValueError(
                "Resume checkpoint has no eye geometry normalization statistics."
            )
        saved_eye_geometry_normalizer = EyeGeometryNormalizer.from_state_dict(
            saved_eye_geometry_state
        )
        if not torch.allclose(
            saved_eye_geometry_normalizer.mean,
            eye_geometry_normalizer.mean,
        ) or not torch.allclose(
            saved_eye_geometry_normalizer.std,
            eye_geometry_normalizer.std,
        ):
            raise ValueError(
                "Resume checkpoint uses different eye geometry normalization "
                "statistics. Use the original split and depth-prior CSV."
            )
    saved_quality_state = checkpoint.get("eye_geometry_quality_normalizer")
    if eye_geometry_quality_normalizer is None:
        if saved_quality_state is not None:
            raise ValueError(
                "Resume checkpoint uses eye geometry quality normalization, "
                "but the current config does not."
            )
    else:
        if saved_quality_state is None:
            raise ValueError(
                "Resume checkpoint has no eye geometry quality normalization "
                "statistics."
            )
        saved_quality_normalizer = EyeGeometryQualityNormalizer.from_state_dict(
            saved_quality_state
        )
        if not torch.allclose(
            saved_quality_normalizer.mean,
            eye_geometry_quality_normalizer.mean,
        ) or not torch.allclose(
            saved_quality_normalizer.std,
            eye_geometry_quality_normalizer.std,
        ):
            raise ValueError(
                "Resume checkpoint uses different eye geometry quality "
                "normalization statistics."
            )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        float(checkpoint.get("best_val_epe_mm", float("inf"))),
    )


def build_epoch_record(
    *,
    epoch: int,
    global_step: int,
    lr: float,
    train: Mapping[str, float],
    val: Mapping[str, float],
    epoch_seconds: float,
    elapsed_seconds: float,
    best_val_epe_mm: float,
) -> dict[str, float | int]:
    record: dict[str, float | int] = {
        "epoch": epoch,
        "global_step": global_step,
        "lr": lr,
        "epoch_seconds": epoch_seconds,
        "elapsed_seconds": elapsed_seconds,
        "best_val_epe_mm": best_val_epe_mm,
    }
    for prefix, values in (("train", train), ("val", val)):
        record[f"{prefix}_loss"] = values["loss"]
        for name in METRIC_NAMES:
            record[f"{prefix}_{name}"] = values[name]
        record[f"{prefix}_seconds"] = values["seconds"]
        record[f"{prefix}_samples_per_second"] = values["samples_per_second"]
    return record


def wandb_metrics(record: Mapping[str, float | int]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "epoch": record["epoch"],
        "global_step": record["global_step"],
        "learning_rate": record["lr"],
        "time/epoch_seconds": record["epoch_seconds"],
        "time/elapsed_seconds": record["elapsed_seconds"],
        "checkpoint/best_val_epe_mm": record["best_val_epe_mm"],
    }
    for prefix in ("train", "val"):
        metrics[f"{prefix}/loss"] = record[f"{prefix}_loss"]
        for name in METRIC_NAMES:
            metrics[f"{prefix}/{name}"] = record[f"{prefix}_{name}"]
        metrics[f"time/{prefix}_seconds"] = record[f"{prefix}_seconds"]
        metrics[f"{prefix}/samples_per_second"] = record[f"{prefix}_samples_per_second"]
    return metrics


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    config = load_config(resolve_project_path(args.config))
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.dry_run:
        config["training"]["epochs"] = 1
        config["logging"]["wandb"]["enabled"] = False
        config["experiment"]["run_name"] = "dry_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    resume_path = resolve_project_path(args.resume) if args.resume is not None else None
    run_dir = make_run_dir(config, resume_path)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_file_logger(run_dir / "train.log")
    if resume_path is None:
        save_config(config, run_dir)

    seed_everything(int(config["experiment"]["seed"]), bool(config["experiment"]["deterministic"]))
    device = resolve_device(str(config["training"]["device"]))
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    data_config = config["data"]
    model_config = make_model_config(config["model"])
    deca_cache_path = resolve_project_path(data_config["deca_cache_path"])
    validate_deca_cache_preprocess(deca_cache_path, data_config)
    depth_prior_csv_path = (
        resolve_project_path(data_config["depth_prior_csv_path"])
        if model_config.use_eye_geometry
        else None
    )
    train_loader, val_loader = build_modelv1_dataloaders(
        csv_path=resolve_project_path(data_config["csv_path"]),
        split_mode=data_config["split_mode"],
        all_datasets=data_config["all_datasets"],
        val_ratio=float(data_config["val_ratio"]),
        split_seed=int(data_config["split_seed"]),
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"]),
        normalize_images=bool(data_config["normalize_images"]),
        load_face_image=bool(data_config["load_face_image"]),
        deca_cache_path=deca_cache_path,
        require_deca_features=True,
        use_eye_geometry=model_config.use_eye_geometry,
        depth_prior_csv_path=depth_prior_csv_path,
        eye_geometry_gate_mode=model_config.eye_geometry_gate_mode,
        eye_geometry_representation=model_config.eye_geometry_representation,
        scene_representation=model_config.scene_representation,
        deca_feature_representation=model_config.deca_feature_representation,
    )
    normalizer = get_uv_target_normalizer(train_loader.dataset)
    if normalizer is None:
        raise RuntimeError("Training requires normalized UV targets.")
    eye_geometry_normalizer = get_eye_geometry_normalizer(train_loader.dataset)
    if model_config.use_eye_geometry and eye_geometry_normalizer is None:
        raise RuntimeError(
            "Eye geometry is enabled but its training normalizer is unavailable."
        )
    eye_geometry_quality_normalizer = get_eye_geometry_quality_normalizer(
        train_loader.dataset
    )
    if (
        model_config.eye_geometry_gate_mode
        == EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
        and eye_geometry_quality_normalizer is None
    ):
        raise RuntimeError(
            "Learned eye geometry gate requires a training quality normalizer."
        )

    model = ModelV1(model_config).to(device)
    loss_config = dict(config["loss"])
    gate_regularization_weight = float(
        loss_config.pop("gate_regularization_weight", 0.0)
    )
    if (
        model_config.eye_geometry_gate_mode
        != EYE_GEOMETRY_GATE_LEARNED_RESIDUAL
    ):
        gate_regularization_weight = 0.0
    criterion = UVRegressionLoss(normalizer, UVLossConfig(**loss_config))
    optimizer_config = config["training"]["optimizer"]
    if optimizer_config["name"].lower() != "adamw":
        raise ValueError("Only AdamW is supported by the V1 training config.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler_config = config["training"]["scheduler"]
    if scheduler_config["name"].lower() != "cosine":
        raise ValueError("Only cosine scheduler is supported by the V1 training config.")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
        eta_min=float(scheduler_config["eta_min"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    global_step = 0
    best_val_epe_mm = float("inf")
    if resume_path is not None:
        start_epoch, global_step, best_val_epe_mm = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            normalizer=normalizer,
            eye_geometry_normalizer=eye_geometry_normalizer,
            eye_geometry_quality_normalizer=eye_geometry_quality_normalizer,
            device=device,
        )

    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    setup_text = (
        f"run_dir={run_dir} device={device} amp={amp_enabled} "
        f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"use_face_image={model_config.use_face_image} "
        f"face_image_backbone={model_config.face_image_backbone} "
        f"eye_backbone={model_config.eye_backbone} "
        f"use_crop_cam={model_config.use_crop_cam} "
        f"scene_representation={model_config.scene_representation} "
        f"eye_geometry_gate_mode={model_config.eye_geometry_gate_mode} "
        f"parameters_total={total_parameter_count:,} "
        f"parameters_trainable={trainable_parameter_count:,} start_epoch={start_epoch}"
    )
    logger.info(setup_text)
    print(setup_text)

    tracker = start_tracker(
        config,
        run_dir,
        static_metadata={
            "use_face_image": model_config.use_face_image,
            "face_image_backbone": model_config.face_image_backbone,
            "eye_backbone": model_config.eye_backbone,
            "parameters_total": total_parameter_count,
            "parameters_trainable": trainable_parameter_count,
        },
    )
    metrics_writer = MetricsCsvWriter(run_dir / "metrics.csv")
    started_at = time.perf_counter()
    epochs = int(config["training"]["epochs"])
    terminal_interval = int(config["logging"]["terminal_every_epochs"])
    grad_clip_value = config["training"].get("grad_clip_norm")
    grad_clip_norm = float(grad_clip_value) if grad_clip_value is not None else None

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started_at = time.perf_counter()
            lr = float(optimizer.param_groups[0]["lr"])
            train_metrics, train_samples = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip_norm=grad_clip_norm,
                gate_regularization_weight=gate_regularization_weight,
            )
            global_step += len(train_loader)
            val_metrics, _ = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip_norm=None,
                gate_regularization_weight=0.0,
            )
            scheduler.step()

            improved = val_metrics["epe_mm"] < best_val_epe_mm
            if improved:
                best_val_epe_mm = val_metrics["epe_mm"]
            record = build_epoch_record(
                epoch=epoch,
                global_step=global_step,
                lr=lr,
                train=train_metrics,
                val=val_metrics,
                epoch_seconds=time.perf_counter() - epoch_started_at,
                elapsed_seconds=time.perf_counter() - started_at,
                best_val_epe_mm=best_val_epe_mm,
            )
            metrics_writer.write(record)
            tracker.log(wandb_metrics(record), step=epoch)
            logger.info(json.dumps(record, ensure_ascii=True))

            if bool(config["training"]["checkpoint"]["save_last"]):
                save_checkpoint(
                    checkpoints_dir / "last.pt",
                    epoch=epoch,
                    global_step=global_step,
                    best_val_epe_mm=best_val_epe_mm,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    normalizer=normalizer,
                    eye_geometry_normalizer=eye_geometry_normalizer,
                    eye_geometry_quality_normalizer=eye_geometry_quality_normalizer,
                    config=config,
                )
            if improved:
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    epoch=epoch,
                    global_step=global_step,
                    best_val_epe_mm=best_val_epe_mm,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    normalizer=normalizer,
                    eye_geometry_normalizer=eye_geometry_normalizer,
                    eye_geometry_quality_normalizer=eye_geometry_quality_normalizer,
                    config=config,
                )

            if epoch % terminal_interval == 0 or epoch == epochs or epoch == start_epoch:
                print(
                    f"epoch {epoch:03d}/{epochs} | lr={lr:.3e} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_epe={val_metrics['epe_mm']:.2f} mm | "
                    f"best={best_val_epe_mm:.2f} mm | "
                    f"epoch_time={record['epoch_seconds']:.1f}s"
                )
    finally:
        metrics_writer.close()
        tracker.finish()

    if bool(config["experiment"]["diary_enabled"]) and not args.dry_run:
        diary_path = resolve_project_path(config["experiment"]["diary_path"])
        diary_updated = append_completed_experiment(
            config=config,
            run_dir=run_dir,
            diary_path=diary_path,
            parameter_count=total_parameter_count,
            train_sample_count=len(train_loader.dataset),
            val_sample_count=len(val_loader.dataset),
        )
        diary_text = (
            f"experiment diary updated: {diary_path}"
            if diary_updated
            else f"experiment diary already contains this run: {diary_path}"
        )
        logger.info(diary_text)
        print(diary_text)

    completed_text = (
        f"completed epochs={epochs} best_val_epe_mm={best_val_epe_mm:.4f} "
        f"run_dir={run_dir}"
    )
    logger.info(completed_text)
    print(completed_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
