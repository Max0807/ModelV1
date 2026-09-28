"""Train the independent supervised Native BlazeGaze Stage-1 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.blazegaze.data_adapter import (  # noqa: E402
    BlazeGazeDataAdapter,
    PogBoundsMM,
)
from baselines.blazegaze.losses import (  # noqa: E402
    NativeBlazeGazeLossConfig,
    NativeBlazeGazeStage1Loss,
    mean_pog_error_mm,
)
from baselines.blazegaze.model import (  # noqa: E402
    NativeBlazeGaze,
    NativeBlazeGazeConfig,
)
from baselines.blazegaze.weighting import (  # noqa: E402
    fit_official_weight_grid,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "blazegaze" / "native_supervised_table7.yaml"
)
METRIC_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_gaze_l2",
    "train_reconstruction_mse",
    "train_embedding_consistency",
    "train_epe_mm",
    "val_loss",
    "val_gaze_l2",
    "val_reconstruction_mse",
    "val_embedding_consistency",
    "val_epe_mm",
    "epoch_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pog-bounds-mm",
        type=float,
        nargs=4,
        metavar=("U_MIN", "U_MAX", "V_MIN", "V_MAX"),
        help="Override fixed physical/protocol PoG bounds from the YAML.",
    )
    parser.add_argument("--device", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one train and one validation batch without writing outputs.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Native BlazeGaze config root must be a mapping.")
    for name in ("experiment", "data", "model", "loss", "training", "logging"):
        if not isinstance(value.get(name), dict):
            raise ValueError(f"Config requires a {name!r} mapping.")
    for split in ("train", "val"):
        if not isinstance(value["data"].get(split), dict):
            raise ValueError(f"data.{split} must be a mapping.")
        for field in ("stage3_csv_paths", "metric_pose_csv_paths"):
            raw_paths = value["data"][split].get(field)
            if not isinstance(raw_paths, list) or not raw_paths:
                raise ValueError(f"data.{split}.{field} must be a non-empty list.")
    manifests = value["data"].get("eye_patch_manifest_paths")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("data.eye_patch_manifest_paths must be a non-empty list.")
    model = value["model"]
    fixed_contract = {
        "architecture": "native_blazegaze",
        "image_shape_chw": [3, 128, 512],
        "embedding_size": 512,
        "gaze_mlp_dims": [518, 16, 16, 2],
    }
    mismatches = {
        name: (model.get(name), expected)
        for name, expected in fixed_contract.items()
        if model.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            "Native BlazeGaze model fields are fixed; mismatches: "
            f"{mismatches}."
        )
    wandb_config = value["logging"].get("wandb")
    if not isinstance(wandb_config, dict):
        raise ValueError("logging.wandb must be a mapping.")
    for field in ("enabled", "project", "entity", "mode", "tags"):
        if field not in wandb_config:
            raise ValueError(f"logging.wandb requires {field!r}.")
    if not isinstance(wandb_config["enabled"], bool):
        raise ValueError("logging.wandb.enabled must be true or false.")
    if not str(wandb_config["project"]).strip():
        raise ValueError("logging.wandb.project must be non-empty.")
    if not str(wandb_config["mode"]).strip():
        raise ValueError("logging.wandb.mode must be non-empty.")
    if not isinstance(wandb_config["tags"], list):
        raise ValueError("logging.wandb.tags must be a list.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_paths(values: list[str]) -> list[Path]:
    paths = [_resolve_path(value) for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required input files are missing: {missing}")
    return paths


def _resolve_bounds(
    config: Mapping[str, Any], override: list[float] | None
) -> PogBoundsMM:
    raw = override if override is not None else config["data"].get("pog_bounds_mm")
    if raw is None:
        raise ValueError(
            "Native BlazeGaze requires fixed physical/protocol PoG bounds. "
            "Set data.pog_bounds_mm in the YAML or pass "
            "--pog-bounds-mm U_MIN U_MAX V_MIN V_MAX. Do not derive them from "
            "the validation/test extrema."
        )
    return PogBoundsMM.from_sequence(raw)


def _seed_everything(seed: int, deterministic: bool) -> None:
    if deterministic and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _make_adapter(
    config: Mapping[str, Any],
    split: str,
    bounds: PogBoundsMM,
) -> BlazeGazeDataAdapter:
    split_config = config["data"][split]
    return BlazeGazeDataAdapter(
        stage3_csv_paths=_resolve_paths(split_config["stage3_csv_paths"]),
        metric_pose_csv_paths=_resolve_paths(
            split_config["metric_pose_csv_paths"]
        ),
        pog_bounds_mm=bounds,
        image_layout="CHW",
        eye_patch_manifest_paths=_resolve_paths(
            config["data"]["eye_patch_manifest_paths"]
        ),
        require_exact_eye_patches=True,
        strict_join=True,
        validate_image_paths=True,
    )


def _validate_bounds(adapter: BlazeGazeDataAdapter, split: str) -> None:
    labels = torch.stack(
        [adapter.pog_norm_at(index) for index in range(len(adapter))]
    )
    invalid = ((labels < -0.5) | (labels > 0.5)).any(dim=-1)
    if invalid.any():
        first = int(torch.nonzero(invalid, as_tuple=False)[0])
        raise ValueError(
            f"{split} sample {adapter.sample_id_at(first)!r} lies outside the "
            "configured physical PoG bounds. Fix the bounds rather than "
            "clipping labels."
        )


def _configure_sample_weights(
    config: Mapping[str, Any],
    train_adapter: BlazeGazeDataAdapter,
    val_adapter: BlazeGazeDataAdapter,
) -> str:
    mode = str(config["data"].get("sample_weighting", "uniform")).strip().lower()
    if mode == "uniform":
        return mode
    if mode != "official_inverse_frequency_30x30":
        raise ValueError(
            "data.sample_weighting must be 'uniform' or "
            "'official_inverse_frequency_30x30'."
        )
    bins = int(config["data"].get("weight_bins", 30))
    if bins != 30:
        raise ValueError("The official Native BlazeGaze weight grid uses 30 bins.")
    grid = fit_official_weight_grid((train_adapter,), bins=bins)
    grid.assign(train_adapter)
    grid.assign(val_adapter)
    return mode


def _make_loader(
    adapter: BlazeGazeDataAdapter,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        adapter,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def _tensor(batch: Mapping[str, object], key: str, device: torch.device) -> Tensor:
    value = batch.get(key)
    if not torch.is_tensor(value):
        raise TypeError(f"Batch field {key!r} must be a tensor.")
    return value.to(device=device, non_blocking=True)


def _run_epoch(
    *,
    model: NativeBlazeGaze,
    loader: DataLoader,
    criterion: NativeBlazeGazeStage1Loss,
    bounds: PogBoundsMM,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float | None,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {
        "loss": 0.0,
        "gaze_l2": 0.0,
        "reconstruction_mse": 0.0,
        "embedding_consistency": 0.0,
        "epe_mm": 0.0,
    }
    sample_count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader):
            image = _tensor(batch, "image", device).float()
            head_vector = _tensor(batch, "head_vector", device).float()
            face_origin = _tensor(batch, "face_origin_3d", device).float()
            target = _tensor(batch, "pog_norm", device).float()
            target_mm = _tensor(batch, "uv_gt_mm", device).float()
            sample_weight = _tensor(batch, "sample_weight", device).float()
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(
                image,
                head_vector,
                face_origin,
                return_reconstruction=(
                    criterion.config.reconstruction_weight > 0
                ),
            )
            losses = criterion(
                outputs,
                image_target=image,
                pog_target=target,
                sample_weight=sample_weight,
            )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(
                    f"Non-finite Native BlazeGaze loss at batch {batch_index}."
                )
            if training:
                losses["loss"].backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            batch_size = int(image.shape[0])
            sample_count += batch_size
            for name in (
                "loss",
                "gaze_l2",
                "reconstruction_mse",
                "embedding_consistency",
            ):
                sums[name] += float(losses[name].detach()) * batch_size
            sums["epe_mm"] += float(
                mean_pog_error_mm(outputs["pog_pred"], target_mm, bounds)
            ) * batch_size
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    if sample_count == 0:
        raise RuntimeError("Native BlazeGaze loader produced no samples.")
    return {name: value / sample_count for name, value in sums.items()}


def _write_metrics_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writeheader()


def _append_metrics(path: Path, values: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writerow(values)


def _save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_payload(
    *,
    model: NativeBlazeGaze,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    bounds: PogBoundsMM,
    val_metrics: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "schema_version": "native_blazegaze_supervised_v1",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": vars(model.config),
        "pog_bounds_mm": [
            bounds.u_min,
            bounds.u_max,
            bounds.v_min,
            bounds.v_max,
        ],
        "val_metrics": dict(val_metrics),
        "config": dict(config),
    }


def _wandb_epoch_metrics(
    record: Mapping[str, float | int],
    *,
    best_val_loss: float,
    best_val_epe_mm: float,
    best_epoch: int,
) -> dict[str, float | int]:
    """Map the local CSV schema to stable W&B metric names."""

    return {
        "epoch": int(record["epoch"]),
        "optimizer/learning_rate": float(record["learning_rate"]),
        "train/loss": float(record["train_loss"]),
        "train/gaze_l2": float(record["train_gaze_l2"]),
        "train/reconstruction_mse": float(
            record["train_reconstruction_mse"]
        ),
        "train/embedding_consistency": float(
            record["train_embedding_consistency"]
        ),
        "train/epe_mm": float(record["train_epe_mm"]),
        "val/loss": float(record["val_loss"]),
        "val/gaze_l2": float(record["val_gaze_l2"]),
        "val/reconstruction_mse": float(record["val_reconstruction_mse"]),
        "val/embedding_consistency": float(
            record["val_embedding_consistency"]
        ),
        "val/epe_mm": float(record["val_epe_mm"]),
        "time/epoch_seconds": float(record["epoch_seconds"]),
        "checkpoint/best_val_loss": float(best_val_loss),
        "checkpoint/best_val_epe_mm": float(best_val_epe_mm),
        "checkpoint/best_epoch": int(best_epoch),
    }


def _start_wandb_run(
    *,
    config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    run_name: str,
    run_dir: Path,
    setup: Mapping[str, object],
) -> Any | None:
    wandb_config = config["logging"]["wandb"]
    if not bool(wandb_config["enabled"]):
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "W&B logging is enabled but wandb is not installed. Run "
            "'python -m pip install -r baselines/blazegaze/requirements.txt'."
        ) from exc
    run = wandb.init(
        project=str(wandb_config["project"]),
        entity=wandb_config.get("entity"),
        mode=str(wandb_config.get("mode", "online")),
        name=run_name,
        group=wandb_config.get("group"),
        tags=[str(value) for value in wandb_config.get("tags", [])],
        dir=str(run_dir),
        config={"native_blazegaze": dict(resolved_config), "setup": dict(setup)},
    )
    run.define_metric("epoch")
    for split in ("train", "val"):
        for name in (
            "loss",
            "gaze_l2",
            "reconstruction_mse",
            "embedding_consistency",
            "epe_mm",
        ):
            run.define_metric(
                f"{split}/{name}", step_metric="epoch", summary="min"
            )
    run.define_metric(
        "optimizer/learning_rate", step_metric="epoch", summary="min"
    )
    run.define_metric(
        "time/epoch_seconds", step_metric="epoch", summary="mean"
    )
    run.define_metric(
        "checkpoint/best_val_loss", step_metric="epoch", summary="min"
    )
    run.define_metric(
        "checkpoint/best_val_epe_mm", step_metric="epoch", summary="min"
    )
    return run


def main() -> int:
    args = parse_args()
    config_path = _resolve_path(args.config)
    config = _load_config(config_path)
    bounds = _resolve_bounds(config, args.pog_bounds_mm)
    seed = int(config["experiment"].get("seed", 42))
    deterministic = bool(config["experiment"].get("deterministic", True))
    _seed_everything(seed, deterministic)
    device_name = args.device or str(config["training"].get("device", "auto"))
    device = _resolve_device(device_name)

    train_adapter = _make_adapter(config, "train", bounds)
    val_adapter = _make_adapter(config, "val", bounds)
    _validate_bounds(train_adapter, "train")
    _validate_bounds(val_adapter, "val")
    weighting_mode = _configure_sample_weights(
        config, train_adapter, val_adapter
    )

    batch_size = int(config["data"].get("batch_size", 8))
    num_workers = int(config["data"].get("num_workers", 0))
    pin_memory = bool(config["data"].get("pin_memory", device.type == "cuda"))
    train_loader = _make_loader(
        train_adapter,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
    val_loader = _make_loader(
        val_adapter,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed + 1,
    )

    model_config = NativeBlazeGazeConfig(
        use_decoder=bool(config["model"].get("use_decoder", True))
    )
    model = NativeBlazeGaze(model_config).to(device)
    loss_config = NativeBlazeGazeLossConfig(
        gaze_weight=float(config["loss"].get("gaze_weight", 1.0)),
        reconstruction_weight=float(
            config["loss"].get("reconstruction_weight", 1.0)
        ),
        consistency_weight=float(
            config["loss"].get("consistency_weight", 0.2)
        ),
    )
    if loss_config.reconstruction_weight > 0 and model.decoder is None:
        raise ValueError("A reconstruction loss requires model.use_decoder=true.")
    criterion = NativeBlazeGazeStage1Loss(loss_config)
    initial_lr = float(config["training"].get("learning_rate", 1e-3))
    if initial_lr <= 0:
        raise ValueError("training.learning_rate must be positive.")
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    decay_rate = float(config["training"].get("decay_rate", 0.95))
    if not 0 < decay_rate <= 1:
        raise ValueError("training.decay_rate must be in (0, 1].")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=decay_rate
    )
    grad_clip_raw = config["training"].get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_raw is None else float(grad_clip_raw)
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("training.grad_clip_norm must be positive or null.")

    inference_parameters = model.inference_parameter_count()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Native BlazeGaze | device={device} | train={len(train_adapter)} | "
        f"val={len(val_adapter)} | batch={batch_size} | "
        f"inference_params={inference_parameters:,} | "
        f"stage1_params={total_parameters:,} | weighting={weighting_mode}"
    )

    if args.dry_run:
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            bounds=bounds,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            max_batches=1,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            bounds=bounds,
            device=device,
            optimizer=None,
            grad_clip_norm=None,
            max_batches=1,
        )
        print(
            "Dry run passed | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_epe_mm={val_metrics['epe_mm']:.3f}"
        )
        return 0

    epochs = int(args.epochs or config["training"].get("epochs", 20))
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    run_name = args.run_name or str(
        config["experiment"].get("run_name", "native_blazegaze_supervised")
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    full_run_name = f"{run_name}_{timestamp}"
    output_root = _resolve_path(
        args.output_root
        or config["experiment"].get("output_root", "outputs/blazegaze_native")
    )
    run_dir = output_root / full_run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    metrics_path = run_dir / "metrics.csv"
    _write_metrics_header(metrics_path)
    with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        resolved_config = dict(config)
        resolved_config["data"] = dict(config["data"])
        resolved_config["data"]["pog_bounds_mm"] = [
            bounds.u_min,
            bounds.u_max,
            bounds.v_min,
            bounds.v_max,
        ]
        yaml.safe_dump(resolved_config, handle, sort_keys=False, allow_unicode=True)

    wandb_run = _start_wandb_run(
        config=config,
        resolved_config=resolved_config,
        run_name=full_run_name,
        run_dir=run_dir,
        setup={
            "device": str(device),
            "train_samples": len(train_adapter),
            "val_samples": len(val_adapter),
            "batch_size": batch_size,
            "sample_weighting": weighting_mode,
            "inference_parameters": inference_parameters,
            "stage1_parameters_with_decoder": total_parameters,
        },
    )
    best_val_loss = float("inf")
    best_val_epe_mm = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            bounds=bounds,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            bounds=bounds,
            device=device,
            optimizer=None,
            grad_clip_norm=None,
        )
        elapsed = perf_counter() - started
        learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
            "epoch_seconds": elapsed,
        }
        _append_metrics(metrics_path, record)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            bounds=bounds,
            val_metrics=val_metrics,
            config=config,
        )
        _save_checkpoint(checkpoint_dir / "last.pt", payload)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            _save_checkpoint(checkpoint_dir / "best.pt", payload)
        best_val_epe_mm = min(best_val_epe_mm, val_metrics["epe_mm"])
        if wandb_run is not None:
            wandb_run.log(
                _wandb_epoch_metrics(
                    record,
                    best_val_loss=best_val_loss,
                    best_val_epe_mm=best_val_epe_mm,
                    best_epoch=best_epoch,
                ),
                step=epoch,
            )
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_epe_mm={val_metrics['epe_mm']:.3f} | "
            f"best={best_val_loss:.6f}@{best_epoch} | {elapsed:.1f}s"
        )

    summary = {
        "schema_version": "native_blazegaze_supervised_run_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "train_samples": len(train_adapter),
        "val_samples": len(val_adapter),
        "pog_bounds_mm": [
            bounds.u_min,
            bounds.u_max,
            bounds.v_min,
            bounds.v_max,
        ],
        "sample_weighting": weighting_mode,
        "inference_parameters": inference_parameters,
        "stage1_parameters_with_decoder": total_parameters,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_epe_mm": best_val_epe_mm,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
    }
    if wandb_run is not None:
        wandb_url = getattr(wandb_run, "url", None)
        if wandb_url:
            summary["wandb_url"] = str(wandb_url)
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["best_val_epe_mm"] = best_val_epe_mm
        wandb_run.summary["best_checkpoint"] = str(checkpoint_dir / "best.pt")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Training complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
