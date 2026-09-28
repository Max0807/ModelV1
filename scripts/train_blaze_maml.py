"""B3: frozen Blaze representation plus first-order MAML personalization."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.blaze_maml import (
    BlazeMAMLGazeHead,
    adapt_blaze_maml_head,
    blaze_maml_l2_loss,
    functional_head_prediction,
    sample_support_query_indices,
)
from modelv1.blaze_style_model import BlazeStyleModel, BlazeStyleModelConfig
from modelv1.data import build_modelv1_dataloaders
from modelv1.data.normalization import UVTargetNormalizer
from modelv1.model_factory import (
    MODEL_ARCHITECTURE_BLAZE_STYLE,
    build_experiment_model,
    canonical_model_architecture,
)
from scripts.train_modelv1 import (
    checkpoint_model_state,
    make_model_config,
    resolve_device,
    resolve_project_path,
    resolve_table7_stage3_artifacts,
    seed_everything,
    validate_config,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "modelv1"
    / "ablations"
    / "train_d13_val_d11_blaze_style_maml.yaml"
)


@dataclass(frozen=True)
class CachedBlazeSplit:
    visual_features: Tensor
    table_frame7_n: Tensor
    targets_normalized: Tensor
    targets_mm: Tensor
    identities: tuple[str, ...]
    sample_ids: tuple[str, ...]
    indices_by_identity: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        sample_count = int(self.visual_features.shape[0])
        if self.visual_features.ndim != 2:
            raise ValueError("Cached visual features must have shape [N, D].")
        if self.table_frame7_n.shape != (sample_count, 7):
            raise ValueError("Cached TableFrame7 tensor must have shape [N, 7].")
        if self.targets_normalized.shape != (sample_count, 2):
            raise ValueError("Cached normalized targets must have shape [N, 2].")
        if self.targets_mm.shape != (sample_count, 2):
            raise ValueError("Cached millimetre targets must have shape [N, 2].")
        if len(self.identities) != sample_count or len(self.sample_ids) != sample_count:
            raise ValueError("Cached metadata length must equal the sample count.")
        tensors = {
            "visual_features": self.visual_features,
            "table_frame7_n": self.table_frame7_n,
            "targets_normalized": self.targets_normalized,
            "targets_mm": self.targets_mm,
        }
        non_finite = [
            name for name, value in tensors.items() if not torch.isfinite(value).all()
        ]
        if non_finite:
            raise ValueError(f"B3 cache contains non-finite tensors: {non_finite}.")


@dataclass(frozen=True)
class EpisodeBatch:
    support_visual: Tensor
    support_scene: Tensor
    support_targets: Tensor
    support_targets_mm: Tensor
    query_visual: Tensor
    query_scene: Tensor
    query_targets: Tensor
    query_targets_mm: Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_b3_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("B3 config root must be a mapping.")
    for section in (
        "experiment",
        "initialization",
        "data",
        "meta_learning",
        "logging",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"B3 config requires a {section!r} mapping.")
    if str(config["meta_learning"].get("algorithm", "")).strip().lower() != (
        "first_order_maml"
    ):
        raise ValueError("B3 requires meta_learning.algorithm='first_order_maml'.")
    positive_integer_fields = {
        "data.support_size": config["data"].get("support_size"),
        "data.query_size": config["data"].get("query_size"),
        "data.feature_cache_batch_size": config["data"].get(
            "feature_cache_batch_size"
        ),
        "meta_learning.outer_steps": config["meta_learning"].get("outer_steps"),
        "meta_learning.inner_steps": config["meta_learning"].get("inner_steps"),
        "meta_learning.validate_every_steps": config["meta_learning"].get(
            "validate_every_steps"
        ),
        "meta_learning.validation_episodes_per_identity": config[
            "meta_learning"
        ].get("validation_episodes_per_identity"),
    }
    for name, value in positive_integer_fields.items():
        if isinstance(value, bool) or int(value or 0) <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("inner_lr", "outer_lr"):
        if float(config["meta_learning"].get(name, 0.0)) <= 0:
            raise ValueError(f"meta_learning.{name} must be positive.")
    allow_single = config["data"].get("allow_single_train_identity", False)
    if not isinstance(allow_single, bool):
        raise ValueError("data.allow_single_train_identity must be boolean.")
    adaptable_components = config["meta_learning"].get("adaptable_components")
    if (
        not isinstance(adaptable_components, list)
        or not adaptable_components
        or not all(str(value).strip() for value in adaptable_components)
    ):
        raise ValueError(
            "meta_learning.adaptable_components must be a non-empty list."
        )
    return config


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"blaze_maml:{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def create_run_dir(config: Mapping[str, Any]) -> Path:
    experiment = config["experiment"]
    run_dir = (
        resolve_project_path(experiment.get("output_dir", "outputs"))
        / str(experiment.get("project", "ModelV1_virtual_camera_cross_camera"))
        / str(experiment["run_name"])
    )
    if run_dir.exists():
        raise FileExistsError(
            f"B3 run directory already exists; choose a new run_name: {run_dir}"
        )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


def load_b2_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    prefer_ema: bool,
) -> tuple[
    BlazeStyleModel,
    UVTargetNormalizer,
    dict[str, Any],
    str,
    Mapping[str, Any],
]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("B2 checkpoint has no embedded training config.")
    validate_config(checkpoint_config)
    architecture = canonical_model_architecture(
        checkpoint_config["model"].get("architecture", "modelv1")
    )
    if architecture != MODEL_ARCHITECTURE_BLAZE_STYLE:
        raise ValueError("B3 initialization checkpoint must use blaze_style.")
    model_config = make_model_config(checkpoint_config["model"])
    if not isinstance(model_config, BlazeStyleModelConfig):
        raise TypeError("B3 requires BlazeStyleModelConfig.")
    model = build_experiment_model(architecture, model_config)
    if not isinstance(model, BlazeStyleModel):
        raise TypeError("B3 expected a BlazeStyleModel checkpoint.")
    state, state_source = checkpoint_model_state(
        checkpoint,
        prefer_ema=prefer_ema,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    normalizer_state = checkpoint.get("normalizer")
    if not isinstance(normalizer_state, Mapping):
        raise ValueError("B2 checkpoint has no UV target normalizer.")
    normalizer = UVTargetNormalizer.from_state_dict(normalizer_state)
    return model, normalizer, checkpoint_config, state_source, checkpoint


def build_feature_loaders(
    checkpoint_config: Mapping[str, Any],
    normalizer: UVTargetNormalizer,
    b3_config: Mapping[str, Any],
    run_dir: Path,
) -> tuple[Any, Any]:
    data_config = checkpoint_config["data"]
    model_config = make_model_config(checkpoint_config["model"])
    if data_config.get("table7_stage3_csv_paths") is None:
        raise ValueError("B3 currently requires Table7 Stage3 CSV inputs.")
    dataset_csv_path = resolve_table7_stage3_artifacts(data_config, run_dir)
    pseudo_paths = data_config.get("eye_geometry_pseudo_label_paths")
    resolved_pseudo_paths = (
        [resolve_project_path(path) for path in pseudo_paths]
        if pseudo_paths
        else None
    )
    loader_config = b3_config["data"]
    return build_modelv1_dataloaders(
        csv_path=dataset_csv_path,
        train_datasets=data_config["train_datasets"],
        val_datasets=data_config["val_datasets"],
        split_mode=str(data_config.get("split_mode", "explicit_datasets")),
        batch_size=int(loader_config["feature_cache_batch_size"]),
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(loader_config.get("persistent_workers", False)),
        normalize_images=bool(data_config.get("normalize_images", True)),
        train_paired_eye_transform=None,
        train_face_transform=None,
        normalize_uv_targets=True,
        target_normalizer=normalizer,
        use_eye_geometry=False,
        use_gaze_geometry=False,
        scene_representation=model_config.scene_representation,
        image_source=str(data_config.get("image_source", "legacy")),
        direct_uv_target_frame=str(
            data_config.get("direct_uv_target_frame", "table_local")
        ),
        use_table_frame_film=model_config.use_table_frame_film,
        precomputed_table_frame7=True,
        eye_geometry_pseudo_label_paths=resolved_pseudo_paths,
        require_eye_geometry_pseudo_labels=bool(
            data_config.get("require_eye_geometry_pseudo_labels", False)
        ),
    )


def cache_blaze_split(
    model: BlazeStyleModel,
    loader: Any,
    device: torch.device,
    *,
    identity_field: str,
) -> CachedBlazeSplit:
    visual_features: list[Tensor] = []
    scenes: list[Tensor] = []
    targets_normalized: list[Tensor] = []
    targets_mm: list[Tensor] = []
    identities: list[str] = []
    sample_ids: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            # Own all CPU metadata before the comparatively slow GPU forward.
            # This prevents persistent workers from recycling pinned buffers.
            scene_cpu = batch["table_frame7_n"].detach().float().clone()
            target_normalized_cpu = (
                batch["uv_target"].detach().float().clone()
            )
            target_mm_cpu = batch["uv_gt"].detach().float().clone()
            batch_identities = batch.get(identity_field)
            if not isinstance(batch_identities, (list, tuple)):
                raise TypeError(
                    f"Batch identity field {identity_field!r} must be strings."
                )
            batch_sample_ids = batch.get("sample_id")
            if not isinstance(batch_sample_ids, (list, tuple)):
                raise TypeError("Batch sample_id metadata must be strings.")

            face = batch["face"].to(device)
            left_eye = batch["left_eye"].to(device)
            right_eye = batch["right_eye"].to(device)
            outputs = model.visual_encoder(
                face,
                left_eye,
                right_eye,
            )
            visual_features.append(
                outputs["visual_features"].detach().float().cpu().clone()
            )
            # DataLoader workers may reuse pinned-memory batch buffers. Clone
            # every CPU tensor so the frozen feature cache owns its storage.
            scenes.append(scene_cpu)
            targets_normalized.append(target_normalized_cpu)
            targets_mm.append(target_mm_cpu)
            identities.extend(str(value) for value in batch_identities)
            sample_ids.extend(str(value) for value in batch_sample_ids)

    groups: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        groups.setdefault(identity, []).append(index)
    return CachedBlazeSplit(
        visual_features=torch.cat(visual_features),
        table_frame7_n=torch.cat(scenes),
        targets_normalized=torch.cat(targets_normalized),
        targets_mm=torch.cat(targets_mm),
        identities=tuple(identities),
        sample_ids=tuple(sample_ids),
        indices_by_identity={
            identity: tuple(indices) for identity, indices in groups.items()
        },
    )


def valid_task_identities(
    cache: CachedBlazeSplit,
    *,
    support_size: int,
    query_size: int,
) -> tuple[str, ...]:
    required = support_size + query_size
    return tuple(
        sorted(
            identity
            for identity, indices in cache.indices_by_identity.items()
            if len(indices) >= required
        )
    )


def build_episode(
    cache: CachedBlazeSplit,
    identity: str,
    *,
    support_size: int,
    query_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> EpisodeBatch:
    support_indices, query_indices = sample_support_query_indices(
        cache.indices_by_identity[identity],
        support_size=support_size,
        query_size=query_size,
        generator=generator,
    )

    def gather(values: Tensor, indices: Tensor) -> Tensor:
        return values.index_select(0, indices).to(device, non_blocking=True)

    return EpisodeBatch(
        support_visual=gather(cache.visual_features, support_indices),
        support_scene=gather(cache.table_frame7_n, support_indices),
        support_targets=gather(cache.targets_normalized, support_indices),
        support_targets_mm=gather(cache.targets_mm, support_indices),
        query_visual=gather(cache.visual_features, query_indices),
        query_scene=gather(cache.table_frame7_n, query_indices),
        query_targets=gather(cache.targets_normalized, query_indices),
        query_targets_mm=gather(cache.targets_mm, query_indices),
    )


def epe_mm(
    predictions_normalized: Tensor,
    targets_mm: Tensor,
    normalizer: UVTargetNormalizer,
) -> float:
    predictions_mm = normalizer.denormalize(predictions_normalized.float())
    return float(torch.linalg.vector_norm(predictions_mm - targets_mm, dim=-1).mean())


def evaluate_personalization(
    head: BlazeMAMLGazeHead,
    cache: CachedBlazeSplit,
    identities: tuple[str, ...],
    *,
    support_size: int,
    query_size: int,
    inner_lr: float,
    inner_steps: int,
    episodes_per_identity: int,
    normalizer: UVTargetNormalizer,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    head.eval()
    sums = {
        "support_loss_after": 0.0,
        "query_loss_before": 0.0,
        "query_loss_after": 0.0,
        "query_epe_before_mm": 0.0,
        "query_epe_after_mm": 0.0,
    }
    episode_count = 0
    generator = torch.Generator().manual_seed(seed)
    for identity in identities:
        for _ in range(episodes_per_identity):
            episode = build_episode(
                cache,
                identity,
                support_size=support_size,
                query_size=query_size,
                generator=generator,
                device=device,
            )
            with torch.no_grad():
                query_before = head(
                    episode.query_visual,
                    episode.query_scene,
                )
                query_loss_before = blaze_maml_l2_loss(
                    query_before,
                    episode.query_targets,
                )
            with torch.enable_grad():
                adapted, support_loss_after = adapt_blaze_maml_head(
                    head,
                    episode.support_visual,
                    episode.support_scene,
                    episode.support_targets,
                    inner_lr=inner_lr,
                    inner_steps=inner_steps,
                    first_order=True,
                    detach_initial=True,
                )
            with torch.no_grad():
                query_after = functional_head_prediction(
                    head,
                    adapted,
                    episode.query_visual,
                    episode.query_scene,
                )
                query_loss_after = blaze_maml_l2_loss(
                    query_after,
                    episode.query_targets,
                )
            sums["support_loss_after"] += float(support_loss_after.detach())
            sums["query_loss_before"] += float(query_loss_before)
            sums["query_loss_after"] += float(query_loss_after)
            sums["query_epe_before_mm"] += epe_mm(
                query_before,
                episode.query_targets_mm,
                normalizer,
            )
            sums["query_epe_after_mm"] += epe_mm(
                query_after,
                episode.query_targets_mm,
                normalizer,
            )
            episode_count += 1
    if episode_count == 0:
        raise RuntimeError("B3 validation produced no episodes.")
    metrics = {name: value / episode_count for name, value in sums.items()}
    metrics["query_epe_improvement_mm"] = (
        metrics["query_epe_before_mm"] - metrics["query_epe_after_mm"]
    )
    return metrics


def save_b3_checkpoint(
    path: Path,
    *,
    step: int,
    best_val_query_epe_mm: float,
    head: BlazeMAMLGazeHead,
    optimizer: torch.optim.Optimizer,
    normalizer: UVTargetNormalizer,
    config: Mapping[str, Any],
    base_checkpoint_path: Path,
    base_state_source: str,
    train_identities: tuple[str, ...],
    val_identities: tuple[str, ...],
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "step": step,
        "best_val_query_epe_mm": best_val_query_epe_mm,
        "maml_head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "config": dict(config),
        "base_checkpoint": str(base_checkpoint_path),
        "base_state_source": base_state_source,
        "train_identities": train_identities,
        "val_identities": val_identities,
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    config = load_b3_config(resolve_project_path(args.config))
    if args.device is not None:
        config["meta_learning"]["device"] = args.device
    if args.dry_run:
        config["experiment"]["run_name"] = (
            "dry_run_b3_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        config["meta_learning"]["outer_steps"] = 2
        config["meta_learning"]["validate_every_steps"] = 1
        config["meta_learning"]["validation_episodes_per_identity"] = 1
        config["logging"]["wandb"]["enabled"] = False

    run_dir = create_run_dir(config)
    logger = setup_logger(run_dir / "train.log")
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    device = resolve_device(str(config["meta_learning"].get("device", "auto")))
    checkpoint_path = resolve_project_path(
        config["initialization"]["checkpoint"]
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"B2 checkpoint does not exist: {checkpoint_path}")
    model, normalizer, checkpoint_config, state_source, _ = load_b2_model(
        checkpoint_path,
        device,
        prefer_ema=bool(config["initialization"].get("prefer_ema", True)),
    )
    head = BlazeMAMLGazeHead.from_blaze_style_model(model).to(device)
    head.set_adaptable_components(meta_components := tuple(
        str(value) for value in config["meta_learning"]["adaptable_components"]
    ))
    train_loader, val_loader = build_feature_loaders(
        checkpoint_config,
        normalizer,
        config,
        run_dir,
    )
    identity_field = str(config["data"].get("identity_field", "dataset"))
    cache_started = time.perf_counter()
    train_cache = cache_blaze_split(
        model,
        train_loader,
        device,
        identity_field=identity_field,
    )
    val_cache = cache_blaze_split(
        model,
        val_loader,
        device,
        identity_field=identity_field,
    )
    cache_seconds = time.perf_counter() - cache_started
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    support_size = int(config["data"]["support_size"])
    query_size = int(config["data"]["query_size"])
    train_identities = valid_task_identities(
        train_cache,
        support_size=support_size,
        query_size=query_size,
    )
    val_identities = valid_task_identities(
        val_cache,
        support_size=support_size,
        query_size=query_size,
    )
    if not train_identities or not val_identities:
        raise ValueError(
            "No train/validation identities contain enough samples for the "
            f"configured {support_size}+{query_size} episode."
        )
    single_identity_protocol = len(train_identities) < 2
    if single_identity_protocol and not bool(
        config["data"].get("allow_single_train_identity", False)
    ):
        raise ValueError(
            "B3 found only one train identity. Add participant data or explicitly "
            "set data.allow_single_train_identity=true for a limited protocol."
        )
    if single_identity_protocol:
        warning = (
            "B3 is running with one meta-training identity; this validates the "
            "mechanism but is not sufficient evidence of cross-user MAML."
        )
        logger.warning(warning)
        print(f"WARNING: {warning}")

    meta = config["meta_learning"]
    inner_lr = float(meta["inner_lr"])
    inner_steps = int(meta["inner_steps"])
    outer_steps = int(meta["outer_steps"])
    validate_every = int(meta["validate_every_steps"])
    validation_episodes = int(meta["validation_episodes_per_identity"])
    optimizer = torch.optim.Adam(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=float(meta["outer_lr"]),
        weight_decay=float(meta.get("weight_decay", 0.0)),
    )
    grad_clip_value = meta.get("grad_clip_norm")
    grad_clip_norm = (
        float(grad_clip_value) if grad_clip_value is not None else None
    )
    train_generator = torch.Generator().manual_seed(seed + 1)
    identity_rng = random.Random(seed + 2)

    setup = {
        "device": str(device),
        "base_checkpoint": str(checkpoint_path),
        "base_state_source": state_source,
        "train_samples": len(train_cache.identities),
        "val_samples": len(val_cache.identities),
        "train_identities": train_identities,
        "val_identities": val_identities,
        "single_identity_protocol": single_identity_protocol,
        "support_size": support_size,
        "query_size": query_size,
        "inner_steps": inner_steps,
        "inner_lr": inner_lr,
        "outer_lr": float(meta["outer_lr"]),
        "adaptable_components": meta_components,
        "adaptable_parameters": sum(
            p.numel() for p in head.parameters() if p.requires_grad
        ),
        "cache_seconds": cache_seconds,
    }
    setup_text = json.dumps(setup, ensure_ascii=False)
    logger.info(setup_text)
    print(setup_text)

    wandb_run = None
    wandb_config = config["logging"]["wandb"]
    if bool(wandb_config.get("enabled", False)):
        import wandb

        wandb_run = wandb.init(
            project=wandb_config["project"],
            entity=wandb_config.get("entity"),
            mode=wandb_config.get("mode", "online"),
            name=config["experiment"]["run_name"],
            tags=wandb_config.get("tags", []),
            dir=str(run_dir),
            config={"b3": config, "setup": setup},
        )

    metric_fields = (
        "step",
        "train_support_loss_after",
        "train_query_loss_after",
        "train_grad_norm",
        "val_support_loss_after",
        "val_query_loss_before",
        "val_query_loss_after",
        "val_query_epe_before_mm",
        "val_query_epe_after_mm",
        "val_query_epe_improvement_mm",
        "best_val_query_epe_mm",
        "elapsed_seconds",
    )
    metrics_handle = (run_dir / "metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    )
    writer = csv.DictWriter(metrics_handle, fieldnames=metric_fields)
    writer.writeheader()
    started_at = time.perf_counter()
    baseline_metrics = evaluate_personalization(
        head,
        val_cache,
        val_identities,
        support_size=support_size,
        query_size=query_size,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        episodes_per_identity=validation_episodes,
        normalizer=normalizer,
        device=device,
        seed=int(meta.get("validation_seed", seed + 100_000)),
    )
    best_val_epe = baseline_metrics["query_epe_after_mm"]
    baseline_record = {
        "step": 0,
        "train_support_loss_after": "",
        "train_query_loss_after": "",
        "train_grad_norm": "",
        "val_support_loss_after": baseline_metrics["support_loss_after"],
        "val_query_loss_before": baseline_metrics["query_loss_before"],
        "val_query_loss_after": baseline_metrics["query_loss_after"],
        "val_query_epe_before_mm": baseline_metrics["query_epe_before_mm"],
        "val_query_epe_after_mm": baseline_metrics["query_epe_after_mm"],
        "val_query_epe_improvement_mm": baseline_metrics[
            "query_epe_improvement_mm"
        ],
        "best_val_query_epe_mm": best_val_epe,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    writer.writerow(baseline_record)
    metrics_handle.flush()
    with (run_dir / "baseline_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": "frozen_b2_head_before_meta_training",
                "validation_seed": int(
                    meta.get("validation_seed", seed + 100_000)
                ),
                **baseline_metrics,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    save_b3_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        step=0,
        best_val_query_epe_mm=best_val_epe,
        head=head,
        optimizer=optimizer,
        normalizer=normalizer,
        config=config,
        base_checkpoint_path=checkpoint_path,
        base_state_source=state_source,
        train_identities=train_identities,
        val_identities=val_identities,
    )
    print(
        "step 0000/"
        f"{outer_steps} | query_before="
        f"{baseline_metrics['query_epe_before_mm']:.2f} mm | query_after="
        f"{baseline_metrics['query_epe_after_mm']:.2f} mm | improvement="
        f"{baseline_metrics['query_epe_improvement_mm']:+.2f} mm | "
        f"best={best_val_epe:.2f} mm"
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "step": 0,
                "val/query_loss_before": baseline_metrics[
                    "query_loss_before"
                ],
                "val/query_loss_after": baseline_metrics["query_loss_after"],
                "val/query_epe_before_mm": baseline_metrics[
                    "query_epe_before_mm"
                ],
                "val/query_epe_after_mm": baseline_metrics[
                    "query_epe_after_mm"
                ],
                "val/query_epe_improvement_mm": baseline_metrics[
                    "query_epe_improvement_mm"
                ],
                "checkpoint/best_val_query_epe_mm": best_val_epe,
            },
            step=0,
        )
    try:
        for step in range(1, outer_steps + 1):
            # BlazeGaze's MAML gaze MLP is deterministic; keep the frozen
            # scene/FiLM prefix in evaluation mode while adapting the UV head.
            head.eval()
            identity = identity_rng.choice(train_identities)
            episode = build_episode(
                train_cache,
                identity,
                support_size=support_size,
                query_size=query_size,
                generator=train_generator,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            adapted, support_loss = adapt_blaze_maml_head(
                head,
                episode.support_visual,
                episode.support_scene,
                episode.support_targets,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                first_order=True,
            )
            query_predictions = functional_head_prediction(
                head,
                adapted,
                episode.query_visual,
                episode.query_scene,
            )
            query_loss = blaze_maml_l2_loss(
                query_predictions,
                episode.query_targets,
            )
            if not torch.isfinite(query_loss):
                raise FloatingPointError(
                    f"Non-finite B3 query loss at outer step {step}."
                )
            query_loss.backward()
            if grad_clip_norm is None:
                squared_norm = sum(
                    float(parameter.grad.detach().square().sum())
                    for parameter in head.parameters()
                    if parameter.grad is not None
                )
                train_grad_norm = squared_norm**0.5
            else:
                train_grad_norm = float(
                    nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in head.parameters()
                            if parameter.requires_grad
                        ],
                        grad_clip_norm,
                    )
                )
            optimizer.step()

            if step % validate_every != 0 and step != outer_steps:
                continue
            val_metrics = evaluate_personalization(
                head,
                val_cache,
                val_identities,
                support_size=support_size,
                query_size=query_size,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                episodes_per_identity=validation_episodes,
                normalizer=normalizer,
                device=device,
                seed=int(meta.get("validation_seed", seed + 100_000)),
            )
            improved = val_metrics["query_epe_after_mm"] < best_val_epe
            if improved:
                best_val_epe = val_metrics["query_epe_after_mm"]
            record = {
                "step": step,
                "train_support_loss_after": float(support_loss.detach()),
                "train_query_loss_after": float(query_loss.detach()),
                "train_grad_norm": train_grad_norm,
                "val_support_loss_after": val_metrics["support_loss_after"],
                "val_query_loss_before": val_metrics["query_loss_before"],
                "val_query_loss_after": val_metrics["query_loss_after"],
                "val_query_epe_before_mm": val_metrics[
                    "query_epe_before_mm"
                ],
                "val_query_epe_after_mm": val_metrics["query_epe_after_mm"],
                "val_query_epe_improvement_mm": val_metrics[
                    "query_epe_improvement_mm"
                ],
                "best_val_query_epe_mm": best_val_epe,
                "elapsed_seconds": time.perf_counter() - started_at,
            }
            writer.writerow(record)
            metrics_handle.flush()
            logger.info(json.dumps(record))
            print(
                f"step {step:04d}/{outer_steps} | "
                f"query_before={record['val_query_epe_before_mm']:.2f} mm | "
                f"query_after={record['val_query_epe_after_mm']:.2f} mm | "
                f"improvement={record['val_query_epe_improvement_mm']:+.2f} mm | "
                f"best={best_val_epe:.2f} mm"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "train/support_loss_after": record[
                            "train_support_loss_after"
                        ],
                        "train/query_loss_after": record[
                            "train_query_loss_after"
                        ],
                        "train/grad_norm": train_grad_norm,
                        "val/query_loss_before": record[
                            "val_query_loss_before"
                        ],
                        "val/query_loss_after": record[
                            "val_query_loss_after"
                        ],
                        "val/query_epe_before_mm": record[
                            "val_query_epe_before_mm"
                        ],
                        "val/query_epe_after_mm": record[
                            "val_query_epe_after_mm"
                        ],
                        "val/query_epe_improvement_mm": record[
                            "val_query_epe_improvement_mm"
                        ],
                        "checkpoint/best_val_query_epe_mm": best_val_epe,
                    },
                    step=step,
                )
            checkpoint_kwargs = {
                "step": step,
                "best_val_query_epe_mm": best_val_epe,
                "head": head,
                "optimizer": optimizer,
                "normalizer": normalizer,
                "config": config,
                "base_checkpoint_path": checkpoint_path,
                "base_state_source": state_source,
                "train_identities": train_identities,
                "val_identities": val_identities,
            }
            save_b3_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                **checkpoint_kwargs,
            )
            if improved:
                save_b3_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    **checkpoint_kwargs,
                )
    finally:
        metrics_handle.close()
        if wandb_run is not None:
            wandb_run.finish()

    print(
        f"completed B3 outer_steps={outer_steps} "
        f"best_val_query_epe_mm={best_val_epe:.4f} run_dir={run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
