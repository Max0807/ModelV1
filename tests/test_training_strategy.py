from __future__ import annotations

import pytest
import torch
from torch import nn

from scripts.train_modelv1 import (
    ExponentialMovingAverage,
    GRADIENT_DIAGNOSTIC_FIELDS,
    MetricsCsvWriter,
    build_training_optimizer,
    build_training_scheduler,
    canonical_amp_dtype_name,
    checkpoint_model_state,
    metric_fields,
    optimizer_group_learning_rates,
    optimizer_group_parameter_counts,
    load_checkpoint,
    resolve_amp_settings,
    save_checkpoint,
    scheduler_steps_per_optimizer_step,
)


def warmup_training_config() -> dict[str, object]:
    return {
        "epochs": 10,
        "eye_backbone_schedule": {
            "enabled": True,
            "frozen_epochs": 0,
            "unfreeze_from": "stem",
            "lr_multiplier": 0.05,
        },
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 1e-4,
        },
        "scheduler": {
            "name": "cosine",
            "eta_min": 1e-6,
            "warmup_epochs": 2,
            "warmup_start_factor": 0.1,
        },
    }


def test_step_scheduler_warms_up_then_preserves_lr_ratios_during_cosine() -> None:
    parameters = [nn.Parameter(torch.ones(())), nn.Parameter(torch.ones(()))]
    optimizer = torch.optim.SGD(
        [
            {"params": [parameters[0]], "lr": 3e-4},
            {"params": [parameters[1]], "lr": 1e-5},
        ]
    )
    config = warmup_training_config()
    scheduler = build_training_scheduler(
        optimizer,
        config,
        steps_per_epoch=5,
    )
    assert scheduler_steps_per_optimizer_step(config)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-6)

    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-5)

    for _ in range(40):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-5 / 300.0)


def test_bf16_amp_settings_disable_scaling_on_cpu_fallback() -> None:
    enabled, dtype_name, dtype = resolve_amp_settings(
        {"amp": True, "amp_dtype": "bf16"},
        torch.device("cpu"),
    )
    assert not enabled
    assert dtype_name == "bfloat16"
    assert dtype == torch.bfloat16
    assert canonical_amp_dtype_name("fp16") == "float16"
    with pytest.raises(ValueError, match="amp_dtype"):
        canonical_amp_dtype_name("float32")


def test_bf16_amp_rejects_unsupported_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError, match="does not report BF16 support"):
        resolve_amp_settings(
            {"amp": True, "amp_dtype": "bfloat16"},
            torch.device("cuda"),
        )


def test_fused_adamw_falls_back_cleanly_on_cpu() -> None:
    model = nn.Linear(4, 2)
    optimizer = build_training_optimizer(
        model,  # type: ignore[arg-type]
        {
            "optimizer": {
                "name": "adamw",
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "fused": True,
            }
        },
        torch.device("cpu"),
    )
    assert optimizer.defaults["fused"] is False


def test_optimizer_group_diagnostics_preserve_names_lrs_and_counts() -> None:
    parameters = [nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3))]
    optimizer = torch.optim.SGD(
        [
            {"params": [parameters[0]], "lr": 3e-4, "group_name": "head"},
            {"params": [parameters[1]], "lr": 3e-5, "group_name": "face"},
        ]
    )

    assert optimizer_group_learning_rates(optimizer) == {
        "head": pytest.approx(3e-4),
        "face": pytest.approx(3e-5),
    }
    assert optimizer_group_parameter_counts(optimizer) == {"head": 2, "face": 3}
    fields = metric_fields(("epe_mm",), ("head", "face"))
    assert "lr_head" in fields
    assert "lr_face" in fields
    assert "train_grad_norm_pre_clip_mean" in fields
    assert "train_grad_clipped_step_fraction" in fields


def test_metrics_writer_keeps_legacy_schema_when_resuming(tmp_path) -> None:
    expected_fields = metric_fields(("epe_mm",), ("head", "face"))
    legacy_fields = [
        field
        for field in expected_fields
        if field not in GRADIENT_DIAGNOSTIC_FIELDS
        and field not in {"lr_head", "lr_face"}
    ]
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(",".join(legacy_fields) + "\n", encoding="utf-8")

    writer = MetricsCsvWriter(
        metrics_path,
        ("epe_mm",),
        ("head", "face"),
    )
    try:
        assert writer.fields == legacy_fields
    finally:
        writer.close()


def test_ema_starts_after_warmup_and_averages_model_weights() -> None:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ExponentialMovingAverage(
        model,
        decay=0.5,
        update_after_step=1,
    )

    with torch.no_grad():
        model.weight.fill_(1.0)
    assert not ema.update(model)
    torch.testing.assert_close(
        ema.module.weight,
        torch.zeros_like(model.weight),
    )

    with torch.no_grad():
        model.weight.fill_(3.0)
    assert ema.update(model)
    torch.testing.assert_close(
        ema.module.weight,
        torch.full_like(model.weight, 3.0),
    )

    with torch.no_grad():
        model.weight.fill_(5.0)
    assert ema.update(model)
    torch.testing.assert_close(
        ema.module.weight,
        torch.full_like(model.weight, 4.0),
    )
    assert ema.optimization_steps == 3
    assert ema.num_updates == 2


def test_checkpoint_model_state_prefers_ema_when_available() -> None:
    raw = {"weight": torch.tensor([1.0])}
    averaged = {"weight": torch.tensor([2.0])}
    checkpoint = {
        "model": raw,
        "ema": {"model": averaged, "num_updates": 1},
    }

    state, source = checkpoint_model_state(checkpoint, prefer_ema=True)
    assert state is averaged
    assert source == "ema"
    state, source = checkpoint_model_state(checkpoint, prefer_ema=False)
    assert state is raw
    assert source == "raw"

    state, source = checkpoint_model_state(
        {"model": raw, "ema": {"model": averaged, "num_updates": 0}},
        prefer_ema=True,
    )
    assert state is raw
    assert source == "raw"


def test_checkpoint_round_trip_restores_raw_and_ema_weights(tmp_path) -> None:
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    ema = ExponentialMovingAverage(model, decay=0.5, update_after_step=0)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        epoch=2,
        global_step=2,
        best_val_epe_mm=12.0,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        normalizer=None,
        eye_geometry_normalizer=None,
        eye_geometry_quality_normalizer=None,
        depth_correction_geometry_normalizer=None,
        config={},
    )

    restored_model = nn.Linear(1, 1, bias=False)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=1e-2)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer,
        step_size=1,
    )
    restored_scaler = torch.cuda.amp.GradScaler(enabled=False)
    restored_ema = ExponentialMovingAverage(
        restored_model,
        decay=0.5,
        update_after_step=0,
    )
    next_epoch, global_step, best_epe = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        normalizer=None,
        eye_geometry_normalizer=None,
        eye_geometry_quality_normalizer=None,
        depth_correction_geometry_normalizer=None,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        restored_model.weight,
        torch.full_like(restored_model.weight, 3.0),
    )
    torch.testing.assert_close(
        restored_ema.module.weight,
        torch.full_like(restored_model.weight, 2.0),
    )
    assert (next_epoch, global_step, best_epe) == (3, 2, 12.0)
    assert restored_ema.num_updates == 2
