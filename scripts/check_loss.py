"""Smoke-test normalized UV targets, physical loss, metrics, and model output."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from modelv1 import ModelV1, UVRegressionLoss, predict_uv_mm
from modelv1.data import build_modelv1_dataloaders, get_uv_target_normalizer


def main() -> int:
    train_loader, _ = build_modelv1_dataloaders(batch_size=4)
    normalizer = get_uv_target_normalizer(train_loader.dataset)
    if normalizer is None:
        raise RuntimeError("Expected a UV target normalizer on the training dataset.")

    batch = next(iter(train_loader))
    model = ModelV1().eval()
    with torch.no_grad():
        uv_pred = model(batch)
    criterion = UVRegressionLoss(normalizer)
    loss = criterion(uv_pred, batch["uv_target"])
    metrics = criterion.metrics(uv_pred, batch["uv_gt"])
    uv_pred_mm = predict_uv_mm(model, batch, normalizer)

    print("uv_pred normalized:", tuple(uv_pred.shape))
    print("uv_target normalized:", tuple(batch["uv_target"].shape))
    print("smooth_l1_normalized:", float(loss))
    print("uv_pred mm:", tuple(uv_pred_mm.shape))
    for name, value in metrics.items():
        print(f"{name}: {float(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
