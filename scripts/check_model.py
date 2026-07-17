"""Run a synthetic forward pass through the ModelV1 network."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow `python scripts\check_model.py` to import the local package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1 import ModelV1, ModelV1Config


def main() -> int:
    config = ModelV1Config()
    model = ModelV1(config)
    model.eval()

    batch_size = 4
    batch = {
        "deca_feat": torch.randn(batch_size, config.deca_feature_dim),
        "left_eye": torch.randn(batch_size, 3, 36, 60),
        "right_eye": torch.randn(batch_size, 3, 36, 60),
        "crop_cam_vec": torch.randn(batch_size, config.crop_cam_dim),
        "scene_vec": torch.randn(batch_size, config.scene_dim),
    }

    with torch.no_grad():
        uv = model(batch)

    print("uv:", tuple(uv.shape))
    print("parameters:", sum(param.numel() for param in model.parameters()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
