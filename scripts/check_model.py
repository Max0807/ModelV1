"""Run a synthetic forward pass through the ModelV1 network."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow `python scripts\check_model.py` to import the local package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1 import ModelV1, ModelV1Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eye-backbone",
        default="resnet18",
        help="Eye image backbone: cnn, resnet18, resnet34, resnet50, resnet101, resnet152.",
    )
    parser.add_argument(
        "--eye-backbone-weights",
        default=None,
        help="Use DEFAULT for ImageNet weights on torchvision ResNet backbones; omit for scratch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ModelV1Config(
        eye_backbone=args.eye_backbone,
        eye_backbone_weights=args.eye_backbone_weights,
    )
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
    print("eye_backbone:", config.eye_backbone)
    print("parameters:", sum(param.numel() for param in model.parameters()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
