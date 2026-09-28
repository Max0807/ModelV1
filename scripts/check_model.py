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
from modelv1.geometry_gate import EYE_GEOMETRY_GATE_MODES
from modelv1.scene import SCENE_REPRESENTATIONS, scene_representation_dim
from modelv1.data.depth_prior import EYE_GEOMETRY_REPRESENTATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eye-backbone",
        default="resnet18",
        help=(
            "Eye image backbone: cnn, resnet18, resnet18_3x3, resnet34, resnet50, "
            "resnet101, resnet152."
        ),
    )
    parser.add_argument(
        "--eye-backbone-weights",
        default=None,
        help="Use DEFAULT for ImageNet weights on torchvision ResNet backbones; omit for scratch.",
    )
    parser.add_argument(
        "--use-face-image",
        action="store_true",
        help="Enable VGGFace2 face features and face-eye cross-attention fusion.",
    )
    parser.add_argument(
        "--freeze-face-image-backbone",
        action="store_true",
        help="Freeze the VGGFace2-pretrained Inception-ResNet face backbone.",
    )
    parser.add_argument(
        "--use-eye-geometry",
        action="store_true",
        help="Enable the normalized 6D binocular geometry input.",
    )
    parser.add_argument(
        "--eye-geometry-representation",
        choices=EYE_GEOMETRY_REPRESENTATIONS,
        default="normalized6d",
        help="normalized6d=center/baseline geometry; raw_eye6d=[Lx,Ly,Lz,Rx,Ry,Rz].",
    )
    parser.add_argument(
        "--no-crop-cam",
        action="store_true",
        help="Disable the 36D crop/camera branch for ablation.",
    )
    parser.add_argument(
        "--eye-geometry-gate-mode",
        choices=EYE_GEOMETRY_GATE_MODES,
        default="none",
        help="none=V2.1, fixed=analytical gate, learned_residual=V2.2.",
    )
    parser.add_argument(
        "--scene-representation",
        choices=SCENE_REPRESENTATIONS,
        default="full25",
        help=(
            "full25=legacy scene vector; table_frame7=orientation plus distance; "
            "orientation6d=table orientation only."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    use_eye_geometry = (
        args.use_eye_geometry or args.eye_geometry_gate_mode != "none"
    )
    config = ModelV1Config(
        use_face_image=args.use_face_image,
        freeze_face_image_backbone=args.freeze_face_image_backbone,
        eye_backbone=args.eye_backbone,
        eye_backbone_weights=args.eye_backbone_weights,
        use_crop_cam=not args.no_crop_cam,
        scene_dim=scene_representation_dim(args.scene_representation),
        scene_representation=args.scene_representation,
        use_eye_geometry=use_eye_geometry,
        eye_geometry_representation=args.eye_geometry_representation,
        eye_geometry_gate_mode=args.eye_geometry_gate_mode,
    )
    model = ModelV1(config)
    model.eval()

    batch_size = 4
    batch = {
        "left_eye": torch.randn(batch_size, 3, 36, 60),
        "right_eye": torch.randn(batch_size, 3, 36, 60),
        "scene_vec": torch.randn(batch_size, config.scene_dim),
    }
    if config.use_face_image:
        batch["face"] = torch.randn(batch_size, 3, 160, 160)
    if config.use_crop_cam:
        batch["crop_cam_vec"] = torch.randn(batch_size, config.crop_cam_dim)
    if config.use_eye_geometry:
        batch["eye_geometry_vec"] = torch.randn(
            batch_size,
            config.eye_geometry_dim,
        )
    if config.eye_geometry_gate_mode != "none":
        batch["eye_geometry_confidence"] = torch.rand(batch_size, 1)
        batch["eye_geometry_valid_mask"] = torch.ones(batch_size, 1)
    if config.eye_geometry_gate_mode == "learned_residual":
        batch["eye_geometry_quality_vec"] = torch.randn(
            batch_size,
            config.eye_geometry_quality_dim,
        )

    with torch.no_grad():
        features = model(batch, return_features=True)
        uv = features["uv"]

    print("uv:", tuple(uv.shape))
    print("use_face_image:", config.use_face_image)
    if "face_image_features" in features:
        print("face_image_features:", tuple(features["face_image_features"].shape))
        print("visual_features:", tuple(features["visual_features"].shape))
        print("face_tokens:", tuple(features["face_tokens"].shape))
        print("eye_tokens:", tuple(features["eye_tokens"].shape))
        print(
            "cross_attention_weights:",
            tuple(features["cross_attention_weights"].shape),
        )
    print("eye_backbone:", config.eye_backbone)
    print("use_crop_cam:", config.use_crop_cam)
    print("scene_representation:", config.scene_representation)
    print("use_eye_geometry:", config.use_eye_geometry)
    print("eye_geometry_representation:", config.eye_geometry_representation)
    print("eye_geometry_gate_mode:", config.eye_geometry_gate_mode)
    if "eye_geometry_gate" in features:
        print("eye_geometry_gate:", tuple(features["eye_geometry_gate"].shape))
        print(
            "eye_geometry_gate_range:",
            float(features["eye_geometry_gate"].min()),
            float(features["eye_geometry_gate"].max()),
        )
    if "eye_geometry_gate_delta" in features:
        print(
            "initial_gate_delta_max_abs:",
            float(features["eye_geometry_gate_delta"].abs().max()),
        )
    print("parameters:", sum(param.numel() for param in model.parameters()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
