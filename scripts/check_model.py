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
from modelv1.model import DECA_BRANCH_MODES
from modelv1.geometry_gate import EYE_GEOMETRY_GATE_MODES
from modelv1.scene import SCENE_REPRESENTATIONS, scene_representation_dim
from modelv1.deca_cache import DECA_FEATURE_REPRESENTATIONS, deca_feature_representation_dim
from modelv1.data.depth_prior import EYE_GEOMETRY_REPRESENTATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deca-feature-representation",
        choices=DECA_FEATURE_REPRESENTATIONS,
        default="full236",
        help="full236=all coarse DECA parameters; geometry156=shape+exp+pose.",
    )
    parser.add_argument(
        "--deca-branch-mode",
        choices=DECA_BRANCH_MODES,
        default="flat",
        help="flat=one DECA MLP; factorized_geometry=separate shape/exp/pose MLPs.",
    )
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
        help="Enable the V3 RGB face encoder and face-eye visual fusion.",
    )
    parser.add_argument(
        "--face-image-backbone",
        default="resnet18",
        help="Face image backbone: resnet18, resnet34, or resnet50.",
    )
    parser.add_argument(
        "--face-image-backbone-weights",
        default=None,
        help="Use DEFAULT for ImageNet weights on the face ResNet; omit for scratch.",
    )
    parser.add_argument(
        "--face-image-freeze-until",
        default="none",
        help="Freeze the face backbone through: none, stem, layer1, layer2, layer3, or layer4.",
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
        deca_feature_dim=deca_feature_representation_dim(args.deca_feature_representation),
        deca_feature_representation=args.deca_feature_representation,
        deca_branch_mode=args.deca_branch_mode,
        use_face_image=args.use_face_image,
        face_image_backbone=args.face_image_backbone,
        face_image_backbone_weights=args.face_image_backbone_weights,
        face_image_freeze_until=args.face_image_freeze_until,
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
        "deca_feat": torch.randn(batch_size, config.deca_feature_dim),
        "left_eye": torch.randn(batch_size, 3, 36, 60),
        "right_eye": torch.randn(batch_size, 3, 36, 60),
        "scene_vec": torch.randn(batch_size, config.scene_dim),
    }
    if config.use_face_image:
        batch["face"] = torch.randn(batch_size, 3, 224, 224)
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
    print("eye_backbone:", config.eye_backbone)
    print("use_crop_cam:", config.use_crop_cam)
    print("scene_representation:", config.scene_representation)
    print("deca_feature_representation:", config.deca_feature_representation)
    print("deca_branch_mode:", config.deca_branch_mode)
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
