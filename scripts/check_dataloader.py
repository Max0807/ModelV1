"""Print one ModelV1 DataLoader batch for a quick smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts\check_dataloader.py` to import the local package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.data import (
    DEFAULT_DECA_CACHE_PATH,
    DEFAULT_DEPTH_PRIOR_PATH,
    build_modelv1_dataloaders,
    get_eye_geometry_normalizer,
    get_eye_geometry_quality_normalizer,
    get_uv_target_normalizer,
)
from modelv1.geometry_gate import EYE_GEOMETRY_GATE_MODES
from modelv1.data.depth_prior import EYE_GEOMETRY_REPRESENTATIONS
from modelv1.scene import SCENE_REPRESENTATIONS
from modelv1.deca_cache import (
    DECA_FEATURE_REPRESENTATIONS,
    DECA_FEATURE_REPRESENTATION_NONE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deca-feature-representation",
        choices=DECA_FEATURE_REPRESENTATIONS,
        default="full236",
        help="full236=all coarse DECA parameters; geometry156=shape+exp+pose.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random_80_20", "dataset_5", "explicit_datasets"],
        default="dataset_5",
        help="Dataset split strategy to smoke-test.",
    )
    parser.add_argument(
        "--train-datasets",
        nargs="+",
        default=["3", "4"],
        help="Training dataset names/indices for explicit_datasets.",
    )
    parser.add_argument(
        "--val-datasets",
        nargs="+",
        default=["5"],
        help="Validation dataset names/indices for explicit_datasets.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used by random_80_20.",
    )
    parser.add_argument(
        "--deca-cache",
        type=Path,
        default=DEFAULT_DECA_CACHE_PATH,
        help="DECA .npz cache produced by cache_deca_features.py.",
    )
    parser.add_argument(
        "--use-eye-geometry",
        action="store_true",
        help="Load and normalize the V2.1 binocular geometry input.",
    )
    parser.add_argument(
        "--eye-geometry-representation",
        choices=EYE_GEOMETRY_REPRESENTATIONS,
        default="normalized6d",
        help="normalized6d=center/baseline geometry; raw_eye6d=[Lx,Ly,Lz,Rx,Ry,Rz] in mm.",
    )
    parser.add_argument(
        "--depth-prior",
        type=Path,
        default=DEFAULT_DEPTH_PRIOR_PATH,
        help="Offline depth-prior CSV keyed by sample_id.",
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
    train_loader, val_loader = build_modelv1_dataloaders(
        batch_size=4,
        train_datasets=args.train_datasets,
        val_datasets=args.val_datasets,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        deca_cache_path=(
            None
            if args.deca_feature_representation
            == DECA_FEATURE_REPRESENTATION_NONE
            else args.deca_cache
        ),
        require_deca_features=(
            args.deca_feature_representation
            != DECA_FEATURE_REPRESENTATION_NONE
        ),
        use_eye_geometry=use_eye_geometry,
        depth_prior_csv_path=args.depth_prior,
        eye_geometry_gate_mode=args.eye_geometry_gate_mode,
        eye_geometry_representation=args.eye_geometry_representation,
        scene_representation=args.scene_representation,
        deca_feature_representation=args.deca_feature_representation,
    )
    batch = next(iter(train_loader))

    print("split mode:", args.split_mode)
    print("train samples:", len(train_loader.dataset))
    print("val samples:", len(val_loader.dataset))
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("face:", tuple(batch["face"].shape))
    print("left_eye:", tuple(batch["left_eye"].shape))
    print("right_eye:", tuple(batch["right_eye"].shape))
    print("crop_cam_vec:", tuple(batch["crop_cam_vec"].shape))
    print("scene_vec:", tuple(batch["scene_vec"].shape))
    print("scene representation:", args.scene_representation)
    print("DECA feature representation:", args.deca_feature_representation)
    print("eye geometry representation:", args.eye_geometry_representation)
    print("uv_gt:", tuple(batch["uv_gt"].shape))
    print("uv_target:", tuple(batch["uv_target"].shape))
    if "deca_feat" in batch:
        print("deca_feat:", tuple(batch["deca_feat"].shape))
    if "eye_geometry_vec" in batch:
        print("eye_geometry_vec:", tuple(batch["eye_geometry_vec"].shape))
        geometry_normalizer = get_eye_geometry_normalizer(train_loader.dataset)
        if geometry_normalizer is None:
            raise RuntimeError("Expected an eye geometry normalizer.")
        print("eye geometry mean:", geometry_normalizer.mean.tolist())
        print("eye geometry std:", geometry_normalizer.std.tolist())
    for key in (
        "eye_geometry_confidence",
        "eye_geometry_valid_mask",
        "eye_geometry_quality_vec",
    ):
        if key in batch:
            print(f"{key}:", tuple(batch[key].shape))
    quality_normalizer = get_eye_geometry_quality_normalizer(
        train_loader.dataset
    )
    if quality_normalizer is not None:
        print("eye quality mean:", quality_normalizer.mean.tolist())
        print("eye quality std:", quality_normalizer.std.tolist())
    normalizer = get_uv_target_normalizer(train_loader.dataset)
    if normalizer is None:
        raise RuntimeError("Expected a UV target normalizer on the training dataset.")
    recovered_uv = normalizer.denormalize(batch["uv_target"])
    print("uv mean mm:", normalizer.mean_mm.tolist())
    print("uv std mm:", normalizer.std_mm.tolist())
    print("uv round-trip max error:", (recovered_uv - batch["uv_gt"]).abs().max().item())
    print("sample ids:", batch["sample_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
