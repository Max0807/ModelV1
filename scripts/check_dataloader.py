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
    build_modelv1_dataloaders,
    get_uv_target_normalizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-mode",
        choices=["random_80_20", "dataset_5"],
        default="dataset_5",
        help="Dataset split strategy to smoke-test.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_loader, val_loader = build_modelv1_dataloaders(
        batch_size=4,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        deca_cache_path=args.deca_cache,
        require_deca_features=True,
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
    print("uv_gt:", tuple(batch["uv_gt"].shape))
    print("uv_target:", tuple(batch["uv_target"].shape))
    if "deca_feat" in batch:
        print("deca_feat:", tuple(batch["deca_feat"].shape))
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
