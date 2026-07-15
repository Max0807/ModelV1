"""Print one ModelV1 DataLoader batch for a quick smoke test."""

from __future__ import annotations

from modelv1.data import build_modelv1_dataloaders


def main() -> int:
    train_loader, val_loader = build_modelv1_dataloaders(batch_size=4)
    batch = next(iter(train_loader))

    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("face:", tuple(batch["face"].shape))
    print("left_eye:", tuple(batch["left_eye"].shape))
    print("right_eye:", tuple(batch["right_eye"].shape))
    print("crop_cam_vec:", tuple(batch["crop_cam_vec"].shape))
    print("scene_vec:", tuple(batch["scene_vec"].shape))
    print("uv_gt:", tuple(batch["uv_gt"].shape))
    print("sample ids:", batch["sample_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
