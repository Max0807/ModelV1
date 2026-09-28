"""Print one ModelV1 DataLoader batch for a quick smoke test."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

# Allow `python scripts\check_dataloader.py` to import the local package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.data import (
    DEFAULT_DEPTH_PRIOR_PATH,
    VirtualCameraManifest,
    build_modelv1_dataloaders,
    get_eye_geometry_normalizer,
    get_eye_geometry_quality_normalizer,
    get_uv_target_normalizer,
)
from modelv1.geometry_gate import EYE_GEOMETRY_GATE_MODES
from modelv1.data.depth_prior import EYE_GEOMETRY_REPRESENTATIONS
from modelv1.scene import SCENE_REPRESENTATIONS
from modelv1.depth_distribution import (
    DEPTH_DISTRIBUTION_LEARNED_REWEIGHT,
    DEPTH_DISTRIBUTION_POINT,
)
from modelv1.model import PREDICTION_MODE_GAZE_GEOMETRY
from scripts.train_modelv1 import (
    load_config,
    make_eye_appearance_augmentation,
    make_model_config,
    resolve_numbered_dataset_artifacts,
    resolve_preprocessed_v2_artifacts,
    resolve_project_path,
    resolve_sample_id_filter_manifest_path,
    resolve_table7_stage3_artifacts,
    resolve_virtual_camera_manifest_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Build the exact DataLoaders described by a training YAML. This is "
            "the recommended mode for preprocessed-v2 data."
        ),
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
    temporary_run: tempfile.TemporaryDirectory[str] | None = None
    if args.config is not None:
        config = load_config(args.config)
        data_config = config["data"]
        model_config = make_model_config(config["model"])
        uses_gaze_geometry = (
            model_config.prediction_mode == PREDICTION_MODE_GAZE_GEOMETRY
        )
        require_depth_uncertainty = uses_gaze_geometry and (
            model_config.depth_distribution_mode != DEPTH_DISTRIBUTION_POINT
            or model_config.use_depth_correction
        )
        require_pnp_quality = uses_gaze_geometry and (
            model_config.depth_distribution_mode
            == DEPTH_DISTRIBUTION_LEARNED_REWEIGHT
            or model_config.use_depth_correction
        )
        temporary_run = tempfile.TemporaryDirectory(
            prefix="modelv1_check_dataloader_"
        )
        run_dir = Path(temporary_run.name)
        uses_table7_stage3 = (
            data_config.get("table7_stage3_csv_paths") is not None
        )
        preprocessed_virtual_manifests: list[Path] = []
        if uses_table7_stage3:
            dataset_csv_path = resolve_table7_stage3_artifacts(
                data_config,
                run_dir,
            )
            depth_prior_csv_path = None
        elif data_config.get("preprocessed_v2_root") is not None:
            (
                dataset_csv_path,
                depth_prior_csv_path,
                preprocessed_virtual_manifests,
            ) = resolve_preprocessed_v2_artifacts(
                data_config,
                run_dir,
                require_depth_prior=(
                    model_config.use_eye_geometry or uses_gaze_geometry
                ),
            )
        else:
            dataset_csv_path, depth_prior_csv_path = (
                resolve_numbered_dataset_artifacts(
                    data_config,
                    run_dir,
                    require_depth_prior=(
                        model_config.use_eye_geometry or uses_gaze_geometry
                    ),
                )
            )
        image_source = str(data_config.get("image_source", "legacy"))
        virtual_camera_manifest_path = resolve_virtual_camera_manifest_path(
            data_config,
            run_dir,
            fallback_paths=(
                preprocessed_virtual_manifests
                if image_source == "virtual_camera"
                else None
            ),
        )
        sample_filter_manifest_path = resolve_sample_id_filter_manifest_path(
            data_config,
            run_dir,
            fallback_paths=(
                preprocessed_virtual_manifests if image_source == "legacy" else None
            ),
        )
        allowed_sample_ids = (
            frozenset(
                VirtualCameraManifest.load(sample_filter_manifest_path).records
            )
            if sample_filter_manifest_path is not None
            else None
        )
        eye_geometry_pseudo_label_paths = [
            resolve_project_path(path)
            for path in data_config.get("eye_geometry_pseudo_label_paths", ())
        ] or None
        train_loader, val_loader = build_modelv1_dataloaders(
            csv_path=dataset_csv_path,
            train_datasets=data_config.get("train_datasets", ("3", "4")),
            val_datasets=data_config.get("val_datasets", ("5",)),
            split_mode=data_config["split_mode"],
            batch_size=min(4, int(data_config["batch_size"])),
            num_workers=0,
            pin_memory=False,
            normalize_images=bool(data_config["normalize_images"]),
            train_paired_eye_transform=make_eye_appearance_augmentation(data_config),
            normalize_uv_targets=not uses_gaze_geometry,
            use_eye_geometry=model_config.use_eye_geometry,
            use_gaze_geometry=uses_gaze_geometry,
            require_depth_uncertainty=require_depth_uncertainty,
            require_pnp_quality=require_pnp_quality,
            use_depth_correction=model_config.use_depth_correction,
            filter_invalid_depth_prior_samples=bool(
                data_config.get("skip_invalid_depth_prior_samples", True)
            ),
            depth_prior_csv_path=depth_prior_csv_path,
            eye_geometry_gate_mode=model_config.eye_geometry_gate_mode,
            eye_geometry_representation=model_config.eye_geometry_representation,
            scene_representation=model_config.scene_representation,
            image_source=image_source,
            virtual_camera_manifest_path=virtual_camera_manifest_path,
            filter_invalid_virtual_camera_samples=bool(
                data_config.get("skip_invalid_virtual_camera_samples", False)
            ),
            allowed_sample_ids=allowed_sample_ids,
            direct_uv_target_frame=str(
                data_config.get("direct_uv_target_frame", "table_local")
            ),
            use_table_frame_film=model_config.use_table_frame_film,
            precomputed_table_frame7=uses_table7_stage3,
            eye_geometry_pseudo_label_paths=(
                eye_geometry_pseudo_label_paths
            ),
            require_eye_geometry_pseudo_labels=bool(
                data_config.get("require_eye_geometry_pseudo_labels", False)
            ),
        )
        split_mode = str(data_config["split_mode"])
        scene_representation = model_config.scene_representation
        eye_representation = model_config.eye_geometry_representation
    else:
        use_eye_geometry = (
            args.use_eye_geometry or args.eye_geometry_gate_mode != "none"
        )
        train_loader, val_loader = build_modelv1_dataloaders(
            batch_size=4,
            train_datasets=args.train_datasets,
            val_datasets=args.val_datasets,
            split_mode="explicit_datasets",
            use_eye_geometry=use_eye_geometry,
            depth_prior_csv_path=args.depth_prior,
            eye_geometry_gate_mode=args.eye_geometry_gate_mode,
            eye_geometry_representation=args.eye_geometry_representation,
            scene_representation=args.scene_representation,
        )
        split_mode = "explicit_datasets"
        scene_representation = args.scene_representation
        eye_representation = args.eye_geometry_representation
    batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("split mode:", split_mode)
    print("train samples:", len(train_loader.dataset))
    print("val samples:", len(val_loader.dataset))
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("face:", tuple(batch["face"].shape))
    print("left_eye:", tuple(batch["left_eye"].shape))
    print("right_eye:", tuple(batch["right_eye"].shape))
    print("crop_cam_vec:", tuple(batch["crop_cam_vec"].shape))
    print("scene_vec:", tuple(batch["scene_vec"].shape))
    print("scene representation:", scene_representation)
    print("eye geometry representation:", eye_representation)
    print("uv_gt:", tuple(batch["uv_gt"].shape))
    print("uv_target:", tuple(batch["uv_target"].shape))
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
    if "eye_pseudo_landmark_valid_mask" in batch:
        base_valid = batch["eye_pseudo_landmark_base_valid_mask"]
        augmented_valid = batch["eye_pseudo_landmark_valid_mask"]
        occluded = batch["eye_pseudo_occluded_mask"]
        print("eye pseudo base-valid points:", int(base_valid.sum().item()))
        print("eye pseudo post-augmentation valid points:", int(augmented_valid.sum().item()))
        print("eye pseudo points hidden by augmentation:", int(occluded.sum().item()))
        if not bool(torch.all(augmented_valid <= base_valid)):
            raise RuntimeError(
                "Eye augmentation created landmark visibility instead of only "
                "preserving/invalidation it."
            )
    for key in (
        "eye_pseudo_landmarks_xy",
        "eye_pseudo_landmarks_local",
        "eye_pseudo_landmark_base_valid_mask",
        "eye_pseudo_landmark_valid_mask",
        "eye_pseudo_occluded_mask",
        "eye_pseudo_geometry_features",
        "eye_pseudo_eye_valid_mask",
        "eye_pseudo_effective_quality",
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
        print("uv normalizer: none")
        print(
            "uv_target equals uv_gt:",
            bool((batch["uv_target"] == batch["uv_gt"]).all()),
        )
    else:
        recovered_uv = normalizer.denormalize(batch["uv_target"])
        print("uv mean mm:", normalizer.mean_mm.tolist())
        print("uv std mm:", normalizer.std_mm.tolist())
        print(
            "uv round-trip max error:",
            (recovered_uv - batch["uv_gt"]).abs().max().item(),
        )
    print("sample ids:", batch["sample_id"])
    print("val sample ids:", val_batch["sample_id"])
    if "rotation_n_from_c" in batch:
        print("rotation_n_from_c:", tuple(batch["rotation_n_from_c"].shape))
    if "raw_eye_geometry_mm" in batch:
        print("raw_eye_geometry_mm:", tuple(batch["raw_eye_geometry_mm"].shape))
        print("table_frame7:", tuple(batch["table_frame7"].shape))
        print(
            "gaze_target_camera_mm:",
            tuple(batch["gaze_target_camera_mm"].shape),
        )
    if temporary_run is not None:
        temporary_run.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
