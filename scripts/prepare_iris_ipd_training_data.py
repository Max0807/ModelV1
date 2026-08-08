"""Build all training artifacts for one newly collected dataset.

This orchestrator deliberately keeps every heavyweight stage in its own Python
process.  In particular, MediaPipe runs separately from PyTorch/DECA, avoiding
the Windows OpenMP-runtime conflict.  It never deletes or edits collection
images.  All outputs follow the numbered artifact convention, for example for
``--dataset-id 11``:

* ``modelv1_dataset11.csv``
* ``deca_features_deca_crop_v1_11.npz``
* ``mediapipe_pnp_landmarks_11.csv``
* either ``depth_priors_iris_ipd_65mm_v1_11.csv`` or
  ``depth_priors_deca_crop_v1_11.csv``
* ``modelv1_dataset_depth_prior_common_11.csv``

The final ``*_common_<id>.csv`` includes only samples with a successful
selected depth reconstruction and all three training images present.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.processed_artifacts import processed_dataset_artifacts


DEFAULT_SOURCE_ROOTS = (
    Path(r"D:\GithubCode\CrossGaze-main\baseline\data_collection"),
    Path(r"D:\GithubCode\CrossGaze-main\CrossGaze-main\data_collection"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True, help="Numeric collection dataset ID, e.g. 11.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Collection data_collection directory. If omitted, known local layouts are checked.",
    )
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument(
        "--depth-prior-kind",
        choices=("iris_ipd_65mm", "pnp1010"),
        default="iris_ipd_65mm",
        help="Depth method. Iris-IPD is the default; pnp1010 uses the legacy fixed 1010 scale.",
    )
    parser.add_argument("--ipd-mm", type=float, default=None, help="Measured interpupillary distance in mm; required for Iris-IPD.")
    parser.add_argument("--max-ipd-residual-mm", type=float, default=3.0)
    parser.add_argument("--max-ray-condition", type=float, default=250.0)
    parser.add_argument("--calibration-start", type=int, default=None, help="Inclusive frame_idx for fixed-shape Iris calibration.")
    parser.add_argument("--calibration-stop", type=int, default=None, help="Inclusive frame_idx for fixed-shape Iris calibration.")
    parser.add_argument(
        "--calibration-index-mode",
        choices=("frame_idx", "position"),
        default="frame_idx",
    )
    parser.add_argument(
        "--estimate-depth-uncertainty",
        action="store_true",
        help=(
            "For Iris-IPD only, generate per-sample label-free log-depth "
            "uncertainty required by fixed_prior/learned_reweight training."
        ),
    )
    parser.add_argument("--uncertainty-bootstrap-samples", type=int, default=64)
    parser.add_argument("--uncertainty-iris-std-px", type=float, default=0.5)
    parser.add_argument("--uncertainty-ipd-std-mm", type=float, default=0.0)
    parser.add_argument("--uncertainty-min-success-count", type=int, default=32)
    parser.add_argument("--uncertainty-seed", type=int, default=42)
    parser.add_argument("--deca-crop-scale", type=float, default=1.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deca-batch-size", type=int, default=32)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing numbered artifacts. Existing collection images are never touched.",
    )
    parser.add_argument(
        "--skip-build-dataset",
        action="store_true",
        help="Reuse an existing modelv1_dataset<ID>.csv instead of rebuilding it from collection files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the exact commands without executing them.")
    return parser.parse_args()


def resolve_source_root(dataset_name: str, requested: Path | None) -> Path:
    """Select the root with the metadata needed to build model samples."""

    candidates: list[Path] = []
    for root in (requested, *DEFAULT_SOURCE_ROOTS):
        if root is not None and root not in candidates:
            candidates.append(root)
    incomplete: list[str] = []
    for root in candidates:
        dataset_dir = root / dataset_name
        has_log = dataset_dir.is_dir() and any(dataset_dir.glob("data_log_*.csv"))
        has_insightface = (dataset_dir / "insightface_coordinates.csv").is_file()
        if has_log and has_insightface:
            return root
        incomplete.append(
            f"{root} (dataset_dir={dataset_dir.is_dir()}, data_log={has_log}, "
            f"insightface_coordinates={has_insightface})"
        )
    checked = "\n  ".join(incomplete)
    raise FileNotFoundError(
        f"Could not find a complete collection dataset {dataset_name}. Checked:\n  {checked}\n"
        "The selected root must contain data_log_*.csv and insightface_coordinates.csv."
    )


def run_step(name: str, command: Sequence[str], *, dry_run: bool) -> None:
    rendered = subprocess.list2cmdline([str(part) for part in command])
    print(f"\n[{name}]\n{rendered}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def count_successes(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    success = sum(
        row.get("depth_prior_status", row.get("status", "")).strip().lower()
        in {"success", "warning"}
        for row in rows
    )
    return success, len(rows)


def count_dataset_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    if args.depth_prior_kind == "iris_ipd_65mm":
        if args.ipd_mm is None or args.ipd_mm <= 0:
            raise ValueError("--ipd-mm must be positive for --depth-prior-kind iris_ipd_65mm.")
        if args.calibration_start is None or args.calibration_stop is None:
            raise ValueError(
                "--calibration-start and --calibration-stop are required for Iris-IPD."
            )
        if args.calibration_stop < args.calibration_start:
            raise ValueError("--calibration-stop must be greater than or equal to --calibration-start.")
    elif args.estimate_depth_uncertainty:
        raise ValueError(
            "--estimate-depth-uncertainty currently supports only "
            "--depth-prior-kind iris_ipd_65mm."
        )

    artifacts = processed_dataset_artifacts(args.dataset_id, args.processed_dir)
    source_root = resolve_source_root(artifacts.dataset_name, args.source_root)
    overwrite_flag = ["--overwrite"] if args.overwrite else []

    if not args.skip_build_dataset:
        if artifacts.dataset_csv.exists() and not args.overwrite:
            raise FileExistsError(
                f"Dataset CSV already exists: {artifacts.dataset_csv}. "
                "Use --skip-build-dataset to reuse it or --overwrite to rebuild it."
            )
        run_step(
            "1/6 build unified training CSV",
            [
                sys.executable,
                "scripts/build_modelv1_dataset.py",
                "--dataset-id",
                artifacts.dataset_id,
                "--source-root",
                str(source_root),
                "--output-dir",
                str(artifacts.processed_dir),
            ],
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            row_count = count_dataset_rows(artifacts.dataset_csv)
            if row_count == 0:
                raise RuntimeError(
                    f"Dataset build produced 0 rows: {artifacts.dataset_csv}. "
                    "Stopping before DECA; inspect the source-root metadata."
                )
    elif not artifacts.dataset_csv.is_file():
        raise FileNotFoundError(
            f"--skip-build-dataset was requested but {artifacts.dataset_csv} does not exist."
        )

    run_step(
        "2/6 cache DECA features",
        [
            sys.executable,
            "scripts/cache_deca_features.py",
            "--dataset-id",
            artifacts.dataset_id,
            "--processed-dir",
            str(artifacts.processed_dir),
            "--face-preprocess",
            "deca",
            "--deca-crop-scale",
            str(args.deca_crop_scale),
            "--batch-size",
            str(args.deca_batch_size),
            "--device",
            args.device,
            *overwrite_flag,
        ],
        dry_run=args.dry_run,
    )
    run_step(
        "3/6 extract MediaPipe PnP landmarks",
        [
            sys.executable,
            "scripts/extract_mediapipe_pnp_landmarks.py",
            "--dataset-id",
            artifacts.dataset_id,
            "--processed-dir",
            str(artifacts.processed_dir),
            *overwrite_flag,
        ],
        dry_run=args.dry_run,
    )
    if args.depth_prior_kind == "iris_ipd_65mm":
        run_step(
            "4/6 extract MediaPipe iris centres",
            [
                sys.executable,
                "scripts/extract_mediapipe_iris_centres.py",
                "--dataset-id",
                artifacts.dataset_id,
                "--processed-dir",
                str(artifacts.processed_dir),
                *overwrite_flag,
            ],
            dry_run=args.dry_run,
        )
        run_step(
            "5/6 reconstruct 65 mm Iris-IPD depth prior",
            [
                sys.executable,
                "scripts/generate_iris_ipd_depth_priors.py",
                "--dataset-id",
                artifacts.dataset_id,
                "--processed-dir",
                str(artifacts.processed_dir),
                "--ipd-mm",
                str(args.ipd_mm),
                "--max-ipd-residual-mm",
                str(args.max_ipd_residual_mm),
                "--max-ray-condition",
                str(args.max_ray_condition),
                "--calibration-start",
                str(args.calibration_start),
                "--calibration-stop",
                str(args.calibration_stop),
                "--calibration-index-mode",
                args.calibration_index_mode,
                "--deca-crop-scale",
                str(args.deca_crop_scale),
                "--device",
                args.device,
                *(
                    [
                        "--estimate-depth-uncertainty",
                        "--uncertainty-bootstrap-samples",
                        str(args.uncertainty_bootstrap_samples),
                        "--uncertainty-iris-std-px",
                        str(args.uncertainty_iris_std_px),
                        "--uncertainty-ipd-std-mm",
                        str(args.uncertainty_ipd_std_mm),
                        "--uncertainty-min-success-count",
                        str(args.uncertainty_min_success_count),
                        "--uncertainty-seed",
                        str(args.uncertainty_seed),
                    ]
                    if args.estimate_depth_uncertainty
                    else []
                ),
                *overwrite_flag,
            ],
            dry_run=args.dry_run,
        )
        depth_prior_path = artifacts.iris_ipd_depth_prior
    else:
        run_step(
            "4/5 reconstruct fixed-1010 PnP depth prior",
            [
                sys.executable,
                "scripts/generate_depth_priors.py",
                "--dataset-id",
                artifacts.dataset_id,
                "--processed-dir",
                str(artifacts.processed_dir),
                "--face-preprocess",
                "deca",
                "--deca-crop-scale",
                str(args.deca_crop_scale),
                "--batch-size",
                str(args.deca_batch_size),
                "--fixed-scale-mm-per-flame-unit",
                "1010",
                "--device",
                args.device,
                *overwrite_flag,
            ],
            dry_run=args.dry_run,
        )
        depth_prior_path = artifacts.pnp_depth_prior
    if not args.dry_run:
        success, total = count_successes(depth_prior_path)
        if success == 0:
            raise RuntimeError(
                f"{args.depth_prior_kind} reconstruction produced 0/{total} valid samples. "
                "Inspect the generated CSV reasons before continuing."
            )
        print(f"{args.depth_prior_kind} depth status: {success}/{total} valid samples", flush=True)
    run_step(
        "6/6 keep trainable depth-prior samples"
        if args.depth_prior_kind == "iris_ipd_65mm"
        else "5/5 keep trainable depth-prior samples",
        [
            sys.executable,
            "scripts/build_depth_prior_comparison_subset.py",
            "--dataset-id",
            artifacts.dataset_id,
            "--processed-dir",
            str(artifacts.processed_dir),
            "--prior-kind",
            args.depth_prior_kind,
            *overwrite_flag,
        ],
        dry_run=args.dry_run,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": artifacts.dataset_id,
        "source_root": str(source_root),
        "depth_prior_kind": args.depth_prior_kind,
        "ipd_mm": args.ipd_mm if args.depth_prior_kind == "iris_ipd_65mm" else None,
        "calibration": (
            {
                "start": args.calibration_start,
                "stop": args.calibration_stop,
                "index_mode": args.calibration_index_mode,
            }
            if args.depth_prior_kind == "iris_ipd_65mm"
            else None
        ),
        "training_inputs": {
            "dataset_csv": str(artifacts.common_depth_prior_dataset_csv),
            "deca_cache": str(artifacts.deca_cache),
            "depth_prior_csv": str(depth_prior_path),
        },
    }
    manifest_path = (
        args.processed_dir
        / f"training_manifest_{args.depth_prior_kind}_{artifacts.dataset_id}.json"
    )
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nCompleted. Training inputs:")
    for key, value in manifest["training_inputs"].items():
        print(f"  {key}: {value}")
    if not args.dry_run:
        print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python scripts/prepare_iris_ipd_training_data.py `
#   --dataset-id 11 `
#   --source-root D:\GithubCode\CrossGaze-main\CrossGaze-main\data_collection `
#   --ipd-mm 65 `
#   --calibration-start 129 `
#   --calibration-stop 231 `
#   --calibration-index-mode frame_idx `
#   --overwrite
