"""Build versioned ModelV1 preprocessing artifacts for one collection dataset.

The selected depth method controls the depth-prior table consumed by training:

* ``pnp1010`` uses the fixed 1010 mm/FLAME-unit PnP reconstruction.
* ``iris65`` uses the 65 mm Iris-IPD reconstruction.
* ``both`` builds a paired common sample set for a controlled comparison.

Virtual-camera normalization always uses the verified PnP1010 ``rotation_*``
and ``tvec_*`` pose. Iris65 changes the training depth prior, not the
normalization centre or camera axes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "modelv1_metric_depth_virtual_camera_v2"
PNP_GEOMETRY_VERSION = "mediapipe_flame_head_local_multistart_v2"
DEFAULT_SOURCE_ROOTS = (
    Path(r"D:\GithubCode\CrossGaze-main\CrossGaze-main\data_collection"),
    Path(r"D:\GithubCode\CrossGaze-main\baseline\data_collection"),
)


def canonical_dataset_id(value: str) -> str:
    text = str(value).strip()
    prefix = "dataset_dual_rigid_body_"
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if not text.isdigit() or int(text) < 0:
        raise argparse.ArgumentTypeError(
            "dataset ID must be a non-negative integer or "
            "dataset_dual_rigid_body_<ID>."
        )
    return str(int(text))


def depth_method(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "pnp": "pnp1010",
        "pnp1010": "pnp1010",
        "iris": "iris65",
        "iris65": "iris65",
        "iris_ipd_65mm": "iris65",
        "both": "both",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "depth method must be pnp1010, iris65, or both."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True, type=canonical_dataset_id)
    parser.add_argument(
        "--depth-method",
        required=True,
        type=depth_method,
        metavar="{pnp1010,iris65,both}",
        help="Training depth prior to build. 'both' creates a paired comparison set.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Root containing dataset_dual_rigid_body_<ID>.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: data/preprocessed_v2/dataset<ID>.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deca-batch-size", type=int, default=8)
    parser.add_argument("--deca-crop-scale", type=float, default=1.25)
    parser.add_argument(
        "--face-image-dir",
        default="insightface_face",
        help=(
            "InsightFace face-crop directory for the model dataset, for example "
            "insightface_face_160. Virtual-camera geometry still uses the "
            "1920x1080 insightface_img source image."
        ),
    )
    parser.add_argument(
        "--insightface-coordinates-file",
        default="insightface_coordinates.csv",
        help=(
            "InsightFace detection CSV filename, for example "
            "insightface_coordinates_224.csv."
        ),
    )
    parser.add_argument("--virtual-face-width", type=int, default=224)
    parser.add_argument("--virtual-face-height", type=int, default=224)
    parser.add_argument("--virtual-focal-length-px", type=float, default=480.0)
    parser.add_argument("--virtual-distance-mm", type=float, default=600.0)
    parser.add_argument("--ipd-mm", type=float, default=65.0)
    parser.add_argument("--max-ipd-residual-mm", type=float, default=4.0)
    parser.add_argument("--max-ray-condition", type=float, default=250.0)
    parser.add_argument(
        "--calibration-start",
        type=int,
        default=None,
        help="Inclusive fixed-shape calibration frame_idx/position; required for Iris65.",
    )
    parser.add_argument(
        "--calibration-stop",
        type=int,
        default=None,
        help="Inclusive fixed-shape calibration frame_idx/position; required for Iris65.",
    )
    parser.add_argument(
        "--calibration-index-mode",
        choices=("frame_idx", "position"),
        default="frame_idx",
    )
    parser.add_argument(
        "--estimate-depth-uncertainty",
        action="store_true",
        help="Generate Iris65 uncertainty for fixed_prior/learned_reweight training.",
    )
    parser.add_argument("--uncertainty-bootstrap-samples", type=int, default=64)
    parser.add_argument("--uncertainty-iris-std-px", type=float, default=0.5)
    parser.add_argument("--uncertainty-ipd-std-mm", type=float, default=0.0)
    parser.add_argument("--uncertainty-min-success-count", type=int, default=32)
    parser.add_argument("--uncertainty-seed", type=int, default=42)
    parser.add_argument(
        "--skip-deca-cache",
        action="store_true",
        help="Skip the optional model DECA feature cache; PnP still runs DECA internally.",
    )
    parser.add_argument(
        "--reuse-base-dataset",
        action="store_true",
        help="Reuse shared/modelv1_dataset<ID>.csv instead of rebuilding changed raw data.",
    )
    parser.add_argument(
        "--virtual-limit-per-dataset",
        type=int,
        default=None,
        help="Development-only limit for virtual-camera output; omit for formal generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace derived files inside --output-root. Collection data is never modified.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.deca_batch_size <= 0:
        parser.error("--deca-batch-size must be positive.")
    if args.deca_crop_scale <= 0:
        parser.error("--deca-crop-scale must be positive.")
    if not args.face_image_dir.strip() or Path(args.face_image_dir).name != args.face_image_dir:
        parser.error("--face-image-dir must be one directory name, not a path.")
    if (
        not args.insightface_coordinates_file.strip()
        or Path(args.insightface_coordinates_file).name
        != args.insightface_coordinates_file
    ):
        parser.error("--insightface-coordinates-file must be one filename, not a path.")
    if args.virtual_face_width <= 0 or args.virtual_face_height <= 0:
        parser.error("--virtual-face-width and --virtual-face-height must be positive.")
    if args.virtual_focal_length_px <= 0 or args.virtual_distance_mm <= 0:
        parser.error("--virtual-focal-length-px and --virtual-distance-mm must be positive.")
    if args.virtual_limit_per_dataset is not None and args.virtual_limit_per_dataset <= 0:
        parser.error("--virtual-limit-per-dataset must be positive.")
    if args.depth_method in {"iris65", "both"}:
        if args.ipd_mm <= 0:
            parser.error("--ipd-mm must be positive for Iris65.")
        if args.calibration_start is None or args.calibration_stop is None:
            parser.error(
                "--calibration-start and --calibration-stop are required for Iris65."
            )
        if args.calibration_stop < args.calibration_start:
            parser.error("--calibration-stop must be >= --calibration-start.")
    elif args.estimate_depth_uncertainty:
        parser.error("--estimate-depth-uncertainty requires Iris65 or both.")
    return args


@dataclass(frozen=True)
class PipelineArtifacts:
    dataset_id: str
    root: Path

    @property
    def dataset_name(self) -> str:
        return f"dataset_dual_rigid_body_{self.dataset_id}"

    @property
    def shared_dir(self) -> Path:
        return self.root / "shared"

    @property
    def pnp_dir(self) -> Path:
        return self.root / "pnp1010"

    @property
    def iris_dir(self) -> Path:
        return self.root / "iris65"

    @property
    def comparison_dir(self) -> Path:
        return self.root / "comparison"

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def dataset_csv(self) -> Path:
        return self.shared_dir / f"modelv1_dataset{self.dataset_id}.csv"

    @property
    def dataset_report(self) -> Path:
        return self.shared_dir / f"modelv1_dataset{self.dataset_id}_report.json"

    @property
    def deca_cache(self) -> Path:
        return self.shared_dir / f"deca_features_deca_crop_v1_dataset{self.dataset_id}.npz"

    @property
    def pnp_landmarks(self) -> Path:
        return self.shared_dir / f"mediapipe_pnp_landmarks_v2_dataset{self.dataset_id}.csv"

    @property
    def iris_centres(self) -> Path:
        return self.shared_dir / f"mediapipe_iris_centres_v2_dataset{self.dataset_id}.csv"

    @property
    def pnp_prior(self) -> Path:
        return self.pnp_dir / f"depth_priors_pnp1010_v2_dataset{self.dataset_id}.csv"

    @property
    def pnp_metadata(self) -> Path:
        return self.pnp_prior.with_suffix(self.pnp_prior.suffix + ".metadata.json")

    @property
    def pnp_validation(self) -> Path:
        return self.pnp_dir / f"pnp_geometry_validation_dataset{self.dataset_id}.json"

    @property
    def pnp_dataset(self) -> Path:
        return self.pnp_dir / f"modelv1_dataset{self.dataset_id}_pnp1010.csv"

    @property
    def iris_prior(self) -> Path:
        return self.iris_dir / f"depth_priors_iris65_pnp_v2_dataset{self.dataset_id}.csv"

    @property
    def iris_validation(self) -> Path:
        return self.iris_dir / f"iris65_pnp_geometry_validation_dataset{self.dataset_id}.json"

    @property
    def iris_dataset(self) -> Path:
        return self.iris_dir / f"modelv1_dataset{self.dataset_id}_iris65.csv"

    @property
    def common_dataset(self) -> Path:
        return (
            self.comparison_dir
            / f"modelv1_dataset{self.dataset_id}_common_pnp1010_iris65.csv"
        )

    def virtual_root(self, method: str) -> Path:
        label = "common_pnp1010_iris65" if method == "both" else method
        return self.root / "virtual_camera" / label

    def training_manifest(self, method: str) -> Path:
        return self.manifest_dir / f"training_inputs_{method}_dataset{self.dataset_id}.json"

    def pipeline_manifest(self, method: str) -> Path:
        return self.manifest_dir / f"pipeline_{method}_dataset{self.dataset_id}.json"


def resolve_source_root(
    dataset_name: str,
    requested: Path | None,
    *,
    insightface_coordinates_file: str,
) -> Path:
    candidates: list[Path] = []
    for candidate in (requested, *DEFAULT_SOURCE_ROOTS):
        if candidate is not None:
            resolved = candidate.resolve()
            if resolved not in candidates:
                candidates.append(resolved)
    failures: list[str] = []
    for root in candidates:
        dataset_dir = root / dataset_name
        logs = sorted(dataset_dir.glob("data_log_*.csv")) if dataset_dir.is_dir() else []
        insightface = dataset_dir / insightface_coordinates_file
        if len(logs) == 1 and insightface.is_file():
            return root
        failures.append(
            f"{root}: dataset_dir={dataset_dir.is_dir()}, data_logs={len(logs)}, "
            f"{insightface_coordinates_file}={insightface.is_file()}"
        )
    raise FileNotFoundError(
        f"No complete source dataset found for {dataset_name}. Checked:\n  "
        + "\n  ".join(failures)
        + "\nExactly one data_log_*.csv and the requested InsightFace CSV are required."
    )


def run_step(name: str, command: Sequence[str], *, dry_run: bool) -> None:
    rendered = subprocess.list2cmdline([str(part) for part in command])
    print(f"\n[{name}]\n{rendered}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_virtual_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    status_counts = report.get("status_counts", {})
    success = int(status_counts.get("success", 0))
    failed = int(status_counts.get("failed", 0))
    if success <= 0 or failed != 0:
        raise RuntimeError(
            f"Virtual-camera generation is incomplete: success={success}, failed={failed}. "
            f"Inspect {path}."
        )
    if report.get("center_source") != "pnp_tvec":
        raise RuntimeError(f"Unexpected virtual-camera centre in {path}.")
    if report.get("pnp_geometry_version") != PNP_GEOMETRY_VERSION:
        raise RuntimeError(f"Unexpected PnP geometry version in {path}.")
    return report


def append_overwrite(command: list[str], overwrite: bool) -> list[str]:
    if overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    args = parse_args()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (PROJECT_ROOT / "data" / "preprocessed_v2" / f"dataset{args.dataset_id}").resolve()
    )
    artifacts = PipelineArtifacts(args.dataset_id, output_root)
    source_root = resolve_source_root(
        artifacts.dataset_name,
        args.source_root,
        insightface_coordinates_file=args.insightface_coordinates_file,
    )
    python = sys.executable

    if not args.reuse_base_dataset:
        if artifacts.dataset_csv.exists() and not args.overwrite:
            raise FileExistsError(
                f"Base dataset exists: {artifacts.dataset_csv}. Use --overwrite to rebuild "
                "changed raw data, or --reuse-base-dataset only when it is intentionally unchanged."
            )
        run_step(
            "1 build unified dataset CSV from data_log + InsightFace",
            [
                python,
                "scripts/build_modelv1_dataset.py",
                "--dataset-id",
                artifacts.dataset_id,
                "--source-root",
                str(source_root),
                "--output-dir",
                str(artifacts.shared_dir),
                "--handeye-translation-scale",
                "1000",
                "--table-z-source",
                "per-sample-target",
                "--face-image-dir",
                args.face_image_dir,
                "--insightface-coordinates-file",
                args.insightface_coordinates_file,
            ],
            dry_run=args.dry_run,
        )
    elif not artifacts.dataset_csv.is_file():
        raise FileNotFoundError(
            f"--reuse-base-dataset requested but file is missing: {artifacts.dataset_csv}"
        )

    if not args.skip_deca_cache:
        run_step(
            "2 cache optional DECA model features",
            append_overwrite(
                [
                    python,
                    "scripts/cache_deca_features.py",
                    "--csv",
                    str(artifacts.dataset_csv),
                    "--output",
                    str(artifacts.deca_cache),
                    "--face-preprocess",
                    "deca",
                    "--deca-crop-scale",
                    str(args.deca_crop_scale),
                    "--batch-size",
                    str(args.deca_batch_size),
                    "--device",
                    args.device,
                ],
                args.overwrite,
            ),
            dry_run=args.dry_run,
        )

    run_step(
        "3 extract shared MediaPipe PnP landmarks",
        append_overwrite(
            [
                python,
                "scripts/extract_mediapipe_pnp_landmarks.py",
                "--csv",
                str(artifacts.dataset_csv),
                "--output",
                str(artifacts.pnp_landmarks),
                "--image-root",
                str(source_root),
            ],
            args.overwrite,
        ),
        dry_run=args.dry_run,
    )

    run_step(
        "4 generate fixed-scale PnP1010 v2 pose/depth geometry",
        append_overwrite(
            [
                python,
                "scripts/generate_depth_priors.py",
                "--csv",
                str(artifacts.dataset_csv),
                "--output",
                str(artifacts.pnp_prior),
                "--metadata-output",
                str(artifacts.pnp_metadata),
                "--pnp-landmark-csv",
                str(artifacts.pnp_landmarks),
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
            ],
            args.overwrite,
        ),
        dry_run=args.dry_run,
    )

    run_step(
        "5 audit successful PnP poses against the v2 geometry contract",
        [
            python,
            "scripts/validate_pnp_geometry.py",
            str(artifacts.pnp_prior),
            "--output",
            str(artifacts.pnp_validation),
            "--allow-failed-rows",
        ],
        dry_run=args.dry_run,
    )

    run_step(
        "6 build the PnP1010-valid model dataset",
        append_overwrite(
            [
                python,
                "scripts/build_depth_prior_comparison_subset.py",
                "--dataset-csv",
                str(artifacts.dataset_csv),
                "--prior-csv",
                str(artifacts.pnp_prior),
                "--output",
                str(artifacts.pnp_dataset),
            ],
            args.overwrite,
        ),
        dry_run=args.dry_run,
    )

    needs_iris = args.depth_method in {"iris65", "both"}
    if needs_iris:
        run_step(
            "7 extract shared MediaPipe iris centres",
            append_overwrite(
                [
                    python,
                    "scripts/extract_mediapipe_iris_centres.py",
                    "--csv",
                    str(artifacts.dataset_csv),
                    "--output",
                    str(artifacts.iris_centres),
                    "--image-root",
                    str(source_root),
                ],
                args.overwrite,
            ),
            dry_run=args.dry_run,
        )
        iris_command = [
            python,
            "scripts/generate_iris_ipd_depth_priors.py",
            "--csv",
            str(artifacts.dataset_csv),
            "--output",
            str(artifacts.iris_prior),
            "--ipd-mm",
            str(args.ipd_mm),
            "--max-ipd-residual-mm",
            str(args.max_ipd_residual_mm),
            "--max-ray-condition",
            str(args.max_ray_condition),
            "--pnp-scale-mm-per-flame-unit",
            "1010",
            "--calibration-dataset",
            artifacts.dataset_name,
            "--calibration-start",
            str(args.calibration_start),
            "--calibration-stop",
            str(args.calibration_stop),
            "--calibration-index-mode",
            args.calibration_index_mode,
            "--face-preprocess",
            "deca",
            "--deca-crop-scale",
            str(args.deca_crop_scale),
            "--device",
            args.device,
            "--image-root",
            str(source_root),
            "--iris-csv",
            str(artifacts.iris_centres),
            "--pnp-landmark-csv",
            str(artifacts.pnp_landmarks),
        ]
        if args.estimate_depth_uncertainty:
            iris_command.extend(
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
            )
        run_step(
            "8 generate Iris65 depth prior with PnP v2 head geometry",
            append_overwrite(iris_command, args.overwrite),
            dry_run=args.dry_run,
        )
        run_step(
            "9 audit the PnP pose saved with Iris65",
            [
                python,
                "scripts/validate_pnp_geometry.py",
                str(artifacts.iris_prior),
                "--output",
                str(artifacts.iris_validation),
                "--allow-failed-rows",
            ],
            dry_run=args.dry_run,
        )
        run_step(
            "10 build the Iris65-valid coverage dataset",
            append_overwrite(
                [
                    python,
                    "scripts/build_depth_prior_comparison_subset.py",
                    "--dataset-csv",
                    str(artifacts.dataset_csv),
                    "--prior-csv",
                    str(artifacts.iris_prior),
                    "--output",
                    str(artifacts.iris_dataset),
                ],
                args.overwrite,
            ),
            dry_run=args.dry_run,
        )
        run_step(
            "11 build the paired PnP1010/Iris65 comparison dataset",
            append_overwrite(
                [
                    python,
                    "scripts/build_depth_prior_comparison_subset.py",
                    "--dataset-csv",
                    str(artifacts.dataset_csv),
                    "--prior-csv",
                    str(artifacts.pnp_prior),
                    str(artifacts.iris_prior),
                    "--output",
                    str(artifacts.common_dataset),
                ],
                args.overwrite,
            ),
            dry_run=args.dry_run,
        )

    selected_dataset = (
        artifacts.pnp_dataset if args.depth_method == "pnp1010" else artifacts.common_dataset
    )
    virtual_root = artifacts.virtual_root(args.depth_method)
    virtual_command = [
        python,
        "scripts/generate_virtual_camera_normalized_images.py",
        "--dataset-csv",
        str(selected_dataset),
        "--depth-prior-csv",
        str(artifacts.pnp_prior),
        "--landmark-csv",
        str(artifacts.pnp_landmarks),
        "--output-root",
        str(virtual_root),
            "--face-width",
            str(args.virtual_face_width),
            "--face-height",
            str(args.virtual_face_height),
            "--virtual-focal-length-px",
            str(args.virtual_focal_length_px),
            "--virtual-distance-mm",
            str(args.virtual_distance_mm),
        "--eye-width",
        "60",
        "--eye-height",
        "36",
        "--eye-span-scale",
        "1.8",
    ]
    if args.virtual_limit_per_dataset is not None:
        virtual_command.extend(
            ["--limit-per-dataset", str(args.virtual_limit_per_dataset)]
        )
    run_step(
        "12 generate shared PnP-centred virtual-camera model images",
        append_overwrite(virtual_command, args.overwrite),
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDry run completed; no files were written.")
        print(f"Planned output root: {artifacts.root}")
        return 0

    virtual_report = verify_virtual_report(virtual_root / "report.json")
    common_training_dataset = selected_dataset
    methods = ("pnp1010", "iris65") if args.depth_method == "both" else (args.depth_method,)
    training_manifests: dict[str, str] = {}
    for method in methods:
        prior_path = artifacts.pnp_prior if method == "pnp1010" else artifacts.iris_prior
        payload = {
            "pipeline_version": PIPELINE_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_id": artifacts.dataset_id,
            "dataset_name": artifacts.dataset_name,
            "depth_method": method,
            "dataset_csv": str(common_training_dataset.resolve()),
            "depth_prior_csv": str(prior_path.resolve()),
            "virtual_camera_manifest": str((virtual_root / "manifest.csv").resolve()),
            "virtual_camera_report": str((virtual_root / "report.json").resolve()),
            "virtual_camera_pose_source": str(artifacts.pnp_prior.resolve()),
            "virtual_camera_center_source": "pnp_tvec",
            "pnp_geometry_version": PNP_GEOMETRY_VERSION,
            "deca_cache": (
                None if args.skip_deca_cache else str(artifacts.deca_cache.resolve())
            ),
            "sample_count": csv_row_count(common_training_dataset),
        }
        manifest_path = artifacts.training_manifest(method)
        write_json(manifest_path, payload, overwrite=args.overwrite)
        training_manifests[method] = str(manifest_path.resolve())

    pipeline_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_depth_method": args.depth_method,
        "dataset_id": artifacts.dataset_id,
        "dataset_name": artifacts.dataset_name,
        "source_root": str(source_root),
        "output_root": str(artifacts.root),
        "pnp_geometry_version": PNP_GEOMETRY_VERSION,
        "virtual_camera_center_source": "pnp_tvec",
        "virtual_camera_status_counts": virtual_report["status_counts"],
        "training_manifests": training_manifests,
        "artifacts": {
            "dataset_csv": str(artifacts.dataset_csv.resolve()),
            "dataset_report": str(artifacts.dataset_report.resolve()),
            "pnp_landmarks": str(artifacts.pnp_landmarks.resolve()),
            "pnp_prior": str(artifacts.pnp_prior.resolve()),
            "pnp_validation": str(artifacts.pnp_validation.resolve()),
            "pnp_dataset": str(artifacts.pnp_dataset.resolve()),
            "iris_centres": str(artifacts.iris_centres.resolve()) if needs_iris else None,
            "iris_prior": str(artifacts.iris_prior.resolve()) if needs_iris else None,
            "iris_validation": str(artifacts.iris_validation.resolve()) if needs_iris else None,
            "iris_dataset": str(artifacts.iris_dataset.resolve()) if needs_iris else None,
            "paired_dataset": str(artifacts.common_dataset.resolve()) if needs_iris else None,
            "virtual_camera_manifest": str((virtual_root / "manifest.csv").resolve()),
        },
        "parameters": {
            "ipd_mm": args.ipd_mm if needs_iris else None,
            "calibration_start": args.calibration_start if needs_iris else None,
            "calibration_stop": args.calibration_stop if needs_iris else None,
            "calibration_index_mode": args.calibration_index_mode if needs_iris else None,
            "deca_crop_scale": args.deca_crop_scale,
            "deca_batch_size": args.deca_batch_size,
            "face_image_dir": args.face_image_dir,
            "insightface_coordinates_file": args.insightface_coordinates_file,
            "virtual_face_width": args.virtual_face_width,
            "virtual_face_height": args.virtual_face_height,
            "virtual_focal_length_px": args.virtual_focal_length_px,
            "virtual_distance_mm": args.virtual_distance_mm,
            "device": args.device,
            "estimate_depth_uncertainty": args.estimate_depth_uncertainty,
        },
    }
    pipeline_manifest = artifacts.pipeline_manifest(args.depth_method)
    write_json(pipeline_manifest, pipeline_payload, overwrite=args.overwrite)

    print("\nCompleted versioned preprocessing.")
    print(f"  output_root: {artifacts.root}")
    print(f"  selected_dataset_csv: {common_training_dataset}")
    print(f"  virtual_camera_manifest: {virtual_root / 'manifest.csv'}")
    for method, manifest in training_manifests.items():
        print(f"  {method}_training_inputs: {manifest}")
    print(f"  pipeline_manifest: {pipeline_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
