"""Build the independent BlazeGaze/TableFrame7 adapter manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.blazegaze.data_adapter import (
    BlazeGazeDataAdapter,
    PogBoundsMM,
    build_blazegaze_manifest,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage3-csv",
        action="append",
        required=True,
        help="Stage 3 TableFrame7 CSV; repeat for multiple participants.",
    )
    parser.add_argument(
        "--metric-pose-csv",
        action="append",
        required=True,
        help="Stage 2 iris65 metric-pose CSV; repeat for multiple participants.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--eye-patch-manifest",
        action="append",
        help="Exact MediaPipe478 eye-patch manifest; repeat if needed.",
    )
    parser.add_argument(
        "--require-exact-eye-patches",
        action="store_true",
        help="Reject every Stage 3 row without a successful exact eye patch.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path (default: output CSV with .report.json suffix).",
    )
    parser.add_argument(
        "--pog-bounds-mm",
        type=float,
        nargs=4,
        metavar=("U_MIN", "U_MAX", "V_MIN", "V_MAX"),
        help=(
            "Fixed physical/protocol bounds used for native centred pog_norm. "
            "Do not estimate these from the test split."
        ),
    )
    parser.add_argument(
        "--allow-missing-pose",
        action="store_true",
        help="Drop Stage 3 rows lacking a successful metric pose instead of failing.",
    )
    parser.add_argument(
        "--skip-image-path-validation",
        action="store_true",
        help="Do not check that both eye-crop files exist.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report or args.output_csv.with_suffix(".report.json")
    if not args.overwrite:
        existing = [path for path in (args.output_csv, report_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing}")
    pog_bounds = (
        PogBoundsMM.from_sequence(args.pog_bounds_mm)
        if args.pog_bounds_mm is not None
        else None
    )
    adapter = BlazeGazeDataAdapter(
        stage3_csv_paths=args.stage3_csv,
        metric_pose_csv_paths=args.metric_pose_csv,
        pog_bounds_mm=pog_bounds,
        eye_patch_manifest_paths=args.eye_patch_manifest,
        require_exact_eye_patches=args.require_exact_eye_patches,
        strict_join=not args.allow_missing_pose,
        validate_image_paths=not args.skip_image_path_validation,
    )
    report = build_blazegaze_manifest(
        adapter, args.output_csv, overwrite=args.overwrite
    )
    write_report(report, report_path)
    print(
        f"Wrote {report['sample_count']} BlazeGaze adapter rows to "
        f"{args.output_csv}"
    )
    if pog_bounds is None:
        print(
            "PoG bounds were not supplied: the manifest contains UV millimetres "
            "but intentionally leaves native pog_norm/screen_info fields empty."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
