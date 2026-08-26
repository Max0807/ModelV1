"""Build a dataset CSV restricted to samples valid in every compared prior.

This avoids confounding a depth-prior comparison with different train/validation
samples. The source dataset CSV is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.depth_prior.pnp import PNP_GEOMETRY_VERSION

EYE_COLUMNS = (
    "left_eye_camera_x_mm", "left_eye_camera_y_mm", "left_eye_camera_z_mm",
    "right_eye_camera_x_mm", "right_eye_camera_y_mm", "right_eye_camera_z_mm",
)
VALID_STATUSES = {"success", "warning"}
TRAINING_IMAGE_FIELDS = ("face_path", "left_eye_path", "right_eye_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument(
        "--prior-csv",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Versioned prior CSVs to compare, normally one verified pnp1010 "
            "table and one regenerated iris65 table."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), reader.fieldnames


def valid_ids(path: Path) -> set[str]:
    rows, fields = read_rows(path)
    required = {
        "sample_id",
        "pnp_geometry_version",
        "pnp_min_object_depth_mm",
        *EYE_COLUMNS,
    }
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"Prior CSV lacks {sorted(missing)}: {path}")
    valid: set[str] = set()
    for row in rows:
        status = row.get("depth_prior_status", row.get("status", "success")).strip().lower()
        if status not in VALID_STATUSES:
            continue
        if row.get("pnp_geometry_version", "").strip() != PNP_GEOMETRY_VERSION:
            continue
        try:
            values = [float(row[column]) for column in EYE_COLUMNS]
            min_object_depth = float(row["pnp_min_object_depth_mm"])
        except (TypeError, ValueError):
            continue
        if (
            all(math.isfinite(value) for value in values)
            and values[2] > 0.0
            and values[5] > 0.0
            and math.isfinite(min_object_depth)
            and min_object_depth > 0.0
        ):
            valid.add(row["sample_id"])
    return valid


def write_atomic(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --overwrite to replace it.")
    dataset_rows, dataset_fields = read_rows(args.dataset_csv)
    if "sample_id" not in dataset_fields:
        raise ValueError(f"Dataset CSV lacks sample_id: {args.dataset_csv}")
    valid_by_prior = {path: valid_ids(path) for path in args.prior_csv}
    shared_ids = set.intersection(*valid_by_prior.values())
    selected: list[dict[str, str]] = []
    image_missing: list[str] = []
    for row in dataset_rows:
        if row["sample_id"] not in shared_ids:
            continue
        try:
            for field in TRAINING_IMAGE_FIELDS:
                image_path = Path(row[field])
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing {field}: {image_path}")
        except (KeyError, FileNotFoundError) as error:
            image_missing.append(f"{row['sample_id']}: {error}")
            continue
        selected.append(row)
    if not selected:
        raise ValueError("No shared valid samples across the supplied prior CSVs.")
    write_atomic(args.output, selected, dataset_fields)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_csv": str(args.dataset_csv.resolve()),
        "prior_csvs": {str(path.resolve()): len(ids) for path, ids in valid_by_prior.items()},
        "shared_valid_sample_count": len(shared_ids),
        "written_sample_count": len(selected),
        "excluded_missing_training_images": len(image_missing),
        "missing_training_image_examples": image_missing[:10],
        "selection": "sample_id is valid in every compared depth-prior CSV and has face/left-eye/right-eye training images",
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.output} ({len(selected)} shared valid samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
