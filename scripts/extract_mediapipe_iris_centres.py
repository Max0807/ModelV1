"""Extract MediaPipe iris-centre observations without loading PyTorch/DECA.

Run this before ``generate_iris_ipd_depth_priors.py`` and pass its CSV through
``--iris-csv``.  Keeping MediaPipe TensorFlow Lite and PyTorch/DECA in
separate processes avoids their Windows Intel OpenMP runtime conflict.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.depth_prior.dataset_assets import resolve_image_asset_path
from modelv1.depth_prior.eye_proxy_validation import detect_iris_centres_rgb
from modelv1.processed_artifacts import processed_dataset_artifacts


DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "modelv1_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "mediapipe_iris_centres.csv"
FIELDS = (
    "sample_id", "dataset", "image_name", "source_image_path", "status", "reason",
    "iris_group_468_x", "iris_group_468_y", "iris_group_473_x", "iris_group_473_y",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-id", default=None, help="Derive input/output names from one dataset ID.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--dataset", default=None, help="Optional dataset name to extract; default is every CSV row.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional root containing dataset directories if saved CSV image paths are stale.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dataset_id is not None:
        artifacts = processed_dataset_artifacts(args.dataset_id, args.processed_dir)
        args.csv = artifacts.dataset_csv
        args.output = artifacts.mediapipe_iris_centres
        args.dataset = artifacts.dataset_name
    return args


def read_rows(path: Path, dataset: str | None, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "dataset", "image_name", "source_dataset_dir", "source_image_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Dataset CSV is empty or lacks the required source-image columns.")
    if dataset is not None:
        rows = [row for row in rows if row["dataset"] == dataset]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive.")
        rows = rows[:limit]
    if not rows:
        raise ValueError("No rows selected for iris extraction.")
    return rows


def write_csv_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --overwrite to replace it.")
    try:
        import mediapipe as mp
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Extraction requires MediaPipe, NumPy, and Pillow.") from error

    rows = read_rows(args.csv, args.dataset, args.limit)
    records: list[dict[str, Any]] = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as detector:
        for index, row in enumerate(rows, start=1):
            record = {field: "" for field in FIELDS}
            record.update({
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "image_name": row["image_name"],
                "source_image_path": row["source_image_path"],
            })
            try:
                image_path = resolve_image_asset_path(row, "source_image_path", args.image_root)
                with Image.open(image_path) as image:
                    source = np.asarray(image.convert("RGB"))
                iris = detect_iris_centres_rgb(source, face_mesh=detector)
                record.update({
                    "status": "success", "reason": "",
                    "iris_group_468_x": iris.group_468_xy[0], "iris_group_468_y": iris.group_468_xy[1],
                    "iris_group_473_x": iris.group_473_xy[0], "iris_group_473_y": iris.group_473_xy[1],
                })
            except Exception as error:
                record.update({"status": "failed", "reason": f"{type(error).__name__}: {error}"})
            records.append(record)
            if index % 20 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)}")
    write_csv_atomic(args.output, records)
    success = sum(record["status"] == "success" for record in records)
    print(f"Wrote {args.output} ({success}/{len(records)} successful)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
