"""Extract the 2D MediaPipe landmarks required by the DECA-FLAME PnP solver.

This is deliberately separate from DECA processing: MediaPipe and PyTorch/DECA
then run in different processes on Windows, avoiding the duplicate OpenMP
runtime issue.  The output is named per dataset and can be reused by both PnP
and Iris-IPD depth-prior generation.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.depth_prior.dataset_assets import resolve_image_asset_path
from modelv1.processed_artifacts import processed_dataset_artifacts


DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "modelv1_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "mediapipe_pnp_landmarks.csv"
# These labels intentionally preserve the legacy CrossGaze *image-side*
# convention used by DEFAULT_PNP_MAPPING: 33/133 are image-left and 362/263
# are image-right. They are not anatomical subject-left/subject-right labels.
# Consumers that produce anatomical left/right eye outputs must swap the two
# groups, as the virtual-camera generator does.
LANDMARK_INDICES = {
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "nose_tip": 1,
    "mouth_left": 61,
    "mouth_right": 291,
    "chin": 152,
}
FIELDS = (
    "sample_id", "dataset", "image_name", "source_image_path", "status", "reason",
    *(f"{label}_{axis}" for label in LANDMARK_INDICES for axis in ("x", "y")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dataset_id is not None:
        artifacts = processed_dataset_artifacts(args.dataset_id, args.processed_dir)
        args.csv = artifacts.dataset_csv
        args.output = artifacts.mediapipe_pnp_landmarks
    return args


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --overwrite to replace it.")
    import mediapipe as mp
    import numpy as np
    from PIL import Image

    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {args.csv}")
    records: list[dict[str, object]] = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for index, row in enumerate(rows, start=1):
            record: dict[str, object] = {field: "" for field in FIELDS}
            record.update({key: row[key] for key in ("sample_id", "dataset", "image_name", "source_image_path")})
            try:
                image_path = resolve_image_asset_path(row, "source_image_path", args.image_root)
                with Image.open(image_path) as image:
                    rgb = np.asarray(image.convert("RGB"))
                result = face_mesh.process(rgb)
                if not result.multi_face_landmarks:
                    raise RuntimeError("MediaPipe Face Mesh found no face")
                height, width = rgb.shape[:2]
                landmarks = result.multi_face_landmarks[0].landmark
                for label, landmark_index in LANDMARK_INDICES.items():
                    point = landmarks[landmark_index]
                    record[f"{label}_x"] = float(point.x * width)
                    record[f"{label}_y"] = float(point.y * height)
                record["status"] = "success"
            except Exception as error:
                record["status"] = "failed"
                record["reason"] = f"{type(error).__name__}: {error}"
            records.append(record)
            if index % 20 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    success = sum(record["status"] == "success" for record in records)
    print(f"Wrote {args.output} ({success}/{len(records)} successful)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
