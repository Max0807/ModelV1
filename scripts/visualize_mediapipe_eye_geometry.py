"""Create visual contact sheets for MediaPipe eye-geometry pseudo-label QA.

The default ``mixed`` selection shows the lowest-quality half first and fills
the other half with deterministic random samples.  A CSV containing a
``sample_id`` column (for example a worst-error CSV) can instead drive an exact
targeted review through ``--sample-id-file``.
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
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.depth_prior.dataset_assets import resolve_image_asset_path
from modelv1.data.eye_geometry_pseudo_labels import (
    EYE_GEOMETRY_PSEUDO_LABEL_SCHEMA_VERSION as SCHEMA_VERSION,
    EYE_PSEUDO_GEOMETRY_FEATURE_NAMES as GEOMETRY_FEATURE_NAMES,
    EYE_PSEUDO_POINT_COUNT as POINT_COUNT_PER_EYE,
    EYE_PSEUDO_POINT_NAMES as EYE_POINT_NAMES,
)
from modelv1.processed_artifacts import canonical_dataset_id
from scripts.extract_mediapipe_eye_geometry import (
    sha256_file,
    write_json_atomic,
)


REQUIRED_ARRAYS = (
    "sample_id",
    "dataset",
    "image_name",
    "detection_success_mask",
    "valid_mask",
    "left_eye_valid_mask",
    "right_eye_valid_mask",
    "left_eye_points_px",
    "right_eye_points_px",
    "left_eye_points_crop_norm",
    "right_eye_points_crop_norm",
    "left_eye_geometry",
    "right_eye_geometry",
    "sample_quality_score",
    "metadata_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Numeric suffix, e.g. 11, or dataset_dual_rigid_body_11.",
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed_v2_160",
    )
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional data_collection root used when saved image paths are stale.",
    )
    parser.add_argument(
        "--selection",
        choices=("mixed", "worst", "random"),
        default="mixed",
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-id-file",
        type=Path,
        default=None,
        help="Optional CSV with sample_id column or one-sample-id-per-line text file.",
    )
    parser.add_argument(
        "--only-valid",
        action="store_true",
        help="Exclude samples that the extractor marked invalid.",
    )
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--samples-per-page", type=int, default=16)
    parser.add_argument("--tile-width", type=int, default=900)
    parser.add_argument("--tile-height", type=int, default=420)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--draw-indices", action="store_true")
    parser.add_argument("--save-individual", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_id = canonical_dataset_id(args.dataset_id)
    suffix = f"dataset{dataset_id}"
    root = args.preprocessed_root.resolve()
    args.dataset_id = dataset_id
    args.dataset_name = f"dataset_dual_rigid_body_{dataset_id}"
    args.npz = (
        args.npz.resolve()
        if args.npz is not None
        else root / f"mediapipe_eye_geometry_v1_{suffix}.npz"
    )
    args.csv = (
        args.csv.resolve()
        if args.csv is not None
        else root / suffix / "shared" / f"modelv1_dataset{dataset_id}.csv"
    )
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / f"mediapipe_eye_geometry_v1_{suffix}_qc"
    )
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.npz.is_file():
        raise FileNotFoundError(f"Eye-geometry NPZ does not exist: {args.npz}")
    if not args.csv.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {args.csv}")
    if args.sample_id_file is not None and not args.sample_id_file.is_file():
        raise FileNotFoundError(f"Sample-ID file does not exist: {args.sample_id_file}")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.columns <= 0:
        raise ValueError("--columns must be positive.")
    if args.samples_per_page <= 0:
        raise ValueError("--samples-per-page must be positive.")
    if args.tile_width < 500 or args.tile_height < 260:
        raise ValueError("--tile-width/--tile-height are too small for three panels.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must lie in [1, 100].")
    known_outputs = [
        args.output_dir / "qc_manifest.json",
        *args.output_dir.glob("qc_page_*.jpg"),
    ]
    existing = [path for path in known_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"QC output exists in {args.output_dir}. Use --overwrite to replace it."
        )


def load_geometry(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_ARRAYS if key not in archive.files]
        if missing:
            raise ValueError(f"Eye-geometry NPZ is missing arrays: {missing}")
        arrays = {key: archive[key].copy() for key in archive.files}
    try:
        metadata = json.loads(str(arrays["metadata_json"].item()))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("NPZ metadata_json is invalid.") from error
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema {SCHEMA_VERSION!r}, got {metadata.get('schema_version')!r}."
        )
    count = len(arrays["sample_id"])
    for key in REQUIRED_ARRAYS:
        if key == "metadata_json":
            continue
        if arrays[key].shape[0] != count:
            raise ValueError(f"NPZ array {key!r} has a different sample count.")
    expected_points = (count, POINT_COUNT_PER_EYE, 2)
    for key in (
        "left_eye_points_px",
        "right_eye_points_px",
        "left_eye_points_crop_norm",
        "right_eye_points_crop_norm",
    ):
        if arrays[key].shape != expected_points:
            raise ValueError(f"NPZ array {key!r} must have shape {expected_points}.")
    return arrays, metadata


def read_rows(path: Path, dataset_name: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset CSV has no header: {path}")
        required = {
            "sample_id",
            "dataset",
            "source_dataset_dir",
            "source_image_path",
            "left_eye_path",
            "right_eye_path",
            "face_bbox_x",
            "face_bbox_y",
            "face_bbox_w",
            "face_bbox_h",
            "left_eye_bbox_x",
            "left_eye_bbox_y",
            "left_eye_bbox_w",
            "left_eye_bbox_h",
            "right_eye_bbox_x",
            "right_eye_bbox_y",
            "right_eye_bbox_w",
            "right_eye_bbox_h",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset CSV is missing columns: {sorted(missing)}")
        rows = {
            row["sample_id"]: row
            for row in reader
            if row["dataset"] == dataset_name
        }
    if not rows:
        raise ValueError(f"No dataset rows found for {dataset_name!r}.")
    return rows


def read_requested_sample_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
                raise ValueError("--sample-id-file CSV must contain a sample_id column.")
            values = [row["sample_id"].strip() for row in reader]
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    values = [value for value in values if value]
    if not values:
        raise ValueError("--sample-id-file selected no sample IDs.")
    if len(values) != len(set(values)):
        raise ValueError("--sample-id-file contains duplicate sample IDs.")
    return values


def select_indices(
    arrays: Mapping[str, np.ndarray],
    *,
    count: int,
    strategy: str,
    seed: int,
    only_valid: bool,
    requested_sample_ids: Sequence[str] | None,
) -> tuple[list[int], list[str]]:
    sample_ids = [str(value) for value in arrays["sample_id"]]
    valid_mask = arrays["valid_mask"].astype(bool)
    candidates = [index for index in range(len(sample_ids)) if not only_valid or valid_mask[index]]
    if requested_sample_ids is not None:
        by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        missing = [sample_id for sample_id in requested_sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"Requested sample IDs are absent from the NPZ: {missing[:10]}")
        selected = [by_id[sample_id] for sample_id in requested_sample_ids]
        if only_valid:
            selected = [index for index in selected if valid_mask[index]]
        selected = selected[:count]
        if not selected:
            raise ValueError("No requested samples remain after filtering.")
        return selected, ["requested"] * len(selected)
    if not candidates:
        raise ValueError("No samples remain after filtering.")

    count = min(count, len(candidates))
    quality = arrays["sample_quality_score"].astype(np.float64)
    worst_order = sorted(candidates, key=lambda index: (quality[index], index))
    rng = np.random.default_rng(seed)
    if strategy == "worst":
        return worst_order[:count], ["worst"] * count
    if strategy == "random":
        selected = [int(index) for index in rng.choice(candidates, size=count, replace=False)]
        return selected, ["random"] * count
    if strategy != "mixed":
        raise ValueError(f"Unknown selection strategy: {strategy}")
    worst_count = min((count + 1) // 2, len(worst_order))
    selected = worst_order[:worst_count]
    remaining = [index for index in candidates if index not in set(selected)]
    random_count = count - len(selected)
    if random_count:
        selected.extend(
            int(index)
            for index in rng.choice(remaining, size=random_count, replace=False)
        )
    return selected, ["worst"] * worst_count + ["random"] * random_count


def _integer_points(points: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(round(float(point[0]))), int(round(float(point[1]))))
        for point in points
    ]


def draw_eye_overlay(
    image: Image.Image,
    points_xy: np.ndarray,
    *,
    draw_indices: bool,
    side_color: tuple[int, int, int],
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape != (POINT_COUNT_PER_EYE, 2) or not np.isfinite(points).all():
        return canvas
    draw = ImageDraw.Draw(canvas)
    radius = max(2, int(round(min(canvas.size) / 180.0)))
    width = max(2, radius)
    integer = _integer_points(points)
    draw.line(integer[:10] + [integer[0]], fill=side_color, width=width)
    iris_center = points[10]
    rim_order = np.argsort(
        np.arctan2(points[11:15, 1] - iris_center[1], points[11:15, 0] - iris_center[0])
    )
    iris_polygon = [integer[11 + int(index)] for index in rim_order]
    draw.line(iris_polygon + [iris_polygon[0]], fill=(255, 165, 0), width=width)
    fitted_center = points[11:15].mean(axis=0)
    fitted_radius = float(np.linalg.norm(points[11:15] - fitted_center, axis=1).mean())
    draw.ellipse(
        (
            int(fitted_center[0] - fitted_radius),
            int(fitted_center[1] - fitted_radius),
            int(fitted_center[0] + fitted_radius),
            int(fitted_center[1] + fitted_radius),
        ),
        outline=(255, 220, 0),
        width=width,
    )
    for point_index, (x, y) in enumerate(integer):
        color = (255, 50, 50) if point_index == 10 else (
            (0, 230, 255) if point_index in (0, 5) else (
                (255, 165, 0) if point_index >= 11 else side_color
            )
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if draw_indices:
            draw.text((x + radius + 1, y - radius - 1), str(point_index), fill=(255, 255, 255))
    return canvas


def _bbox(row: Mapping[str, str], prefix: str) -> tuple[int, int, int, int]:
    x, y, width, height = (
        float(row[f"{prefix}_bbox_{axis}"]) for axis in ("x", "y", "w", "h")
    )
    return (
        int(math.floor(x)),
        int(math.floor(y)),
        int(math.ceil(x + width)),
        int(math.ceil(y + height)),
    )


def _expanded_crop_box(
    row: Mapping[str, str], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    left, top, right, bottom = _bbox(row, "face")
    width = right - left
    height = bottom - top
    margin_x = int(round(0.08 * width))
    margin_y = int(round(0.08 * height))
    image_width, image_height = image_size
    return (
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(image_width, right + margin_x),
        min(image_height, bottom + margin_y),
    )


def _fit_into(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Cannot resize an empty image.")
    scale = min(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, (25, 25, 25))
    canvas.paste(
        resized,
        ((target_width - resized.width) // 2, (target_height - resized.height) // 2),
    )
    return canvas


def _load_eye_panel(
    row: dict[str, str],
    side: str,
    source: Image.Image,
    image_root: Path | None,
) -> tuple[Image.Image, str]:
    field = f"{side}_eye_path"
    try:
        path = resolve_image_asset_path(row, field, image_root)
        with Image.open(path) as image:
            return image.convert("RGB"), str(path.resolve())
    except (FileNotFoundError, KeyError, OSError):
        return source.crop(_bbox(row, f"{side}_eye")), "source_bbox_fallback"


def make_sample_tile(
    *,
    row: dict[str, str],
    index: int,
    arrays: Mapping[str, np.ndarray],
    role: str,
    tile_size: tuple[int, int],
    image_root: Path | None,
    draw_indices: bool,
) -> tuple[Image.Image, dict[str, Any]]:
    source_path = resolve_image_asset_path(row, "source_image_path", image_root)
    with Image.open(source_path) as image:
        source = image.convert("RGB")
    left_px = arrays["left_eye_points_px"][index]
    right_px = arrays["right_eye_points_px"][index]
    annotated_source = draw_eye_overlay(
        source,
        left_px,
        draw_indices=draw_indices,
        side_color=(80, 220, 80),
    )
    annotated_source = draw_eye_overlay(
        annotated_source,
        right_px,
        draw_indices=draw_indices,
        side_color=(80, 160, 255),
    )
    face_panel = annotated_source.crop(_expanded_crop_box(row, source.size))

    left_eye, left_eye_path = _load_eye_panel(row, "left", source, image_root)
    right_eye, right_eye_path = _load_eye_panel(row, "right", source, image_root)
    left_norm = arrays["left_eye_points_crop_norm"][index]
    right_norm = arrays["right_eye_points_crop_norm"][index]
    left_eye_points = left_norm * np.asarray(left_eye.size, dtype=np.float32)
    right_eye_points = right_norm * np.asarray(right_eye.size, dtype=np.float32)
    left_panel = draw_eye_overlay(
        left_eye,
        left_eye_points,
        draw_indices=draw_indices,
        side_color=(80, 220, 80),
    )
    right_panel = draw_eye_overlay(
        right_eye,
        right_eye_points,
        draw_indices=draw_indices,
        side_color=(80, 160, 255),
    )

    tile_width, tile_height = tile_size
    margin = 10
    title_height = 54
    content_height = tile_height - title_height - margin
    face_width = int(round(tile_width * 0.56))
    eye_width = tile_width - face_width - 3 * margin
    eye_height = (content_height - margin) // 2
    tile = Image.new("RGB", tile_size, (15, 15, 15))
    tile.paste(
        _fit_into(face_panel, (face_width, content_height)),
        (margin, title_height),
    )
    tile.paste(
        _fit_into(left_panel, (eye_width, eye_height)),
        (face_width + 2 * margin, title_height),
    )
    tile.paste(
        _fit_into(right_panel, (eye_width, eye_height)),
        (face_width + 2 * margin, title_height + eye_height + margin),
    )
    draw = ImageDraw.Draw(tile)
    quality = float(arrays["sample_quality_score"][index])
    valid = bool(arrays["valid_mask"][index])
    sample_id = str(arrays["sample_id"][index])
    draw.text(
        (margin, 7),
        f"{sample_id} | role={role} | valid={valid} | quality={quality:.3f}",
        fill=(255, 255, 255),
    )
    feature_index = {name: idx for idx, name in enumerate(GEOMETRY_FEATURE_NAMES)}
    left_geometry = arrays["left_eye_geometry"][index]
    right_geometry = arrays["right_eye_geometry"][index]
    draw.text(
        (margin, 28),
        (
            "green=anatomical-left, blue=anatomical-right | "
            f"iris_y L/R={left_geometry[feature_index['iris_center_y_local']]:.3f}/"
            f"{right_geometry[feature_index['iris_center_y_local']]:.3f}"
        ),
        fill=(210, 210, 210),
    )
    manifest_record = {
        "sample_id": sample_id,
        "dataset": str(arrays["dataset"][index]),
        "image_name": str(arrays["image_name"][index]),
        "selection_role": role,
        "npz_index": index,
        "detection_success": bool(arrays["detection_success_mask"][index]),
        "valid": valid,
        "left_eye_valid": bool(arrays["left_eye_valid_mask"][index]),
        "right_eye_valid": bool(arrays["right_eye_valid_mask"][index]),
        "quality_score": quality,
        "detection_reason": (
            str(arrays["detection_reason"][index])
            if "detection_reason" in arrays
            else ""
        ),
        "left_eye_invalid_reason": (
            str(arrays["left_eye_invalid_reason"][index])
            if "left_eye_invalid_reason" in arrays
            else ""
        ),
        "right_eye_invalid_reason": (
            str(arrays["right_eye_invalid_reason"][index])
            if "right_eye_invalid_reason" in arrays
            else ""
        ),
        "source_image_path": str(source_path.resolve()),
        "left_eye_path": left_eye_path,
        "right_eye_path": right_eye_path,
    }
    return tile, manifest_record


def save_jpeg_atomic(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        image.save(temporary, format="JPEG", quality=quality, subsampling=0)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    validate_args(args)
    arrays, metadata = load_geometry(args.npz)
    if str(metadata.get("dataset_id")) != args.dataset_id:
        raise ValueError(
            f"NPZ dataset_id={metadata.get('dataset_id')!r} does not match "
            f"--dataset-id={args.dataset_id!r}."
        )
    rows = read_rows(args.csv, args.dataset_name)
    npz_sample_ids = [str(value) for value in arrays["sample_id"]]
    missing_rows = [sample_id for sample_id in npz_sample_ids if sample_id not in rows]
    if missing_rows:
        raise ValueError(f"NPZ sample IDs are missing from the dataset CSV: {missing_rows[:10]}")
    requested_ids = (
        read_requested_sample_ids(args.sample_id_file)
        if args.sample_id_file is not None
        else None
    )
    selected, roles = select_indices(
        arrays,
        count=args.num_samples,
        strategy=args.selection,
        seed=args.seed,
        only_valid=args.only_valid,
        requested_sample_ids=requested_ids,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = args.output_dir / "individual"
    if args.save_individual:
        individual_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[Image.Image] = []
    records: list[dict[str, Any]] = []
    for position, (index, role) in enumerate(zip(selected, roles), start=1):
        sample_id = npz_sample_ids[index]
        tile, record = make_sample_tile(
            row=rows[sample_id],
            index=index,
            arrays=arrays,
            role=role,
            tile_size=(args.tile_width, args.tile_height),
            image_root=args.image_root,
            draw_indices=args.draw_indices,
        )
        tiles.append(tile)
        record["selection_position"] = position
        records.append(record)
        if args.save_individual:
            individual_path = individual_dir / f"{position:04d}_{Path(record['image_name']).stem}.jpg"
            save_jpeg_atomic(tile, individual_path, args.jpeg_quality)
            record["individual_image"] = str(individual_path.resolve())
        if position % 10 == 0 or position == len(selected):
            print(f"Rendered {position}/{len(selected)} samples")

    page_paths: list[Path] = []
    for page_start in range(0, len(tiles), args.samples_per_page):
        page_tiles = tiles[page_start : page_start + args.samples_per_page]
        columns = min(args.columns, len(page_tiles))
        rows_per_page = int(math.ceil(len(page_tiles) / columns))
        sheet = Image.new(
            "RGB",
            (columns * args.tile_width, rows_per_page * args.tile_height),
            (8, 8, 8),
        )
        for tile_index, tile in enumerate(page_tiles):
            sheet.paste(
                tile,
                (
                    (tile_index % columns) * args.tile_width,
                    (tile_index // columns) * args.tile_height,
                ),
            )
        page_number = len(page_paths) + 1
        page_path = args.output_dir / f"qc_page_{page_number:03d}.jpg"
        save_jpeg_atomic(sheet, page_path, args.jpeg_quality)
        page_paths.append(page_path)
        for offset in range(len(page_tiles)):
            records[page_start + offset]["page"] = page_number
            records[page_start + offset]["page_tile_index"] = offset

    manifest_path = args.output_dir / "qc_manifest.json"
    manifest = {
        "schema_version": "mediapipe_eye_geometry_visual_qc_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "input_npz": str(args.npz),
        "input_npz_sha256": sha256_file(args.npz),
        "dataset_csv": str(args.csv),
        "selection": {
            "strategy": "requested" if requested_ids is not None else args.selection,
            "requested_count": args.num_samples,
            "selected_count": len(selected),
            "seed": args.seed,
            "only_valid": args.only_valid,
            "sample_id_file": (
                str(args.sample_id_file.resolve())
                if args.sample_id_file is not None
                else None
            ),
        },
        "legend": {
            "anatomical_left_contour": "green",
            "anatomical_right_contour": "blue",
            "iris_rim_and_fitted_circle": "orange/yellow",
            "iris_center": "red",
            "canthi": "cyan",
            "point_names_by_index": list(EYE_POINT_NAMES),
        },
        "pages": [str(path.resolve()) for path in page_paths],
        "samples": records,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"Wrote {len(page_paths)} QC page(s) to: {args.output_dir}")
    print(f"Wrote QC manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
