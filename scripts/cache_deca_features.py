"""Extract frozen 236-D DECA ``E_flame`` features for ModelV1 samples.

The input CSV must contain ``sample_id`` and ``face_path``.  With
``--face-preprocess deca`` it must also contain the source image path and face
bounding box columns. The output is one compressed NPZ cache aligned by sample
id, so it can be consumed directly by ``ModelV1Dataset(deca_cache_path=...)``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.deca_cache import (
    DECA_CACHE_FORMAT_VERSION,
    DECA_FEATURE_DIM,
    DECA_PARAMETER_LAYOUT,
    DecaFeatureCache,
)
from modelv1.depth_prior.face_preprocess import (
    DEFAULT_DECA_CROP_SCALE,
    FACE_PREPROCESS_CHOICES,
    FACE_PREPROCESS_DECA,
    FACE_PREPROCESS_LEGACY,
    prepare_deca_face_image,
)


DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "modelv1_dataset.csv"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "deca_features_v1.npz"
DEFAULT_DECA_ROOT = PROJECT_ROOT / "DECA-master"
DECA_PREPROCESS_VERSION = "deca_bbox_crop_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--deca-root", type=Path, default=DEFAULT_DECA_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <deca-root>/data/deca_model.tar.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--face-preprocess",
        choices=FACE_PREPROCESS_CHOICES,
        default=FACE_PREPROCESS_LEGACY,
        help=(
            "DECA input crop. 'legacy' reproduces the existing saved face crop; "
            "'deca' renders the official square bbox crop from source_image_path."
        ),
    )
    parser.add_argument(
        "--deca-crop-scale",
        type=float,
        default=DEFAULT_DECA_CROP_SCALE,
        help="Official-style DECA bbox crop scale, used only with --face-preprocess deca.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cuda, cuda:0, cpu, or auto (cuda when available).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N CSV rows."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore reusable entries from an existing output cache.",
    )
    parser.add_argument(
        "--verify-cache",
        action="store_true",
        help="Validate that --output covers every sample in --csv; do not run DECA.",
    )
    return parser.parse_args()


def read_rows(
    csv_path: Path,
    limit: int | None,
    face_preprocess: str,
) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Dataset CSV has no rows: {csv_path}")
    required = {"sample_id", "face_path"}
    if face_preprocess == FACE_PREPROCESS_DECA:
        required.update(
            {
                "source_image_path",
                "face_bbox_x",
                "face_bbox_y",
                "face_bbox_w",
                "face_bbox_h",
            }
        )
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Dataset CSV is missing columns: {sorted(missing)}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Dataset CSV contains duplicate sample_id values")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def face_bbox_xywh(row: dict[str, str]) -> tuple[float, float, float, float]:
    """Read the detector box used by the official-style DECA crop."""

    columns = ("face_bbox_x", "face_bbox_y", "face_bbox_w", "face_bbox_h")
    try:
        return tuple(float(row[column]) for column in columns)  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid face bounding box for sample_id={row.get('sample_id')!r}"
        ) from error


def collect_input_hashes(
    rows: Iterable[dict[str, str]],
    *,
    face_preprocess: str,
    deca_crop_scale: float,
) -> list[str]:
    """Hash every input affecting the rendered DECA tensor."""

    hashes: list[str] = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(
            row["source_image_path"]
            if face_preprocess == FACE_PREPROCESS_DECA
            else row["face_path"]
        )
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing DECA input image for sample_id={row['sample_id']!r}: {image_path}"
            )
        digest = hashlib.sha256()
        digest.update(sha256_file(image_path).encode("ascii"))
        digest.update(face_preprocess.encode("ascii"))
        digest.update(DECA_PREPROCESS_VERSION.encode("ascii"))
        if face_preprocess == FACE_PREPROCESS_DECA:
            bbox_text = ",".join(f"{value:.8f}" for value in face_bbox_xywh(row))
            digest.update(bbox_text.encode("ascii"))
            digest.update(f"{deca_crop_scale:.8f}".encode("ascii"))
        hashes.append(digest.hexdigest())
        if index % 100 == 0:
            print(f"Hashed {index} DECA inputs")
    return hashes


def validate_deca_checkout(deca_root: Path, checkpoint: Path) -> None:
    required_files = [
        deca_root / "decalib" / "models" / "encoders.py",
        deca_root / "decalib" / "models" / "resnet.py",
        checkpoint,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The DECA checkout is incomplete. Missing:\n- "
            + "\n- ".join(missing)
            + "\nUse a complete official DECA checkout and place deca_model.tar under data/."
        )


def import_torch_modules() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DECA caching requires torch, torchvision, numpy, and Pillow. "
            "Install requirements.txt plus a torch/torchvision build for your device."
        ) from exc
    return np, torch, Image


def load_encoder(deca_root: Path, checkpoint: Path, device: str, torch: Any) -> Any:
    deca_root_text = str(deca_root.resolve())
    if deca_root_text not in sys.path:
        sys.path.insert(0, deca_root_text)
    importlib.invalidate_caches()
    try:
        from decalib.models.encoders import ResnetEncoder
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Unable to import DECA's ResnetEncoder. Ensure torchvision is installed "
            "and --deca-root points at the official DECA source checkout."
        ) from exc

    encoder = ResnetEncoder(outsize=DECA_FEATURE_DIM).to(device)
    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
    if not isinstance(checkpoint_data, dict) or "E_flame" not in checkpoint_data:
        raise ValueError(f"Checkpoint lacks the expected E_flame state dict: {checkpoint}")
    result = encoder.load_state_dict(checkpoint_data["E_flame"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "E_flame checkpoint mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return encoder.eval()


def resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is not available")
    return requested


def load_deca_face_batch(
    rows: list[dict[str, str]],
    *,
    face_preprocess: str,
    deca_crop_scale: float,
    image_cls: Any,
    np: Any,
    torch: Any,
) -> Any:
    """Render one batch exactly as the selected DECA input convention requires."""

    images = []
    for row in rows:
        if face_preprocess == FACE_PREPROCESS_DECA:
            source_path = Path(row["source_image_path"])
            with image_cls.open(source_path) as source_image:
                source_rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
            image_rgb, _ = prepare_deca_face_image(
                source_rgb,
                face_bbox_xywh(row),
                mode=FACE_PREPROCESS_DECA,
                input_size=224,
                deca_crop_scale=deca_crop_scale,
            )
        else:
            path = Path(row["face_path"])
            with image_cls.open(path) as source_image:
                image = source_image.convert("RGB")
            if image.size != (224, 224):
                image = image.resize((224, 224), image_cls.Resampling.BILINEAR)
            image_rgb = np.asarray(image, dtype=np.uint8)
        array = np.asarray(image_rgb, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(array.transpose(2, 0, 1)))
    return torch.stack(images, dim=0)


def reusable_features(
    cache_path: Path,
    sample_ids: list[str],
    image_hashes: list[str],
    checkpoint_hash: str,
    face_preprocess: str,
    deca_crop_scale: float,
    overwrite: bool,
) -> dict[str, Any]:
    if overwrite or not cache_path.exists():
        return {}
    cache = DecaFeatureCache.load(cache_path)
    old_checkpoint = str(cache.metadata.get("deca_checkpoint_sha256", ""))
    if old_checkpoint != checkpoint_hash:
        print("DECA checkpoint changed or is unknown; regenerating all features")
        return {}
    cached_preprocess = str(
        cache.metadata.get("face_preprocess", FACE_PREPROCESS_LEGACY)
    )
    if cached_preprocess != face_preprocess:
        print("DECA face preprocessing changed or is unknown; regenerating all features")
        return {}
    if face_preprocess == FACE_PREPROCESS_DECA:
        cached_scale = cache.metadata.get("deca_crop_scale")
        if cached_scale is None or abs(float(cached_scale) - deca_crop_scale) > 1e-8:
            print("DECA crop scale changed or is unknown; regenerating all features")
            return {}
    result = {}
    for sample_id, image_hash in zip(sample_ids, image_hashes):
        if not cache.has_sample_id(sample_id):
            continue
        if cache.image_digest(sample_id) == image_hash:
            result[sample_id] = cache.lookup(sample_id).copy()
    return result


def write_cache(
    output: Path,
    features: Any,
    sample_ids: list[str],
    image_hashes: list[str],
    metadata: dict[str, object],
    np: Any,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_suffix(output.suffix + ".tmp")
    try:
        with temp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                deca_feat=np.asarray(features, dtype=np.float32),
                sample_id=np.asarray(sample_ids, dtype=str),
                image_sha256=np.asarray(image_hashes, dtype=str),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=str),
            )
        os.replace(temp_path, output)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    args = parse_args()
    if args.deca_crop_scale <= 0:
        raise ValueError("--deca-crop-scale must be positive")
    rows = read_rows(args.csv, args.limit, args.face_preprocess)
    sample_ids = [row["sample_id"] for row in rows]
    if args.verify_cache:
        cache = DecaFeatureCache.load(args.output)
        cache.require_sample_ids(sample_ids)
        cached_preprocess = str(
            cache.metadata.get("face_preprocess", FACE_PREPROCESS_LEGACY)
        )
        if cached_preprocess != args.face_preprocess:
            raise ValueError(
                "DECA cache preprocessing does not match --face-preprocess. "
                "Rebuild the cache with the intended preprocessing."
            )
        if args.face_preprocess == FACE_PREPROCESS_DECA:
            cached_scale = cache.metadata.get("deca_crop_scale")
            if cached_scale is None or abs(float(cached_scale) - args.deca_crop_scale) > 1e-8:
                raise ValueError(
                    "DECA cache crop scale does not match --deca-crop-scale. "
                    "Rebuild the cache with the intended preprocessing."
                )
        print(f"Valid cache: {args.output}")
        print(f"Samples: {len(sample_ids)}, feature dim: {cache.features.shape[1]}")
        return 0

    np, torch, image_cls = import_torch_modules()
    checkpoint = args.checkpoint or args.deca_root / "data" / "deca_model.tar"
    validate_deca_checkout(args.deca_root, checkpoint)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = resolve_device(args.device, torch)
    image_hashes = collect_input_hashes(
        rows,
        face_preprocess=args.face_preprocess,
        deca_crop_scale=args.deca_crop_scale,
    )
    checkpoint_hash = sha256_file(checkpoint)
    cached = reusable_features(
        args.output,
        sample_ids,
        image_hashes,
        checkpoint_hash,
        args.face_preprocess,
        args.deca_crop_scale,
        args.overwrite,
    )
    encoder = load_encoder(args.deca_root, checkpoint, device, torch)

    features_by_id = dict(cached)
    pending = [row for row in rows if row["sample_id"] not in features_by_id]
    print(
        f"Using {device}; samples={len(rows)}, reused={len(cached)}, "
        f"to_extract={len(pending)}"
    )
    with torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch_rows = pending[start : start + args.batch_size]
            images = load_deca_face_batch(
                batch_rows,
                face_preprocess=args.face_preprocess,
                deca_crop_scale=args.deca_crop_scale,
                image_cls=image_cls,
                np=np,
                torch=torch,
            ).to(device)
            encoded = encoder(images).detach().cpu().to(dtype=torch.float32).numpy()
            if encoded.shape != (len(batch_rows), DECA_FEATURE_DIM):
                raise RuntimeError(f"Unexpected E_flame output shape: {encoded.shape}")
            for row, feature in zip(batch_rows, encoded):
                features_by_id[row["sample_id"]] = feature
            print(f"Extracted {min(start + len(batch_rows), len(pending))}/{len(pending)}")

    features = np.stack([features_by_id[sample_id] for sample_id in sample_ids], axis=0)
    metadata: dict[str, object] = {
        "cache_format_version": DECA_CACHE_FORMAT_VERSION,
        "feature_dim": DECA_FEATURE_DIM,
        "parameter_layout": DECA_PARAMETER_LAYOUT,
        "feature_source": "DECA E_flame coarse parameter vector",
        "deca_root": str(args.deca_root.resolve()),
        "deca_checkpoint": str(checkpoint.resolve()),
        "deca_checkpoint_sha256": checkpoint_hash,
        "face_preprocess": args.face_preprocess,
        "deca_crop_scale": args.deca_crop_scale,
        "preprocess_version": DECA_PREPROCESS_VERSION,
        "preprocess": (
            "legacy: saved detector face crop resized to 224x224; "
            "deca: official-style square bbox crop from source image, then resized to 224x224; "
            "RGB float32 pixels in [0, 1]"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(sample_ids),
    }
    write_cache(args.output, features, sample_ids, image_hashes, metadata, np)
    print(f"Wrote cache: {args.output}")
    print(f"Feature matrix: {features.shape}, dtype={features.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
