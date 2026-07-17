"""Extract frozen 236-D DECA ``E_flame`` features for ModelV1 face crops.

The input CSV must contain ``sample_id`` and ``face_path``.  The output is one
compressed NPZ cache aligned by sample id, so it can be consumed directly by
``ModelV1Dataset(deca_cache_path=...)``.
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


DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "modelv1_dataset.csv"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "deca_features_v1.npz"
DEFAULT_DECA_ROOT = PROJECT_ROOT / "DECA-master"


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


def read_rows(csv_path: Path, limit: int | None) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Dataset CSV has no rows: {csv_path}")
    required = {"sample_id", "face_path"}
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


def collect_image_hashes(rows: Iterable[dict[str, str]]) -> list[str]:
    hashes: list[str] = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(row["face_path"])
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing face image for sample_id={row['sample_id']!r}: {image_path}"
            )
        hashes.append(sha256_file(image_path))
        if index % 100 == 0:
            print(f"Hashed {index} face images")
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


def load_face_batch(paths: list[Path], image_cls: Any, np: Any, torch: Any) -> Any:
    images = []
    for path in paths:
        with image_cls.open(path) as image:
            image = image.convert("RGB")
            if image.size != (224, 224):
                image = image.resize((224, 224), image_cls.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(array.transpose(2, 0, 1)))
    return torch.stack(images, dim=0)


def reusable_features(
    cache_path: Path,
    sample_ids: list[str],
    image_hashes: list[str],
    checkpoint_hash: str,
    overwrite: bool,
) -> dict[str, Any]:
    if overwrite or not cache_path.exists():
        return {}
    cache = DecaFeatureCache.load(cache_path)
    old_checkpoint = str(cache.metadata.get("deca_checkpoint_sha256", ""))
    if old_checkpoint != checkpoint_hash:
        print("DECA checkpoint changed or is unknown; regenerating all features")
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
    rows = read_rows(args.csv, args.limit)
    sample_ids = [row["sample_id"] for row in rows]
    if args.verify_cache:
        cache = DecaFeatureCache.load(args.output)
        cache.require_sample_ids(sample_ids)
        print(f"Valid cache: {args.output}")
        print(f"Samples: {len(sample_ids)}, feature dim: {cache.features.shape[1]}")
        return 0

    np, torch, image_cls = import_torch_modules()
    checkpoint = args.checkpoint or args.deca_root / "data" / "deca_model.tar"
    validate_deca_checkout(args.deca_root, checkpoint)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = resolve_device(args.device, torch)
    image_hashes = collect_image_hashes(rows)
    checkpoint_hash = sha256_file(checkpoint)
    cached = reusable_features(
        args.output, sample_ids, image_hashes, checkpoint_hash, args.overwrite
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
            batch_paths = [Path(row["face_path"]) for row in batch_rows]
            images = load_face_batch(batch_paths, image_cls, np, torch).to(device)
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
        "preprocess": "RGB, resize to 224x224 if needed, float32 pixels in [0, 1]",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(sample_ids),
    }
    write_cache(args.output, features, sample_ids, image_hashes, metadata, np)
    print(f"Wrote cache: {args.output}")
    print(f"Feature matrix: {features.shape}, dtype={features.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
