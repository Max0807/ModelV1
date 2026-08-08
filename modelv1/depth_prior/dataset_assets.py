"""Path resolution for dataset assets recorded in processed CSV files."""

from __future__ import annotations

from pathlib import Path


def resolve_image_asset_path(row: dict[str, str], field: str, image_root: Path | None = None) -> Path:
    """Resolve a saved CSV asset path, including the known moved-data layout."""
    original = Path(row[field])
    if original.is_file():
        return original

    source_dataset_dir = Path(row["source_dataset_dir"])
    try:
        asset_relative_path = original.relative_to(source_dataset_dir)
    except ValueError:
        asset_relative_path = Path(original.name)

    candidate_roots: list[Path] = []
    if image_root is not None:
        candidate_roots.append(image_root)

    # Historical CSVs point to <repo>/baseline/data_collection, whereas the
    # retained source images may live at <repo>/CrossGaze-main/data_collection.
    if source_dataset_dir.parent.name == "data_collection" and source_dataset_dir.parent.parent.name == "baseline":
        candidate_roots.append(source_dataset_dir.parent.parent.parent / "CrossGaze-main" / "data_collection")

    checked = [original]
    for root in candidate_roots:
        candidate = root / row["dataset"] / asset_relative_path
        checked.append(candidate)
        if candidate.is_file():
            return candidate

    checked_paths = "\n  ".join(str(path) for path in checked)
    raise FileNotFoundError(f"Could not find {field} for sample {row['sample_id']}. Checked:\n  {checked_paths}")
