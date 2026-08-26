"""Audit saved PnP pose tables against ModelV1's versioned geometry contract.

This validator is deliberately independent of DECA, MediaPipe, OpenCV, and
PyTorch. It checks provenance, one-to-one sample identity, finite values,
proper rotations, front-facing face/eye depths, and saved reprojection quality.
It exits non-zero when any hard contract violation is found.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelv1.depth_prior.pnp import PNP_GEOMETRY_VERSION, PnpConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-failed-rows",
        action="store_true",
        help=(
            "Audit every successful pose but do not treat failed reconstruction "
            "rows as hard geometry violations. Failed rows remain counted in the report."
        ),
    )
    return parser.parse_args()


def quantile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = sorted(value for value in values if math.isfinite(value))
    return {
        "count": len(finite),
        "min": quantile(finite, 0.0),
        "p05": quantile(finite, 0.05),
        "median": quantile(finite, 0.5),
        "p95": quantile(finite, 0.95),
        "max": quantile(finite, 1.0),
    }


def float_value(row: dict[str, str], *columns: str) -> float:
    for column in columns:
        raw = row.get(column, "").strip()
        if raw:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"non_finite:{column}")
            return value
    raise ValueError(f"missing:{'|'.join(columns)}")


def rotation_from_row(row: dict[str, str]) -> list[list[float]]:
    prefix = "rotation_" if row.get("rotation_00", "").strip() else "pnp_rotation_"
    return [
        [float_value(row, f"{prefix}{r}{c}") for c in range(3)]
        for r in range(3)
    ]


def matmul_transpose(rotation: list[list[float]]) -> list[list[float]]:
    return [
        [sum(rotation[k][i] * rotation[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def determinant(rotation: list[list[float]]) -> float:
    a = rotation
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def audit(path: Path, *, allow_failed_rows: bool = False) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    violations: Counter[str] = Counter()
    sample_ids: set[str] = set()
    versions: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    origin_depths: list[float] = []
    mean_errors: list[float] = []
    max_errors: list[float] = []
    confidences: list[float] = []
    orthogonality_errors: list[float] = []
    determinant_errors: list[float] = []
    success_count = 0
    failed_count = 0
    thresholds = PnpConfig()

    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            violations["missing_sample_id"] += 1
        elif sample_id in sample_ids:
            violations["duplicate_sample_id"] += 1
        else:
            sample_ids.add(sample_id)
        datasets[row.get("dataset", "").strip() or "missing"] += 1
        status = row.get("depth_prior_status", row.get("status", "")).strip().lower()
        if status != "success":
            failed_count += 1
            if not allow_failed_rows:
                violations["non_success_row"] += 1
            continue
        success_count += 1
        version = row.get("pnp_geometry_version", "").strip() or "missing"
        versions[version] += 1
        if version != PNP_GEOMETRY_VERSION:
            violations["pnp_geometry_version_mismatch"] += 1
        try:
            z = float_value(row, "tvec_z_mm", "pnp_tvec_z_mm")
            origin_depths.append(z)
            if not (
                thresholds.min_plausible_depth_mm
                <= z
                <= thresholds.max_plausible_depth_mm
            ):
                violations["face_origin_not_in_front"] += 1
            for side in ("left", "right"):
                eye_z = float_value(row, f"{side}_eye_camera_z_mm")
                if eye_z <= 0.0:
                    violations[f"{side}_eye_not_in_front"] += 1
            if row.get("pnp_min_object_depth_mm", "").strip():
                if float_value(row, "pnp_min_object_depth_mm") <= 0.0:
                    violations["pnp_landmark_not_in_front"] += 1
            else:
                violations["missing_pnp_min_object_depth"] += 1

            rotation = rotation_from_row(row)
            gram = matmul_transpose(rotation)
            orthogonality_error = max(
                abs(gram[i][j] - float(i == j))
                for i in range(3)
                for j in range(3)
            )
            det_error = abs(determinant(rotation) - 1.0)
            orthogonality_errors.append(orthogonality_error)
            determinant_errors.append(det_error)
            if orthogonality_error >= 1e-5:
                violations["rotation_not_orthonormal"] += 1
            if det_error >= 1e-5:
                violations["rotation_not_proper"] += 1

            mean_error = float_value(
                row,
                "reprojection_error_mean_px",
                "pnp_reprojection_error_mean_px",
            )
            max_error = float_value(
                row,
                "reprojection_error_max_px",
                "pnp_reprojection_error_max_px",
            )
            mean_errors.append(mean_error)
            max_errors.append(max_error)
            if mean_error > thresholds.max_reprojection_error_mean_px:
                violations["mean_reprojection_error_too_large"] += 1
            if max_error > thresholds.max_reprojection_error_max_px:
                violations["max_reprojection_error_too_large"] += 1
            scale_disagreement = float_value(row, "scale_disagreement_ratio")
            if scale_disagreement > thresholds.max_scale_disagreement_ratio:
                violations["scale_disagreement_too_large"] += 1
            if row.get("pnp_confidence", "").strip():
                confidence = float_value(row, "pnp_confidence")
                confidences.append(confidence)
                if not 0.0 <= confidence <= 1.0:
                    violations["pnp_confidence_out_of_range"] += 1
        except (KeyError, TypeError, ValueError):
            violations["missing_or_nonfinite_geometry"] += 1

    if success_count == 0:
        violations["no_success_rows"] += 1

    report = {
        "path": str(path.resolve()),
        "expected_pnp_geometry_version": PNP_GEOMETRY_VERSION,
        "hard_thresholds": {
            "face_origin_depth_mm": [
                thresholds.min_plausible_depth_mm,
                thresholds.max_plausible_depth_mm,
            ],
            "reprojection_error_mean_px_max": thresholds.max_reprojection_error_mean_px,
            "reprojection_error_max_px_max": thresholds.max_reprojection_error_max_px,
            "scale_disagreement_ratio_max": thresholds.max_scale_disagreement_ratio,
        },
        "row_count": len(rows),
        "success_count": success_count,
        "failed_count": failed_count,
        "allow_failed_rows": allow_failed_rows,
        "datasets": dict(datasets),
        "versions": dict(versions),
        "violations": dict(violations),
        "passed": not violations,
        "face_origin_depth_mm": distribution(origin_depths),
        "reprojection_error_mean_px": distribution(mean_errors),
        "reprojection_error_max_px": distribution(max_errors),
        "pnp_confidence": distribution(confidences),
        "rotation_orthogonality_error": distribution(orthogonality_errors),
        "rotation_determinant_error": distribution(determinant_errors),
    }
    return report


def main() -> int:
    args = parse_args()
    reports = [
        audit(path, allow_failed_rows=args.allow_failed_rows)
        for path in args.csv
    ]
    payload = {
        "pnp_geometry_version": PNP_GEOMETRY_VERSION,
        "passed": all(report["passed"] for report in reports),
        "files": reports,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
