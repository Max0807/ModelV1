"""Real-camera calibration helpers used by preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CROSSGAZE_CAMERA_MATRIX = (
    (1367.8584, 0.0, 957.9159),
    (0.0, 1369.0087, 543.3381),
    (0.0, 0.0, 1.0),
)
CROSSGAZE_DIST_COEFFS = (
    1.78483784e-01,
    -5.90774600e-01,
    -5.11403240e-04,
    -1.08456025e-03,
    5.95053471e-01,
)
CROSSGAZE_IMAGE_WIDTH_PX = 1920
CROSSGAZE_IMAGE_HEIGHT_PX = 1080


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """确认相机 YAML 根节点是映射类型并返回该映射。"""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping.")
    return value


def _matrix3x3(value: Any, name: str) -> tuple[tuple[float, float, float], ...]:
    """把 YAML 值转换为严格的 3×3 浮点矩阵并验证行列数。"""

    if value is None:
        raise ValueError(f"{name} is required.")
    rows = list(value)
    if len(rows) != 3:
        raise ValueError(f"{name} must have exactly 3 rows.")
    matrix: list[tuple[float, float, float]] = []
    for row in rows:
        values = tuple(float(item) for item in row)
        if len(values) != 3:
            raise ValueError(f"{name} rows must have exactly 3 values.")
        matrix.append(values)
    return tuple(matrix)


@dataclass(frozen=True)
class PinholeCameraCalibration:
    name: str
    image_width_px: int
    image_height_px: int
    camera_matrix: tuple[tuple[float, float, float], ...]
    distortion_coefficients: tuple[float, ...]

    def validate_image_size(self, width_px: int, height_px: int) -> None:
        """确认原图尺寸与这份内参标定时使用的分辨率完全一致。"""

        actual = (int(width_px), int(height_px))
        expected = (self.image_width_px, self.image_height_px)
        if actual != expected:
            raise ValueError(
                f"Camera calibration {self.name!r} expects "
                f"{expected[0]}x{expected[1]} px, got "
                f"{actual[0]}x{actual[1]} px."
            )


def load_camera_calibration(path: str | Path) -> PinholeCameraCalibration:
    """Load an OpenCV pinhole camera calibration from YAML."""

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read camera calibration files.") from error

    calibration_path = Path(path).resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Camera calibration file does not exist: {calibration_path}"
        )
    payload = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    root = _mapping(payload, "camera calibration")
    matrix = _matrix3x3(root.get("camera_matrix"), "camera_matrix")
    distortion_raw = root.get("distortion_coefficients", ())
    distortion = tuple(float(item) for item in distortion_raw)
    if len(distortion) not in {4, 5, 8, 12, 14}:
        raise ValueError(
            "distortion_coefficients must contain 4, 5, 8, 12, or 14 values."
        )
    width = int(root.get("image_width_px", 0))
    height = int(root.get("image_height_px", 0))
    if width <= 0 or height <= 0:
        raise ValueError("image_width_px and image_height_px must be positive.")
    if matrix[0][0] <= 0 or matrix[1][1] <= 0:
        raise ValueError("camera_matrix focal lengths must be positive.")
    if matrix[2][2] == 0:
        raise ValueError("camera_matrix[2][2] must be non-zero.")
    return PinholeCameraCalibration(
        name=str(root.get("name", calibration_path.stem)),
        image_width_px=width,
        image_height_px=height,
        camera_matrix=matrix,
        distortion_coefficients=distortion,
    )


def validate_crossgaze_image_size(width_px: int, height_px: int) -> None:
    """校验图像是否符合 CrossGaze 固定的 1920×1080 标定分辨率。"""

    actual = (int(width_px), int(height_px))
    expected = (CROSSGAZE_IMAGE_WIDTH_PX, CROSSGAZE_IMAGE_HEIGHT_PX)
    if actual != expected:
        raise ValueError(
            f"CrossGaze calibration expects {expected[0]}x{expected[1]} px, "
            f"got {actual[0]}x{actual[1]} px."
        )
