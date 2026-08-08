"""Delete corresponding raw, face, and eye images for selected frame indices.

The CrossGaze collection stores one sample in four places. This utility keeps
them in sync for both naming formats used by the collection:
``frame_000123_<timestamp>.jpg`` and ``img_<timestamp>_00021.jpg``.

Run without ``--delete`` to preview the files first.  Add ``--delete`` only
after checking the preview, because deletion cannot be undone by this script.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(r"D:\GithubCode\CrossGaze-main\CrossGaze-main\data_collection")
FRAME_INDEX_WIDTH = 6

# 修改这两项后直接运行脚本即可。DATASET 可以是 3，也可以是完整目录名。
DATASET = "5"
# 支持单个索引、范围和混用，例如："1-5,8,10-12"。
# INDICES = "21,26-34,91-101,167-193,267-282,339-353,394-405,453-466,505-514,553-561,601-602"  # 3
# INDICES = "54-61,100-104,149-159,194,256-261,333-357,401-408"  # 4
# INDICES = "33-37,65-73,140,212-222,230"  # 5
# INDICES = "428,458-459,518-522,548-550,577-581,711-760,864-892,1020"  # 11 384

IMAGE_DIRECTORIES = {
    "original": Path("insightface_img"),
    "face": Path("insightface_face"),
    "left_eye": Path("insightface_eyes") / "left_eye",
    "right_eye": Path("insightface_eyes") / "right_eye",
}


def parse_frame_indices(specification: str) -> list[int]:
    """Parse ``1-5,8,10-12`` into sorted, de-duplicated frame indices."""
    if not specification or not specification.strip():
        raise argparse.ArgumentTypeError("索引不能为空，例如：1-5,8,10-12")

    indices: set[int] = set()
    tokens = re.split(r"[,，\s]+", specification.strip())
    for token in tokens:
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", token)
        if match is None:
            raise argparse.ArgumentTypeError(
                f"无效索引片段：{token!r}；请使用 1、1-5 或 1-5,8,10-12"
            )

        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            raise argparse.ArgumentTypeError(f"索引范围必须从小到大：{token!r}")
        indices.update(range(start, end + 1))

    return sorted(indices)


def normalize_dataset_name(value: str) -> str:
    """Accept a dataset number (``3``) or a full dataset directory name."""
    value = value.strip()
    if value.isdigit():
        return f"dataset_dual_rigid_body_{int(value)}"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="真正删除文件；未提供时仅显示将要处理的文件。",
    )
    return parser.parse_args()


def matching_name_patterns(index: int) -> tuple[str, ...]:
    """Return patterns for the legacy and current CrossGaze file names."""
    return (
        f"frame_{index:06d}_*",  # frame_000123_<timestamp>.jpg
        f"*_{index:05d}.*",  # img_<timestamp>_00123.jpg
    )


def iter_matching_files(dataset_dir: Path, frame_indices: Iterable[int]) -> Iterable[tuple[str, Path]]:
    for index in frame_indices:
        for kind, relative_dir in IMAGE_DIRECTORIES.items():
            image_dir = dataset_dir / relative_dir
            if not image_dir.is_dir():
                continue
            seen: set[Path] = set()
            for pattern in matching_name_patterns(index):
                for path in sorted(image_dir.glob(pattern)):
                    if path.is_file() and path not in seen:
                        seen.add(path)
                        yield kind, path


def main() -> int:
    args = parse_args()
    frame_indices = parse_frame_indices(INDICES)
    dataset_name = normalize_dataset_name(DATASET)
    dataset_dir = DEFAULT_SOURCE_ROOT / dataset_name
    if not dataset_dir.is_dir():
        raise SystemExit(f"找不到数据集目录：{dataset_dir}")

    matches = list(iter_matching_files(dataset_dir, frame_indices))
    expected_per_index = len(IMAGE_DIRECTORIES)
    expected_total = len(frame_indices) * expected_per_index

    print(f"数据集：{dataset_dir}")
    print(f"帧索引：{', '.join(map(str, frame_indices))}")
    print(f"匹配到 {len(matches)} / {expected_total} 个图像文件：")
    for kind, path in matches:
        print(f"  [{kind}] {path}")

    if not args.delete:
        print("\n这是预览，未删除任何文件。确认无误后加 --delete 执行删除。")
        return 0

    for _, path in matches:
        path.unlink()
    print(f"\n已删除 {len(matches)} 个图像文件。")
    if len(matches) != expected_total:
        print("注意：部分索引在一个或多个图像目录中未找到对应文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
