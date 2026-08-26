"""Quarantine image files and remove matching data-log rows for selected indices.

The CrossGaze collection stores one sample in four image directories and in a
``data_log_*.csv`` file. This utility keeps them in sync for both image naming
formats used by the collection:
``frame_000123_<timestamp>.jpg`` and ``img_<timestamp>_00021.jpg``.

Run without ``--delete`` to preview the files first.  Add ``--delete`` only
after checking the preview. Images are moved into a timestamped quarantine
directory, and each modified CSV is backed up there before being rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile


DEFAULT_SOURCE_ROOT = Path(r"D:\GithubCode\CrossGaze-main\CrossGaze-main\data_collection")
FRAME_INDEX_WIDTH = 6
QUARANTINE_DIRECTORY = "_low_quality_quarantine"

DATASET = "13"  # # img 过滤
# INDICES = "21,26-34,91-101,167-193,267-282,338,339-353,354,355,394-405,453-466,505-514,553-561,601-602"  # 3 51
# INDICES = ""  # 4  166
# INDICES = ","  # 5
# INDICES = "226-233,236-257,299-300,326-327,331,336,345-346,348,350-351,356,376,391-436,440-443,445-446,449-452,463-467,469-470,473-477,483-484,490-491,495-500," \
# "505-537,542,607-649,802-837,856-863,893-896,902-905,909-911,927,982-990,993-998"  # 11 384
# INDICES = "2-3,5-7,10-12,15-18,22-24,29-30,32-34,37-39,43-44,46-49,52-53,55,57,62-63,66-68,72-73,75-78,81,83,88,91-93,96-97,101-102,104,106,108,110,112"  # 11 
# INDICES = "40-45,91-98,103-107,113-114,138-151,165-173,186-197,199,221-230,238-247,275-276,292,296-303,309,382-393,400-401,414-421,427-435,439-469," \
# "480-486,490-496,499502,578,621-626,635-638,645-666,719-725,748-754,760-761,791-803,856-875,881-900,909-911,1060-1107,1137,1142-1149,1172-1175,1178-1179," \
# "1183,1191-1194,1249-1261,1265-1269,1272-1294,1315-1329,1346-1354,1376"  # 13 -404


# DATASET = "3"  # # face 过滤
# INDICES = "64,66,136,153,201,206,208,214,219,229,262,293,302,320-321,322,377,382,445,486,540,592"  # 3 22
# INDICES = "76,78,216,218-219,221-222,"  # 4
# INDICES = "88,89,190-193,196-197"  # 5
# INDICES = "369,371-372,375,377,384-390,462,480,482,564-567,582,"  # 11 384
# INDICES = "50-52,101-102,108,497-504,1138-1141,"  # 13


# DATASET = "3"  # # left eye 过滤
# INDICES = "89-109,160,165-197,314,317,323,393-411,467-468,562-568,605-606"  # 3 26
# INDICES = "10,53,170,195,197-198,"  # 4
# INDICES = "33,210-218,220,229"  # 5
# INDICES = ""  # 11 384
# INDICES = "470-474,708,735,738,790,848-851,877-878,1108-1109,1232-1233,1236-1271,1298,1300,1307,1330,1333,1396,1409,1414-1415"  # 11 384

# DATASET = "3"  # # right eye 过滤
# INDICES = "71,75,192,240,384-388,424-426 "# 3 11
# INDICES = "422,423"  # 4
# INDICES = "65-67,139"  # 5
# INDICES = "761"  # 11 384
# INDICES = "115-124,162-164,397-399,520,522-526,528,530,532-533,537,540,582-583,733-745,759,880,901,1413-1417,"  # 11 384

# DATASET = "3"  # data 数据异常
# INDICES = "36,37,126-134,458-459,469,490-496,506-519,561"  # 3 13
# INDICES = "12-23,42-47,51-52,59-60,86-94,98-100,107-118,180-181,200-210,238-246,254-297,303-421,"  # 4
# INDICES = "54-61,100-104,149-159,194,256-261,333-357,401-408"  # 4
# INDICES = "8-11,28,38,61,71-72,104,108,161-165,224,227-228"  # 5
# INDICES = "458-459,548-550,577-581,711-760,864-896,1020"  # 11 384

IMAGE_DIRECTORIES = {
    "original": Path("insightface_img"),
    "face": Path("insightface_face"),
    "left_eye": Path("insightface_eyes") / "left_eye",
    "right_eye": Path("insightface_eyes") / "right_eye",
}


@dataclass
class DataLogUpdate:
    """Rows to retain/remove when atomically rewriting one data-log CSV."""

    path: Path
    header: list[str]
    retained_rows: list[list[str]]
    removed_rows: list[list[str]]
    encoding: str
    line_terminator: str


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
        help="移动图像到隔离目录并更新日志；未提供时仅预览，不会永久删除图像。",
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


def frame_index_from_image_filename(image_filename: str) -> int | None:
    """Extract the frame index from either supported image filename format."""
    name = Path(image_filename.strip()).name
    match = re.fullmatch(r"frame_(\d{6})_.+", name)
    if match is None:
        match = re.fullmatch(r"img_.+_(\d{5})\.[^.]+", name)
    return int(match.group(1)) if match is not None else None


def row_matches_indices(
    row: list[str],
    frame_indices: set[int],
    image_filename_column: int | None,
    frame_idx_column: int | None,
) -> bool:
    """Match by image filename, falling back to ``frame_idx`` when necessary."""
    if image_filename_column is not None and image_filename_column < len(row):
        index = frame_index_from_image_filename(row[image_filename_column])
        if index is not None:
            return index in frame_indices

    if frame_idx_column is not None and frame_idx_column < len(row):
        try:
            return int(row[frame_idx_column].strip()) in frame_indices
        except ValueError:
            pass
    return False


def plan_data_log_updates(dataset_dir: Path, frame_indices: Iterable[int]) -> list[DataLogUpdate]:
    """Read every data log and identify rows corresponding to selected images."""
    selected = set(frame_indices)
    updates: list[DataLogUpdate] = []

    for path in sorted(dataset_dir.glob("data_log_*.csv")):
        with path.open("rb") as raw_file:
            sample = raw_file.read(8192)
        encoding = "utf-8-sig" if sample.startswith(b"\xef\xbb\xbf") else "utf-8"
        line_terminator = "\r\n" if b"\r\n" in sample else "\n"

        with path.open("r", encoding=encoding, newline="") as csv_file:
            reader = csv.reader(csv_file)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"数据日志为空：{path}") from exc
            rows = list(reader)

        image_filename_column = (
            header.index("image_filename") if "image_filename" in header else None
        )
        frame_idx_column = header.index("frame_idx") if "frame_idx" in header else None
        if image_filename_column is None and frame_idx_column is None:
            raise ValueError(
                f"数据日志缺少 image_filename 和 frame_idx 列，无法匹配图像：{path}"
            )

        removed_rows: list[list[str]] = []
        retained_rows: list[list[str]] = []
        for row in rows:
            destination = (
                removed_rows
                if row_matches_indices(row, selected, image_filename_column, frame_idx_column)
                else retained_rows
            )
            destination.append(row)

        updates.append(
            DataLogUpdate(
                path=path,
                header=header,
                retained_rows=retained_rows,
                removed_rows=removed_rows,
                encoding=encoding,
                line_terminator=line_terminator,
            )
        )

    return updates


def describe_log_row(update: DataLogUpdate, row: list[str]) -> str:
    """Return the useful identifiers from a data-log row for dry-run output."""
    details: list[str] = []
    for column in ("frame_idx", "image_filename"):
        if column in update.header:
            position = update.header.index(column)
            if position < len(row):
                details.append(f"{column}={row[position]}")
    return ", ".join(details)


def apply_data_log_update(update: DataLogUpdate) -> None:
    """Atomically replace one CSV with its selected rows removed."""
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding=update.encoding,
            newline="",
            dir=update.path.parent,
            prefix=f".{update.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.writer(temporary_file, lineterminator=update.line_terminator)
            writer.writerow(update.header)
            writer.writerows(update.retained_rows)
        temporary_path.replace(update.path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_quarantine_batch(dataset_dir: Path) -> Path:
    """Create a unique timestamped directory for one quarantine operation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_dir = dataset_dir / QUARANTINE_DIRECTORY / timestamp
    batch_dir.mkdir(parents=True, exist_ok=False)
    return batch_dir


def quarantine_changes(
    dataset_dir: Path,
    matches: Iterable[tuple[str, Path]],
    log_updates: Iterable[DataLogUpdate],
    frame_indices: Iterable[int],
) -> Path:
    """Back up logs, rewrite them, and move images into a recoverable batch."""
    image_matches = list(matches)
    changed_logs = [update for update in log_updates if update.removed_rows]
    batch_dir = create_quarantine_batch(dataset_dir)
    images_dir = batch_dir / "images"
    data_logs_dir = batch_dir / "data_logs"

    image_moves: list[tuple[Path, Path]] = []
    image_manifest: list[dict[str, str]] = []
    for kind, source_path in image_matches:
        relative_path = source_path.relative_to(dataset_dir)
        destination_path = images_dir / relative_path
        if destination_path.exists():
            raise FileExistsError(f"隔离目录中已存在同名图像：{destination_path}")
        image_moves.append((source_path, destination_path))
        image_manifest.append(
            {
                "kind": kind,
                "original_path": relative_path.as_posix(),
                "quarantine_path": destination_path.relative_to(batch_dir).as_posix(),
            }
        )

    log_manifest: list[dict[str, str | int]] = []
    for update in changed_logs:
        backup_path = data_logs_dir / update.path.name
        log_manifest.append(
            {
                "original_path": update.path.relative_to(dataset_dir).as_posix(),
                "backup_path": backup_path.relative_to(batch_dir).as_posix(),
                "removed_rows": len(update.removed_rows),
            }
        )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_dir": str(dataset_dir),
        "frame_indices": list(frame_indices),
        "images": image_manifest,
        "data_logs": log_manifest,
    }
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 所有原始 CSV 都成功备份后，才开始改写任何日志。
    if changed_logs:
        data_logs_dir.mkdir(parents=True, exist_ok=True)
    for update in changed_logs:
        shutil.copy2(update.path, data_logs_dir / update.path.name)
    for update in changed_logs:
        apply_data_log_update(update)

    # 日志不再引用这些图像后，再将图像移入隔离目录。
    for source_path, destination_path in image_moves:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(destination_path)

    return batch_dir


def main() -> int:
    args = parse_args()
    frame_indices = parse_frame_indices(INDICES)
    dataset_name = normalize_dataset_name(DATASET)
    dataset_dir = DEFAULT_SOURCE_ROOT / dataset_name
    if not dataset_dir.is_dir():
        raise SystemExit(f"找不到数据集目录：{dataset_dir}")

    matches = list(iter_matching_files(dataset_dir, frame_indices))
    try:
        log_updates = plan_data_log_updates(dataset_dir, frame_indices)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise SystemExit(f"读取数据日志失败：{exc}") from exc
    expected_per_index = len(IMAGE_DIRECTORIES)
    expected_total = len(frame_indices) * expected_per_index
    removed_log_rows = sum(len(update.removed_rows) for update in log_updates)

    print(f"数据集：{dataset_dir}")
    print(f"帧索引：{', '.join(map(str, frame_indices))}")
    print(f"匹配到 {len(matches)} / {expected_total} 个图像文件：")
    for kind, path in matches:
        print(f"  [{kind}] {path}")

    print(f"\n匹配到 {removed_log_rows} 条 data_log 记录：")
    if not log_updates:
        print("  未找到 data_log_*.csv")
    for update in log_updates:
        print(f"  [{update.path.name}] {len(update.removed_rows)} 条")
        for row in update.removed_rows:
            print(f"    {describe_log_row(update, row)}")

    if not args.delete:
        print(
            "\n这是预览，未移动图像或修改日志。"
            "确认无误后加 --delete 执行可恢复的隔离操作。"
        )
        return 0

    if not matches and removed_log_rows == 0:
        print("\n没有匹配到需要处理的图像或 data_log 记录。")
        return 0

    try:
        quarantine_dir = quarantine_changes(
            dataset_dir,
            matches,
            log_updates,
            frame_indices,
        )
    except OSError as exc:
        raise SystemExit(f"隔离操作失败：{exc}") from exc

    print(
        f"\n已将 {len(matches)} 个图像移入隔离目录，"
        f"并从工作日志移除 {removed_log_rows} 条记录。"
    )
    print(f"隔离批次与 CSV 备份：{quarantine_dir}")
    if len(matches) != expected_total:
        print("注意：部分索引在一个或多个图像目录中未找到对应文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
