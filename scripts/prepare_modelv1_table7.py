"""Run the final three-stage ModelV1 TableFrame7 preprocessing pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.table7_preprocessing_config import (
    DEFAULT_CONFIG_PATH,
    load_table7_preprocessing_config,
)


def parse_args() -> argparse.Namespace:
    """解析统一配置文件路径，供完整 Stage 1–3 调度入口使用。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _run(name: str, script: str, config_path: Path) -> None:
    """用当前 Python 解释器启动一个阶段子进程，并在失败时立即终止流水线。"""

    command = [sys.executable, script, "--config", str(config_path)]
    print(f"\n[{name}]\n{subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    """读取配置并按 execution.stages 顺序调度 Stage 1、2A、2B 和 Stage 3。"""

    args = parse_args()
    config = load_table7_preprocessing_config(args.config)
    tasks = {
        1: (
            ("1 base data integration", "scripts/build_modelv1_table7_base.py"),
        ),
        2: (
            ("2A one-pass MediaPipe eye15 + iris + PnP observations", "scripts/extract_mediapipe_table7_observations.py"),
            ("2B Iris65 metric head pose", "scripts/build_iris65_metric_head_pose.py"),
        ),
        3: (
            ("3 virtual camera images + table_frame7_n[7]", "scripts/build_virtual_camera_table7.py"),
        ),
    }
    selected = [task for stage in config.stages for task in tasks[stage]]  # 按配置顺序展开选定的阶段任务
    if config.dry_run:
        print(f"Config: {config.config_path}")
        print(f"Output root: {config.output_root}")
        for name, script in selected:
            command = [sys.executable, script, "--config", str(config.config_path)]
            print(f"[dry-run] {name}: {subprocess.list2cmdline(command)}")
        return 0
    for name, script in selected:
        _run(name, script, config.config_path)
    print(f"\nCompleted stages {list(config.stages)}. Final CSV: {config.final_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
