# ModelV1 预处理流水线 v2

入口：[prepare_modelv1_preprocessing.py](../scripts/prepare_modelv1_preprocessing.py)
默认配置：[modelv1_preprocessing_v2.yaml](../configs/preprocessing/modelv1_preprocessing_v2.yaml)

## 调用方式

默认参数全部来自 YAML，正常运行无需再填写数据集、路径、设备或图像尺寸：

```powershell
& "D:\software\Anaconda\envs\cross_gaze\python.exe" scripts\prepare_modelv1_preprocessing.py
```

要保留多套环境，只需选择另一份配置：

```powershell
& "D:\software\Anaconda\envs\cross_gaze\python.exe" scripts\prepare_modelv1_preprocessing.py `
  --config configs\preprocessing\modelv1_preprocessing_v2.yaml
```

入口唯一的 CLI 参数是 `--config`。原来的 `--device`、路径、标定范围、图像尺寸等选项均已移到 YAML。

## 配置分组

| 配置段 | 作用 |
|---|---|
| `dataset` | 数据集 ID，以及 `pnp1010`、`iris65` 或 `both` 深度方案 |
| `paths` | 原始采集数据根目录、派生结果根目录 |
| `input` | 人脸图像目录名、InsightFace 坐标 CSV 文件名 |
| `dataset_build` | 手眼平移单位换算、桌面 Z 值来源 |
| `pnp1010` | PnP 固定尺度 |
| `iris65` | IPD、质量阈值、标定区间和不确定性选项 |
| `virtual_camera` | face/eye 输出尺寸和虚拟相机参数 |
| `execution` | 复用基础表、覆盖已有结果、dry run |

相对路径以仓库根目录为基准；`{dataset_id}` 会替换为 `dataset.id`，也支持环境变量。换机器时先改 `paths.source_root`；换数据集时还要检查 `dataset.id` 和 `iris65.calibration_*`。

## 输入契约

```text
<source_root>/
└── dataset_dual_rigid_body_<ID>/
    ├── data_log_*.csv                 # 必须恰好一个
    ├── <insightface_coordinates_file>
    ├── <face_image_dir>/
    └── insightface_img/
```

入口会验证数据集目录、唯一 `data_log_*.csv` 和配置指定的 InsightFace CSV，不再在多个硬编码根目录之间自动猜测。

## 深度方案

- `pnp1010`：训练使用固定尺度的 PnP 深度先验。
- `iris65`：训练使用 65 mm Iris-IPD 深度先验。
- `both`：生成两套先验，并生成共同有效样本比较集。

三种模式的虚拟相机都使用 PnP1010 `rotation_*` 和 `tvec_*`。Iris65 只改变训练深度先验，不改变虚拟相机中心和坐标轴。

## DECA 删除范围

入口已删除独立 DECA 特征缓存：

- 不再调用 `scripts/cache_deca_features.py`；
- 不再生成 `deca_features_*.npz`；
- 不再接收或转发 device、DECA batch size、DECA crop scale；
- training manifest 不再包含 `deca_cache`。

PnP/Iris 深度生成器仍保留算法自身的模型实现和默认人脸预处理；这些不再由本入口配置，也不会产生独立 DECA 特征缓存。

## `both` 模式的 11 个步骤

1. 合并 data log 与 InsightFace，生成统一数据表；
2. 提取 MediaPipe PnP landmarks；
3. 生成 PnP1010 pose/depth geometry；
4. 审计 PnP v2 几何；
5. 生成 PnP1010 有效样本表；
6. 提取 MediaPipe iris centres；
7. 生成 Iris65 深度先验；
8. 审计 Iris65 表中的 PnP pose；
9. 生成 Iris65 有效覆盖表；
10. 生成 PnP1010/Iris65 共同样本表；
11. 生成 PnP 中心的虚拟相机 face/eye 图像及 manifest。

`pnp1010` 模式跳过第 6～10 步。

## 输出

```text
<output_root>/
├── shared/          # 统一数据表、PnP landmarks、Iris centres
├── pnp1010/         # PnP 先验、审计报告、有效样本表
├── iris65/          # Iris 先验、审计报告、有效样本表
├── comparison/      # 两种深度方案的共同样本表
├── virtual_camera/  # face/left_eye/right_eye、manifest.csv、report.json
└── manifests/       # training_inputs_*.json、pipeline_*.json
```

输出中不再有 `deca_features_*.npz`。`training_inputs_*.json` 是训练输入契约；pipeline manifest 会记录配置文件绝对路径和解析后的完整参数。

## 安全开关

- 只打印命令：`execution.dry_run: true`。
- 确认基础 CSV 未变化时才使用：`execution.reuse_base_dataset: true`。
- 替换已有派生文件：`execution.overwrite: true`。
- `overwrite` 不会删除采集数据，也不会清理旧运行遗留但本次未引用的图片。
