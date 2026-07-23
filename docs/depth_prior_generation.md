# 离线深度先验生成

`scripts/generate_depth_priors.py` 会遍历 `modelv1_dataset.csv` 的每一条
样本，并生成一个独立的深度先验表。它不会修改或覆盖原始训练数据表。

## 前置数据

每条样本需要以下已有数据：

- `source_image_path` 指向的原始图像，以及 `face_bbox_x/y/w/h` 检测框。
- `face_path` 指向的 `224x224` 人脸图；仅在兼容旧版 `legacy` 裁剪时使用。
- `source_dataset_dir/mediapipe_pnp_landmarks.csv` 中保存的 8 个原图 2D PnP 点。
- 本地官方 DECA 代码与其 `deca_model.tar` 权重。
- 相机内参、畸变参数、真实眼角距离或已验证的固定尺度。

当前工程的 `modelv1_dataset.csv` 已包含前三项路径；三个已有数据集的
MediaPipe PnP 文件位于各自的 `dataset_dual_rigid_body_3`、`4`、`5` 目录。

## 运行前环境

项目主 `requirements.txt` 只覆盖训练主流程。生成深度先验还需要官方 DECA
与 OpenCV 依赖，至少包括：

```powershell
python -m pip install yacs opencv-python
```

如果 DECA 继续提示缺包，请参考 `DECA-master/requirements.txt` 补齐与本地
PyTorch 版本兼容的官方 DECA 依赖。不要直接将该文件中的旧版 PyTorch 固定版本
覆盖到当前训练环境。

项目的 DECA 包装层已经兼容 NumPy 1.24+ 与旧版 `chumpy` 的别名问题；不需要仅
为了 `ImportError: cannot import name 'bool' from numpy` 而降低整个环境的 NumPy。

## 推荐运行命令

在项目根目录执行：

```powershell
python scripts/generate_depth_priors.py --device auto --batch-size 8
```

默认 `--face-preprocess deca`：从原图检测框建立官方 DECA 风格的正方形
similarity crop（默认 `--deca-crop-scale 1.25`），再缩放到 `224x224` 输入 DECA。
该方式保留纵横比并在下方加入官方的中心偏移，推荐用于新的深度先验。每条输出还会
保存 `deca_face_preprocess` 与 `deca_crop_*_px`，用于复现与可视化反投影。

若要复现实验中旧的“检测框矩形直接压缩为 `224x224`”输入：

```powershell
python scripts/generate_depth_priors.py --face-preprocess legacy
```

默认输出：

```text
data/processed/depth_prior_v1.csv
data/processed/depth_prior_v1.csv.metadata.json
```

默认使用固定尺度 `1010.0 mm/FLAME unit`，与当前 CrossGaze 基线最终选定的
尺度保持一致。该策略避免了逐帧尺度估计抖动，同时仍会保存内外眼角尺度诊断。

如需重新按每一帧的内外眼角测量值估计尺度：

```powershell
python scripts/generate_depth_priors.py --device auto --use-measured-scale-per-frame
```

如需先做小规模检查：

```powershell
python scripts/generate_depth_priors.py --device cpu --limit 10 --output data/processed/depth_prior_debug.csv
```

脚本不会覆盖已存在的输出。确认需要更新时，增加 `--overwrite`：

```powershell
python scripts/generate_depth_priors.py --device auto --overwrite
```

## CSV 保存字段与类型

`depth_prior_v1.csv` 每个样本一行，主要字段如下：

| 字段组 | 类型 | 含义 |
|---|---|---|
| `sample_id`、`dataset`、路径字段 | string | 样本和输入来源追溯信息 |
| `depth_prior_status`、`reason` | string | `success` 或 `failed` 及失败原因 |
| `deca_face_preprocess`、`deca_crop_*_px` | string / float | DECA 输入裁剪模式及其在原图中的可逆裁剪范围 |
| `left_eye_camera_*_mm` | float | 图像左眼内外眼角中点的相机坐标，单位 mm |
| `right_eye_camera_*_mm` | float | 图像右眼内外眼角中点的相机坐标，单位 mm |
| `face_depth_z_mm` | float | FLAME 局部原点的相机 z 深度，单位 mm |
| `rvec_*_rad`、`rotation_**` | float | PnP 旋转结果，用于调试与追溯 |
| `tvec_*_mm` | float | PnP 平移结果，单位 mm |
| `reprojection_error_*_px` | float | PnP 重投影误差，单位像素 |
| `pnp_num_points`、`pnp_inlier_count` | integer | 实际 PnP 对应点数与内点数 |
| `pnp_confidence` | float | 基于误差、内点比例、尺度一致性的启发式质量分数 |
| `depth_is_plausible` | boolean | 深度是否通过基础合理性检查 |
| `*_scale_mm_per_flame_unit` | float | 固定尺度及内、外眼角诊断尺度 |
| `scale_disagreement_ratio` | float | 内外眼角尺度之间的相对差异 |

`depth_prior_v1.csv.metadata.json` 保存字段类型、相机参数、尺度策略、DECA
来源、处理时间、样本状态计数和坐标系约定。

## 与训练数据的关系

该生成器的结果是离线深度先验，不是深度真值。当前脚本故意输出到独立 CSV，
以免修改 `modelv1_dataset.csv`。训练阶段应按 `sample_id` 关联此表，并将
`pnp_confidence`、重投影误差和尺度分歧作为过滤或损失加权依据。

当前预处理不会产生 `z_sigma_mm` 或 `eye_center_sigma_mm`。它们应由后续训练期
的不确定性头、验证集校准或时间序列统计生成，而不是凭 PnP 单帧结果伪造。
