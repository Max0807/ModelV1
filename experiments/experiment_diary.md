# ModelV1 实验日记

从下一次正式训练开始自动更新。此前实验不回填，`--dry-run` 不记录。

## 实验索引

| 完成时间 | Run | 实验说明 | 最佳 Val EPE | 相比上一实验 | 最佳 Epoch |
|---|---|---|---:|---:|---:|
| 2026-07-23T17:11:37+08:00 | 20260723_164504 | V2.2：双眼6D几何加入PnP质量门控与可学习残差修正。 | 20.694 mm | — | 217 |
| 2026-07-23T18:25:55+08:00 | 20260723_175740 | V2.2：双眼6D几何加入PnP质量门控与可学习残差修正。 | 21.211 mm | +0.517 mm | 269 |
| 2026-07-23T21:41:08+08:00 | 20260723_211817 | Crop消融：V2.2关闭crop_cam分支，保留scene、双眼6D几何和质量门控。 | 22.290 mm | +1.079 mm | 289 |
| 2026-07-23T23:06:05+08:00 | 20260723_224136 | Scene消融：恢复crop_cam，将Scene从full25替换为物理TableFrame 7D。 | 20.954 mm | -1.336 mm | 260 |
| 2026-07-24T00:09:08+08:00 | 20260723_234608 | Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。 | 21.948 mm | +0.994 mm | 233 |
| 2026-07-24T10:22:37+08:00 | 20260724_095707 | Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。 | 22.114 mm | +0.166 mm | 244 |
| 2026-07-24T11:29:41+08:00 | 20260724_110737 | Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。 | 20.783 mm | -1.331 mm | 248 |
| 2026-07-25T12:52:31+08:00 | 20260725_122430 | 6D几何量纲消融：log(Cz)改为log(Cz / 1000 mm)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。 | 21.622 mm | +0.839 mm | 263 |
| 2026-07-28T10:59:01+08:00 | 20260728_103423 | 6D几何消融：第三维使用固定范围线性深度(Cz-500)/(1000-500)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。 | 21.228 mm | -0.394 mm | 263 |
| 2026-07-28T19:56:09+08:00 | 20260728_193536 | 6D几何消融：第三维使用固定范围线性深度(Cz-600)/(850-600)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。 | 21.550 mm | +0.322 mm | 263 |
| 2026-07-28T20:53:28+08:00 | 20260728_203116 | 6D几何消融：第三维使用线性绝对深度Cz/100 mm，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。 | 21.266 mm | -0.284 mm | 287 |
| 2026-07-28T21:27:39+08:00 | 20260728_210429 | 当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。 | 20.783 mm | -0.483 mm | 248 |
| 2026-07-29T00:12:49+08:00 | 20260728_232915 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 23.271 mm | +2.487 mm | 232 |
| 2026-07-29T10:33:11+08:00 | 20260729_100651 | 当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。 | 20.783 mm | -2.487 mm | 248 |
| 2026-07-29T11:05:39+08:00 | 20260729_104153 | 当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。 | 20.783 mm | +0.000 mm | 248 |
| 2026-07-29T14:49:23+08:00 | 20260729_142525 | DECA消融：仅使用shape+exp+pose的Geometry 156D特征；保留TableFrame 7D、无crop_cam与V2.2质量门控。 | 21.363 mm | +0.580 mm | 210 |
| 2026-07-29T16:36:13+08:00 | 20260729_161308 | 几何表示消融：双眼先验从中心—基线6D切换为原始左右眼相机坐标RawEye 6D；保留DECA Geometry156、TableFrame7D、无crop_cam与V2.2质量门控。 | 20.796 mm | -0.567 mm | 246 |
| 2026-07-29T19:58:22+08:00 | 20260729_192939 | DECA分解编码消融：shape、exp、pose分别经小型MLP后拼接融合为128D；保留RawEye 6D、TableFrame7D、无crop_cam与V2.2质量门控。 | 22.311 mm | +1.515 mm | 248 |
| 2026-07-29T20:35:55+08:00 | 20260729_200634 | DECA分解编码消融：shape、exp、pose分别经小型MLP后拼接融合为128D；保留RawEye 6D、TableFrame7D、无crop_cam与V2.2质量门控。 | 21.944 mm | -0.367 mm | 248 |
| 2026-07-29T22:17:36+08:00 | 20260729_212830 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 23.133 mm | +1.189 mm | 141 |
| 2026-07-30T11:05:12+08:00 | 20260730_102300 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 21.093 mm | -2.039 mm | 248 |
| 2026-07-30T15:04:00+08:00 | 20260730_140918 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 23.133 mm | +2.039 mm | 141 |
| 2026-07-30T19:41:19+08:00 | 20260730_185157 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 23.023 mm | -0.110 mm | 201 |
| 2026-07-30T21:00:48+08:00 | 20260730_202154 | V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。 | 20.796 mm | -2.227 mm | 246 |
| 2026-08-03T08:26:05+08:00 | 20260803_081359 | V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze. | 66.797 mm | +46.002 mm | 6 |
| 2026-08-03T10:13:45+08:00 | 20260803_095629 | Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline. | 78.787 mm | +11.989 mm | 31 |
| 2026-08-03T16:04:09+08:00 | 20260803_154452 | V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze. | 45.754 mm | -33.033 mm | 17 |
| 2026-08-03T19:18:27+08:00 | 20260803_185853 | V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze. | 45.754 mm | +0.000 mm | 17 |
| 2026-08-03T19:34:32+08:00 | 20260803_192317 | Stage 2 depth correction: load and freeze the dataset3 gaze model, then train only a bounded shared PnP eye scale; validate on dataset4. | 35.684 mm | -10.070 mm | 87 |
| 2026-08-03T21:48:37+08:00 | 20260803_213438 | Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline. | 64.639 mm | +28.955 mm | 38 |
| 2026-08-03T22:06:05+08:00 | 20260803_215445 | Stage 2 depth correction: load and freeze the dataset3 gaze model, then train only a bounded shared PnP eye scale; validate on dataset4. | 64.649 mm | +0.010 mm | 1 |
| 2026-08-04T10:25:46+08:00 | 20260804_100848 | Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline. | 63.392 mm | -1.257 mm | 87 |
| 2026-08-04T10:59:04+08:00 | 20260804_104428 | Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline. | 68.590 mm | +5.198 mm | 52 |
| 2026-08-04T23:19:12+08:00 | dataset5_v4_point_iris_ipd65_common971 | V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV. | 41.292 mm | -27.298 mm | 21 |
| 2026-08-05T08:55:02+08:00 | dataset3_4_v4_point_pnp1010_common842 | V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV. | 71.784 mm | +30.492 mm | 83 |
| 2026-08-05T09:07:34+08:00 | dataset3_4_v4_point_iris_ipd65_common842 | V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV. | 69.063 mm | -2.721 mm | 53 |
| 2026-08-06T12:04:43+08:00 | numbered_train13_val3_v4_iris_ipd_65mm_fixed_prior_20260806_114607 | Auto-generated numbered-artifact split: train=13, val=3, prior=iris_ipd_65mm, depth_mode=fixed_prior. | 83.640 mm | +14.577 mm | 30 |
| 2026-08-07T09:42:52+08:00 | numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_093016 | Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point. | 69.342 mm | -14.298 mm | 68 |
| 2026-08-07T11:42:28+08:00 | numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_112933 | Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point. | 73.782 mm | +4.440 mm | 56 |
| 2026-08-07T15:31:36+08:00 | numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_151924 | Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point. | 69.342 mm | -4.440 mm | 68 |
| 2026-08-07T23:50:26+08:00 | 20260807_233119 | Virtual-camera integration baseline: train dataset 3, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 39.946 mm | -29.395 mm | 58 |
| 2026-08-08T09:51:35+08:00 | numbered_train13_val3_v4_iris_ipd_65mm_point_20260808_092612 | Auto-generated numbered-artifact split: train=13, val=3, prior=iris_ipd_65mm, depth_mode=point. | 83.234 mm | +43.288 mm | 39 |
| 2026-08-08T11:27:27+08:00 | 20260808_105208 | Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 3; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 106.195 mm | +22.961 mm | 31 |
| 2026-08-08T12:16:37+08:00 | 20260808_114943 | Virtual-camera integration baseline: train dataset 11, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 58.620 mm | -47.576 mm | 28 |
| 2026-08-08T21:07:45+08:00 | 20260808_203708 | Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 3; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 106.195 mm | +47.576 mm | 31 |
| 2026-08-08T23:14:10+08:00 | 20260808_225452 | Virtual-camera integration baseline: train dataset 12, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB. | 79.562 mm | -26.633 mm | 34 |
<!-- AUTO-INDEX-END -->

## 实验详情

## 2026-07-23T17:11:37+08:00 · 20260723_164504

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260723_164504","completed_at":"2026-07-23T17:11:37+08:00","run_name":"20260723_164504","best_val_epe_mm":20.69390106201172,"best_epoch":217} -->

- 实验目的：V2.2：双眼6D几何加入PnP质量门控与可学习残差修正。
- 模型变化：V2.2：双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'eye_geometry_6d', 'quality_gate', 'learned_residual', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260723_164504`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,522,325 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 217） | 0.038 | 20.694 mm | 16.601 mm | 10.365 mm | 15.841 mm |
| 最终训练（epoch 300） | 0.020 | 15.695 mm | 13.774 mm | 10.264 mm | 9.709 mm |
| 最终验证（epoch 300） | 0.039 | 20.984 mm | 17.646 mm | 10.593 mm | 16.026 mm |

相比上一实验：这是自动日记记录的首个实验，暂无上一实验可比较。

## 2026-07-23T18:25:55+08:00 · 20260723_175740

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260723_175740","completed_at":"2026-07-23T18:25:55+08:00","run_name":"20260723_175740","best_val_epe_mm":21.21085548400879,"best_epoch":269} -->

- 实验目的：V2.2：双眼6D几何加入PnP质量门控与可学习残差修正。
- 模型变化：双眼 6D 几何使用固定 PnP 质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'eye_geometry_6d', 'quality_gate', 'learned_residual', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260723_175740`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| 几何分支 | use=True，gate=`fixed` |
| 融合层 | `[256, 128]`；总参数=11,522,228 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 269） | 0.038 | 21.211 mm | 17.680 mm | 10.558 mm | 16.126 mm |
| 最终训练（epoch 300） | 0.022 | 16.462 mm | 14.760 mm | 10.927 mm | 10.132 mm |
| 最终验证（epoch 300） | 0.039 | 21.271 mm | 18.051 mm | 10.705 mm | 16.075 mm |

相比上一实验：最佳 Val EPE 上升 0.517 mm，性能下降。

## 2026-07-23T21:41:08+08:00 · 20260723_211817

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260723_211817","completed_at":"2026-07-23T21:41:08+08:00","run_name":"20260723_211817","best_val_epe_mm":22.29008674621582,"best_epoch":289} -->

- 实验目的：Crop消融：V2.2关闭crop_cam分支，保留scene、双眼6D几何和质量门控。
- 模型变化：移除 crop_cam 36D 分支；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260723_211817`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,492,493 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 289） | 0.042 | 22.290 mm | 18.103 mm | 11.278 mm | 16.949 mm |
| 最终训练（epoch 300） | 0.022 | 16.413 mm | 14.367 mm | 10.699 mm | 10.325 mm |
| 最终验证（epoch 300） | 0.042 | 22.423 mm | 18.623 mm | 11.384 mm | 17.044 mm |

相比上一实验：最佳 Val EPE 上升 1.079 mm，性能下降。

## 2026-07-23T23:06:05+08:00 · 20260723_224136

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260723_224136","completed_at":"2026-07-23T23:06:05+08:00","run_name":"20260723_224136","best_val_epe_mm":20.95374298095703,"best_epoch":260} -->

- 实验目的：Scene消融：恢复crop_cam，将Scene从full25替换为物理TableFrame 7D。
- 模型变化：Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260723_224136`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=True，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,519,985 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 260） | 0.038 | 20.954 mm | 17.121 mm | 10.705 mm | 15.743 mm |
| 最终训练（epoch 300） | 0.022 | 16.569 mm | 15.200 mm | 10.925 mm | 10.273 mm |
| 最终验证（epoch 300） | 0.039 | 21.207 mm | 17.562 mm | 10.690 mm | 16.087 mm |

相比上一实验：最佳 Val EPE 下降 1.336 mm，性能提升。

## 2026-07-24T00:09:08+08:00 · 20260723_234608

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260723_234608","completed_at":"2026-07-24T00:09:08+08:00","run_name":"20260723_234608","best_val_epe_mm":21.947961807250977,"best_epoch":233} -->

- 实验目的：Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。
- 模型变化：Scene 替换为桌面 Orientation 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_orientation6d', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260723_234608`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=True，input_dim=36 |
| Scene分支 | representation=`orientation6d`，input_dim=6 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,519,855 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 233） | 0.040 | 21.948 mm | 17.284 mm | 11.965 mm | 15.753 mm |
| 最终训练（epoch 300） | 0.024 | 17.451 mm | 15.452 mm | 11.399 mm | 10.862 mm |
| 最终验证（epoch 300） | 0.041 | 22.069 mm | 17.949 mm | 11.870 mm | 15.922 mm |

相比上一实验：最佳 Val EPE 上升 0.994 mm，性能下降。

## 2026-07-24T10:22:37+08:00 · 20260724_095707

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260724_095707","completed_at":"2026-07-24T10:22:37+08:00","run_name":"20260724_095707","best_val_epe_mm":22.113685607910156,"best_epoch":244} -->

- 实验目的：Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。
- 模型变化：移除 crop_cam 36D 分支；Scene 替换为桌面 Orientation 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_orientation6d', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260724_095707`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`orientation6d`，input_dim=6 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,023 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 244） | 0.042 | 22.114 mm | 18.253 mm | 10.929 mm | 16.904 mm |
| 最终训练（epoch 300） | 0.023 | 16.349 mm | 14.335 mm | 10.459 mm | 10.344 mm |
| 最终验证（epoch 300） | 0.043 | 22.438 mm | 18.104 mm | 11.198 mm | 17.131 mm |

相比上一实验：最佳 Val EPE 上升 0.166 mm，性能下降。

## 2026-07-24T11:29:41+08:00 · 20260724_110737

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260724_110737","completed_at":"2026-07-24T11:29:41+08:00","run_name":"20260724_110737","best_val_epe_mm":20.783071517944336,"best_epoch":248} -->

- 实验目的：Scene消融：保持crop_cam和V2.2质量门控，将Scene替换为桌面Orientation 6D。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_orientation6d', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260724_110737`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.038 | 20.783 mm | 17.092 mm | 10.427 mm | 15.970 mm |
| 最终训练（epoch 300） | 0.021 | 16.034 mm | 14.394 mm | 10.778 mm | 9.718 mm |
| 最终验证（epoch 300） | 0.039 | 21.105 mm | 16.963 mm | 10.630 mm | 16.047 mm |

相比上一实验：最佳 Val EPE 下降 1.331 mm，性能提升。

## 2026-07-25T12:52:31+08:00 · 20260725_122430

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260725_122430","completed_at":"2026-07-25T12:52:31+08:00","run_name":"20260725_122430","best_val_epe_mm":21.621980667114258,"best_epoch":263} -->

- 实验目的：6D几何量纲消融：log(Cz)改为log(Cz / 1000 mm)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'log_depth_ref_1m', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260725_122430`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 263） | 0.040 | 21.622 mm | 19.004 mm | 11.106 mm | 16.297 mm |
| 最终训练（epoch 300） | 0.021 | 16.091 mm | 14.580 mm | 10.834 mm | 9.734 mm |
| 最终验证（epoch 300） | 0.041 | 21.844 mm | 18.600 mm | 11.072 mm | 16.622 mm |

相比上一实验：最佳 Val EPE 上升 0.839 mm，性能下降。

## 2026-07-28T10:59:01+08:00 · 20260728_103423

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260728_103423","completed_at":"2026-07-28T10:59:01+08:00","run_name":"20260728_103423","best_val_epe_mm":21.228282928466797,"best_epoch":263} -->

- 实验目的：6D几何消融：第三维使用固定范围线性深度(Cz-500)/(1000-500)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'linear_depth_500_1000mm', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260728_103423`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 263） | 0.039 | 21.228 mm | 17.253 mm | 10.884 mm | 15.907 mm |
| 最终训练（epoch 300） | 0.021 | 16.172 mm | 14.193 mm | 11.014 mm | 9.640 mm |
| 最终验证（epoch 300） | 0.040 | 21.451 mm | 17.115 mm | 10.898 mm | 16.167 mm |

相比上一实验：最佳 Val EPE 下降 0.394 mm，性能提升。

## 2026-07-28T19:56:09+08:00 · 20260728_193536

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260728_193536","completed_at":"2026-07-28T19:56:09+08:00","run_name":"20260728_193536","best_val_epe_mm":21.550310134887695,"best_epoch":263} -->

- 实验目的：6D几何消融：第三维使用固定范围线性深度(Cz-600)/(850-600)，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'linear_depth_600_850mm', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260728_193536`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 263） | 0.040 | 21.550 mm | 16.618 mm | 10.860 mm | 16.567 mm |
| 最终训练（epoch 300） | 0.021 | 16.199 mm | 14.096 mm | 10.990 mm | 9.726 mm |
| 最终验证（epoch 300） | 0.041 | 21.735 mm | 17.307 mm | 10.861 mm | 16.771 mm |

相比上一实验：最佳 Val EPE 上升 0.322 mm，性能下降。

## 2026-07-28T20:53:28+08:00 · 20260728_203116

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260728_203116","completed_at":"2026-07-28T20:53:28+08:00","run_name":"20260728_203116","best_val_epe_mm":21.26618766784668,"best_epoch":287} -->

- 实验目的：6D几何消融：第三维使用线性绝对深度Cz/100 mm，保留当前TableFrame 7D、无crop_cam与V2.2质量门控。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'linear_depth_div100mm', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260728_203116`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 287） | 0.039 | 21.266 mm | 17.454 mm | 10.697 mm | 16.167 mm |
| 最终训练（epoch 300） | 0.021 | 16.104 mm | 14.308 mm | 10.816 mm | 9.659 mm |
| 最终验证（epoch 300） | 0.040 | 21.466 mm | 17.358 mm | 10.629 mm | 16.365 mm |

相比上一实验：最佳 Val EPE 下降 0.284 mm，性能提升。

## 2026-07-28T21:27:39+08:00 · 20260728_210429

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260728_210429","completed_at":"2026-07-28T21:27:39+08:00","run_name":"20260728_210429","best_val_epe_mm":20.783071517944336,"best_epoch":248} -->

- 实验目的：当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260728_210429`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.038 | 20.783 mm | 17.092 mm | 10.427 mm | 15.970 mm |
| 最终训练（epoch 300） | 0.021 | 16.034 mm | 14.394 mm | 10.778 mm | 9.718 mm |
| 最终验证（epoch 300） | 0.039 | 21.105 mm | 16.963 mm | 10.630 mm | 16.047 mm |

相比上一实验：最佳 Val EPE 下降 0.483 mm，性能提升。

## 2026-07-29T00:12:49+08:00 · 20260728_232915

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260728_232915","completed_at":"2026-07-29T00:12:49+08:00","run_name":"20260728_232915","best_val_epe_mm":23.270545959472656,"best_epoch":232} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V3', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260728_232915`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=22,832,553 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 232） | 0.046 | 23.271 mm | 19.386 mm | 10.313 mm | 18.740 mm |
| 最终训练（epoch 300） | 0.020 | 15.512 mm | 13.660 mm | 10.072 mm | 9.651 mm |
| 最终验证（epoch 300） | 0.047 | 23.732 mm | 19.746 mm | 10.189 mm | 19.365 mm |

相比上一实验：最佳 Val EPE 上升 2.487 mm，性能下降。

## 2026-07-29T10:33:11+08:00 · 20260729_100651

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260729_100651","completed_at":"2026-07-29T10:33:11+08:00","run_name":"20260729_100651","best_val_epe_mm":20.783071517944336,"best_epoch":248} -->

- 实验目的：当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260729_100651`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.038 | 20.783 mm | 17.092 mm | 10.427 mm | 15.970 mm |
| 最终训练（epoch 300） | 0.021 | 16.034 mm | 14.394 mm | 10.778 mm | 9.718 mm |
| 最终验证（epoch 300） | 0.039 | 21.105 mm | 16.963 mm | 10.630 mm | 16.047 mm |

相比上一实验：最佳 Val EPE 下降 2.487 mm，性能提升。

## 2026-07-29T11:05:39+08:00 · 20260729_104153

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_V3/20260729_104153","completed_at":"2026-07-29T11:05:39+08:00","run_name":"20260729_104153","best_val_epe_mm":20.783071517944336,"best_epoch":248} -->

- 实验目的：当前配置：TableFrame 7D、无crop_cam与V2.2质量门控；双眼6D第三维使用log(Cz)。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_V3`，mode=`online`，tags=['ModelV1', 'V2_2', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_V3/20260729_104153`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.038 | 20.783 mm | 17.092 mm | 10.427 mm | 15.970 mm |
| 最终训练（epoch 300） | 0.021 | 16.034 mm | 14.394 mm | 10.778 mm | 9.718 mm |
| 最终验证（epoch 300） | 0.039 | 21.105 mm | 16.963 mm | 10.630 mm | 16.047 mm |

相比上一实验：最佳 Val EPE 与上一实验相同。

## 2026-07-29T14:49:23+08:00 · 20260729_142525

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260729_142525","completed_at":"2026-07-29T14:49:23+08:00","run_name":"20260729_142525","best_val_epe_mm":21.36295509338379,"best_epoch":210} -->

- 实验目的：DECA消融：仅使用shape+exp+pose的Geometry 156D特征；保留TableFrame 7D、无crop_cam与V2.2质量门控。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'deca_geometry156', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260729_142525`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,469,513 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 210） | 0.040 | 21.363 mm | 17.607 mm | 11.189 mm | 15.994 mm |
| 最终训练（epoch 300） | 0.021 | 16.261 mm | 14.142 mm | 10.616 mm | 10.083 mm |
| 最终验证（epoch 300） | 0.041 | 21.863 mm | 18.474 mm | 10.986 mm | 16.639 mm |

相比上一实验：最佳 Val EPE 上升 0.580 mm，性能下降。

## 2026-07-29T16:36:13+08:00 · 20260729_161308

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260729_161308","completed_at":"2026-07-29T16:36:13+08:00","run_name":"20260729_161308","best_val_epe_mm":20.79582977294922,"best_epoch":246} -->

- 实验目的：几何表示消融：双眼先验从中心—基线6D切换为原始左右眼相机坐标RawEye 6D；保留DECA Geometry156、TableFrame7D、无crop_cam与V2.2质量门控。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'deca_geometry156', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260729_161308`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,469,513 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 246） | 0.038 | 20.796 mm | 16.225 mm | 10.520 mm | 15.802 mm |
| 最终训练（epoch 300） | 0.021 | 16.119 mm | 14.408 mm | 10.566 mm | 10.012 mm |
| 最终验证（epoch 300） | 0.039 | 20.997 mm | 16.370 mm | 10.362 mm | 16.232 mm |

相比上一实验：最佳 Val EPE 下降 0.567 mm，性能提升。

## 2026-07-29T19:58:22+08:00 · 20260729_192939

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260729_192939","completed_at":"2026-07-29T19:58:22+08:00","run_name":"20260729_192939","best_val_epe_mm":22.311243057250977,"best_epoch":248} -->

- 实验目的：DECA分解编码消融：shape、exp、pose分别经小型MLP后拼接融合为128D；保留RawEye 6D、TableFrame7D、无crop_cam与V2.2质量门控。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；DECA 的 shape、exp、pose 分别经小型 MLP 编码后再融合为 128D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'deca_geometry156', 'deca_factorized_geometry', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260729_192939`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,502,537 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.042 | 22.311 mm | 18.966 mm | 10.914 mm | 16.973 mm |
| 最终训练（epoch 300） | 0.022 | 16.699 mm | 14.828 mm | 11.019 mm | 10.386 mm |
| 最终验证（epoch 300） | 0.043 | 22.578 mm | 19.107 mm | 10.960 mm | 17.081 mm |

相比上一实验：最佳 Val EPE 上升 1.515 mm，性能下降。

## 2026-07-29T20:35:55+08:00 · 20260729_200634

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_100_deca_crop/20260729_200634","completed_at":"2026-07-29T20:35:55+08:00","run_name":"20260729_200634","best_val_epe_mm":21.943836212158203,"best_epoch":248} -->

- 实验目的：DECA分解编码消融：shape、exp、pose分别经小型MLP后拼接融合为128D；保留RawEye 6D、TableFrame7D、无crop_cam与V2.2质量门控。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；DECA 的 shape、exp、pose 分别经小型 MLP 编码后再融合为 128D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_100_deca_crop`，mode=`online`，tags=['ModelV1', 'V2_2', 'deca_geometry156', 'deca_factorized_geometry', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_100_deca_crop/20260729_200634`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,502,537 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.041 | 21.944 mm | 18.444 mm | 10.999 mm | 16.722 mm |
| 最终训练（epoch 300） | 0.023 | 16.874 mm | 15.305 mm | 11.219 mm | 10.464 mm |
| 最终验证（epoch 300） | 0.042 | 22.573 mm | 19.228 mm | 11.277 mm | 17.144 mm |

相比上一实验：最佳 Val EPE 下降 0.367 mm，性能提升。

## 2026-07-29T22:17:36+08:00 · 20260729_212830

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_V3/20260729_212830","completed_at":"2026-07-29T22:17:36+08:00","run_name":"20260729_212830","best_val_epe_mm":23.132816314697266,"best_epoch":141} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_V3`，mode=`online`，tags=['ModelV1', 'V3', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_V3/20260729_212830`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=22,832,553 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 141） | 0.045 | 23.133 mm | 18.122 mm | 10.380 mm | 18.323 mm |
| 最终训练（epoch 300） | 0.020 | 15.613 mm | 14.078 mm | 10.179 mm | 9.867 mm |
| 最终验证（epoch 300） | 0.047 | 23.746 mm | 19.983 mm | 10.083 mm | 19.413 mm |

相比上一实验：最佳 Val EPE 上升 1.189 mm，性能下降。

## 2026-07-30T11:05:12+08:00 · 20260730_102300

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_V3/20260730_102300","completed_at":"2026-07-30T11:05:12+08:00","run_name":"20260730_102300","best_val_epe_mm":21.093454360961914,"best_epoch":248} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_V3`，mode=`online`，tags=['ModelV1', 'V3', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_V3/20260730_102300`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,490,153 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 248） | 0.039 | 21.093 mm | 17.326 mm | 10.610 mm | 15.941 mm |
| 最终训练（epoch 300） | 0.021 | 16.074 mm | 14.339 mm | 10.982 mm | 9.439 mm |
| 最终验证（epoch 300） | 0.041 | 21.549 mm | 17.619 mm | 10.760 mm | 16.347 mm |

相比上一实验：最佳 Val EPE 下降 2.039 mm，性能提升。

## 2026-07-30T15:04:00+08:00 · 20260730_140918

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_V3/20260730_140918","completed_at":"2026-07-30T15:04:00+08:00","run_name":"20260730_140918","best_val_epe_mm":23.132816314697266,"best_epoch":141} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_V3`，mode=`online`，tags=['ModelV1', 'V3', 'deca_geometry156', 'deca_factorized_geometry', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_V3/20260730_140918`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=22,832,553 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 141） | 0.045 | 23.133 mm | 18.122 mm | 10.380 mm | 18.323 mm |
| 最终训练（epoch 300） | 0.020 | 15.613 mm | 14.078 mm | 10.179 mm | 9.867 mm |
| 最终验证（epoch 300） | 0.047 | 23.746 mm | 19.983 mm | 10.083 mm | 19.413 mm |

相比上一实验：最佳 Val EPE 上升 2.039 mm，性能下降。

## 2026-07-30T19:41:19+08:00 · 20260730_185157

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_V3/20260730_185157","completed_at":"2026-07-30T19:41:19+08:00","run_name":"20260730_185157","best_val_epe_mm":23.022624969482422,"best_epoch":201} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_V3`，mode=`online`，tags=['ModelV1', 'V3', 'deca_geometry156', 'deca_factorized_geometry', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_V3/20260730_185157`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=22,811,913 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 201） | 0.045 | 23.023 mm | 19.736 mm | 11.107 mm | 18.152 mm |
| 最终训练（epoch 300） | 0.021 | 16.332 mm | 15.073 mm | 10.615 mm | 10.114 mm |
| 最终验证（epoch 300） | 0.046 | 23.475 mm | 20.185 mm | 11.010 mm | 18.571 mm |

相比上一实验：最佳 Val EPE 下降 0.110 mm，性能提升。

## 2026-07-30T21:00:48+08:00 · 20260730_202154

<!-- experiment-record: {"run_dir":"outputs/ModelV1_random_80_20_V3/20260730_202154","completed_at":"2026-07-30T21:00:48+08:00","run_name":"20260730_202154","best_val_epe_mm":20.79582977294922,"best_epoch":246} -->

- 实验目的：V3：在V2.2基础上增加RGB人脸ResNet18编码器；face与双眼特征先融合为128D视觉特征，再与DECA、TableFrame和质量门控双眼几何拼接。
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；双眼几何从中心—基线 6D 切换为原始左右眼相机坐标 RawEye 6D；V2.2 双眼 6D 几何使用可学习残差质量门控。
- W&B：project=`ModelV1_random_80_20_V3`，mode=`online`，tags=['ModelV1', 'V3', 'deca_geometry156', 'deca_factorized_geometry', 'face_image_resnet18', 'face_eye_visual_fusion', 'scene_table_frame7', 'no_crop_cam', 'eye_geometry_raw_eye6d', 'quality_gate', 'random_80_20']
- 产物目录：`outputs/ModelV1_random_80_20_V3/20260730_202154`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `random_80_20`；train=906，val=227，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`DEFAULT`，freeze_until=`layer3` |
| 眼部骨干 | `resnet18`，weights=`None` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=True，gate=`learned_residual` |
| 融合层 | `[256, 128]`；总参数=11,469,513 |
| 训练 | epochs=300，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | beta_mm=30.0，gate_reg=0.001 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 246） | 0.038 | 20.796 mm | 16.225 mm | 10.520 mm | 15.802 mm |
| 最终训练（epoch 300） | 0.021 | 16.119 mm | 14.408 mm | 10.566 mm | 10.012 mm |
| 最终验证（epoch 300） | 0.039 | 20.997 mm | 16.370 mm | 10.362 mm | 16.232 mm |

相比上一实验：最佳 Val EPE 下降 2.227 mm，性能提升。

## 2026-08-03T08:26:05+08:00 · 20260803_081359

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset5_V4/20260803_081359","completed_at":"2026-08-03T08:26:05+08:00","run_name":"20260803_081359","best_val_epe_mm":66.7973403930664,"best_epoch":6} -->

- 实验目的：V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset5_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_5', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset5_V4/20260803_081359`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `dataset_5`；train=904，val=229，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=False，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=11,441,691 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 6） | 30.885 | 66.797 mm | 58.004 mm | 40.834 mm | 45.708 mm |
| 最终训练（epoch 100） | 8.265 | 28.573 mm | 25.732 mm | 18.083 mm | 18.212 mm |
| 最终验证（epoch 100） | 41.946 | 82.022 mm | 77.584 mm | 55.292 mm | 52.658 mm |

相比上一实验：最佳 Val EPE 上升 46.002 mm，性能下降。

## 2026-08-03T10:13:45+08:00 · 20260803_095629

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4/20260803_095629","completed_at":"2026-08-03T10:13:45+08:00","run_name":"20260803_095629","best_val_epe_mm":78.78666687011719,"best_epoch":31} -->

- 实验目的：Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'explicit_holdout', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'rgb_face', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4/20260803_095629`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`None`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 31） | 37.981 | 78.787 mm | 73.948 mm | 47.770 mm | 53.085 mm |
| 最终训练（epoch 100） | 7.288 | 26.202 mm | 23.722 mm | 16.042 mm | 17.192 mm |
| 最终验证（epoch 100） | 43.957 | 87.920 mm | 83.778 mm | 56.349 mm | 56.380 mm |

相比上一实验：最佳 Val EPE 上升 11.989 mm，性能下降。

## 2026-08-03T16:04:09+08:00 · 20260803_154452

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset5_V4/20260803_154452","completed_at":"2026-08-03T16:04:09+08:00","run_name":"20260803_154452","best_val_epe_mm":45.75394821166992,"best_epoch":17} -->

- 实验目的：V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset5_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_5', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset5_V4/20260803_154452`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `dataset_5`；train=904，val=229，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v2.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.1，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 17） | 18.613 | 45.754 mm | 39.142 mm | 25.947 mm | 31.846 mm |
| 最终训练（epoch 100） | 5.790 | 21.340 mm | 19.902 mm | 12.463 mm | 14.682 mm |
| 最终验证（epoch 100） | 24.481 | 56.151 mm | 54.834 mm | 39.040 mm | 32.651 mm |

相比上一实验：最佳 Val EPE 下降 33.033 mm，性能提升。

## 2026-08-03T19:18:27+08:00 · 20260803_185853

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset5_V4/20260803_185853","completed_at":"2026-08-03T19:18:27+08:00","run_name":"20260803_185853","best_val_epe_mm":45.75394821166992,"best_epoch":17} -->

- 实验目的：V4 point-depth ablation: add weak cosine supervision from the PnP-eye-to-Vicon-target pseudo gaze.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset5_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_5', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset5_V4/20260803_185853`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `dataset_5`；train=904，val=229，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v2.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.1，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 17） | 18.613 | 45.754 mm | 39.142 mm | 25.947 mm | 31.846 mm |
| 最终训练（epoch 100） | 5.790 | 21.340 mm | 19.902 mm | 12.463 mm | 14.682 mm |
| 最终验证（epoch 100） | 24.481 | 56.151 mm | 54.834 mm | 39.040 mm | 32.651 mm |

相比上一实验：最佳 Val EPE 与上一实验相同。

## 2026-08-03T19:34:32+08:00 · 20260803_192317

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4_depth_correction/20260803_192317","completed_at":"2026-08-03T19:34:32+08:00","run_name":"20260803_192317","best_val_epe_mm":35.684207916259766,"best_epoch":87} -->

- 实验目的：Stage 2 depth correction: load and freeze the dataset3 gaze model, then train only a bounded shared PnP eye scale; validate on dataset4.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4_depth_correction`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'depth_correction', 'shared_scale', 'PnP_10D', 'K1', 'frozen_gaze']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4_depth_correction/20260803_192317`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v2.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,786,908 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0001，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 87） | 11.314 | 35.684 mm | 34.263 mm | 27.551 mm | 17.425 mm |
| 最终训练（epoch 100） | 10.543 | 33.557 mm | 31.046 mm | 23.319 mm | 19.414 mm |
| 最终验证（epoch 100） | 11.315 | 35.685 mm | 34.263 mm | 27.552 mm | 17.426 mm |

相比上一实验：最佳 Val EPE 下降 10.070 mm，性能提升。

## 2026-08-03T21:48:37+08:00 · 20260803_213438

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4/20260803_213438","completed_at":"2026-08-03T21:48:37+08:00","run_name":"20260803_213438","best_val_epe_mm":64.63905334472656,"best_epoch":38} -->

- 实验目的：Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'explicit_holdout', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'rgb_face', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4/20260803_213438`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 38） | 27.809 | 64.639 mm | 59.384 mm | 22.436 mm | 57.429 mm |
| 最终训练（epoch 100） | 6.606 | 24.974 mm | 22.215 mm | 14.984 mm | 16.608 mm |
| 最终验证（epoch 100） | 28.906 | 66.934 mm | 63.512 mm | 20.999 mm | 60.918 mm |

相比上一实验：最佳 Val EPE 上升 28.955 mm，性能下降。

## 2026-08-03T22:06:05+08:00 · 20260803_215445

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4_depth_correction/20260803_215445","completed_at":"2026-08-03T22:06:05+08:00","run_name":"20260803_215445","best_val_epe_mm":64.6488265991211,"best_epoch":1} -->

- 实验目的：Stage 2 depth correction: load and freeze the dataset3 gaze model, then train only a bounded shared PnP eye scale; validate on dataset4.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4_depth_correction`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'depth_correction', 'shared_scale', 'PnP_10D', 'K1', 'frozen_gaze']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4_depth_correction/20260803_215445`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v2.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,786,908 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0001，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 1） | 27.388 | 64.649 mm | 59.384 mm | 22.433 mm | 57.437 mm |
| 最终训练（epoch 100） | 2.621 | 15.542 mm | 13.838 mm | 10.420 mm | 9.396 mm |
| 最终验证（epoch 100） | 29.933 | 69.496 mm | 65.063 mm | 22.141 mm | 62.777 mm |

相比上一实验：最佳 Val EPE 上升 0.010 mm，性能下降。

## 2026-08-04T10:25:46+08:00 · 20260804_100848

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4/20260804_100848","completed_at":"2026-08-04T10:25:46+08:00","run_name":"20260804_100848","best_val_epe_mm":63.39213562011719,"best_epoch":87} -->

- 实验目的：Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'explicit_holdout', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'rgb_face', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4/20260804_100848`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,478 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 87） | 39.316 | 63.392 mm | 58.346 mm | 20.508 mm | 57.075 mm |
| 最终训练（epoch 100） | 14.852 | 23.742 mm | 21.218 mm | 14.952 mm | 15.286 mm |
| 最终验证（epoch 100） | 39.721 | 63.902 mm | 59.279 mm | 20.735 mm | 57.550 mm |

相比上一实验：最佳 Val EPE 下降 1.257 mm，性能提升。

## 2026-08-04T10:59:04+08:00 · 20260804_104428

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_train_dataset4_val_V4/20260804_104428","completed_at":"2026-08-04T10:59:04+08:00","run_name":"20260804_104428","best_val_epe_mm":68.59001159667969,"best_epoch":52} -->

- 实验目的：Controlled camera/session holdout: train only on dataset 3 and validate only on dataset 4; otherwise identical to the RGB-face V4 point+pseudo-gaze baseline.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_train_dataset4_val_V4`，mode=`online`，tags=['ModelV1', 'V4', 'train_dataset_3', 'val_dataset_4', 'explicit_holdout', 'gaze_geometry', 'point_pnp', 'pseudo_gaze_cosine_100', 'rgb_face', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_train_dataset4_val_V4/20260804_104428`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=506，val=398，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,478 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 52） | 41.968 | 68.590 mm | 64.393 mm | 25.218 mm | 60.202 mm |
| 最终训练（epoch 100） | 14.968 | 24.268 mm | 22.632 mm | 15.022 mm | 15.907 mm |
| 最终验证（epoch 100） | 43.213 | 70.028 mm | 65.539 mm | 20.233 mm | 64.106 mm |

相比上一实验：最佳 Val EPE 上升 5.198 mm，性能下降。

## 2026-08-04T23:19:12+08:00 · dataset5_v4_point_iris_ipd65_common971

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset5_V4/dataset5_v4_point_iris_ipd65_common971","completed_at":"2026-08-04T23:19:12+08:00","run_name":"dataset5_v4_point_iris_ipd65_common971","best_val_epe_mm":41.2922477722168,"best_epoch":21} -->

- 实验目的：V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset5_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_5', 'gaze_geometry', 'point_pnp', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset5_V4/dataset5_v4_point_iris_ipd65_common971`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `dataset_5`；train=782，val=189，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_iris_ipd_65mm_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 21） | 15.428 | 41.292 mm | 32.598 mm | 22.042 mm | 31.029 mm |
| 最终训练（epoch 100） | 5.379 | 22.686 mm | 20.080 mm | 14.270 mm | 14.403 mm |
| 最终验证（epoch 100） | 20.161 | 51.244 mm | 47.343 mm | 32.559 mm | 32.430 mm |

相比上一实验：最佳 Val EPE 下降 27.298 mm，性能提升。

## 2026-08-05T08:55:02+08:00 · dataset3_4_v4_point_pnp1010_common842

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_4_V4/dataset3_4_v4_point_pnp1010_common842","completed_at":"2026-08-05T08:55:02+08:00","run_name":"dataset3_4_v4_point_pnp1010_common842","best_val_epe_mm":71.78421783447266,"best_epoch":83} -->

- 实验目的：V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_4_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_3_4', 'gaze_geometry', 'point_pnp', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_4_V4/dataset3_4_v4_point_pnp1010_common842`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=370，val=309，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_deca_crop_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 83） | 31.389 | 71.784 mm | 68.760 mm | 24.702 mm | 63.418 mm |
| 最终训练（epoch 100） | 7.475 | 27.063 mm | 24.661 mm | 16.811 mm | 17.956 mm |
| 最终验证（epoch 100） | 32.058 | 73.098 mm | 73.128 mm | 25.380 mm | 64.029 mm |

相比上一实验：最佳 Val EPE 上升 30.492 mm，性能下降。

## 2026-08-05T09:07:34+08:00 · dataset3_4_v4_point_iris_ipd65_common842

<!-- experiment-record: {"run_dir":"outputs/ModelV1_dataset3_4_V4/dataset3_4_v4_point_iris_ipd65_common842","completed_at":"2026-08-05T09:07:34+08:00","run_name":"dataset3_4_v4_point_iris_ipd65_common842","best_val_epe_mm":69.06285095214844,"best_epoch":53} -->

- 实验目的：V4 stage 1: image predicts camera-space gaze; one PnP eye midpoint and parameter-free ray/table geometry predict UV.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_dataset3_4_V4`，mode=`online`，tags=['ModelV1', 'V4', 'dataset_3_4', 'gaze_geometry', 'point_pnp', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_dataset3_4_V4/dataset3_4_v4_point_iris_ipd65_common842`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=370，val=309，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`data/processed/deca_features_deca_crop_v1.npz` |
| 深度先验 | `data/processed/depth_priors_iris_ipd_65mm_v1.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 53） | 29.714 | 69.063 mm | 65.344 mm | 22.612 mm | 61.280 mm |
| 最终训练（epoch 100） | 7.126 | 26.256 mm | 24.731 mm | 16.450 mm | 17.502 mm |
| 最终验证（epoch 100） | 30.400 | 70.104 mm | 67.628 mm | 23.810 mm | 61.748 mm |

相比上一实验：最佳 Val EPE 下降 2.721 mm，性能提升。

## 2026-08-06T12:04:43+08:00 · numbered_train13_val3_v4_iris_ipd_65mm_fixed_prior_20260806_114607

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train13_val3_v4_iris_ipd_65mm_fixed_prior_20260806_114607","completed_at":"2026-08-06T12:04:43+08:00","run_name":"numbered_train13_val3_v4_iris_ipd_65mm_fixed_prior_20260806_114607","best_val_epe_mm":83.64015197753906,"best_epoch":30} -->

- 实验目的：Auto-generated numbered-artifact split: train=13, val=3, prior=iris_ipd_65mm, depth_mode=fixed_prior.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_fixed_prior', 'train_dataset_13', 'val_dataset_3', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train13_val3_v4_iris_ipd_65mm_fixed_prior_20260806_114607`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1047，val=305，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`numbered datasets [3, 4, 11, 13]; data/processed/deca_features_deca_crop_v1_<id>.npz` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸图像骨干 | use=True，`resnet18`，weights=`DEFAULT`，freeze_until=`none` |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=22,784,091 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 30） | 39.958 | 83.640 mm | 78.504 mm | 63.520 mm | 42.964 mm |
| 最终训练（epoch 100） | 2.482 | 14.990 mm | 13.529 mm | 9.793 mm | 9.192 mm |
| 最终验证（epoch 100） | 44.940 | 91.296 mm | 89.783 mm | 71.639 mm | 45.296 mm |

相比上一实验：最佳 Val EPE 上升 14.577 mm，性能下降。

## 2026-08-07T09:42:52+08:00 · numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_093016

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_093016","completed_at":"2026-08-07T09:42:52+08:00","run_name":"numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_093016","best_val_epe_mm":69.34169006347656,"best_epoch":68} -->

- 实验目的：Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_point', 'train_dataset_3', 'val_dataset_4', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_093016`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=310，val=277，seed=42 |
| DECA | preprocess=`deca`，crop_scale=1.25，cache=`numbered datasets [3, 4, 11, 13]; data/processed/deca_features_deca_crop_v1_<id>.npz` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,263,739 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 68） | 30.117 | 69.342 mm | 66.234 mm | 24.599 mm | 60.729 mm |
| 最终训练（epoch 100） | 7.947 | 28.124 mm | 26.276 mm | 17.848 mm | 18.400 mm |
| 最终验证（epoch 100） | 31.173 | 71.210 mm | 66.199 mm | 24.929 mm | 62.103 mm |

相比上一实验：最佳 Val EPE 下降 14.298 mm，性能提升。

## 2026-08-07T11:42:28+08:00 · numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_112933

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_112933","completed_at":"2026-08-07T11:42:28+08:00","run_name":"numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_112933","best_val_epe_mm":73.78179931640625,"best_epoch":56} -->

- 实验目的：Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_point', 'train_dataset_3', 'val_dataset_4', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_112933`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=310，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,156,803 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 56） | 32.157 | 73.782 mm | 69.730 mm | 21.772 mm | 67.516 mm |
| 最终训练（epoch 100） | 6.929 | 25.789 mm | 23.176 mm | 15.255 mm | 17.551 mm |
| 最终验证（epoch 100） | 33.557 | 77.798 mm | 77.226 mm | 18.651 mm | 72.689 mm |

相比上一实验：最佳 Val EPE 上升 4.440 mm，性能下降。

## 2026-08-07T15:31:36+08:00 · numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_151924

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_151924","completed_at":"2026-08-07T15:31:36+08:00","run_name":"numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_151924","best_val_epe_mm":69.34169006347656,"best_epoch":68} -->

- 实验目的：Auto-generated numbered-artifact split: train=3, val=4, prior=iris_ipd_65mm, depth_mode=point.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_point', 'train_dataset_3', 'val_dataset_4', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train3_val4_v4_iris_ipd_65mm_point_20260807_151924`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=310，val=277，seed=42 |
| DECA | representation=`geometry156`，preprocess=`deca`，crop_scale=1.25，cache=`numbered datasets [3, 4, 11, 13]; data/processed/deca_features_deca_crop_v1_<id>.npz` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,263,739 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 68） | 30.117 | 69.342 mm | 66.234 mm | 24.599 mm | 60.729 mm |
| 最终训练（epoch 100） | 7.947 | 28.124 mm | 26.276 mm | 17.848 mm | 18.400 mm |
| 最终验证（epoch 100） | 31.173 | 71.210 mm | 66.199 mm | 24.929 mm | 62.103 mm |

相比上一实验：最佳 Val EPE 下降 4.440 mm，性能提升。

## 2026-08-07T23:50:26+08:00 · 20260807_233119

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260807_233119","completed_at":"2026-08-07T23:50:26+08:00","run_name":"20260807_233119","best_val_epe_mm":39.94624328613281,"best_epoch":58} -->

- 实验目的：Virtual-camera integration baseline: train dataset 3, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_3', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260807_233119`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=370，val=309，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `data/processed/modelv1_dataset_depth_prior_common.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 58） | 24.110 | 39.946 mm | 35.789 mm | 20.744 mm | 29.649 mm |
| 最终训练（epoch 100） | 14.578 | 23.086 mm | 19.999 mm | 13.127 mm | 15.925 mm |
| 最终验证（epoch 100） | 29.101 | 48.045 mm | 40.765 mm | 21.991 mm | 37.821 mm |

相比上一实验：最佳 Val EPE 下降 29.395 mm，性能提升。

## 2026-08-08T09:51:35+08:00 · numbered_train13_val3_v4_iris_ipd_65mm_point_20260808_092612

<!-- experiment-record: {"run_dir":"outputs/ModelV1_numbered_V4/numbered_train13_val3_v4_iris_ipd_65mm_point_20260808_092612","completed_at":"2026-08-08T09:51:35+08:00","run_name":"numbered_train13_val3_v4_iris_ipd_65mm_point_20260808_092612","best_val_epe_mm":83.23388671875,"best_epoch":39} -->

- 实验目的：Auto-generated numbered-artifact split: train=13, val=3, prior=iris_ipd_65mm, depth_mode=point.
- 模型变化：DECA 特征从完整 236D 切换为 shape+exp+pose 的 Geometry 156D；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_numbered_V4`，mode=`online`，tags=['ModelV1', 'v4', 'numbered_artifacts', 'gaze_geometry', 'iris_ipd_65mm', 'depth_point', 'train_dataset_13', 'val_dataset_3', 'datasets_3_4_11_13', 'imagenet_resnet18', 'eye_augmentation']
- 产物目录：`outputs/ModelV1_numbered_V4/numbered_train13_val3_v4_iris_ipd_65mm_point_20260808_092612`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1069，val=310，seed=42 |
| DECA | representation=`geometry156`，preprocess=`deca`，crop_scale=1.25，cache=`numbered datasets [3, 4, 11, 13]; data/processed/deca_features_deca_crop_v1_<id>.npz` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,263,739 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=1.0，gaze_angular_weight=0.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 39） | 39.799 | 83.234 mm | 82.168 mm | 63.594 mm | 41.533 mm |
| 最终训练（epoch 100） | 1.887 | 13.008 mm | 11.783 mm | 8.281 mm | 8.238 mm |
| 最终验证（epoch 100） | 44.628 | 92.270 mm | 89.941 mm | 69.929 mm | 45.623 mm |

相比上一实验：最佳 Val EPE 上升 43.288 mm，性能下降。

## 2026-08-08T11:27:27+08:00 · 20260808_105208

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260808_105208","completed_at":"2026-08-08T11:27:27+08:00","run_name":"20260808_105208","best_val_epe_mm":106.19518280029297,"best_epoch":31} -->

- 实验目的：Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 3; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_13', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260808_105208`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1066，val=310，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 31） | 80.654 | 106.195 mm | 104.974 mm | 64.268 mm | 71.203 mm |
| 最终训练（epoch 100） | 9.959 | 12.754 mm | 11.781 mm | 8.441 mm | 7.820 mm |
| 最终验证（epoch 100） | 98.839 | 124.722 mm | 122.931 mm | 63.203 mm | 95.347 mm |

相比上一实验：最佳 Val EPE 上升 22.961 mm，性能下降。

## 2026-08-08T12:16:37+08:00 · 20260808_114943

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260808_114943","completed_at":"2026-08-08T12:16:37+08:00","run_name":"20260808_114943","best_val_epe_mm":58.61956787109375,"best_epoch":28} -->

- 实验目的：Virtual-camera integration baseline: train dataset 11, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_11', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260808_114943`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=819，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [4, 11]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 28） | 38.172 | 58.620 mm | 53.381 mm | 36.908 mm | 38.645 mm |
| 最终训练（epoch 100） | 10.226 | 13.522 mm | 12.146 mm | 8.601 mm | 8.658 mm |
| 最终验证（epoch 100） | 50.009 | 73.105 mm | 67.556 mm | 37.167 mm | 54.523 mm |

相比上一实验：最佳 Val EPE 下降 47.576 mm，性能提升。

## 2026-08-08T21:07:45+08:00 · 20260808_203708

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260808_203708","completed_at":"2026-08-08T21:07:45+08:00","run_name":"20260808_203708","best_val_epe_mm":106.19518280029297,"best_epoch":31} -->

- 实验目的：Virtual-camera integration baseline: train dataset 13, validate unseen-camera dataset 3; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_13', 'val_dataset_3', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260808_203708`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=1066，val=310，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 11, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 31） | 80.654 | 106.195 mm | 104.974 mm | 64.268 mm | 71.203 mm |
| 最终训练（epoch 100） | 9.959 | 12.754 mm | 11.781 mm | 8.441 mm | 7.820 mm |
| 最终验证（epoch 100） | 98.839 | 124.722 mm | 122.931 mm | 63.203 mm | 95.347 mm |

相比上一实验：最佳 Val EPE 上升 47.576 mm，性能下降。

## 2026-08-08T23:14:10+08:00 · 20260808_225452

<!-- experiment-record: {"run_dir":"outputs/ModelV1_virtual_camera_cross_camera/20260808_225452","completed_at":"2026-08-08T23:14:10+08:00","run_name":"20260808_225452","best_val_epe_mm":79.56175231933594,"best_epoch":34} -->

- 实验目的：Virtual-camera integration baseline: train dataset 12, validate unseen-camera dataset 4; predict gaze in N and rotate deterministically to C before geometry. DECA is disabled to isolate normalized RGB.
- 模型变化：完全移除离线 DECA 特征及其 MLP 分支；增加 RGB 人脸编码器及 face-eye 视觉融合；移除 crop_cam 36D 分支；Scene 从 full25 替换为 TableFrame 7D；不使用双眼 6D 几何先验。
- W&B：project=`ModelV1_virtual_camera_cross_camera`，mode=`online`，tags=['ModelV1', 'V4', 'virtual_camera', 'gaze_N_to_C', 'train_dataset_12', 'val_dataset_4', 'no_DECA']
- 产物目录：`outputs/ModelV1_virtual_camera_cross_camera/20260808_225452`

### 关键参数

| 类别 | 设置 |
|---|---|
| 数据划分 | `explicit_datasets`；train=572，val=277，seed=42 |
| DECA | representation=`none`，preprocess=`deca`，crop_scale=1.25，cache=`disabled (deca_feature_representation=none)` |
| 深度先验 | `numbered datasets [3, 4, 5, 11, 12, 13]; data/processed/depth_priors_iris_ipd_65mm_v1_<id>.csv` |
| 人脸/双眼视觉融合 | use=True，face=`inception_resnet_v1(vggface2)`，eye=`resnet18`，cross_attention=8x128，face_frozen=False |
| 眼部骨干 | `resnet18`，weights=`DEFAULT` |
| Crop分支 | use=False，input_dim=36 |
| Scene分支 | representation=`table_frame7`，input_dim=7 |
| 几何分支 | use=False，gate=`none` |
| 融合层 | `[256, 128]`；总参数=35,157,190 |
| 训练 | epochs=100，batch=32，optimizer=`adamw`，lr=0.0003，weight_decay=0.0001 |
| 调度器 | `cosine`，eta_min=1e-06 |
| 损失 | uv_huber_beta_mm=30.0，uv_huber_weight=1.0，mixture_nll_weight=0.0，gaze_direction_weight=0.0，gaze_angular_weight=100.0，depth_prior_kl_weight=0.0，ray_validity_weight=0.1 |

### 性能指标

| 记录 | Loss | EPE | Median EPE | MAE-U | MAE-V |
|---|---:|---:|---:|---:|---:|
| 最佳验证（epoch 34） | 53.611 | 79.562 mm | 77.388 mm | 43.537 mm | 57.734 mm |
| 最终训练（epoch 100） | 10.670 | 14.947 mm | 13.487 mm | 10.191 mm | 8.743 mm |
| 最终验证（epoch 100） | 74.855 | 101.530 mm | 99.695 mm | 49.118 mm | 81.107 mm |

相比上一实验：最佳 Val EPE 下降 26.633 mm，性能提升。

<!-- AUTO-DETAILS-END -->
