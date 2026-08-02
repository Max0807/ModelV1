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

<!-- AUTO-DETAILS-END -->
