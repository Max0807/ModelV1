# ModelV1 V4：gaze 学习与相机/桌面几何解耦

## 数据流

```text
DECA 特征 + 左右眼图像
        │
        └── 可学习 gaze 网络 ──> 单位向量 g_c（眼睛指向注视点）

左右眼 PnP 坐标（mm）+ PnP 对数尺度不确定性
        └── 无参数候选生成 ──> K 个双眼中点 E_c^(k)

g_c + E_c^(k) + TableFrame7(e1_c,e2_c,d_c)
        └── 无参数射线/平面求交 ──> K 个 uv^(k)（mm）
```

由于直线/平面交点对 `g_c` 与 `-g_c` 完全相同，V4 在无参数几何层内使用基准尺度的 PnP 双眼中点和桌面方程，将视觉网络预测的 gaze 轴自动定向到 `lambda > 0` 的桌面半球。原始轴保存在 `gaze_axis_direction_c`，最终物理方向保存在 `gaze_direction_c`。该定向不包含可学习参数。

V4 构造时强制执行以下约束：

- `use_crop_cam=false`；相机/裁剪元数据不进入 gaze MLP。
- `use_eye_geometry=false`；PnP 双眼坐标不进入 gaze MLP。
- `scene_representation=table_frame7`，但不创建 `scene_branch`。
- `TableFrame7` 只传给 `RayTableGeometry`；该模块参数量为 0。
- 桌面距离从 CSV 的米制缩放恢复到毫米，几何计算全程使用毫米。
- UV 原点是相机光心在桌面上的正交投影；UV 轴是 Vicon 世界 X/Y 在相机系中的方向。

训练时 `gaze_target_camera_mm` 只可作为监督量；推理不需要它。`uv_gt` 只用于损失和指标。验证/测试不得用自身标签拟合归一化量或不确定性参数。

方向监督采用权重为 1.0 的半球约束：

```text
L_direction = ReLU(-dot(gaze_direction_c, gaze_gt_c))
```

它只惩罚与“眼睛→注视点”GT 相反的半球，不强迫网络拟合受 PnP 深度误差影响的精确角度。

V4 的训练、验证 CSV/W&B 额外记录：`physical_valid_rate`、`mean_lambda_mm`、`behind_ray_rate`、`parallel_ray_rate`、`gaze_axis_error_deg` 和 `directed_gaze_error_deg`。

## PnP 不确定性

`scripts/generate_depth_priors.py` 对每一帧执行：

1. 固定 FLAME 尺度 `1010 mm / FLAME unit` 求基准 PnP；
2. 从基准重投影得到二维残差；
3. 做 64 次中心化残差 bootstrap；
4. 做逐 PnP 点留一 jackknife；
5. 计算双眼中点深度的共同 `log(z/z0)` 标准差。

该估计不读取深度 GT、UV GT、桌面方程或数据集划分统计。生成命令：

```powershell
python scripts/generate_depth_priors.py --device auto --batch-size 8
```

默认会新建 `data/processed/depth_priors_deca_crop_v2.csv`，不会覆盖 V3 的 v1 表。新 CSV 的关键字段是 `depth_log_scale_std`、`depth_scale_p05/p50/p95` 和各类成功样本数。旧深度表仍可运行 V3 和 V4 point；17 候选模式使用带不确定性字段的 v2 表。

## 三阶段训练

1. `train_dataset5_v4_point.yaml`：一个 PnP 深度点，先学习 gaze。
2. `train_dataset5_v4_fixed_prior.yaml`：K=17，使用固定 PnP 先验权重；可通过 `training.initialization_checkpoint` 载入阶段 1 的 `best.pt`。
3. `train_dataset5_v4_learned_reweight.yaml`：K=17，只训练共享候选权重器；必须显式载入阶段 2 检查点。

第三阶段会同时冻结 gaze 网络参数、BatchNorm 运行统计量和 Dropout 状态。权重器只读取冻结的视觉上下文、训练集归一化的 PnP 质量、候选对数尺度和先验权重；它不读取 `TableFrame7`、相机位姿或候选 UV。

## 推理输出

V4 返回：

- `gaze_direction_c`
- `eye_hypotheses_c_mm`、`depth_scales`
- `uv_hypotheses_mm`、`ray_valid_mask`
- `uv_mean_mm`、`uv_map_mm`、`uv_covariance_mm2`
- `depth_effective_weights`

`predict_uv_mm` 会自动识别 V4 映射输出；旧 V3 张量输出仍要求传入训练集 UV normalizer。
