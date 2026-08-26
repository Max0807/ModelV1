# ModelV1 第二部分验证：MediaPipe–FLAME–solvePnP

验证日期：2026-08-12  
验证契约：`mediapipe_flame_head_local_multistart_v2`

## 1. 这一段验证什么

第二部分验证训练前离线几何链路：

```text
1920×1080 原始非镜像人脸图像
→ MediaPipe 2D 像素关键点
→ DECA/FLAME 头部局部 3D 关键点
→ 固定 2D–3D 语义对应
→ 毫米尺度
→ OpenCV solvePnP
→ R_camera_from_head、t_camera_from_head
→ PnP 人脸原点
→ 虚拟相机归一化中心
```

它验证的是先验几何和数据契约，不是模型的 gaze EPE，也不是 Iris/IPD 重建本身。
正式协议仍是 dataset13 训练、dataset4 跨相机验证；本阶段不能替代后续训练对照实验。

## 2. 怎么验证

### 2.1 输入图像、相机与关键点语义

验证内容：原图必须与相机标定的 `1920×1080` 分辨率一致；MediaPipe 索引与
FLAME 68 点必须逐项同义；CSV 中遗留的 `left/right` 是图像侧，而训练输出的
`left/right` 是人物解剖侧。

权威定义：

- 相机内参与分辨率：`modelv1/depth_prior/pnp.py` 的
  `CROSSGAZE_CAMERA_MATRIX`、`CROSSGAZE_DIST_COEFFS`、
  `validate_crossgaze_image_size`。
- MediaPipe 索引：同文件的 `MEDIAPIPE_PNP_LANDMARK_INDICES`。
- 2D–3D 对应：同文件的 `DEFAULT_PNP_MAPPING`。
- 解剖左右眼：`FLAME_LEFT_EYE_CANTHUS_INDICES=(42,45)`，
  `FLAME_RIGHT_EYE_CANTHUS_INDICES=(36,39)`。
- 虚拟相机换侧：`scripts/generate_virtual_camera_normalized_images.py` 的
  `eye_corners_from_landmark_row`。

手动验证方法：

1. 在原图上画出 33、133、362、263、1、61、291、152 八个点。
2. 非镜像原图中，33/133 应位于图像左眼，362/263 应位于图像右眼。
3. 输出人物解剖左眼时必须使用图像右侧组；人物解剖右眼反之。
4. 任意图片不是 `1920×1080` 时，应显示
   `CrossGaze camera calibration expects 1920x1080 px`，而不是继续求解。

正式 dataset13/4 共 1584 条成功关键点记录，图像左右组横坐标次序违规数为 0。

为什么验证：常见错误是把图像左眼当作人物左眼、把归一化坐标当像素、在缩放图
上使用原图内参、或者更新一处索引却遗漏另一个脚本。这些错误可能仍得到低重投影误差。

训练位置与价值：这是所有 PnP、Iris/IPD、虚拟相机输入之前的数据语义层。其本身是
必要工程校验，不是独立算法创新；统一版本化语义契约具有很高可行性和复现实验价值。

### 2.2 FLAME 必须是头部局部坐标

验证内容：传给 PnP 的 3D 点不能已经包含 DECA 预测的全局头部旋转，否则
`solvePnP` 只会估计残余旋转，输出的 `R` 不再是头部局部到相机的完整变换。

最重要代码：

- `modelv1/depth_prior/deca_flame.py` 的 `head_local_pose_parameters`：只把姿态
  前三维全局旋转清零，保留下颌姿态。
- `DecaFlameExtractor.extract`：生成唯一允许进入 PnP 的
  `head_local_landmarks3d`。
- `scripts/generate_depth_priors.py` 和
  `scripts/generate_iris_ipd_depth_priors.py` 均只传该字段。

手动断点检查：

```text
code_dict["pose"][:, :3]       != 0（一般情况）
head_local_pose[:, :3]         == 0
head_local_pose[:, 3:]         == code_dict["pose"][:, 3:]
landmarks3d input to PnP       == output.head_local_landmarks3d
```

为什么验证：旧实现将全局旋转过的 FLAME 点直接传入 PnP，数值可以收敛，但 `R/t`
的物理含义错误，后续头部旋转归一化也会错误。

训练位置与价值：这是 DECA 特征提取和 PnP 之间的坐标系边界。修复是正确性前提，
不是论文创新；“头部局部形状 + 相机外参”解耦后，虚拟相机实验才具有可行性。

### 2.3 尺度、R/t 方向与物理解

验证内容：FLAME unit 先乘固定 `1010 mm/unit`；求解满足
`X_camera_mm = R_camera_from_head @ X_head_mm + t_camera_from_head_mm`；人脸原点、
全部 PnP 点和双眼都必须位于 OpenCV 相机前方。

最重要代码：

- `modelv1/depth_prior/pnp.py` 的 `PnpConfig`：硬阈值集中定义。
- `_evaluate_pnp_candidate`：深度、正前方和重投影硬过滤。
- `_solve_physical_pnp_candidates`：ITERATIVE/EPNP/SQPNP 多初值及迭代细化。
- `solve_pnp_face_depth`：选择物理有效候选并输出 R/t、双眼和质量字段。

硬验收条件：

```text
pnp_geometry_version == mediapipe_flame_head_local_multistart_v2
100 <= tvec_z_mm <= 2000
pnp_min_object_depth_mm > 0
left_eye_camera_z_mm > 0 and right_eye_camera_z_mm > 0
max(abs(R.T @ R - I)) < 1e-5
abs(det(R) - 1) < 1e-5
reprojection_error_mean_px <= 8
reprojection_error_max_px <= 20
scale_disagreement_ratio <= 0.25
```

必须重点查看 CSV 的这些列：

```text
depth_prior_status, reason, pnp_geometry_version
pnp_solver_method, pnp_candidate_count
tvec_x_mm, tvec_y_mm, tvec_z_mm
rotation_00 ... rotation_22
pnp_min_object_depth_mm
left_eye_camera_z_mm, right_eye_camera_z_mm
reprojection_error_mean_px, reprojection_error_max_px
pnp_inlier_count, pnp_num_points, scale_disagreement_ratio, pnp_confidence
```

通过时，生成器最后应显示 `Status counts: {'success': N}`；审计器顶层和每个文件
都应显示 `"passed": true`、`"violations": {}`，进程退出码为 0。

运行正式训练/验证组合：

```powershell
D:\software\Anaconda\envs\cross_gaze\python.exe scripts\validate_pnp_geometry.py `
  data\validation_stage2\depth_priors_pnp_v2_dataset13.csv `
  data\validation_stage2\depth_priors_pnp_v2_dataset4.csv `
  --output data\validation_stage2\pnp_geometry_validation_report_train13_val4.json
```

失败时最先看 `violations`，再按 `sample_id` 在 CSV 中检查上述列。不要只看 OpenCV
返回的 `success=True`。

为什么验证：裸 `SOLVEPNP_ITERATIVE` 在旧 dataset11/12/13 中产生了 27/56/18，
合计 101 条整张脸位于相机后方的镜像局部解，但旧 CSV 仍写成 `success`。旧文件也没有
版本号和最小物点深度，无法证明来源。多初值加 cheirality 检查专门阻断这一类错误。

训练位置与价值：R/t 既可生成 PnP 眼部几何先验，也直接驱动虚拟相机旋转和中心。
多初值不是新的 PnP 算法，创新价值有限；将物理解、版本、质量过滤和训练数据契约整合，
是支持可复现消融实验的工程贡献。

### 2.4 PnP 人脸原点必须是唯一虚拟相机中心

验证内容：虚拟相机必须直接读取 PnP 的 `tvec`，不能使用双眼中点、检测框中心、
旧 CSV 深度或另一套 PnP 结果。

最重要代码：

- `scripts/generate_virtual_camera_normalized_images.py` 的
  `center_from_depth_row`：唯一返回 `[tvec_x,tvec_y,tvec_z]`。
- `validate_depth_row_for_normalization`：使用前再次执行版本、深度、旋转、误差和
  尺度硬校验。
- `modelv1/data/virtual_camera_manifest.py`：训练加载时再次要求完全相同的版本号和
  人物解剖左右眼语义。

通过输出必须包含：

```text
center_source: pnp_tvec
pnp_geometry_version: mediapipe_flame_head_local_multistart_v2
virtual_camera.focal_length_px: 480
virtual_camera.distance_mm: 600
status_counts.success == selected sample count
manifest.csv 中 normalization_status == success
```

dataset13 和 dataset4 各抽取 10 条端到端归一化，均为 10/10 成功；训练侧 manifest
加载结果均为 10 条有效、0 条无效。人工查看时脸应完整、双眼 crop 应分别落在人物解剖
左/右眼，不能靠修改 480 px 或 600 mm 来掩盖 PnP 抖动。

为什么验证：仓库旧虚拟相机目录引用了旧的无版本 PnP 表；文件名相似并不能保证
几何语义相同。新 loader 会明确拒绝旧 manifest，防止静默混用。

训练位置与价值：这是第二部分到图像归一化/训练输入层的交界。Sugano/Zhang 归一化
本身不是本项目创新；以可追溯 PnP 原点为唯一中心的实验设计可行，但需要后续 EPE 对照
才能判断是否带来模型收益。

### 2.5 PnP 质量字段是否真正参与处理

当前分为两层：

1. `mean/max reprojection error`、正深度、旋转和尺度分歧是硬过滤，生成虚拟相机或
   加载 `DepthPriorTable` 时不满足就拒绝。
2. `pnp_confidence` 是诊断值；只有启用 `require_pnp_quality`、深度修正或 learned
   reweighting 的配置，才会组成 `pnp_quality_vec` 进入模型。当前 point/virtual-camera
   主配置的 `use_eye_geometry=false`、`use_depth_correction=false`，因此它不参与加权。

手动 debug：先在 `modelv1/data/depth_prior.py::DepthPriorTable.load` 查看样本是否进入
`geometry_by_sample_id`；再在 `modelv1/data/dataset.py` 查看是否产生
`pnp_quality_vec`；最后检查训练 YAML 的 `use_eye_geometry`、`use_depth_correction`。

为什么验证：仅把质量字段写进 CSV 或 manifest 不等于训练实际使用。把启发式
`pnp_confidence` 当校准概率也会产生错误结论。

训练位置与价值：硬过滤属于数据层；可学习质量加权属于模型输入/融合层，可能有创新
空间，但必须在相同 sample ID、split、初始化、超参和 checkpoint 规则下单独消融。

## 3. 全量验证结果

正式协议组合：

| 数据集 | 角色 | 成功/总数 | tvec Z 中位数 mm | mean reproj 中位数/最大 px | max reproj 最大 px |
|---|---|---:|---:|---:|---:|
| dataset13 | train | 1418/1418 | 741.00 | 3.12 / 7.77 | 14.98 |
| dataset4 | cross-camera val | 166/166 | 737.42 | 3.40 / 5.27 | 8.39 |

合计 1584/1584 通过；sample ID 与各自数据集 CSV 完全一致，train/val 交集为 0；
最小 PnP 物点深度 607.25 mm，最小眼部深度 629.27 mm。所有旋转正交误差与
`abs(det(R)-1)` 均远低于 `1e-5`。

补充复建 dataset3/11 也分别为 383/383、583/583 通过。权威机器可读结果：

- `data/validation_stage2/pnp_geometry_validation_report_train13_val4.json`
- `data/validation_stage2/pnp_geometry_validation_report.json`
- `data/validation_stage2/legacy_pnp_rejection_report.json`

## 4. 代码简化结论

已完成：

- MediaPipe 索引、相机分辨率、PnP 映射和阈值集中到 `pnp.py`，删除脚本间重复常量。
- 固定 PnP 导出器删除从未生成有效值的 bootstrap 参数、伪不确定性空字段和错误说明。
- DECA 输出删除无人使用的完整网格、全局姿态关键点、动态关键点和硬编码眼球顶点；
  只保留头部局部 68 点和 Iris 固定形状校准所需参数。
- PnP/Iris 与虚拟相机都使用同一 `PNP_GEOMETRY_VERSION`、同一物理求解器和同一质量列。

保留、不应为了“代码短”而删：有限值、维度、分辨率、sample ID 唯一性、版本、正深度、
旋转、误差、尺度阈值以及下游 loader 的重复校验。这些分别保护生成边界和训练加载边界。

仍需处理：旧 `depth_priors_deca_crop_v1*.csv`、旧 Iris/IPD 表和旧虚拟相机 manifest
应视为历史产物。不要删除以便审计，但不能用于新正式实验；必须用 v2 PnP 契约重新生成
dataset13/4 的 Iris/IPD 和完整虚拟相机文件后再更新训练配置。

## 5. 验证总结

第二部分核心 PnP 几何链路结论为 **通过**：输入语义、分辨率、头部局部 FLAME、毫米
尺度、R/t 方向、正深度、多初值、重投影、PnP 原点及版本唯一性均已由代码、单元测试和
正式 dataset13/4 全量数据共同验证。

第二部分下游正式训练产物结论为 **待重新生成**：新 PnP 已验证，但旧 Iris/IPD 与旧
虚拟相机产物不能继承这个结论。完成 v2 全量重建并做训练 EPE 对照后，才能评价创新收益。
