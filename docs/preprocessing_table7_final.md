# ModelV1 最终版 TableFrame7 预处理

## 1. 目标

这套流水线只为当前训练所需的三张虚拟相机图像、`table_frame7_n[7]` 和
二维注视标签服务。绝对深度的唯一来源是 **Iris65/IPD**；PnP 只提供头部旋转
`R_C_H`，其内部求解得到的平移不保存，也不参与虚拟相机中心计算。

总流程：

```text
采集日志 + InsightFace 检测
              |
              v
Stage 1: 基础数据整合、真实相机/桌面坐标
              |
              v
Stage 2A: 单次 MediaPipe FaceMesh
          |-- 左右眼各 15 点
          |-- 左右虹膜中心（15 点中的 index=10）
          `-- PnP 所需 8 个二维点
              |
              v
Stage 2B: DECA/FLAME + PnP rotation + Iris65
          |-- R_C_H：PnP/FLAME
          |-- left/right iris proxy 3D：Iris65
          `-- t_C_H：Iris midpoint + FLAME local offset
              |
              v
Stage 3: 虚拟相机图像 + table_frame7_n[7] + UV 标签
```

## 2. 配置和入口

默认配置：
[modelv1_table7_final.yaml](../configs/preprocessing/modelv1_table7_final.yaml)

真实相机标定配置：
[crossgaze_camera.yaml](../configs/camera/crossgaze_camera.yaml)

`camera.calibration_file` 指向离线标定好的真实相机内参和畸变参数。
Iris65 的像素射线、DECA 输入图像尺寸校验、虚拟相机反畸变/warp 都使用这一份配置。
运行预处理时不会重新用标定板图像估计内参；换相机、换分辨率或换裁剪链路时，应先在外部完成 OpenCV 相机标定，再替换这份 YAML。

完整运行：

```powershell
python scripts/prepare_modelv1_table7.py
```

指定另一份配置：

```powershell
python scripts/prepare_modelv1_table7.py --config configs/preprocessing/my_dataset.yaml
```

只运行某一阶段时，修改配置中的：

```yaml
execution:
  stages: [2]       # 可写 [1]、[2]、[3] 或 [1, 2, 3]
  overwrite: false
  dry_run: false
  limit: null
```

也可以直接运行阶段脚本；它们都只接受同一个 `--config` 参数：

```powershell
python scripts/build_modelv1_table7_base.py --config configs/preprocessing/modelv1_table7_final.yaml
python scripts/extract_mediapipe_table7_observations.py --config configs/preprocessing/modelv1_table7_final.yaml
python scripts/build_iris65_metric_head_pose.py --config configs/preprocessing/modelv1_table7_final.yaml
python scripts/build_virtual_camera_table7.py --config configs/preprocessing/modelv1_table7_final.yaml
```

## 3. 三个阶段

### Stage 1：基础数据整合

代码：
[build_modelv1_table7_base.py](../scripts/build_modelv1_table7_base.py)

输入：

- `data_log_*.csv`：采集帧、真实相机 Vicon 位姿、真实注视点。
- `insightface_coordinates.csv`：人脸与左右眼 bbox。
- 原始图像和已有 face crop 路径。

输出：`stage1_base/base_dataset<ID>.csv`。

主要字段：

- 标识与路径：`sample_id`、`dataset`、`image_name`、`source_image_path`。
- bbox：人脸/左右眼各 `[x,y,w,h]`。
- 相机：`R_C_W[3,3]`、`R_W_C[3,3]`、`t_C_W[3]`、`t_W_C[3]`。
- 桌面：`table_u_axis_camera[3]`、`table_v_axis_camera[3]`、
  `table_distance_camera_mm[1]`。
- 标签：`uv_gt_[u,v]_mm[2]`、`gaze_target_W[3]`。

桌面高度按当前约定使用每帧的 `gaze_target_tz`，配置值为
`table_z_source: per-sample-target`。

### Stage 2：Iris65 metric head pose

MediaPipe 代码：
[extract_mediapipe_table7_observations.py](../scripts/extract_mediapipe_table7_observations.py)

Metric pose 代码：
[build_iris65_metric_head_pose.py](../scripts/build_iris65_metric_head_pose.py)

Stage 2A 对每张原图只执行一次 `FaceMesh(refine_landmarks=True)`，输出
`mediapipe_eye15_dataset<ID>.npz`：

- `left/right_eye_points_px`：`[N,15,2]`，源图像像素坐标。
- `left/right_eye_points_crop_norm`：`[N,15,2]`，眼 crop 归一化坐标。
- `left/right_eye_points_local`：`[N,15,2]`，眼角局部坐标。
- `left/right_eye_point_valid_mask`：`[N,15]`。
- `left/right_eye_geometry`：`[N,10]`，质量/形态特征。
- `pnp_points_px`：`[N,8,2]`，同次检测得到的 PnP 二维点。
- 虹膜中心就是左右眼 15 点中的 `index=10`。

Stage 2B 输出 `iris65_metric_head_pose_dataset<ID>.csv`，核心字段：

- `left/right_iris_proxy_camera_[x,y,z]_mm`：左右虹膜/瞳孔代理点，各 `[3]`。
- `iris_midpoint_camera_[x,y,z]_mm`：左右代理点中点 `[3]`。
- `head_rotation_c_from_h_00..22`：`R_C_H[3,3]`，来自 PnP。
- `head_origin_camera_[x,y,z]_mm`：`t_C_H[3]`，来自 Iris65 和 FLAME 偏移。
- PnP reprojection、Iris baseline residual、ray condition 等质量字段。

关键计算：

```text
s = IPD_mm / ||right_eye_proxy_H - left_eye_proxy_H||
eye_proxy_mid_H_mm = s * (left_eye_proxy_H + right_eye_proxy_H) / 2
t_C_H = iris_mid_C_mm - R_C_H * eye_proxy_mid_H_mm
```

FLAME 68 点没有真正的虹膜中心，因此这里的局部 eye proxy 是每只眼两个眼角的
中点。这是显式的几何近似；`flame_scale_from_ipd_mm_per_unit` 会写入 CSV 供检查。

Rotation-only PnP 会先将选中的 FLAME 点减去质心，再除以 RMS 半径，形成无量纲
目标点。候选解只检查所有归一化目标点位于相机前方、旋转矩阵有效以及二维重投影
误差；不使用 FLAME-to-mm 固定尺度，也不使用绝对平移或绝对深度阈值。求解器内部
平移属于归一化目标点单位，会被直接丢弃。

### Stage 3：虚拟相机和 TableFrame7

代码：
[build_virtual_camera_table7.py](../scripts/build_virtual_camera_table7.py)

输入：Stage 1 CSV、Stage 2 metric pose CSV 和原图。

输出目录：

```text
stage3_virtual_table7/
├── images/
│   └── face/<dataset>/*.png
├── training_table7_dataset<ID>.csv
└── training_table7_dataset<ID>_report.json
```

最终 CSV 只写成功样本，字段包括：

- 样本标识、源图路径和虚拟人脸图路径。
- `table_frame7_n_0..6`：模型输入 `[7]`。
- `uv_gt_u_mm, uv_gt_v_mm`：训练标签 `[2]`。

`table_frame7_n[7]` 的顺序是：

```text
[e_u_N.x, e_u_N.y, e_u_N.z,
 e_v_N.x, e_v_N.y, e_v_N.z,
 d_C_m]
```

其中 `e_u_N = R_N_C e_u_C`，`e_v_N = R_N_C e_v_C`，最后一维是相机原点到
桌面平面的有符号距离，单位由 mm 转为 m。最终 CSV 不保存
`virtual_camera_pose_table[9]`；如果后续网络仍需要 pose9，应从 TableFrame7
即时推导，而不是离线重复保存。

## 4. 主要参数

- `paths.source_root`：采集数据根目录。
- `paths.output_root`：最终预处理输出目录。
- `camera.calibration_file`：真实相机 OpenCV 标定 YAML。Iris65 深度依赖这里的 K、畸变参数和源图尺寸。
- `metric_head_pose.ipd_mm`：被试 IPD；当前默认 65 mm。若有实测值，应改成实测值。
- `fixed_shape_calibration.start/stop`：每个数据集必须检查的固定 shape 标定帧范围。
- `max_ipd_residual_mm`、`max_ray_condition`：Iris65 几何质量阈值。
- `virtual_camera.*`：虚拟人脸输出尺寸、虚拟焦距和标准距离。
- `execution.overwrite`：是否允许覆盖本流水线已有输出。
- `execution.limit`：仅限制 Stage 2/3 的昂贵处理；Stage 1 始终完整整合基础
  CSV，避免截断 fixed-shape 标定帧。正式生成保持 `null`。

## 5. 旧代码复用审计

最终入口已经解除以下冲突复用：

- Stage 1 不再调用旧 `build_modelv1_dataset.py`，不会扫描历史 PnP/depth CSV，
  也不会先构造无用的 `scene25` 和 `crop36`。
- Stage 2B 不再调用 `solve_pnp_face_depth()`，改用独立的无尺度
  `solve_head_rotation_pnp()`。
- Stage 2B 不再从旧 `generate_iris_ipd_depth_priors.py` 导入图像/shape
  标定助手，这些通用功能已经迁入最终版 `preprocessing/deca_inputs.py`。

仍然保留的复用都是与当前要求一致的基础能力：

- DECA/FLAME：只产生去除全局旋转后的 head-local 68 点。
- Iris/IPD：只负责虹膜射线的 metric 重建。
- virtual camera：只负责统一图像坐标变换与可选眼部裁剪。
- asset resolver 和 face preprocess：只负责路径恢复与 DECA 输入裁剪。

## 6. 环境依赖与检查

运行环境需要：NumPy、PyYAML、Pillow、OpenCV、legacy MediaPipe、PyTorch，
以及仓库内可用的官方 DECA/FLAME 代码和 checkpoint。

纯几何单元测试：

```powershell
python -m unittest tests.test_table7_preprocessing_geometry -v
```

建议逐阶段检查：

1. Stage 1 的相机旋转是否正交，table U/V 是否单位正交，距离单位是否为 mm。
2. Stage 2A 的左右眼语义、虹膜 index=10、15 点质量和 PnP 8 点覆盖率。
3. Stage 2B 的 PnP 重投影误差、Iris residual/condition、头部深度分布。
4. Stage 3 随机查看虚拟 face 图，并核验 TableFrame7 前 6 维仍为两条单位正交轴。

当前 Direct-UV 训练已经支持通过
`data.table7_stage3_csv_paths` 直接读取最终 CSV。loader 使用
`virtual_face_path`、`table_frame7_n_0..6` 和 UV 标签；左右眼输入依据
`source_image_path` 保留对应的 legacy 90x56 左右眼裁剪，从而继续
匹配既有 eye15 伪标签。该模式不再加载或重复应用旧 PnP1010 virtual-camera
manifest。


