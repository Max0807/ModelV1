# TableFrame7 Stage 1–3 完整生成与 Debug 说明书

## 1. 本文范围

本文只解释离线数据生成链路：怎样从采集日志、已有原图、InsightFace 检测结果和
相机标定出发，经过 Stage 1、Stage 2A、Stage 2B、Stage 3，最终生成模型需要的
`table_frame7_n[7]`、虚拟相机人脸图和 UV 标签。

它不把 DataLoader 到 `ModelV1.forward()` 当作主流程。训练读取属于本流水线之后的
另一个阶段，可参考 `modelv1_pre_model_debug_guide.md`。

有两个必须先说清的事实：

1. `table_frame7_n` 不是只从 RGB 图像推理出来的。桌面方向、桌面距离和 UV 真值依赖
   `data_log_*.csv` 中的相机 Vicon 位姿与注视目标。
2. 当前 Stage 1 不运行 InsightFace，也不重新标定相机。它假定
   `insightface_coordinates.csv`、`insightface_img/` 和相机标定 YAML 已经存在。
   MediaPipe 与 DECA/FLAME 才是本流水线中直接读取原图并执行模型推理的步骤。

## 2. 一眼看懂完整数据流

```text
外部已有数据
├─ data_log_*.csv
│  ├─ frame/image 名称
│  ├─ 相机刚体在世界系的 R、t
│  └─ gaze_target_W
├─ insightface_coordinates.csv ── face/left-eye/right-eye bbox
├─ insightface_img/*.jpg ───────── 1920×1080 原图
├─ crossgaze_camera.yaml ───────── K、畸变、图像尺寸
└─ DECA-master/data/deca_model.tar ─ DECA/FLAME 权重
          │
          ▼
Stage 1  build_modelv1_table7_base.py
          │  日志与 bbox 按 image_name 连接；算 C/W 变换、桌面轴、桌面距离、UV
          ▼
stage1_base/base_dataset<ID>.csv
          │
          ├──────────────────────────────────────────────┐
          ▼                                              │
Stage 2A extract_mediapipe_table7_observations.py        │
          │  原图 -> FaceMesh(refine_landmarks=True)     │
          │  -> 双眼 Eye15 + 虹膜中心 + 8 个 PnP 像素点 │
          ▼                                              │
mediapipe_eye15_dataset<ID>.npz                          │
          │                                              │
          └──────────────────────┬───────────────────────┘
                                 ▼
Stage 2B build_iris65_metric_head_pose.py
          ├─ 原图+bbox -> DECA -> 固定 shape 的 FLAME 68 点
          ├─ FLAME 3D + MediaPipe 2D -> rotation-only PnP -> R_C_H
          ├─ 虹膜像素 + K/畸变 + 65 mm IPD -> 双眼 metric 3D
          └─ 虹膜中点 - FLAME 局部眼中点偏移 -> t_C_H
                                 ▼
iris65_metric_head_pose_dataset<ID>.csv
                                 │
                                 ▼
Stage 3  build_virtual_camera_table7.py
          ├─ R_C_H,t_C_H -> R_N_C 与单应矩阵 -> 160×160 虚拟人脸
          └─ R_N_C × 桌面 U/V 轴 + 桌面距离 mm→m -> table_frame7_n[7]
                                 ▼
stage3_virtual_table7/
├─ images/face/.../*.png
├─ training_table7_dataset<ID>.csv
└─ training_table7_dataset<ID>_report.json
```

阶段之间统一使用 `sample_id` 连接，不应使用 CSV 行号连接。

## 3. 坐标系与最终 7 维契约

- `W`：世界/Vicon 坐标系。
- `C`：标定后的真实相机光学坐标系。
- `H`：移除 DECA 全局头旋转后的 FLAME 头部局部坐标系。
- `N`：以头原点为中心构造的虚拟相机坐标系。
- `R_A_B`：把 B 系向量变换到 A 系，例如 `R_C_H`。
- 空间平移、IPD、桌面距离在 Stage 1/2 中使用 mm；最终第 7 维使用 m。
- 图像像素原点在左上，x 向右、y 向下。

最终字段严格是：

```text
table_frame7_n = [
  e_u_N.x, e_u_N.y, e_u_N.z,
  e_v_N.x, e_v_N.y, e_v_N.z,
  d_C_m
]

e_u_N = R_N_C @ e_u_C
e_v_N = R_N_C @ e_v_C
d_C_m = table_distance_camera_mm / 1000
```

前 6 维是两条单位正交桌面轴；最后 1 维是从真实相机原点到桌面平面的有符号距离。
虚拟相机只是旋转坐标轴，因此平面距离不随 `R_N_C` 改变。

## 4. Debug 文件与运行方式

本仓库现在有两份配置：

- 正式配置：`configs/preprocessing/modelv1_table7_final.yaml`，输出到
  `data/preprocessed_table7/dataset<ID>`，处理全部样本。
- 调试配置：`configs/preprocessing/modelv1_table7_debug.yaml`，输出到
  `data/debug_table7/dataset<ID>`，昂贵的 Stage 2/3 只处理 3 个样本。

调试配置不会覆盖正式结果。它把固定 shape 标定临时缩短为 `frame_idx=0..4`，仅为
缩短单步调试时间；正式数据必须恢复正式配置中的被试标定范围。

VS Code 中选择 `D:/software/Anaconda/envs/cross_gaze/python.exe`，打开“运行和调试”，
按顺序使用：

1. `Table7 01: Stage1 基础数据整合`
2. `Table7 02A: MediaPipe Eye15 + Iris + PnP 2D`
3. `Table7 02B: DECA/FLAME + PnP + Iris65`
4. `Table7 03: 虚拟相机 + table_frame7_n`

也可以使用 `Table7 00` 一次执行完整链路；它通过子进程运行各阶段，已启用
`subProcess: true`。真正逐句理解时仍建议运行单阶段入口，因为调用栈更干净。

第一次调试建议在 VS Code 的 Breakpoints 面板打开 “Raised Exceptions”，这样每个样本
内部被 `try/except` 转为失败记录之前，会先停在原始异常位置。理解正常流程后再关掉，
否则 5 条预期质量门失败也会暂停。

## 5. Stage 0：统一配置解析与总调度

入口 `scripts/prepare_modelv1_table7.py:33`：

1. `parse_args()` 只接收一个 `--config`。
2. `load_table7_preprocessing_config()` 读取 YAML、把 `{dataset_id}` 展开并将相对路径
   转成仓库绝对路径。
3. `execution.stages` 被展开为 `1 -> 2A -> 2B -> 3`。
4. `_run()` 用当前 Python 解释器启动阶段脚本，因此完整入口中的真正工作都发生在
   Python 子进程。

推荐断点：

- `scripts/table7_preprocessing_config.py:133`：开始解析 YAML。
- `scripts/table7_preprocessing_config.py:211`：检查路径、阈值和 stage 合法性。
- `scripts/prepare_modelv1_table7.py:33`：观察最终 `config`。
- `scripts/prepare_modelv1_table7.py:27`：观察将要启动的 `command`。

重点查看：`config.dataset_name`、`source_root`、`output_root`、四个阶段产物路径、
`limit`、`overwrite`、`calibration_start/stop`。

## 6. Stage 1：基础数据整合

### 6.1 从哪里到哪里

代码范围：`scripts/build_modelv1_table7_base.py:251-415`。

输入：

- `<source_root>/dataset_dual_rigid_body_<ID>/data_log_*.csv`
- 同目录的 `insightface_coordinates.csv`
- `insightface_img/<image_name>`，这里只检查路径存在，不解码图像
- 常量 `HAND_EYE_RIGID_TO_OPTICAL`，把相机刚体坐标转为相机光学坐标

输出：

- `stage1_base/base_dataset<ID>.csv`，每个有效样本 59 个字段
- `stage1_base/base_dataset<ID>_report.json`，输入、写入、跳过数量及原因

### 6.2 每一步做什么

1. `:254-267` 定位唯一的采集日志，读取两个 CSV，并把 InsightFace 行按
   `image_name` 建唯一索引。
2. `:277-288` 遍历采集日志，过滤缺失图名、缺失 bbox 行和 InsightFace 失败行。
3. `:290-296` 读取 `gaze_target_W=[x,y,z]`，并把三个 bbox 裁到
   `1920×1080` 图像范围内。
4. `:297-299` 在 `_camera_pose()` 中组合 Vicon 刚体位姿与 hand-eye 外参：

   ```text
   R_W_C = R_W_R @ R_R_C
   t_W_C = R_W_R @ t_R_C + t_W_R
   R_C_W = R_W_C^T
   t_C_W = -R_C_W @ t_W_C
   ```

5. `:300-324` 构建桌面。当前配置令每帧桌面高度
   `z_table = gaze_target_tz`；世界系桌面 U/V/normal 分别为 X/Y/Z 轴。把 U/V/normal
   旋转到相机系，计算有符号距离并校验它等于 `z_table - camera_z_world`。
6. 以 `[camera_x_world,camera_y_world,z_table]` 为桌面 UV 原点，将注视点投影到世界
   X/Y 桌面轴得到 `uv_gt_u_mm, uv_gt_v_mm`。
7. `:333-370` 写入标识、路径、bbox、相机双向位姿、桌面轴/距离和标签。
8. `:371-385` 生成统计报告；`:404-413` 先写临时文件再原子替换正式 CSV/JSON。

注意：`execution.limit` 不截断 Stage 1。这保证 Stage 2B 按 `frame_idx` 做固定 shape
标定时仍能访问完整基础表。

### 6.3 推荐断点与观察值

| 断点 | 观察内容 | 正常条件 |
|---|---|---|
| `:251` `_build_rows` | `dataset_dir`, `logs`, `insight_path` | 路径指向同一数据集 |
| `:277` 样本循环 | `image_name`, `data`, `insight` | 两表图名一致 |
| `:297` `_camera_pose` 后 | `r_wc,t_wc,r_cw,t_cw` | `r_cw @ r_wc ≈ I` |
| `:313` 距离 | `table_distance_camera`, `expected_distance` | 差值小于 `1e-3 mm` |
| `:333` 组装行 | `uv`, `gaze_camera`, `source_path` | 全部有限、原图存在 |
| `:398` 返回主函数 | `len(rows)`, `report['skipped']` | 跳过原因可解释 |

## 7. Stage 2A：一次 MediaPipe 得到全部 2D 观测

### 7.1 从哪里到哪里

代码范围：`scripts/extract_mediapipe_table7_observations.py:60-224`。

输入是 Stage 1 CSV 中的 `source_image_path`、左右眼 bbox 与 `sample_id`。每张原图读取
为 RGB `[H,W,3]`，交给 legacy MediaPipe
`FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True)`。

输出是：

- `mediapipe_eye15_dataset<ID>.npz`
- `mediapipe_eye15_dataset<ID>_report.json`

NPZ 保留成功和失败的所有选中样本，靠 mask 表示可用性。主要数组为：

| 数组 | shape | 含义 |
|---|---:|---|
| `sample_id` | `[N]` | 阶段间主键 |
| `image_size_px` | `[N,2]` | `[width,height]` |
| `left/right_eye_points_px` | `[N,15,2]` | 原图像素坐标 |
| `left/right_eye_points_crop_norm` | `[N,15,2]` | 相对眼 bbox 的归一化坐标 |
| `left/right_eye_points_local` | `[N,15,2]` | 以眼角中点为原点、眼角距离为尺度 |
| `left/right_eye_point_valid_mask` | `[N,15]` | 点同时落在图像与 eye bbox 内 |
| `left/right_eye_geometry` | `[N,10]` | 虹膜/眼睑形态与质量特征 |
| `sample_quality_score` | `[N]` | 两眼 quality 的较小值 |
| `pnp_points_px` | `[N,8,2]` | PnP 的 8 个 MediaPipe 像素点 |
| `pnp_point_valid_mask` | `[N,8]` | PnP 点有限性 |

Eye15 顺序为：眼外角、4 个上眼睑点、眼内角、4 个下眼睑点、虹膜中心、4 个虹膜
边缘点。虹膜中心固定是 `index=10`。

解剖学左右非常重要：subject-left 是图像右侧、MediaPipe 473 虹膜组；
subject-right 是图像左侧、468 虹膜组。这里不能按图像左右交换。

### 7.2 每一步做什么

1. `:93-105` 从 Stage 1 选行并预分配 Eye15 与 PnP 数组。
2. `:114-119` 只创建一次 FaceMesh 检测器。
3. `:120-139` 读取一张原图、执行一次 `detector.process(rgb)`，从同一套 landmark
   中抽取左右 Eye15 和 8 个 PnP 点。
4. `:140-157` 调用 `fit_eye_geometry()`。该函数建立眼部局部坐标，拟合虹膜圆形质量，
   计算点在图像/bbox 中的比例、眼角距离和眼睑开合。
5. `:158-179` 写入数组与 mask；只有左右眼都通过阈值时 `valid_mask=True`。
6. `:196-222` 将数组、失败原因和完整元数据原子写成 NPZ/JSON。

`fit_eye_geometry()` 的无效条件位于
`scripts/extract_mediapipe_eye_geometry.py:354-366`：眼角小于 8 px、点出图、bbox 内比例
小于 0.8、虹膜半径不合理、开合非正或综合质量低于 0.25。

### 7.3 推荐断点与观察值

- `extract_mediapipe_table7_observations.py:120`：查看当前 `row`。
- `:129`：单步进入 MediaPipe；返回后检查是否有 `multi_face_landmarks`。
- `:133`：检查 `left/right.shape == (15,2)`，并确认左右语义。
- `:139`：检查 `pnp.shape == (8,2)`。
- `extract_mediapipe_eye_geometry.py:262`：进入 Eye15 质量计算。
- `:318-337`：观察 `iris_center`、`iris_radius`、`quality`。
- `extract_mediapipe_table7_observations.py:172`：检查双眼门控结果。
- `:196`：写文件前检查所有数组第一维都等于 `len(rows)`。

## 8. Stage 2B：DECA/FLAME、旋转 PnP 与 Iris65

### 8.1 这个阶段解决什么问题

Stage 3 构造虚拟相机需要头部在真实相机系中的旋转 `R_C_H` 和毫米级中心
`t_C_H`。这里把两个信息源严格分工：

- PnP 只给旋转，内部 `tvec` 是归一化 FLAME 单位，直接丢弃。
- Iris65 只给绝对毫米深度与双眼位置。

输入为完整 Stage 1 CSV、Stage 2A NPZ、相机 K/畸变、DECA checkpoint。输出
`iris65_metric_head_pose_dataset<ID>.csv` 与报告 JSON。CSV 保留每个选中样本；失败行
写 `metric_pose_status=failed` 与 `reason`，而不是从文件中删除。

### 8.2 DECA 输入与固定 shape

1. `build_iris65_metric_head_pose.py:181` 初始化 `DecaFlameExtractor`。设备为
   `cuda if available else cpu`，加载官方 `E_flame` 和 FLAME。
2. `:183` 调用 `calibrate_fixed_shape()`。在指定 `frame_idx` 闭区间内，每张图先按
   face bbox、`crop_scale=1.25` 做官方风格裁剪并缩放为 RGB `224×224`，变成
   `tensor [1,3,224,224]`、float32、范围 `[0,1]`。
3. 对标定帧分别运行 DECA，取编码参数前 100 维 shape，沿样本维取 median，得到
   `fixed_shape [100]`。这样同一个人的形状不随帧抖动。
4. 对当前样本再次运行 DECA，但用 `fixed_shape` 覆盖本帧 shape；保留表情和下颌姿态。
   `head_local_pose_parameters()` 把 pose 的前三维全局旋转清零，FLAME 输出
   `head_local_landmarks3d [1,68,3]`。

推荐断点：

- `modelv1/preprocessing/deca_inputs.py:41`：看裁剪前原图、bbox 与 `deca_image`。
- `deca_inputs.py:59` 和 `:118`：看标定帧集合与 `fixed_shape.shape`。
- `modelv1/depth_prior/deca_flame.py:303`：DECA encoder 输出。
- `deca_flame.py:328-329`：全局旋转清零后进入 FLAME。
- `build_iris65_metric_head_pose.py:224`：当前样本的最终 FLAME 输出。

### 8.3 Rotation-only PnP 得到 R_C_H

`solve_head_rotation_pnp()` 将 MediaPipe 2D 的眼角、鼻尖、嘴角、下巴共 8 点，与
FLAME 68 点中的对应 3D 点配对。3D 点先减质心，再除以 RMS 半径，因此尺度为 1，
不会把 FLAME 单位误当 mm。

代码依次尝试 ITERATIVE、EPNP、SQPNP，并用非 ITERATIVE 候选再做 ITERATIVE
refine。候选必须满足：

- 全部归一化 3D 点在相机前方；
- `R^T R≈I` 且 `det(R)≈1`；
- 平均重投影误差不超过 8 px；
- 最大重投影误差不超过 20 px。

从合格候选中选平均误差最小者，只返回其 `rotation_camera_from_head`。推荐断点：

- `build_iris65_metric_head_pose.py:226`
- `modelv1/preprocessing/rotation_pnp.py:215`：检查对应点 shape。
- `rotation_pnp.py:220`：逐个 PnP 方法。
- `rotation_pnp.py:167-184`：检查正深度和重投影门限。
- `rotation_pnp.py:282`：查看最终候选、误差、confidence。

### 8.4 Iris65 得到 metric 双眼 3D

Stage 2A 的左右 `eye_points_px[index,10]` 是两个虹膜中心像素。先用 K 和畸变参数
反投影为单位射线 `r_L,r_R`。FLAME 左右眼 proxy 的方向经 `R_C_H` 转到相机系，并
把长度替换为配置的 65 mm，得到目标 baseline `b`。求最小二乘系统：

```text
lambda_R * r_R - lambda_L * r_L = b
p_L = lambda_L * r_L
p_R = lambda_R * r_R
```

两个 lambda 必须为正，射线系统 condition 不得超过 250，且
`||(p_R-p_L)-b||` 不得超过 4 mm。

推荐断点：

- `build_iris65_metric_head_pose.py:233-236`：眼 proxy、虹膜像素。
- `modelv1/depth_prior/iris_ipd.py:145-153`：反畸变射线与 metric baseline。
- `iris_ipd.py:155-175`：lambda、condition、metric 点和 residual。

### 8.5 从虹膜中点得到头原点 t_C_H

FLAME 68 没有真正的虹膜顶点，所以每只眼使用两个眼角的中点作为 head-local eye
proxy。用它们的 FLAME 距离估计局部到 mm 的比例：

```text
s = 65 / ||right_proxy_H - left_proxy_H||
eye_proxy_mid_H_mm = s * (left_proxy_H + right_proxy_H) / 2
t_C_H = iris_mid_C_mm - R_C_H @ eye_proxy_mid_H_mm
```

代码在 `modelv1/preprocessing/table7_geometry.py:65-125`；最终要求
`t_C_H.z > 0`。调试时观察：

- `landmarks.shape == (68,3)`
- `rotation_c_h.T @ rotation_c_h ≈ I`
- `det(rotation_c_h) ≈ 1`
- `left_iris_xy/right_iris_xy.shape == (2,)`
- `left/right iris z > 0`
- `iris.baseline_residual_mm <= 4`
- `pose.head_origin_camera_mm[2] > 0`

## 9. Stage 3：虚拟相机与 table_frame7_n

### 9.1 输入、筛选与输出

代码范围：`scripts/build_virtual_camera_table7.py:155-324`。

必需输入：Stage 1 CSV、Stage 2B CSV和原图。Stage 3 以 `sample_id` 连接
Stage 1/2，只处理 `metric_pose_status=success` 的行。因此 Stage 2B
失败样本不会进入最终训练 CSV，但会进入 Stage 3 report 的 `failures`。

Stage 3 只生成 `160×160` 虚拟人脸图，不生成虚拟左右眼图；训练所需左右眼仍由
Dataset 根据 `source_image_path` 解析 legacy eye crop。

### 9.2 构造虚拟相机

`build_virtual_camera_transform()` 使用 Stage 2B 的
`head_origin_camera_mm` 作为中心：

1. `z_N` 指向真实相机原点到头原点的方向。
2. 把头部局部 X 轴与 `z_N` 叉乘，构造正交的 `y_N`、`x_N`。
3. 将三条 N 轴按行堆叠成 `R_N_C`。
4. `distance_scale = 600 / ||t_C_H||`。
5. 构造 `H = K_N @ diag(1,1,distance_scale) @ R_N_C @ inv(K_C)`。
6. 先用真实相机畸变参数反畸变整图，再用 H warp 为 `160×160`。

推荐断点：

- `build_virtual_camera_table7.py:223`：逐样本入口。
- `:238`：进入 transform 构造。
- `modelv1/data/virtual_camera.py:337-371`：中心、N 轴、scale、homography。
- `build_virtual_camera_table7.py:279`：得到 `face_image` 后检查 shape、黑边与脸居中。

### 9.3 生成 7 维字段

`build_virtual_camera_table7.py:283` 把 Stage 1 的 `e_u_C,e_v_C,d_C_mm` 和刚生成的
`R_N_C` 交给 `build_table_frame7_n()`：

```text
table7[:3] = normalize(R_N_C @ normalize(e_u_C))
table7[3:6] = normalize(R_N_C @ normalize(e_v_C))
table7[6] = d_C_mm / 1000
```

`modelv1/preprocessing/table7_geometry.py:129-158` 会拒绝非有限、非单位可归一化、
不正交或错误 shape。推荐在 Stage 3 `:283` 与 `:303` 两处查看：

```text
table7.shape == (7,)
||table7[:3]|| ≈ 1
||table7[3:6]|| ≈ 1
dot(table7[:3], table7[3:6]) ≈ 0
table7[6] == table_distance_camera_mm / 1000
face_image.shape == (160,160,3)
```

最终 CSV 只有 16 列：标识/路径、`table_frame7_n_0..6` 和
`uv_gt_u_mm,uv_gt_v_mm`。模型直接需要的核心产物是虚拟 face、7 维桌面描述与 2 维
标签；Stage 1/2 的详细位姿、Eye15、质量分数与报告属于可审计的辅助数据。

## 10. 推荐的完整 Debug 顺序

### 第一次：确认输入与坐标变换

1. 打开调试配置，确认 dataset 11、source root、独立 debug output root。
2. 启动 `Table7 01`，在 Stage 1 `:277` 固定第一条样本。
3. 单步进入 `_camera_pose()`，确认旋转互逆、平移单位为 mm。
4. 到 `:313` 检查桌面距离恒等式，到 `:333` 检查 UV。
5. 运行完成后先看 Stage 1 report，再看 CSV 第一行。

### 第二次：确认图像观测与左右眼语义

1. 启动 `Table7 02A`。
2. 在 `:129` 检查输入 RGB 尺寸应为 `1080×1920×3`。
3. 在 `:133-139` 重点确认 subject-left 对应图像右侧。
4. 进入 `fit_eye_geometry()`，核对 index 10 位于四个 iris rim 中心附近。
5. 写 NPZ 前确认 3 条样本的 detection/valid mask。

### 第三次：分别验证旋转与 metric 深度

1. 启动 `Table7 02B`。
2. 先走完固定 shape 标定：看入选 frame_idx、`[1,3,224,224]` tensor 和
   `fixed_shape[100]`。
3. 对第一条正式样本停在 `extractor.extract()` 后，确认 68 点为 head-local。
4. 单步 PnP，明确只拿 `rotation_c_h`，不要使用内部 `tvec`。
5. 单步 Iris65，观察两条射线、condition、lambda、IPD residual。
6. 在 `build_metric_head_pose()` 后验证公式重新计算值与 `origin` 一致。

### 第四次：确认最终图像与 7 维字段

1. 启动 `Table7 03`。
2. 在虚拟相机构造中检查 `R_N_C` 正交、中心 z 为正、distance scale 合理。
3. 打开生成的 160×160 PNG，确认脸居中、朝向合理、黑边不过多。
4. 在 `table7` 生成后验证两条轴单位正交和第 7 维单位转换。
5. 对照 Stage 3 report，确保 success 数与 Stage 2B success 数一致。

### 第五次：自动化回归

- 运行 `Table7 05: 纯几何单元测试`。
- 对正式产物运行 `Table7 04: 校验正式 Stage2/3 数学契约`。
- 正式全量重建前，把正式配置的 `execution.dry_run` 临时设为 true 检查命令和路径；
  确认后恢复 false。正式配置当前 `overwrite: true`，运行会覆盖同名正式产物。

## 11. 常见问题与定位方法

| 现象 | 首先检查 | 常见原因 |
|---|---|---|
| Stage 1 找不到日志 | `dataset_dir`, `glob` 结果 | source root/数据集 ID 错，或同目录有多份日志 |
| Stage 1 大量 missing row | report `skipped` | 日志图名与 InsightFace CSV 图名不一致 |
| 相机/桌面距离校验失败 | `translation_scale`, hand-eye 外参 | m/mm 混用或刚体到光学外参不匹配 |
| MediaPipe 无脸 | 原图 RGB、分辨率、bbox | 图像损坏、脸太小或 MediaPipe 版本不兼容 |
| Eye15 invalid | invalid reason、10 维 quality | 左右眼 bbox 错、点出 crop、闭眼、虹膜拟合差 |
| DECA checkpoint missing | `cfg.pretrained_modelpath` | `DECA-master/data/deca_model.tar` 不存在 |
| DECA shape 不稳定 | calibration rows、frame_idx | 标定区间太短、跨人、遮挡帧过多 |
| PnP no solution | 8 组点、重投影误差 | 2D/3D 语义错位、左右交换、畸变/K 错 |
| Iris ray ill-conditioned | 两虹膜像素间距、condition | 双眼像素过近、检测错误或相机内参错误 |
| IPD residual > 4 mm | `baseline`, `R_C_H`, 虹膜点 | 虹膜定位、头旋转或固定 shape 不一致 |
| Stage 3 metric_pose_failed | 对应 Stage 2B `reason` | 这是上游质量门传播，不是 warp 自身错误 |
| 虚拟脸偏移/黑边大 | `head_origin`, `R_N_C`, H | metric 中心错误、图像尺寸/标定不匹配 |
| table7 轴不正交 | Stage 1 U/V 与 `R_N_C` | 坐标系方向写反或输入字段被污染 |

不要为了提高成功率直接放宽 IPD/PnP 阈值。应先可视化对应原图与 MediaPipe 点，并检查
左右语义、K/畸变、固定 shape 和 PnP 重投影；阈值是最后才调整的策略参数。

## 12. 当前 Dataset 11 的基线结果

现有正式产物已经过只读一致性校验：

- Stage 1：输入 583，写入 583，无跳过。
- Stage 2A：583 次检测成功，583 条双眼都通过 Eye15 门控。
- Stage 2B：578 成功，5 失败。
- 5 条失败全部是 Iris/IPD baseline residual 超过 4 mm，实际约
  4.04–4.57 mm；这是明确的质量筛选，不是文件连接缺失。
- Stage 3：输入 583，成功写 578，5 条失败与 Stage 2B 完全对应。
- `validate_table7_stage23_math.py`：21 项检查全部通过；578 条最终记录的旋转、头原点
  公式、虚拟相机轴、路径、桌面轴、距离和重算 table7 均一致。

这组数字可以作为以后修改 Stage 1–3 后的回归基线。若代码、相机标定、IPD 或固定
shape 区间有意变化，数值可以变化，但每个变化都应由 report 与抽样可视化解释。

## 13. 正式生成命令

确认调试结果后，正式全量执行：

```powershell
D:\software\Anaconda\envs\cross_gaze\python.exe scripts\prepare_modelv1_table7.py `
  --config configs\preprocessing\modelv1_table7_final.yaml
```

只运行某阶段时，直接运行相应脚本；Stage 2 仍按 2A 后 2B 的顺序：

```powershell
D:\software\Anaconda\envs\cross_gaze\python.exe scripts\build_modelv1_table7_base.py --config configs\preprocessing\modelv1_table7_final.yaml
D:\software\Anaconda\envs\cross_gaze\python.exe scripts\extract_mediapipe_table7_observations.py --config configs\preprocessing\modelv1_table7_final.yaml
D:\software\Anaconda\envs\cross_gaze\python.exe scripts\build_iris65_metric_head_pose.py --config configs\preprocessing\modelv1_table7_final.yaml
D:\software\Anaconda\envs\cross_gaze\python.exe scripts\build_virtual_camera_table7.py --config configs\preprocessing\modelv1_table7_final.yaml
```

换数据集时至少重新检查：数据集 ID、source/output root、相机标定、实际 IPD、固定
shape 标定帧范围，以及 InsightFace bbox/原图是否属于同一版本。
