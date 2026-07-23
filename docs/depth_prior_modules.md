# 离线深度先验模块说明

本文说明 `modelv1.depth_prior` 中的两个离线几何模块。它们负责在
模型训练之前生成几何深度先验；当前不包含训练期深度预测头，也不直接
生成经过校准的预测不确定性。

## 模块范围与职责

`modelv1.depth_prior.deca_flame.DecaFlameExtractor` 是对仓库内官方 DECA
实现的轻量包装，实际代码位于 `DECA-master/decalib`。其中的 FLAME Layer
由官方 DECA 提供，本项目不会重新实现它。

`modelv1.depth_prior.pnp.solve_pnp_face_depth` 是项目侧的相机几何模块。
它使用 OpenCV 的 `solvePnP`、DECA-FLAME 的三维几何以及真实尺度信息，
将 FLAME 局部坐标转换为相机坐标系下的毫米单位坐标。

运行 DECA 前，Python 环境还需要具备官方 DECA 的依赖，包括 `yacs`、
PyTorch，以及本地 DECA 项目依赖列表中的其他包。

## 1. DECA-FLAME 提取模块

### 输入

- 对齐后的人脸 RGB 图像批次，形状为 `[B, 3, 224, 224]`。
- `float32` 类型，数值范围为 `[0, 1]`；裁剪、颜色通道和归一化方式应与
  官方 DECA 预处理保持一致。
- 本地 DECA 代码目录及其预训练权重。

### 内部过程

1. 官方 DECA 的 `ResnetEncoder` 从人脸图像预测参数向量。
2. 参数向量被拆分为形状、纹理、表情、姿态、相机和光照参数。
3. 官方 `FLAME` Layer 接收形状、表情和姿态参数，输出人脸网格与关键点。
4. 保留当前 CrossGaze 基线的两个网格顶点代理：`3933` 与 `3930`，仅用于
   兼容性诊断；它们不再用于 PnP 的双眼参考点输出。

### 输出

输出为 `DecaFlameOutput`，其中所有张量均已从计算图分离并迁移到 CPU：

- `vertices`：通常为 `[B, 5023, 3]`，完整 FLAME 人脸网格顶点。
- `landmarks3d`：通常为 `[B, 68, 3]`，68 个三维人脸关键点。
- `landmarks2d`：官方 FLAME Layer 返回的动态二维关键点集合。其坐标值仍是
  FLAME 局部三维坐标，并不是图像像素坐标；需要结合 DECA 的弱透视相机参数
  才能叠加到 `224x224` Face 图上。
- `left_eye_vertex`、`right_eye_vertex`：均为 `[B, 3]`，旧基线的网格顶点代理，
  不应视为已验证的解剖学眼球中心。
- `parameters`：原始 DECA 参数向量，用于调试和追溯。

这些输出都处于 FLAME 局部坐标系，单位仍然是 FLAME unit；它们既不是
毫米，也不是相机坐标。

## 2. solvePnP 人脸深度模块

### 输入

- `image_points_by_label`：检测到的二维像素关键点。默认键为
  `left_eye_outer`、`left_eye_inner`、`right_eye_inner`、
  `right_eye_outer`、`nose_tip`、`mouth_left`、`mouth_right`、`chin`。
- `landmarks3d` 中的一条 `[68, 3]` 数据。
- `PnpCamera`：标定后的相机内参矩阵 `K` 和畸变系数。
- 真实测量的眼角距离，或经独立验证后选定的固定尺度。

### 内部过程

1. 根据 FLAME 的眼角关键点计算内眼角距离和外眼角距离。
2. 用真实测量距离得到 FLAME 到毫米的尺度。当前基线默认的测量值为：
   外眼角距离 `105.0 mm`，内眼角距离 `38.0 mm`。
3. 构建默认的 2D-3D 对应关系，所用 FLAME 68 点索引为
   `36, 39, 42, 45, 30, 48, 54, 8`。
4. 使用 OpenCV 的 `solvePnP` 求解投影模型 `s p = K [R|t] P` 中的旋转
   `R` 和平移 `t`。
5. 计算图像左眼 `FLAME[36, 39]` 与图像右眼 `FLAME[42, 45]` 的内外眼角
   三维中点，并用下式变换到相机坐标系：
   `X_camera_mm = R * (scale_mm_per_flame_unit * X_flame) + t_mm`。
6. 将求得的三维点重新投影到图像，计算拟合质量指标。

### 输出

输出为 `PnpFaceDepthResult`：

- `left_eye_camera_xyz_mm [3]` 和 `right_eye_camera_xyz_mm [3]`：图像左/右眼的
  内外眼角三维中点在相机坐标系中的毫米坐标。字段名为兼容训练数据而保留；它们是
  gaze reference point，不应直接解释为解剖学眼球中心。
- `face_depth_z_mm [1]`：`tvec_mm` 的 z 分量，即 FLAME 人脸局部原点的
  相机深度。
- `rotation_matrix`、`rvec`、`tvec_mm`：用于追溯和后续调试的位姿结果。
- 质量指标：平均与最大重投影误差、PnP 对应点数、内点数、启发式
  `pnp_confidence`、尺度诊断字段和基本深度合理性标记。

`as_record()` 可以将结果展平为标量字段，直接写入 CSV 或 Parquet。

## 两个模块的连接方式

```text
对齐后的人脸图像 --> DecaFlameExtractor
                         |-- landmarks3d [68, 3]
                         |-- vertices [5023, 3]
二维人脸关键点 ----------+
相机内参与畸变参数 -------+--> solve_pnp_face_depth
真实人脸尺度信息 ---------+         |-- left_eye_camera_xyz_mm
                                      |-- right_eye_camera_xyz_mm
                                      |-- face_depth_z_mm
                                      +-- PnP 质量指标
```

完整链路属于离线预处理阶段。训练数据集读取由该过程生成的表格。后续训练模块
可以学习深度残差修正和不确定性，但应将该离线几何结果视为先验，而不是深度真值。

## 最小调用示例

```python
from modelv1.depth_prior import (
    DecaFlameExtractor,
    PnpCamera,
    solve_pnp_face_depth,
)

deca = DecaFlameExtractor()
deca_output = deca.extract(face_batch_rgb_224)

result = solve_pnp_face_depth(
    image_points_by_label=detected_2d_landmarks,
    landmarks3d=deca_output.landmarks3d[0].numpy(),
    vertices=deca_output.vertices[0].numpy(),
    camera=PnpCamera.crossgaze_default(),
)
record = result.as_record()
```

## 重要限制与使用原则

- PnP 的重投影误差不能确定绝对尺度。将所有三维点与 `tvec` 同时按同一比例
  缩放，投影到图像后的结果不变。因此必须使用真实测量的眼距、经验证的固定尺度，
  或其他具有物理单位的参考信息。
- `pnp_confidence` 是基于重投影误差、内点比例和尺度一致性构成的透明质量分数，
  不是经过校准的概率，也不是 `z_sigma_mm`。
- 不应将该先验直接当作深度真值进行监督。更合理的方式是按质量指标过滤或加权样本，
  再基于适当的监督或校准信号学习深度残差和校准后的不确定性。
