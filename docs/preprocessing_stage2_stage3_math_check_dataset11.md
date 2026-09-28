# Stage2 / Stage3 新数学定义审计记录（dataset11）

检查时间：2026-08-28  
数据根目录：`data/preprocessed_table7/dataset11`

这份记录的目的不是强行让新预处理结果等于旧预处理结果，而是确认新版 Stage2/3 在当前定义下没有坐标、尺度、字段转化上的歧义。

## 1. 本次确认的新定义

新版流程采用下面的责任划分：

1. **PnP / FLAME 只负责头部旋转**
   - 输出：`R_C_H`
   - 含义：把头部/FLAME 局部坐标系 `H` 的向量旋转到真实相机坐标系 `C`
   - 注意：不再使用 PnP translation，不再使用 1010 mm/FLAME-unit 的绝对深度逻辑

2. **Iris65 / IPD=65mm 负责唯一绝对尺度来源**
   - 输入：左右 iris 2D 点、真实相机内参/畸变、`R_C_H`、FLAME 左右眼代理点方向
   - 输出：左右 iris proxy 在相机坐标系下的 3D 点
   - 约束：左右 iris proxy 的相对方向由 `R_C_H @ FLAME_eye_baseline_direction` 给出，尺度由 65mm IPD 给出

3. **头部平移由 iris midpoint 反推**
   - 公式：

     ```text
     t_head_C = iris_mid_C - R_C_H @ eye_proxy_mid_H_mm
     ```

   - 其中 `eye_proxy_mid_H_mm` 来自 FLAME 左右眼 canthus midpoint 的中点，并按 IPD 缩放到 mm

4. **虚拟相机归一化使用 `R_C_H + t_head_C`**
   - 虚拟相机中心目标点：`t_head_C`
   - 虚拟相机 `z` 轴：真实相机原点指向 `t_head_C` 的方向
   - 虚拟相机 `x/y` 轴：结合头部 `x` 轴构造稳定正交基
   - `table_frame7_n[0:6]`：真实相机下桌面两个单位轴旋转到虚拟相机坐标系
   - `table_frame7_n[6]`：真实相机中桌面距离，单位从 mm 转为 m

## 2. 代码级审计结论

### Stage2：`scripts/build_iris65_metric_head_pose.py`

关键调用链：

- `solve_head_rotation_pnp(...)`
  - 来自 `modelv1/preprocessing/rotation_pnp.py`
  - 当前实现使用居中 / RMS 归一化后的 FLAME 点做 PnP
  - 返回的 translation 只作为内部求解变量，不进入输出 CSV

- `reconstruct_iris_centres_from_ipd(...)`
  - 来自 `modelv1/depth_prior/iris_ipd.py`
  - 使用真实相机内参和畸变把左右 iris 2D 点变成相机射线
  - 用 `R_C_H @ head_left_to_right_direction_flame * IPD` 作为 metric baseline 约束

- `build_metric_head_pose(...)`
  - 来自 `modelv1/preprocessing/table7_geometry.py`
  - 只用 Iris65 得到的 `iris_mid_C` 和 FLAME 眼部代理点反推 `t_head_C`

审计结论：

- Stage2 中没有 1010 绝对深度逻辑。
- PnP 的作用边界清楚：只提供 `R_C_H`。
- `iris_baseline_residual_mm` 的含义是射线方程的 3D baseline residual：

  ```text
  || (right_iris_C - left_iris_C) - R_C_H @ baseline_H_65mm ||
  ```

  它不是 `| ||right-left|| - 65 |`。这一点已在验证脚本中按源码定义修正。

### Stage3：`scripts/build_virtual_camera_table7.py`

关键调用链：

- `build_virtual_camera_transform(...)`
  - 来自 `modelv1/data/virtual_camera.py`
  - 输入 `R_C_H` 和 `head_origin_camera_mm`
  - 输出 `R_N_C`、homography、归一化相机内参等

- `build_table_frame7_n(...)`
  - 来自 `modelv1/preprocessing/table7_geometry.py`
  - 输入 Stage1 的 `table_u_axis_camera`、`table_v_axis_camera`、`table_distance_camera_mm` 和 Stage3 的 `R_N_C`
  - 输出 7 维 `table_frame7_n`

审计结论：

- Stage3 不再依赖 PnP1010 depth。
- Stage3 的虚拟图像生成只依赖新版 metric head pose。
- 当前配置下 `generate_eye_images=false`，所以最终 CSV 中 `virtual_left_eye_path` / `virtual_right_eye_path` 为空是正确行为，不是缺失。

## 3. 数据级逐样本验证结果

验证脚本：

```bash
python scripts/validate_table7_stage23_math.py --dataset-id 11
```

### 总体数量

| 项目 | 数量 |
|---|---:|
| Stage1 base rows | 583 |
| Stage2 pose rows | 583 |
| Stage2 success | 578 |
| Stage2 failed | 5 |
| Stage3 final rows | 578 |

Stage3 的 578 行 sample_id 与 Stage2 成功样本完全一致，没有多样本、少样本、错样本。

### Stage2 通过项

| 检查项 | 结果 |
|---|---:|
| `R_C_H` 有限、正交、det=1 | 578 / 578 |
| 左右 iris、iris midpoint、head origin 的 z 坐标为正 | 578 / 578 |
| `iris_mid_C = 0.5 * (left_iris_C + right_iris_C)` | 578 / 578 |
| Iris65 residual / condition / baseline length gate 合格 | 578 / 578 |
| `t_head_C + R_C_H @ eye_proxy_mid_H_mm = iris_mid_C` | 578 / 578 |
| PnP reprojection gate 合格 | 578 / 578 |
| PnP confidence 在 `[0, 1]` | 578 / 578 |

### Stage3 通过项

| 检查项 | 结果 |
|---|---:|
| 源图和虚拟 face 图路径有效，face 图为 160x160 | 578 / 578 |
| 眼图路径符合 `generate_eye_images=false` 契约 | 578 / 578 |
| `R_N_C` 有限、正交、det=1 | 578 / 578 |
| 虚拟相机 z 轴与 `head_origin_C` 方向一致 | 578 / 578 |
| `table_frame7_n[0:3]` 与 `[3:6]` 是单位正交轴 | 578 / 578 |
| `table_frame7_n[6] = table_distance_camera_mm / 1000` | 578 / 578 |
| 用 Stage1 + Stage2 + virtual camera 逐行重算 `table_frame7_n` 与 CSV 一致 | 578 / 578 |

最终结果：

```text
checks_total = 21
checks_failed = 0
```

## 4. 相比旧流程，为什么这个结果更可信

在当前目标定义下，新结果应当优于旧流程，核心原因是：

1. 绝对尺度来源唯一：只来自 Iris65 / IPD=65mm，不再混入 PnP1010 深度。
2. PnP 只负责旋转，避免把 FLAME 模型尺度错误传播成绝对深度错误。
3. Stage2 显式保存了 `R_C_H`、左右 iris metric point、`iris_mid_C`、`t_head_C`、FLAME-to-mm scale、Iris residual 和 PnP reprojection 质量。
4. Stage3 的 `table_frame7_n` 可以完全由 Stage1 + Stage2 逐行复算出来，没有隐藏状态。
5. 当前失败样本在 Stage2 被过滤，Stage3 只接收 Stage2 成功样本。

需要保留的边界条件：

- 如果真实相机内参 / 畸变参数错误，Iris65 的绝对深度会系统性偏移。
- 如果用户真实 IPD 明显不是 65mm，绝对尺度会按比例偏移。
- 如果 MediaPipe iris 点检测错误，Stage2 residual / condition / PnP reprojection gate 会过滤一部分坏样本，但不能保证发现所有细小错误。
- 当前验证证明“代码转化和数学自洽”，不能替代真实标尺或已知距离样本的物理标定实验。

## 5. 本次修改

新增：

- `scripts/validate_table7_stage23_math.py`
  - 只读验证脚本
  - 不修改任何生成数据
  - 用于后续每次重跑预处理后检查 Stage2/3 数学自洽性

新增：

- `docs/preprocessing_stage2_stage3_math_check_dataset11.md`
  - 本次 Stage2/3 审计记录

未修改：

- Stage2 生成逻辑
- Stage3 生成逻辑
- 已生成 CSV / JSON 数据
