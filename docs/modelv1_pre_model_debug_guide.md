# ModelV1 进入模型前 Debug 说明书

> 核对日期：2026-09-04  
> 适用入口：`scripts/train_modelv1.py` 与当前 Direct-UV + Table7 Stage3 配置  
> 范围：从训练入口开始，到 `ModelV1.forward(device_batch, ...)` 被调用为止

## 1. 当前真实主链路

```text
launch.json / 命令行参数
  -> 读取实验 YAML，合并 defaults_direct_uv.yaml
  -> 校验配置契约
  -> 合并 Dataset13 + Dataset11 的 Stage3 CSV
  -> 加载两个 MediaPipe 眼部伪标签 NPZ
  -> 显式划分 train=13、val=11
  -> 仅使用 Dataset13 的 UV 拟合 mean/std
  -> 创建 train_loader 和 val_loader
  -> Dataset.__getitem__ 读取单个样本
       -> 虚拟相机人脸图
       -> legacy 左右眼裁剪图
       -> 15 点眼部伪标签
       -> 训练集外观增强
       -> 人脸/眼图归一化
       -> 读取预计算 TableFrame7_N
       -> 构造并标准化 table-local UV 标签
  -> DataLoader 拼成 batch
  -> Tensor 搬到 CPU/CUDA
  -> ModelV1.forward(device_batch, ...)
```

“进入模型前”的边界是：

- 最后一个模型外断点：[`scripts/train_modelv1.py`](../scripts/train_modelv1.py#L1993) 的 `model_kwargs`；
- 真正进入模型：[`scripts/train_modelv1.py`](../scripts/train_modelv1.py#L1996) 的 `model_output = model(...)`；
- 模型读取 batch 的第一行：[`modelv1/model.py`](../modelv1/model.py#L2027)。

当前 `ModelV1.forward()` 实际强制取出四项：

```python
face = batch["face"]                     # [B,3,160,160]
left_eye = batch["left_eye"]             # [B,3,56,90]
right_eye = batch["right_eye"]           # [B,3,56,90]
table_frame7_n = batch["table_frame7_n"] # [B,7]
```

随后 `scene_vec = table_frame7_n`。Dataset 仍返回 `crop_cam_vec`，但当前前向入口不使用它。

## 2. 当前实际输入

### 2.1 配置与数据划分

默认配置：
[`configs/modelv1/train_dataset11_val_dataset3_direct_uv_virtual_camera_film_v2.yaml`](../configs/modelv1/train_dataset11_val_dataset3_direct_uv_virtual_camera_film_v2.yaml)

它通过 `defaults: defaults_direct_uv.yaml` 继承
[`configs/modelv1/defaults_direct_uv.yaml`](../configs/modelv1/defaults_direct_uv.yaml)：

| 配置项 | 有效值 | 含义 |
|---|---|---|
| `split_mode` | `explicit_datasets` | 显式且互斥地划分训练/验证集 |
| `direct_uv_target_frame` | `table_local` | 监督目标为桌面局部 UV，单位毫米 |
| `require_eye_geometry_pseudo_labels` | `true` | 最终样本必须有眼部伪标签 |
| `skip_invalid_virtual_camera_samples` | `true` | 虚拟相机路线的有效样本约定 |

配置文件名保留了早期的 `dataset11_val_dataset3` 字样，但当前 YAML 内容实际是
`train_datasets: ["13"]`、`val_datasets: ["11"]`。Debug 时以内容为准。

### 2.2 Stage3 CSV

- Dataset 13：`data/preprocessed_table7/dataset13/stage3_virtual_table7/training_table7_dataset13.csv`，887 条；
- Dataset 11：`data/preprocessed_table7/dataset11/stage3_virtual_table7/training_table7_dataset11.csv`，578 条。

必需字段：

```text
sample_id, dataset, image_name, source_image_path, virtual_face_path,
uv_gt_u_mm, uv_gt_v_mm, table_frame7_n_0 ... table_frame7_n_6
```

### 2.3 图像真实来源

当前 YAML 的 `image_source: legacy` 容易产生误解。进入
`precomputed_table_frame7=True` 分支后，代码实际选择：

| 输入 | 路径来源 | 单样本形状 |
|---|---|---|
| 人脸 | CSV 的 `virtual_face_path` | `[3,160,160]` |
| 左眼 | `source_image_path` 映射到 `insightface_eyes/left_eye` | `[3,56,90]` |
| 右眼 | `source_image_path` 映射到 `insightface_eyes/right_eye` | `[3,56,90]` |

即：人脸是 Stage3 虚拟相机图，眼睛是同一样本的 legacy 眼部裁剪。

### 2.4 眼部伪标签 NPZ

当前加载：

- `data/preprocessed_v2_160/mediapipe_eye_geometry_v1_dataset13.npz`；
- `data/preprocessed_v2_160/mediapipe_eye_geometry_v1_dataset11.npz`。

每个样本按 anatomical `[left,right]` 顺序保存：

| 内容 | 形状 |
|---|---|
| crop-normalized 15 点坐标 | `[2,15,2]` |
| eye-local 15 点坐标 | `[2,15,2]` |
| 点有效性 | `[2,15]` |
| 每眼几何/质量特征 | `[2,10]` |
| 每眼有效性 | `[2]` |

## 3. VS Code 启动方式

项目的 [`.vscode/launch.json`](../.vscode/launch.json) 提供三个相关入口。

### `ModelV1 01: 训练前数据链路（推荐）`

运行 [`scripts/check_dataloader.py`](../scripts/check_dataloader.py)：

- 使用与正式训练相同的 YAML、工件解析和 Dataset；
- 强制 `num_workers=0`，`__getitem__` 在当前 Debug 进程执行；
- batch size 最大为 4；
- 不初始化模型，不下载模型权重；
- 只取训练和验证各一个 batch；
- 临时合并 CSV，退出时清理临时目录；
- `stopOnEntry=true`，可从入口开始按 F10/F11。

这是排查配置、路径、样本、增强、标签、归一化和 batch 的首选入口。

### `ModelV1 02: 完整训练 Dry Run（CUDA）`

从正式 [`scripts/train_modelv1.py`](../scripts/train_modelv1.py) 运行：

- `--dry-run` 将 epoch 改为 1，关闭 W&B 和 checkpoint；
- 仍创建 `outputs/dry_run_*`，保存解析配置与训练日志；
- 使用 YAML 中的 CUDA/BF16、batch size 64、2 个 persistent workers；
- `subProcess=true` 可跟踪 Windows DataLoader 子进程。

适合确认“真实入口 -> batch -> 模型 -> loss -> backward”。`--dry-run` 不会减少
一个 epoch 内的 batch 数，所以单看 Dataset 时应使用入口 01。

### `ModelV1 03: 完整训练 Dry Run（CPU）`

额外传入 `--device cpu`，用于区分数据/模型错误和 CUDA/BF16 错误。模型较大，CPU
完整 dry run 会明显更慢。

### Python 解释器

已验证可运行项目的解释器是：

```text
D:\software\Anaconda\envs\cross_gaze\python.exe
```

该环境包含 PyTorch 1.13.0 且可见 CUDA。VS Code 先执行 `Python: Select Interpreter`
并选择 `cross_gaze`。若出现 `No module named 'torch'`，说明选错了解释器，不是代码错误。

## 4. 四轮断点流程

### 第一轮：配置与数据工件

| 顺序 | 位置 | 输入 | 工作 | 输出/预期 |
|---:|---|---|---|---|
| 1 | `train_modelv1.py:2891` 或 `check_dataloader.py:105` | launch args | 解析命令行 | `args.config` 指向目标 YAML |
| 2 | `train_modelv1.py:339` | YAML 路径 | 读取实验配置 | 原始 `dict` |
| 3 | `train_modelv1.py:352` | defaults + override | 递归合并 | 完整 `config` |
| 4 | `train_modelv1.py:357` | 完整 config | 校验训练契约 | 无异常返回 |
| 5 | `train_modelv1.py:980` | 两个 CSV 路径、run dir | 存在性/表头/重复 ID 审计并合并 | `dataset_csv_path` |
| 6 | `train_modelv1.py:1008` | 合并 CSV | 读取记录 | 总行数 1465 |

### 第二轮：DataLoader 构建

| 顺序 | 位置 | 观察重点 |
|---:|---|---|
| 1 | `dataset.py:931` | `build_modelv1_dataloaders` 的有效参数 |
| 2 | `eye_geometry_pseudo_labels.py:99` | NPZ schema、形状、重复 sample ID |
| 3 | `dataset.py:1090` | `13/11` 规范化为完整数据集名称 |
| 4 | `dataset.py:1103` | 训练 Dataset，有增强 |
| 5 | `dataset.py:1134` | 验证 Dataset，无增强 |
| 6 | `dataset.py:1164` | 只用训练 UV 拟合 normalizer |
| 7 | `dataset.py:1235` | train loader，`shuffle=True` |
| 8 | `dataset.py:1244` | val loader，`shuffle=False` |

输出应为 `train_loader`、`val_loader`，以及挂在 Dataset 上的 UV normalizer。

### 第三轮：一个样本如何形成

使用入口 01：

| 顺序 | 位置 | 输入 | 输出 |
|---:|---|---|---|
| 1 | `dataset.py:577` | `index` | 当前 CSV `row` |
| 2 | `dataset.py:584` | Stage3 row | 三张图像路径 |
| 3 | `dataset.py:597` | 路径、目标尺寸 | `[0,1]` CHW float32 图像 |
| 4 | `dataset.py:605` | `sample_id` | 眼部伪标签 record |
| 5 | `dataset.py:618` | 图像、点、valid mask | 训练增强后的图像和 mask |
| 6 | `dataset.py:641` | `[0,1]` 图像 | FaceNet/ImageNet 归一化图像 |
| 7 | `dataset.py:647` | CSV UV 列 | `uv_gt_table_mm`、`uv_gt` |
| 8 | `dataset.py:649` | Stage3 的 7 列 | 正交性检查后的 `table_frame7_n` |
| 9 | `dataset.py:681` | 全部处理结果 | 单样本 `item` |
| 10 | `dataset.py:849` | 完整 item | 返回 DataLoader |

### 第四轮：batch 到模型边界

| 顺序 | 位置 | 工作 |
|---:|---|---|
| 1 | `train_modelv1.py:1978` | 默认 collate 已生成 CPU batch |
| 2 | `train_modelv1.py:1979` | Tensor 搬到 device |
| 3 | `train_modelv1.py:1980` | 从 `uv_gt.shape[0]` 得到真实 batch size |
| 4 | `train_modelv1.py:1993` | 决定模型返回附加结果 |
| 5 | `train_modelv1.py:1996` | 跨过此行进入模型 |
| 6 | `model.py:2027` | 模型读取四个必需输入 |

## 5. 单样本与 batch 的完整输入输出

在 `dataset.py:849` 返回前，当前 Stage3 单样本的主要字段为：

| key | 单样本形状/类型 | 工作与去向 |
|---|---|---|
| `face` | `[3,160,160] float32` | FaceNet 标准化的虚拟相机人脸图；模型输入 |
| `left_eye` | `[3,56,90] float32` | ImageNet 标准化的 legacy 左眼图；模型输入 |
| `right_eye` | `[3,56,90] float32` | ImageNet 标准化的 legacy 右眼图；模型输入 |
| `scene_vec` | `[7] float32` | 当前等于 `table_frame7_n` |
| `table_frame7_n` | `[7] float32` | `[e1_N,e2_N,d_m]`；模型场景输入和 FiLM 条件 |
| `crop_cam_vec` | `[36] float32` 全零 | Stage3 兼容字段；当前模型不使用 |
| `uv_gt` | `[2] float32` | table-local UV，毫米；用于指标 |
| `uv_gt_table_mm` | `[2] float32` | 原始 table-local UV 审计副本 |
| `uv_target` | `[2] float32` | 训练集统计量 z-score 后的监督目标 |
| `gaze_target_w` | `[3] float32` 全零 | Stage3 兼容字段；当前 Direct-UV 不使用 |
| `table_origin_w` | `[3] float32` 全零 | Stage3 兼容字段；当前 Direct-UV 不使用 |
| `eye_pseudo_landmarks_xy` | `[2,15,2]` | crop-normalized 伪关键点；辅助监督 |
| `eye_pseudo_landmarks_local` | `[2,15,2]` | eye-local 点；验证/辅助几何 |
| `eye_pseudo_landmark_base_valid_mask` | `[2,15]` | 增强前有效性 |
| `eye_pseudo_landmark_valid_mask` | `[2,15]` | 增强后有效性 |
| `eye_pseudo_occluded_mask` | `[2,15]` | 被本次增强遮挡的点 |
| `eye_pseudo_geometry_features` | `[2,10]` | 每眼几何与质量特征 |
| `eye_pseudo_eye_valid_mask` | `[2]` | 每眼是否有效 |
| `eye_pseudo_detection_mask` | `[1]` | MediaPipe 检测是否成功 |
| `eye_pseudo_sample_valid_mask` | `[1]` | 整个样本是否有效 |
| `eye_pseudo_sample_quality` | `[1]` | 样本质量 |
| `eye_pseudo_eye_quality` | `[2]` | 每眼质量 |
| `eye_pseudo_effective_quality` | `[2]` | 质量 × 眼有效 × 可见比例 |
| `sample_id` / `dataset` / `image_name` | `str` | 对齐和追踪元数据 |
| `paths` | `dict[str,str]` | 实际读取的图像路径 |

默认 collate 后增加 batch 维：

```text
face                               [B,3,160,160]
left_eye / right_eye               [B,3,56,90]
scene_vec / table_frame7_n         [B,7]
crop_cam_vec                       [B,36]
uv_gt / uv_gt_table_mm / uv_target [B,2]
eye_pseudo_landmarks_xy            [B,2,15,2]
eye_pseudo_landmark_valid_mask     [B,2,15]
eye_pseudo_geometry_features       [B,2,10]
eye_pseudo_effective_quality       [B,2]
```

字符串字段变成长度为 `B` 的列表；`paths` 递归 collate 成嵌套字典，叶子是长度为
`B` 的路径列表。

## 6. 逐函数解释

### 6.1 `parse_args()`：命令行到 Namespace

位置：[`scripts/train_modelv1.py`](../scripts/train_modelv1.py#L303)

输入是 launch 的 `args`，输出类似：

```python
Namespace(config=Path(...), resume=None, device=None, dry_run=True)
```

`--dry-run` 只把 epoch 改为 1，并关闭 W&B 和 checkpoint；不会减少每个 epoch 的 batch 数。

### 6.2 `load_config()`：YAML 到最终配置字典

位置：[`scripts/train_modelv1.py`](../scripts/train_modelv1.py#L339)

1. 读取实验 YAML；
2. 弹出顶层 `defaults`；
3. 以实验 YAML 为 override 递归合并默认 YAML；
4. 调用 `validate_config()`；
5. 返回最终 `dict`。

字典递归合并，标量和列表整体覆盖。默认文件不允许再包含另一个 `defaults`。

### 6.3 `validate_config()`：在读取样本前失败

位置：[`scripts/train_modelv1.py`](../scripts/train_modelv1.py#L357)

这里验证 section、batch/worker、AMP、模型开关、伪标签、Stage3、目标坐标系、
TableFrame FiLM、优化器、EMA 和 scheduler 契约。配置错误应在此解决，不要绕过。

当前 Stage3 路线要求：

- `prediction_mode=direct_uv`；
- `use_eye_geometry=false`；
- `use_crop_cam=false`；
- `scene_representation=table_frame7`；
- `direct_uv_target_frame=table_local`；
- 不再传旧 virtual-camera manifest。

### 6.4 `resolve_table7_stage3_artifacts()`：CSV 审计与合并

位置：[`scripts/train_modelv1.py`](../scripts/train_modelv1.py#L980)

输入是两个 CSV 路径和 `run_dir`。它会检查文件存在、表头一致和 `sample_id` 不重复，
再合并 CSV，并写 `sources.json` 记录来源和总样本数。

正式训练输出在 `run_dir/resolved_table7_stage3/`。检查脚本使用系统临时目录并在结束时清理。

### 6.5 `EyeGeometryPseudoLabelTable.load()`：NPZ 到查找表

位置：[`modelv1/data/eye_geometry_pseudo_labels.py`](../modelv1/data/eye_geometry_pseudo_labels.py#L99)

验证 schema version、点名/特征名顺序、数组形状、有限值和重复 sample ID，然后构造：

```python
records: dict[sample_id, EyeGeometryPseudoLabelRecord]
```

Dataset `lookup()` 会 clone Tensor，避免增强原地污染全局记录。

### 6.6 `ModelV1Dataset.__init__()`：过滤和对齐

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L183)

输入是合并 CSV、当前 split 名称、增强、伪标签表和模型数据开关。它会读取 CSV、过滤
split、验证 Stage3 字段，并确认最终每个 `sample_id` 都在伪标签表中。

训练集和验证集是独立实例：训练实例有图像增强，验证实例没有增强。

### 6.7 `UVTargetNormalizer.fit()`：只拟合训练集

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L1164)

代码遍历训练 Dataset 的 `raw_uv_target()`，只使用 Dataset 13 的 UV 拟合逐轴均值和
标准差，再把同一个 normalizer 挂到训练和验证 Dataset，避免验证集信息泄漏。

本机 smoke test 得到：

```text
mean_mm = [-256.82275390625, 63.48074722290039]
std_mm  = [94.47730255126953, 133.45648193359375]
```

验证 round-trip：

```python
normalizer.denormalize(batch["uv_target"]) - batch["uv_gt"]
```

最大绝对误差应接近 0；不同随机 batch 会有 float32 舍入差异，建议判定为 `<=1e-5`。

### 6.8 `_load_image()`：路径到 float32 CHW

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L851)

PIL 打开 -> RGB -> 必要时 resize -> HWC 转 CHW -> `uint8/255`。刚返回时值域在
`[0,1]`，此处尚未做 ImageNet/FaceNet 归一化。

### 6.9 训练增强：图像和 valid mask 同步

位置：[`modelv1/data/augmentation.py`](../modelv1/data/augmentation.py#L229)

`apply_with_keypoints()` 做成对外观增强。当前策略不移动点坐标；局部遮挡可把有效点
变为无效点。应始终满足：

```python
torch.all(augmented_valid <= base_valid)
```

增强只能保持或减少可见性，不能创建有效关键点。

### 6.10 图像归一化

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L867)

- 人脸：`mean=std=(0.5,0.5,0.5)`，把 `[0,1]` 映射到约 `[-1,1]`；
- 双眼：ImageNet mean/std，匹配预训练 ResNet18；
- 输出形状不变。

### 6.11 `precomputed_table_frame7_tensor()`

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L1436)

输入为 `[e1_N.x,e1_N.y,e1_N.z,e2_N.x,e2_N.y,e2_N.z,d_m]`。函数检查有限值、
基向量单位长度及正交性，输出 `[7] float32`。Stage3 已完成虚拟相机坐标变换，
这里不会再加载旧 manifest，也不会重复旋转。

### 6.12 `raw_uv_target()` 与 `uv_target`

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L881)

当前 `direct_uv_target_frame=table_local`：

```python
uv_gt = [uv_gt_u_mm, uv_gt_v_mm]
uv_target = (uv_gt - mean_mm) / std_mm
```

`uv_gt` 用于毫米指标，`uv_target` 用于回归损失；两者都不是视觉输入。

### 6.13 DataLoader 默认 collate

DataLoader 对 Tensor 执行 stack，对字符串形成 list，对嵌套 dict 递归 collate。正式训练
为 `batch_size=64, num_workers=2, persistent_workers=true`；入口 01 使用
`batch_size<=4, num_workers=0, pin_memory=false`。

### 6.14 `move_batch_to_device()`：最后的模型外步骤

位置：[`scripts/train_modelv1.py`](../scripts/train_modelv1.py#L1825)

只对 Tensor 执行 `value.to(device, non_blocking=True)`；字符串和路径字典保持 Python
对象。输出 `device_batch` 的 key 不变，Tensor device 改变。

## 7. Debug Console 推荐表达式

### 配置与路径

```python
args
config["data"]
config["model"]
uses_table7_stage3
dataset_csv_path
eye_geometry_pseudo_label_paths
```

### Dataset 与 split

```python
len(train_loader.dataset)
len(val_loader.dataset)
train_loader.dataset.rows[0]["sample_id"]
train_loader.dataset.rows[0]["dataset"]
train_loader.dataset.target_normalizer.mean_mm
train_loader.dataset.target_normalizer.std_mm
```

预期样本数为 `887`、`578`，训练行属于 `dataset_dual_rigid_body_13`。

### 单样本路径和张量

```python
row["sample_id"]
face_path, left_eye_path, right_eye_path
tuple(face.shape), face.dtype, float(face.min()), float(face.max())
tuple(left_eye.shape), left_eye.dtype
tuple(pseudo_points.shape), tuple(pseudo_augmented_valid.shape)
item.keys()
```

### TableFrame7 几何自检

```python
torch.linalg.vector_norm(table_frame7_n[:3])
torch.linalg.vector_norm(table_frame7_n[3:6])
torch.dot(table_frame7_n[:3], table_frame7_n[3:6])
table_frame7_n[6]
```

前两项应接近 1，点积接近 0；最后一项是米制有符号距离。

### 整个 device batch

```python
{
    k: (tuple(v.shape), str(v.dtype), str(v.device), bool(torch.isfinite(v).all()))
    for k, v in device_batch.items()
    if torch.is_tensor(v)
}
```

进入模型前，所有模型输入和监督 Tensor 都应是有限值。

## 8. 已验证的 smoke test 基线

使用 `cross_gaze` 环境运行当前检查入口，结果为：

```text
train samples: 887
val samples: 578
train batches: 222
val batches: 145
face: (4, 3, 160, 160)
left_eye: (4, 3, 56, 90)
right_eye: (4, 3, 56, 90)
crop_cam_vec: (4, 36)
scene_vec: (4, 7)
uv_gt: (4, 2)
uv_target: (4, 2)
eye_pseudo_landmarks_xy: (4, 2, 15, 2)
eye_pseudo_landmark_valid_mask: (4, 2, 15)
eye_pseudo_geometry_features: (4, 2, 10)
eye_pseudo_effective_quality: (4, 2)
uv round-trip max error: 约 0 到 3.1e-6（应 <= 1e-5）
```

训练 loader 使用 `shuffle=True`，所以训练 sample ID 每次可以不同；验证 loader 第一批顺序应稳定。

## 9. 常见问题定位

### `No module named torch`

在 VS Code 选择 `cross_gaze` 解释器，不要因此修改项目 import。

### 断点进不了 `__getitem__`

正式配置有两个 worker，`__getitem__` 在子进程执行。切到入口 01；它固定
`num_workers=0`，是最稳定的 Dataset 单步方式。

### `Missing Stage3 TableFrame7 CSV`

停在 `resolve_table7_stage3_artifacts()`，检查相对项目根目录解析后的两个路径。

### `Cannot merge ... different columns`

两个 Stage3 CSV 必须有完全一致且顺序一致的表头。先修正数据工件，不要跳过校验。

### `Duplicate sample_id`

两个 CSV 有样本交叉或单个 CSV 内重复。训练前必须修正数据工件。

### `Missing image`

在 `dataset.py:584-599` 检查三条最终路径。Stage3 左右眼不是 CSV 的 `virtual_*_eye_path`，
而是从 `source_image_path` 推导出的 legacy 路径。

### 伪标签 sample ID 缺失

对比 CSV 的 `row["sample_id"]` 与 NPZ 加载出的 `records`。当前配置要求所有过滤后的
训练/验证样本都有记录。

### UV 数值异常

依次观察 `uv_gt_table_mm`、`uv_gt`、`uv_target`，再做 normalizer round-trip。不要把
标准化值当作毫米值。

### `TableFrame7_N must be finite and orthonormal`

检查 7 个 CSV 字段是否错列、NaN 或来自错误坐标系。不要在 Dataset 中关闭该校验。

### CUDA/BF16 启动失败

入口 01 不依赖 CUDA。先确认数据链路，再用 CUDA dry run；也可用 CPU dry run 区分
数据/模型错误与 CUDA/AMP 错误。

## 10. 一次完整 Debug 的操作顺序

1. VS Code 选择 `cross_gaze` 解释器。
2. 选择 `ModelV1 01: 训练前数据链路（推荐）`。
3. 第一遍只看 YAML、defaults、两个 CSV 与两个 NPZ。
4. 第二遍检查 split、887/578 样本数和 UV normalizer。
5. 第三遍进入一个训练样本，检查图像路径、增强前后 mask 和单样本 item。
6. 在 `check_dataloader.py:249` 检查 collate 后的 batch。
7. 确认输出与第 8 节基线一致。
8. 切换到入口 02，在 `train_modelv1.py:1978、1979、1996` 下断点。
9. 在 1996 行跨入模型，检查 `model.py:2027-2031` 的四个必需 Tensor。
10. 数据错误回入口 01；模型、loss、backward 错误留在完整 dry run 中处理。

模型内部从图像到 UV 的继续调试顺序见
[`docs/modelv1_training_forward_to_uv.md`](modelv1_training_forward_to_uv.md)。
