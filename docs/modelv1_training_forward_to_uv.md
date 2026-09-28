# ModelV1 训练前向链路：从输入图像到 UV 输出

> 整理日期：2026-08-29  
> 分析范围：当前默认训练配置实际启用的前向路径，不包含仅存在于代码中但未启用的可选分支。

## 原始问题

请梳理预训练之后，真正进入训练的模型：从输入图像和相关数据，到输出 UV，这一阶段各个模块都在哪些位置；最主要的函数或者 class 分别有什么作用，属于哪一个模块。请按照训练时经过的顺序，一个函数一个函数列出来。

换种说法：图像经过某个函数时，输入了什么、输出了什么、做了哪些事情，在训练中接下来又进入哪个模块或函数。

---

## 结论：当前真实主链路

```text
虚拟相机人脸图 + legacy 左右眼图
        ↓
Dataset 读取、增强、归一化
        ↓
人脸 InceptionResnetV1
左右眼共享 ResNet18 + FPN
        ↓
概率眼部关键点预测
        ↓
关键点引导的眼特征融合
        ↓
双眼 Query ↔ 人脸 Key/Value 交叉注意力
        ↓
视觉特征 [B,128]
        +
TableFrame7 场景特征 [B,64]
        ↓
拼接 [B,192]
        ↓
TableFrame7 条件 FiLM 融合
        ↓
融合特征 [B,128]
        ↓
Linear UV Head
        ↓
归一化桌面局部坐标 UV [B,2]
```

---

## 1. 当前真正使用的训练配置

默认配置入口：

- [`scripts/train_modelv1.py`](../scripts/train_modelv1.py#L113)
- [`configs/modelv1/train_dataset11_val_dataset3_direct_uv_virtual_camera_film_v2.yaml`](../configs/modelv1/train_dataset11_val_dataset3_direct_uv_virtual_camera_film_v2.yaml)

当前关键配置：

| 配置项 | 当前值 | 含义 |
|---|---:|---|
| `train_datasets` | `["13"]` | Dataset 13 用于训练 |
| `val_datasets` | `["11"]` | Dataset 11 用于验证 |
| `table7_stage3_csv_paths` | Dataset13、11 Stage3 CSV | Iris65 metric head-origin 最终训练数据 |
| `image_source` | `virtual_camera` | 人脸使用虚拟相机归一化图 |
| `prediction_mode` | `direct_uv` | 直接回归 UV，不先预测 gaze ray |
| `direct_uv_target_frame` | `table_local` | UV 位于桌面局部坐标系 |
| `scene_representation` | `table_frame7` | 场景使用7维桌面坐标系表示 |
| `use_face_image` | `true` | 启用人脸图像分支 |
| `freeze_face_image_backbone` | `false` | 人脸骨干参与训练 |
| `use_eye_keypoint_auxiliary` | `true` | 启用概率眼部关键点模块及辅助监督 |
| `use_landmark_guided_eye_fusion` | `true` | 用关键点 token 引导眼部特征 |
| `use_table_frame_film` | `true` | 启用 TableFrame7 条件 FiLM |
| `use_crop_cam` | `false` | 不使用 crop/camera 数值分支 |
| `use_eye_geometry` | `false` | 不使用独立眼部几何数值分支 |
| `initialization_checkpoint` | `null` | 不加载完整 ModelV1 checkpoint |
| `trainable_components` | `all` | 所有当前启用模块都可训练 |

需要注意：配置文件名中写着 `train_dataset11_val_dataset3`，但文件当前实际内容是 Dataset 13 训练、Dataset 11 验证。

---

# 第一部分：数据进入模型前

## 2. `build_modelv1_dataloaders(...)`

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L879)  
所属模块：数据集和 DataLoader 构建模块。

### 输入

- 数据配置；
- Dataset 13、11 的 Stage3 TableFrame7 CSV；
- 眼部伪关键点 `.npz`；
- batch size、数据增强等配置。

### 处理

1. 读取训练和验证样本；
2. 分别创建训练、验证 `ModelV1Dataset`；
3. 只使用训练集真实 UV 拟合 UV 均值和标准差；
4. 训练集和验证集共享训练集得到的 UV normalizer；
5. 创建 PyTorch `DataLoader`；
6. 训练集 `shuffle=True`，验证集 `shuffle=False`。

### 输出

```python
train_loader
val_loader
metadata
```

### 下一步

训练循环从 `train_loader` 取 batch。DataLoader 为每个样本调用 `ModelV1Dataset.__getitem__()`。

---

## 3. `ModelV1Dataset.__getitem__(index)`

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L540)  
所属模块：单样本读取与预处理模块。

### 输入

```python
index: int
```

### 处理

先根据索引读取 CSV 中的一行，再确定图像路径。

当前人脸和眼图来源不同：

```python
face_path = row["virtual_face_path"]
left_eye_path = resolve_legacy_eye_crop_path(row, "left_eye")
right_eye_path = resolve_legacy_eye_crop_path(row, "right_eye")
```

因此当前真实图像输入是：

- 人脸：虚拟相机归一化人脸图；
- 左右眼：由 `source_image_path` 定位到同一样本的 legacy 眼部裁剪图；
- 不加载旧 PnP1010 虚拟相机 manifest。

### 下一步

分别调用 `_load_image()` 读取人脸、左眼和右眼。

---

## 4. `_load_image(path, size)`

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L799)  
所属模块：图像读取模块。

### 输入

```python
path
size
```

当前目标尺寸：

```text
人脸：160 × 160
眼图：90 × 56
```

### 处理

1. 使用 PIL 打开图片；
2. 强制转换为 RGB；
3. resize 到目标大小；
4. 从 HWC 转换成 CHW；
5. 从 `uint8 [0,255]` 转换为 `float32 [0,1]`。

### 输出

```text
face_image       [3,160,160]
left_eye_image   [3,56,90]
right_eye_image  [3,56,90]
```

### 下一步

训练样本的左右眼图进入 `PairedEyeAppearanceAugmentation.apply_with_keypoints()`。

---

## 5. `PairedEyeAppearanceAugmentation.apply_with_keypoints(...)`

位置：[`modelv1/data/augmentation.py`](../modelv1/data/augmentation.py#L139)  
所属模块：双眼联合外观增强模块。

当前增强策略：

```yaml
eye_augmentation: appearance_mild_v1
face_augmentation: appearance_mild_v1
```

人脸图在 FaceNet 标准化前独立经过轻量外观增强：亮度、对比度、gamma、
轻微模糊和低幅噪声。该路径不执行翻转、旋转、裁剪、平移或缩放，且验证集
不注入任何 Face transform。

### 输入

```python
left_eye_image
right_eye_image
eye_keypoints
eye_keypoint_valid
```

### 处理

- 左右眼共享亮度变化；
- 左右眼共享对比度变化；
- 左右眼共享 gamma 变化；
- 模糊；
- 局部噪声；
- 局部遮挡。

该策略主要改变图像外观，不直接移动关键点坐标。发生局部遮挡时，会更新关键点有效性标记。

### 输出

```python
augmented_left_eye
augmented_right_eye
keypoints
updated_valid_mask
```

### 下一步

图像进入 `_normalize_image()`。

---

## 6. `_normalize_image(image, mean, std)`

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L815)  
所属模块：图像数值归一化模块。

### 计算

```python
normalized = (image - mean) / std
```

### 人脸图归一化

使用 FaceNet 风格参数：

```text
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

大致将输入从 `[0,1]` 映射到 `[-1,1]`。

### 左右眼归一化

使用 ImageNet mean/std，因为眼部骨干是 ImageNet 预训练的 ResNet18。

### 输出

尺寸不变：

```text
face_image       [3,160,160]
left_eye_image   [3,56,90]
right_eye_image  [3,56,90]
```

---

## 7. `precomputed_table_frame7_tensor(...)`

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L1189)  
所属模块：场景几何输入构造模块。

### 输入

```python
table_frame7_n_0 ... table_frame7_n_6
```

### 处理

直接读取 Stage3 已经在 Iris65 metric head-origin 虚拟相机坐标系中构造好的：

- 桌面第一个基向量 `e1`：3维；
- 桌面第二个基向量 `e2`：3维；
- 相机到桌面的有符号距离：1维。

最后一维在 Stage3 中已经由毫米转换为米。

函数还会检查：

- `e1`、`e2` 是否接近单位向量；
- 两个向量是否正交；
- 构造的桌面坐标系是否满足右手系。

### 输出

```text
table_frame7 = [e1_x,e1_y,e1_z,e2_x,e2_y,e2_z,d]
shape = [7]
```

### 输出

```text
table_frame7_n [7]
```

loader 不再通过旧 `scene25` 重建 TableFrame7，也不会再次乘旧 manifest 的
`rotation_n_from_c`。

### 后续用途

这个向量在模型中使用两次：

1. 输入 `scene_branch`，得到场景 embedding；
2. 直接作为 FiLM 的条件向量。

---

## 8. `raw_uv_target(...)`

位置：[`modelv1/data/dataset.py`](../modelv1/data/dataset.py#L829)  
所属模块：UV 标签构造模块。

当前目标坐标系：

```yaml
direct_uv_target_frame: table_local
```

### 输入

```text
uv_gt_table_mm = [u_mm, v_mm]
```

### 输出

```text
raw_uv_target [2]
```

UV normalizer 随后将毫米值标准化：

```python
uv_target = (uv_gt_table_mm - train_mean) / train_std
```

需要区分：

- `uv_gt_table_mm`：桌面局部坐标，单位毫米；
- `uv_target`：标准化后的训练监督目标；
- 二者都是监督标签，不是模型视觉输入。

---

## 9. `ModelV1Dataset.__getitem__()` 返回样本字典

主要字段：

```python
{
    "face_image":       [3,160,160],
    "left_eye_image":   [3,56,90],
    "right_eye_image":  [3,56,90],
    "scene_vec":        [7],
    "table_frame7_n":   [7],
    "uv_target":        [2],
    "uv_gt_table_mm":   [2],

    # 关键点辅助监督
    "eye_keypoints":       [2,15,2],
    "eye_keypoint_valid":  [2,15],
}
```

经过 DataLoader 拼接后增加 batch 维：

```text
face_image       [B,3,160,160]
left_eye_image   [B,3,56,90]
right_eye_image  [B,3,56,90]
scene_vec        [B,7]
table_frame7_n   [B,7]
uv_target        [B,2]
```

---

# 第二部分：从训练 batch 到 UV 输出

## 10. `run_epoch(...)`

位置：[`scripts/train_modelv1.py`](../scripts/train_modelv1.py#L1408)  
模型调用位置：[`scripts/train_modelv1.py`](../scripts/train_modelv1.py#L1518)  
所属模块：训练循环。

### 输入

```python
batch
model
optimizer
loss
```

### 处理

先将 batch 中的 Tensor 移到运行设备，然后执行：

```python
model_output = model(
    device_batch,
    return_auxiliary=True,
)
```

之所以设置 `return_auxiliary=True`，是因为当前启用了概率眼部关键点辅助训练。

### 下一步

进入 `ModelV1.forward()`。

---

## 11. `ModelV1.forward(...)`

位置：[`modelv1/model.py`](../modelv1/model.py#L1909)  
所属模块：整个 ModelV1 的总调度模块。

### 输入

```python
batch = {
    "face_image": ...,
    "left_eye_image": ...,
    "right_eye_image": ...,
    "scene_vec": ...,
    "table_frame7_n": ...,
}
```

### 当前有效分支

```text
visual_encoder
scene_branch
direct_uv_table_frame_film_fusion
uv_head
```

首先调用视觉编码器：

```python
visual_outputs = self.visual_encoder(
    face,
    left_eye,
    right_eye,
    ...
)
```

调用位置：[`modelv1/model.py`](../modelv1/model.py#L2095)

---

## 12. `CrossAttentionVisualEncoder.forward(...)`

位置：[`modelv1/model.py`](../modelv1/model.py#L1172)  
所属模块：眼脸视觉特征编码与融合模块。

### 输入

```text
face       [B,3,160,160]
left_eye   [B,3,56,90]
right_eye  [B,3,56,90]
```

### 实际调用顺序

1. 人脸编码器；
2. 左眼编码器；
3. 右眼编码器；
4. 概率关键点头；
5. 关键点引导眼特征融合；
6. 眼脸交叉注意力。

---

## 13. `InceptionResnetFaceEncoder.forward(face)`

Class：[`modelv1/model.py`](../modelv1/model.py#L853)  
forward：[`modelv1/model.py`](../modelv1/model.py#L885)  
所属模块：人脸图像骨干网络。

初始化时加载：

```python
InceptionResnetV1(
    pretrained="vggface2",
    classify=False,
)
```

### 输入

```text
face [B,3,160,160]
```

### 处理

- 通过 InceptionResnetV1 各卷积阶段提取人脸空间特征；
- 不使用原分类器；
- 保留最后的空间特征图，而不是只保留全局分类向量。

### 骨干输出

```text
face_map [B,1792,3,3]
```

随后经过：

```text
Linear(1792 → 128)
LayerNorm(128)
```

将 3×3 特征图展开为空间 token：

```text
face_tokens [B,9,128]
```

### 下一步

这9个人脸 token 将作为眼脸交叉注意力的 Key 和 Value。

---

## 14. `ResNetEyeImageEncoder.forward_feature_maps(eye)`

Class：[`modelv1/model.py`](../modelv1/model.py#L597)  
函数：[`modelv1/model.py`](../modelv1/model.py#L741)  
所属模块：左右眼共享的 ResNet18 + FPN 编码器。

左右眼调用同一个 encoder，参数共享。

### 输入

```text
left_eye/right_eye [B,3,56,90]
```

### 骨干来源

```text
torchvision ResNet18
ImageNet DEFAULT 预训练权重
```

为了适应小尺寸眼图，原 ResNet stem 被修改为：

```text
3×3 convolution, stride=2
不使用原 max-pooling
```

### ResNet 中间输出

```text
layer2 [B,128,14,23]
layer3 [B,256,7,12]
layer4 [B,512,4,6]
```

### FPN 处理

1. `layer2`、`layer3`、`layer4` 分别投影到128通道；
2. 将空间尺寸统一为 7×12；
3. 拼接为384通道；
4. 使用卷积融合到128通道。

### 输出

```python
{
    "feature_map": [B,128,7,12],
    "layer2":      [B,128,14,23],
    "layer3":      [B,256,7,12],
    "layer4":      [B,512,4,6],
}
```

### 下一步

- `feature_map` 用于生成眼部视觉 token；
- `layer2 + layer3` 输入概率关键点头。

---

## 15. `ProbabilisticEyeLandmarkHead.forward(...)`

Class：[`modelv1/vertical_eye_geometry.py`](../modelv1/vertical_eye_geometry.py#L25)  
forward：[`modelv1/vertical_eye_geometry.py`](../modelv1/vertical_eye_geometry.py#L97)  
所属模块：概率眼部关键点预测模块。

当前真正实例化的是这个 class，而不是 `model.py` 中旧的 `ProbabilisticEyeKeypointHead`。

每只眼分别调用一次。

### 输入

```text
layer2 [B,128,14,23]
layer3 [B,256,7,12]
```

### 处理

1. 将 `layer2`、`layer3` 投影到相同通道数；
2. 将 `layer3` 上采样到 `layer2` 尺寸；
3. 拼接并融合；
4. 上采样到关键点热图尺寸 28×45；
5. 为15个眼部关键点分别输出热图 logits；
6. 对热图做空间 softmax；
7. 计算坐标期望、协方差、熵和置信度；
8. 预测关键点可见性；
9. 生成眼部 canonical shape embedding。

### 输出

```text
heatmap_logits    [B,15,28,45]
mean              [B,15,2]
covariance        [B,15,2,2]
visibility        [B,15]
entropy           [B,15]
confidence        [B,15]
shape_embedding   [B,32]
```

关键点坐标 `mean` 是归一化眼图坐标，范围接近 `[0,1]`。

### 下一步

- 这些输出用于关键点辅助损失；
- 同时进入 `LandmarkGuidedEyeFusion`。

---

## 16. `spatial_probability_moments(...)`

位置：[`modelv1/vertical_eye_geometry.py`](../modelv1/vertical_eye_geometry.py#L564)  
所属模块：关键点概率分布统计模块。

### 输入

```text
heatmap_logits [B,15,28,45]
```

### 处理

```python
probability = softmax(heatmap_logits)
```

然后计算每个关键点的：

- X、Y 坐标期望；
- X/Y 方差；
- XY 协方差；
- 熵；
- 置信度。

### 输出

```text
mean       [B,15,2]
covariance [B,15,2,2]
entropy    [B,15]
confidence [B,15]
```

### 下一步

返回关键点头，然后送入关键点 tokenizer。

---

## 17. `ProbabilisticLandmarkTokenizer.forward(...)`

Class：[`modelv1/vertical_eye_geometry.py`](../modelv1/vertical_eye_geometry.py#L160)  
forward：[`modelv1/vertical_eye_geometry.py`](../modelv1/vertical_eye_geometry.py#L353)  
所属模块：关键点特征 token 化模块。

### 输入

- 左右眼 `layer2/layer3` 特征图；
- 关键点概率分布；
- 坐标均值；
- 协方差；
- 可见性；
- 熵；
- 左眼/右眼类型。

### 处理

1. 根据每个关键点的空间概率，对眼部特征图加权采样；
2. 得到每个关键点附近的 appearance feature；
3. 将坐标、协方差、可见性和熵编码为统计特征；
4. 加入关键点编号 embedding；
5. 加入左眼/右眼 embedding；
6. 计算关键点质量分数。

统计特征由以下信息组成：

```text
坐标均值       2维
协方差压缩表示 3维
可见性         1维
熵             1维
```

### 输出

```text
left_landmark_tokens   [B,15,128]
right_landmark_tokens  [B,15,128]
all_landmark_tokens    [B,30,128]
quality                [B,30]
```

### 下一步

进入 `LandmarkGuidedEyeFusion.forward()`。

---

## 18. `LandmarkGuidedEyeFusion.forward(...)`

Class：[`modelv1/vertical_eye_geometry.py`](../modelv1/vertical_eye_geometry.py#L396)  
forward：[`modelv1/vertical_eye_geometry.py`](../modelv1/vertical_eye_geometry.py#L462)  
所属模块：关键点引导的眼图视觉特征融合模块。

左右眼 FPN 输出首先展开为空间 token：

```text
左眼 token  [B,84,128]  # 7×12
右眼 token  [B,84,128]
双眼 token  [B,168,128]
```

### 输入

```text
eye_tokens       [B,168,128]
landmark_tokens  [B,30,128]
quality          [B,30]
```

### 注意力关系

```text
Query = 168个眼部图像token
Key   = 30个眼部关键点token
Value = 30个眼部关键点token
```

关键点质量作为 attention bias，使高质量关键点具有更大的影响。

注意力结果先生成：

```text
landmark_delta [B,168,128]
```

然后通过可学习 gate 融合：

```python
refined_eye_tokens = eye_tokens + gate * landmark_delta
```

gate 初始接近0，使模型在训练初期接近原始眼图特征，之后再学习是否以及多大程度依赖关键点。

### 输出

```text
refined_eye_tokens [B,168,128]
```

### 下一步

进入眼脸交叉注意力。

---

## 19. 眼脸 `MultiheadAttention`

位置：[`modelv1/model.py`](../modelv1/model.py#L1248)  
所属模块：`CrossAttentionVisualEncoder` 的主交叉注意力。

### 输入

```text
Query = refined_eye_tokens [B,168,128]
Key   = face_tokens        [B,9,128]
Value = face_tokens        [B,9,128]
```

### 含义

每个眼部位置主动查询人脸的9个空间区域，将人脸姿态、头部外观和双眼局部特征结合起来。

### 处理

注意力之后依次经过：

```text
残差连接
LayerNorm
FeedForward Network
残差连接
LayerNorm
```

输出仍为：

```text
[B,168,128]
```

最后对168个眼部 token 做平均池化：

```text
visual_features [B,128]
```

### 下一步

返回 `ModelV1.forward()`，与场景特征汇合。

---

## 20. `VectorBranch.forward(scene_vec)`

Class：[`modelv1/model.py`](../modelv1/model.py#L1385)  
forward：[`modelv1/model.py`](../modelv1/model.py#L1408)  
调用位置：[`modelv1/model.py`](../modelv1/model.py#L2156)  
所属模块：场景向量编码模块。

### 输入

```text
scene_vec/table_frame7_n [B,7]
```

### 处理

网络结构可以概括为：

```text
LayerNorm
7 → 128
激活与 Dropout
128 → 64
```

### 输出

```text
scene_features [B,64]
```

### 下一步

与视觉特征拼接。

---

## 21. `torch.cat(fusion_parts, dim=-1)`

位置：[`modelv1/model.py`](../modelv1/model.py#L2257)  
所属模块：ModelV1 多分支特征汇合处。

### 输入

```text
visual_features [B,128]
scene_features  [B,64]
```

由于 `crop_cam` 和独立 `eye_geometry` 分支都未启用，当前只拼接这两部分：

```python
fusion_input = torch.cat(
    [visual_features, scene_features],
    dim=-1,
)
```

### 输出

```text
fusion_input [B,192]
```

### 下一步

进入 TableFrame7 FiLM。

---

## 22. `DirectUVTableFrameFiLMFusion.forward(...)`

Class：[`modelv1/model.py`](../modelv1/model.py#L1417)  
forward：[`modelv1/model.py`](../modelv1/model.py#L1464)  
调用位置：[`modelv1/model.py`](../modelv1/model.py#L2261)  
所属模块：TableFrame7 条件特征调制模块。

### 输入

```text
fusion_input    [B,192]
table_frame7_n  [B,7]
```

### 第一步：编码融合特征

```text
fusion_input [B,192]
      ↓
pre_film
      ↓
hidden [B,256]
```

### 第二步：编码 TableFrame7 条件

```text
table_frame7_n [B,7]
       ↓
condition_encoder
       ↓
condition [B,32]
```

### 第三步：生成 FiLM 参数

```text
gamma [B,256]
beta  [B,256]
```

调制公式：

```python
modulated = (1 + gamma) * hidden + beta
```

`gamma`、`beta` 的初始影响被限制在较小范围。FiLM 初始化时接近恒等映射，不会在训练开始时剧烈破坏融合特征。

最后经过 post-FiLM MLP：

```text
[B,256] → [B,128]
```

### 输出

```text
fused_features [B,128]
```

### 下一步

进入最终 UV Head。

---

## 23. `uv_head(fused_features)`

调用位置：[`modelv1/model.py`](../modelv1/model.py#L2486)  
所属模块：直接 UV 回归头。

模块定义：

```python
nn.Linear(128, 2)
```

### 输入

```text
fused_features [B,128]
```

### 处理

```python
uv = self.uv_head(fused_features)
```

### 输出

```text
uv [B,2]
```

两个数分别表示：

```text
u_normalized
v_normalized
```

它们不是毫米值，而是使用 Dataset 13 训练集 UV 均值和标准差做 z-score 标准化后的桌面局部坐标。

恢复为毫米值需要：

```python
uv_mm = uv * train_uv_std + train_uv_mean
```

因为训练时 `return_auxiliary=True`，模型实际返回一个字典：

```python
{
    "uv":                  [B,2],
    "visual_features":     [B,128],
    "scene_features":      [B,64],
    "fused_features":      [B,128],
    "eye_keypoint_*":      ...,
    "landmark_guided_*":   ...,
    "table_frame_film_*":  ...,
}
```

至此完成从输入图像、场景数据到 UV 输出的完整前向传播。

---

# 第三部分：预训练参数与新增模块

## 24. 哪些模块来自预训练

| 模块 | 初始化来源 | 当前是否训练 |
|---|---|---:|
| 人脸 InceptionResnetV1 | VGGFace2 预训练 | 是，不冻结 |
| 眼部 ResNet18 骨干 | ImageNet DEFAULT | 是 |
| 眼部 FPN | 新建 | 是 |
| 概率关键点头 | 新建 | 是 |
| Landmark Tokenizer | 新建 | 是 |
| Landmark Guided Fusion | 新建 | 是 |
| 眼脸交叉注意力 | 新建 | 是 |
| Scene VectorBranch | 新建 | 是 |
| TableFrame7 FiLM | 新建，FiLM 接近恒等初始化 | 是 |
| UV Head | 新建 `Linear(128,2)` | 是 |

眼部 ResNet 使用更小的学习率：

```text
主学习率：       3e-4
眼部骨干倍率：   0.05
眼部骨干学习率： 1.5e-5
```

眼部骨干配置为：

```yaml
frozen_epochs: 0
unfreeze_from: stem
```

因此从第一个 epoch 开始，眼部 ResNet 各阶段就参与训练，但学习率较小。

人脸骨干当前没有冻结，也没有使用眼部骨干的0.05倍率。

---

# 第四部分：当前配置没有进入 UV 主链路的模块

以下模块虽然存在于代码中，但当前配置不会执行：

- `crop_cam_branch`：`use_crop_cam: false`；
- 独立 `eye_geometry` 数值分支：`use_eye_geometry: false`；
- `gaze_head`；
- gaze uncertainty head；
- 深度生成器；
- 深度重加权模块；
- 深度修正模块；
- `RayTableGeometry`；
- gaze ray 与桌面求交模块；
- 旧版 `ProbabilisticEyeKeypointHead`；
- 仅使用左右眼、不使用人脸的 `EyeBranch`。

还需要注意：

- `crop_cam_vec` 即使由 Dataset 生成，当前 forward 也不会使用；
- 伪关键点标签用于关键点辅助损失，不是直接送入主模型的输入特征；
- 主模型内部用于引导眼部 token 的是模型自己预测出的概率关键点；
- `uv_target` 和 `uv_gt_table_mm` 是监督标签，不参与 UV 的前向预测。

---

# 第五部分：完整函数调用顺序速查

```text
scripts/train_modelv1.py
└─ build_modelv1_dataloaders(...)
   └─ ModelV1Dataset.__getitem__(index)
      ├─ _load_image(face_path, FACE_SIZE)
      ├─ _load_image(left_eye_path, EYE_SIZE)
      ├─ _load_image(right_eye_path, EYE_SIZE)
      ├─ PairedEyeAppearanceAugmentation.apply_with_keypoints(...)
      ├─ _normalize_image(face_image, FACENET_MEAN, FACENET_STD)
      ├─ _normalize_image(left_eye_image, IMAGENET_MEAN, IMAGENET_STD)
      ├─ _normalize_image(right_eye_image, IMAGENET_MEAN, IMAGENET_STD)
      ├─ build_scene_input_vector(..., "table_frame7")
      ├─ transform_table_frame7_to_virtual_camera(...)
      └─ raw_uv_target(...)

run_epoch(...)
└─ ModelV1.forward(batch)
   ├─ CrossAttentionVisualEncoder.forward(...)
   │  ├─ InceptionResnetFaceEncoder.forward(face)
   │  ├─ ResNetEyeImageEncoder.forward_feature_maps(left_eye)
   │  ├─ ResNetEyeImageEncoder.forward_feature_maps(right_eye)
   │  ├─ ProbabilisticEyeLandmarkHead.forward(left_maps)
   │  │  └─ spatial_probability_moments(...)
   │  ├─ ProbabilisticEyeLandmarkHead.forward(right_maps)
   │  │  └─ spatial_probability_moments(...)
   │  ├─ LandmarkGuidedEyeFusion.forward(...)
   │  │  └─ ProbabilisticLandmarkTokenizer.forward(...)
   │  ├─ MultiheadAttention(
   │  │      query=eye_tokens,
   │  │      key=face_tokens,
   │  │      value=face_tokens
   │  │  )
   │  └─ mean_pool → visual_features [B,128]
   ├─ VectorBranch.forward(table_frame7_n)
   │  └─ scene_features [B,64]
   ├─ torch.cat([visual_features, scene_features])
   │  └─ fusion_input [B,192]
   ├─ DirectUVTableFrameFiLMFusion.forward(
   │      fusion_input,
   │      table_frame7_n
   │  )
   │  └─ fused_features [B,128]
   └─ uv_head(fused_features)
      └─ normalized table-local UV [B,2]
```

## 最终张量尺寸汇总

| 阶段 | 张量 | 尺寸 |
|---|---|---|
| Dataset | 人脸图 | `[B,3,160,160]` |
| Dataset | 单只眼图 | `[B,3,56,90]` |
| 人脸骨干 | 人脸特征图 | `[B,1792,3,3]` |
| 人脸投影 | 人脸 tokens | `[B,9,128]` |
| 眼部 ResNet layer2 | 单眼特征 | `[B,128,14,23]` |
| 眼部 ResNet layer3 | 单眼特征 | `[B,256,7,12]` |
| 眼部 ResNet layer4 | 单眼特征 | `[B,512,4,6]` |
| 眼部 FPN | 单眼特征 | `[B,128,7,12]` |
| 双眼展开 | 双眼 tokens | `[B,168,128]` |
| 关键点头 | 单眼热图 | `[B,15,28,45]` |
| 关键点头 | 单眼坐标均值 | `[B,15,2]` |
| 关键点 tokenizer | 双眼关键点 tokens | `[B,30,128]` |
| 眼脸交叉注意力 | 输出 tokens | `[B,168,128]` |
| 视觉池化 | visual features | `[B,128]` |
| 场景分支 | scene features | `[B,64]` |
| 特征拼接 | fusion input | `[B,192]` |
| FiLM 融合 | fused features | `[B,128]` |
| UV Head | normalized UV | `[B,2]` |
