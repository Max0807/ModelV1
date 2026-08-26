# ModelV1 当前训练全流程与模块调试地图

> 基线日期：2026-08-25  
> 基线配置：`configs/modelv1/train_dataset11_val_dataset3_direct_uv_virtual_camera_film_v2.yaml`  
> 目标：按照“原始采集输出 → 离线预处理 → Dataset/DataLoader → 模型每个执行模块 → 输出 → loss → backward/optimizer”的顺序，逐项检查当前 direct-UV 工程。

## 0. 阅读约定

- 本文只把**当前主配置实际执行的路径**写进主流程；条件分支和关闭的模块集中放在第 12 节。
- 张量维度使用 PyTorch 顺序：图像为 `[B, C, H, W]`，序列为 `[B, N, D]`。
- 配置的 `batch_size=64`，但 `drop_last=False`，因此本文中的 `B` 通常为 64，最后一个 batch 可以更小。
- 当前训练集 Dataset13 有 798 条，验证集 Dataset11 有 505 条，因此每个 epoch 是 13 个训练 batch（最后一个 `B=30`）和 8 个验证 batch（最后一个 `B=57`）。
- 文件位置采用 `文件:行号 / 类或函数`。代码继续整理后行号可能变化，优先按类名或函数名搜索。

## 1. 先确认基线：当前代码有 5 个容易误判的地方

### 1.1 必须显式传配置

训练入口是 `scripts/train_modelv1.py:3253 / main()`，但默认配置常量仍指向已经不存在的：

```text
configs/modelv1/train_random_80_20_100.yaml
```

位置：`scripts/train_modelv1.py:138 / DEFAULT_CONFIG_PATH`。

当前主流程应显式启动：

```powershell
python scripts/train_modelv1.py --config configs/modelv1/train_dataset11_val_dataset3_direct_uv_virtual_camera_film_v2.yaml
```

### 1.2 配置文件名与真实 split 不一致

文件名写的是 `dataset11_val_dataset3`，但 YAML 真正执行的是：

- `train_datasets: ["13"]`
- `val_datasets: ["11"]`

位置：主 YAML `:43-46`。本文全部以**训练 Dataset13、验证 Dataset11**为准。

### 1.3 实际 eye token 不是注释中的 4×6/48

主 YAML `:116` 明确设置 `eye_token_grid: [7, 12]`：

- 每只眼：`7 × 12 = 84` 个 token；
- 两只眼：`2 × 84 = 168` 个 token。

YAML `:4` 的“4x6 tokens”和 `:133` 的“48 binocular tokens”都是过时描述。模型实际会检查并使用 `[7,12]`，位置：`modelv1/model.py:1622-1629, 1844-1853`。

### 1.4 distance-FiLM 和 pose-FiLM 同时开启时，实际只使用 pose 条件

主 YAML 同时设置：

```yaml
use_virtual_distance_film: true
use_virtual_pose_film: true
```

但 `DirectUVFiLMFusion` 在 `use_pose_condition=True` 时只读取：

```text
virtual_camera_pose_table: [B, 9]
```

不会读取或生成 `virtual_log_scale_normalized: [B,1]`。位置：

- Dataset：`modelv1/data/dataset.py:920-935`；
- Model：`modelv1/model.py:2290-2331`。

因此当前路径更准确的名称是“**pose-conditioned FiLM**”。9D pose 的平移部分已经包含相机到桌面平面的距离信息，但它不是独立的 scalar distance-FiLM 分支。

### 1.5 `depth_method: iris65` 选择数据契约，但 direct-UV 前向不读取深度先验

主 YAML `:12-14` 已说明这一点。当前：

- `prediction_mode=direct_uv`；
- `use_eye_geometry=false`；
- `use_depth_correction=false`；
- `deca_feature_representation=none`。

所以 Iris65 depth CSV 用来确定 preprocessing-v2 的权威样本集合和 manifest，不会进入模型 forward，也不会执行 gaze ray/table 求交。

## 2. 当前端到端执行图

```text
原始采集目录
  ├─ data_log / 标定 / gaze target / table pose
  ├─ 1920×1080 source image
  └─ InsightFace face/eye bbox 与 legacy eye crop 路径
          │
          ▼
preprocessing-v2 离线流水线
  ├─ 统一 dataset CSV
  ├─ MediaPipe PnP landmarks → PnP1010 pose/depth
  ├─ MediaPipe iris → Iris65 depth contract
  ├─ PnP/Iris 公共有效样本集
  ├─ PnP-centred virtual-camera face 160×160 + manifest
  └─ 独立生成 legacy-eye 15 点 MediaPipe pseudo labels
          │
          ▼
训练入口解析 Dataset13 + Dataset11 manifests
  ├─ 合并 CSV（798 + 505）
  ├─ 合并 virtual-camera manifests
  └─ 加载两个 pseudo-label NPZ
          │
          ▼
ModelV1Dataset.__getitem__
  ├─ virtual face → [3,160,160] → [-1,1]
  ├─ legacy left/right eye → 各 [3,56,90] → appearance aug → ImageNet normalize
  ├─ table-local UV → [2] mm → 训练集 z-score target [2]
  ├─ TableFrame7_C → TableFrame7_N [7]
  ├─ virtual camera/table pose [9]
  └─ pseudo eye labels [2,15,2] + masks/quality
          │ collate
          ▼
ModelV1.forward
  ├─ Face Inception-ResNet V1 → [B,9,128] face tokens
  ├─ shared Eye ResNet18 H5/FPN → [B,168,128] binocular tokens
  ├─ 15-point probabilistic head → 每眼 [B,15,28,45] 等
  ├─ landmark-guided eye fusion → [B,168,128]
  ├─ eye-query / face-key-value CrossAttention → [B,168,128]
  ├─ mean pool → visual [B,128]
  ├─ TableFrame7_N MLP → scene [B,64]
  ├─ concat → [B,192]
  ├─ pose FiLM → fused [B,128]
  ├─ UV head → normalized uv_base [B,2]
  └─ vertical geometry residual → delta_v_mm [B,1]
          │
          ▼
DirectTableUVLoss
  ├─ denormalize uv_base → [B,2] mm
  ├─ 只对 v 加 delta_v_mm
  ├─ main SmoothL1(beta=30 mm)
  └─ 加权叠加 vertical + landmark auxiliary losses
          │
          ▼
loss.backward → clip_grad_norm_(1.0) → AdamW.step → epoch-end cosine scheduler
```

## 3. 离线 preprocessing-v2

### 3.1 总入口与版本契约

- 总入口：`scripts/prepare_modelv1_preprocessing.py:369 / main()`。
- pipeline version：`modelv1_metric_depth_virtual_camera_v2`，定义于 `scripts/prepare_modelv1_preprocessing.py:28`。
- 训练侧验证：`modelv1/preprocessed_v2.py:13, 59-190 / PreprocessedV2TrainingInput.load()`。
- 每个 Dataset 的权威训练入口：

```text
data/preprocessed_v2_160/dataset<ID>/manifests/
  training_inputs_iris65_dataset<ID>.json
```

训练侧会验证：版本号、Dataset ID/name、depth method、所有引用文件存在、`sample_count > 0`，位置：`modelv1/preprocessed_v2.py:76-190`。

### 3.2 标准 12 步离线流水线

| 步骤 | 代码位置 | 主要作用 | 主要输入 | 主要输出/维度 | 当前主训练是否直接消费 |
|---|---|---|---|---|---|
| 1 | `scripts/prepare_modelv1_preprocessing.py:390-410` → `scripts/build_modelv1_dataset.py` | 合并 data_log、标定、InsightFace 检测/裁剪路径和 gaze/table 标签 | 原始采集目录；每张图一条记录 | `modelv1_dataset<ID>.csv`；每行一个 sample | 是，后续筛选后的 CSV 是 Dataset 根表 |
| 2 | `:417-439` → `scripts/cache_deca_features.py` | 可选缓存 DECA 表征 | face image | NPZ，按 sample 存特征 | 否，当前 `representation=none`，manifest 中也是 `deca_cache=null` |
| 3 | `:442-458` → `scripts/extract_mediapipe_pnp_landmarks.py` | 提取 PnP 使用的 MediaPipe 面部点 | source RGB image | landmark CSV；每行固定 PnP 点坐标 | 间接，用于步骤 4/12 |
| 4 | `:460-488` → `scripts/generate_depth_priors.py` | 固定 1010 mm/FLAME-unit 的 PnP pose/depth | dataset CSV + PnP landmarks | PnP prior CSV；含 `R_C_H [3,3]`、`tvec_C [3]`、眼中心/质量等 | depth 不进当前模型；pose 用于虚拟相机 |
| 5 | `:490-501` → `scripts/validate_pnp_geometry.py` | 审核 PnP 几何契约 | PnP prior CSV | validation JSON | 只作离线质量门槛 |
| 6 | `:503-519` → `scripts/build_depth_prior_comparison_subset.py` | 生成 PnP 有效样本集 | dataset CSV + PnP prior | `modelv1_dataset<ID>_pnp1010.csv` | 当前用公共 PnP/Iris 子集，不直接用此表 |
| 7 | `:521-539` → `scripts/extract_mediapipe_iris_centres.py` | 提取两眼虹膜中心 | source image | iris CSV；每样本左右眼 2D 中心 | 用于步骤 8；旧单点 iris aux 当前关闭 |
| 8 | `:540-596` → `scripts/generate_iris_ipd_depth_priors.py` | 用 65 mm IPD 生成 Iris65 metric depth，同时保留 PnP pose | dataset/PnP/iris/calibration | Iris65 prior CSV | 只用于选择样本契约；不进 direct-UV forward |
| 9 | `:597-608` | 审核 Iris65 文件里的 PnP pose | Iris65 prior | validation JSON | 离线审核 |
| 10 | `:609-625` | 生成 Iris65 有效覆盖集 | dataset + Iris65 prior | `modelv1_dataset<ID>_iris65.csv` | 当前最终使用步骤 11 的公共集 |
| 11 | `:626-643` | 取 PnP1010 与 Iris65 都有效的交集 | dataset + 两个 prior | `modelv1_dataset<ID>_common_pnp1010_iris65.csv` | 是 |
| 12 | `:649-682` → `scripts/generate_virtual_camera_normalized_images.py` | 用 PnP face centre/pose 把 source image warp 到共享虚拟相机，并生成 manifest | source RGB、相机标定、PnP pose、眼角点 | 当前 face `[160,160,3]`；另生成 virtual eyes `[36,60,3]`；manifest | 当前只使用 virtual face 和 manifest，virtual eyes 不使用 |

标准产物名定义在 `scripts/prepare_modelv1_preprocessing.py:200-291 / PipelineArtifacts`。

注意：`scripts/prepare_modelv1_preprocessing.py:645-647` 在 `depth_method=iris65` 时也选择 `common_dataset`，所以当前 Iris65 manifest 实际指向 PnP/Iris 公共样本集。这与当前文件中的 798/505 条一致。

### 3.3 虚拟相机图像模块

核心位置：`modelv1/data/virtual_camera.py`。

1. `VirtualCameraConfig`（`:81-108`）定义输出宽高、焦距和固定虚拟距离。
2. `build_virtual_camera_transform()`（`:312-382`）：
   - 输入 `R_C_H [3,3]`、`face_center_C_mm [3]`、相机内参 `K_C [3,3]`；
   - 构造 `R_N_C [3,3]`；
   - `distance_scale = virtual_distance_mm / ||face_center_C_mm||`，scalar；
   - 构造 homography `H_N_undistorted [3,3]`。
3. `VirtualCameraTransform.render()`（`:262-287`）：undistort 后 `warpPerspective`，当前产出 RGB `[160,160,3]`。
4. `crop_normalized_eye()`（`:385-429`）：在 virtual face 上按眼角点生成 `[36,60,3]` virtual eye。
5. `render_shared_normalized_inputs()`（`:432-462`）：一次生成 face + left/right virtual eyes，并保持 anatomical left/right，不镜像。

当前主配置设置 `image_source=virtual_camera`、`eye_image_source=legacy`，因此：

- face path 来自 manifest 的 `normalized_face_path`；
- left/right eye path 仍来自 dataset CSV 的 `left_eye_path/right_eye_path`；
- manifest 中的 `normalized_left_eye_path/normalized_right_eye_path` 不进入当前模型。

### 3.4 15 点眼伪标签是额外的离线阶段

它没有被上面的 12 步 orchestrator 自动调用，生成入口是：

```text
scripts/extract_mediapipe_eye_geometry.py:500 / main()
```

它从 source image 跑 `MediaPipe FaceMesh(refine_landmarks=True)`，主循环在 `:516-585`。每只 anatomical eye 取 15 点：

```text
0 outer canthus
1-4 upper eyelid
5 inner canthus
6-9 lower eyelid
10 iris center
11-14 iris rim
```

NPZ 的核心数组定义在 `scripts/extract_mediapipe_eye_geometry.py:377-404`：

| 数组 | 单个数据集整体形状 | 单样本加载形状 | 说明 |
|---|---:|---:|---|
| left/right `points_px` | `[N,15,2]` | 两眼合并 `[2,15,2]` | source image 像素坐标 |
| left/right `points_crop_norm` | `[N,15,2]` | `[2,15,2]` | legacy eye bbox 内归一化坐标，训练主监督使用 |
| left/right `points_local` | `[N,15,2]` | `[2,15,2]` | 眼角中心为原点、canthus distance 归一化 |
| left/right point valid | `[N,15]` | `[2,15]` | 每点有效 mask |
| left/right geometry | `[N,10]` | `[2,10]` | iris/eyelid/crop containment/quality 特征 |
| eye valid | `[N]` each | `[2]` | 每眼有效性 |
| sample quality | `[N]` | `[1]` | 两眼质量的较小值 |

加载契约：`modelv1/data/eye_geometry_pseudo_labels.py:16-18, 71-82`。当前 NPZ 覆盖原始 Dataset13 的 903 条和 Dataset11 的 583 条，并且全部检测成功；训练使用的 798/505 条公共子集均可按 `sample_id` 查到。

## 4. 训练入口如何解析当前数据

### 4.1 配置加载与 artifact 合并

`scripts/train_modelv1.py:3253-3338`：

1. 加载 YAML；
2. 固定 seed、选择 device；
3. `make_model_config()` 把 YAML model section 转成 `ModelV1Config`；
4. 调用 `resolve_preprocessed_v2_artifacts()`。

`resolve_preprocessed_v2_artifacts()` 位于 `scripts/train_modelv1.py:1393-1485`：

- 读取 Dataset13/11 的 `training_inputs_iris65_*.json`；
- 验证各自 `sample_count`；
- 在本次 run 目录的 `resolved_preprocessed_v2/dataset.csv` 合并两个 CSV；
- 当前不合并 DECA cache，也不合并 depth prior，因为主模型不需要；
- 保存 sources.json，记录实际输入来源。

两个当前 manifest：

| Dataset | role | CSV/manifest sample_count |
|---|---|---:|
| Dataset13 | train | 798 |
| Dataset11 | validation | 505 |

虚拟相机 manifests 在 `scripts/train_modelv1.py:3365-3376` 解析/合并；两个 pseudo-label NPZ 在 `:3410-3423` 检查后传给 DataLoader。

### 4.2 split、normalizer 和 loader

`build_modelv1_dataloaders()`：`modelv1/data/dataset.py:1154-1695`。

当前 `split_mode=explicit_datasets`：

- `train_set` 只保留 Dataset13；
- `val_set` 只保留 Dataset11；
- 两者必须不重叠，检查位置 `:1496-1508`。

UV normalizer 只在 Dataset13 的 table-local UV 上拟合：

```text
mean_mm = mean(train_uv_mm, dim=0)        # [2]
std_mm  = std(train_uv_mm, unbiased=False) # [2]
uv_target = (uv_mm - mean_mm) / std_mm    # [2]
```

位置：

- 拟合/共享给 val：`modelv1/data/dataset.py:1590-1602`；
- 公式：`modelv1/data/normalization.py:20-69 / UVTargetNormalizer`。

Loader 设置：`modelv1/data/dataset.py:1679-1694`。

| loader | shuffle | augmentation | batch size | drop_last |
|---|---:|---:|---:|---:|
| train | true | 是 | 64 | false |
| val | false | 否 | 64 | false |

## 5. `ModelV1Dataset.__getitem__`：单样本到 batch

主位置：`modelv1/data/dataset.py:748-940`。

### 5.1 图像读取、resize、增强、归一化

`_load_image()` 位于 `modelv1/data/dataset.py:1065-1078`：

```text
PIL open → RGB → bilinear resize → uint8 HWC → float32 CHW / 255
```

#### Face

- 来源：virtual manifest `normalized_face_path`；
- 单样本输入：`[3,160,160]`，值域 `[0,1]`；
- 不做 eye augmentation；
- FaceNet normalization：`(x-0.5)/0.5`，输出约 `[-1,1]`；
- 位置：`modelv1/data/dataset.py:755-780, 823-826`。

#### Left/right eye

- 来源：legacy `left_eye_path/right_eye_path`；
- 配置尺寸是 `(width,height)=(90,56)`；
- 单样本张量各为 `[3,56,90]`；
- 训练时先做 paired appearance augmentation，再做 ImageNet normalization；
- ImageNet mean `(0.485,0.456,0.406)`，std `(0.229,0.224,0.225)`；
- 位置：`modelv1/data/dataset.py:760-783, 804-829`。

眼增强实现在 `modelv1/data/augmentation.py:79-195 / PairedEyeAppearanceAugmentation`：

- shared：brightness/contrast/gamma、Gaussian blur；
- independent：noise、mean-filled random occlusion；
- 不允许 translation/rotation/crop/flip，所以 `[2,15,2]` pseudo 坐标不变；
- 被 occlusion 覆盖的点只把 valid mask 置 0，位置 `:98-128`。

### 5.2 单样本核心字段与 batch 维度

| key | 单样本形状 | collate 后形状 | 当前作用 |
|---|---:|---:|---|
| `face` | `[3,160,160]` | `[B,3,160,160]` | face encoder 输入 |
| `left_eye` | `[3,56,90]` | `[B,3,56,90]` | shared eye encoder 输入 |
| `right_eye` | `[3,56,90]` | `[B,3,56,90]` | shared eye encoder 输入 |
| `scene_vec` | `[7]` | `[B,7]` | 当前等于 `table_frame7_n`；scene MLP 和 vertical residual 输入 |
| `table_frame7_n` | `[7]` | `[B,7]` | `[e1_N(3),e2_N(3),signed_distance_m(1)]` |
| `virtual_camera_pose_table` | `[9]` | `[B,9]` | pose-FiLM 条件 `[R_6D,t_3D]` |
| `rotation_n_from_c` | `[3,3]` | `[B,3,3]` | 当前 direct-UV forward 不使用；保留用于其他分支/审计 |
| `rotation_c_from_n` | `[3,3]` | `[B,3,3]` | 当前 forward 不使用 |
| `virtual_distance_scale` | `[1]` | `[B,1]` | 当前不送入 FiLM，仅作记录 |
| `uv_gt_table_mm` | `[2]` | `[B,2]` | 主 loss 的物理监督 |
| `uv_gt` | `[2]` | `[B,2]` | 当前同 table-local mm；metric target |
| `uv_target` | `[2]` | `[B,2]` | Dataset13 统计 z-score 后的监督空间；模型 head 的输出空间 |
| `crop_cam_vec` | `[36]` | `[B,36]` | 仍在 batch，但 `use_crop_cam=false`，模型不读取 |
| `gaze_target_w` | `[3]` | `[B,3]` | metric/debug 字段，当前 direct-UV 模型不读取 |
| `table_origin_w` | `[3]` | `[B,3]` | debug 字段，当前模型不读取 |

构造位置：`modelv1/data/dataset.py:831-875, 912-935`。

`TableFrame7_C → TableFrame7_N` 只旋转两个基向量，保留 signed plane distance：`modelv1/data/dataset.py:1773-1817`。

9D pose 构造：`modelv1/data/dataset.py:1820-1864`：

```text
virtual_camera_pose_table = concat(R_table_from_N first two columns, t_table_m)
                          = [R_6D, t_3D]
                          = [9]
```

### 5.3 pseudo-label 字段

| key | batch 形状 | 说明 |
|---|---:|---|
| `eye_pseudo_landmarks_xy` | `[B,2,15,2]` | legacy crop normalized target，主关键点监督 |
| `eye_pseudo_landmarks_local` | `[B,2,15,2]` | 眼局部坐标，当前主 loss 不直接用 |
| `eye_pseudo_landmark_base_valid_mask` | `[B,2,15]` | 增强前有效 mask |
| `eye_pseudo_landmark_valid_mask` | `[B,2,15]` | 遮挡增强更新后的 mask |
| `eye_pseudo_occluded_mask` | `[B,2,15]` | base valid 但被增强遮住的点 |
| `eye_pseudo_geometry_features` | `[B,2,10]` | 离线教师几何/质量特征 |
| `eye_pseudo_eye_valid_mask` | `[B,2]` | 每眼有效性 |
| `eye_pseudo_detection_mask` | `[B,1]` | FaceMesh 是否检测成功 |
| `eye_pseudo_sample_valid_mask` | `[B,1]` | 双眼整体有效性 |
| `eye_pseudo_sample_quality` | `[B,1]` | sample quality |
| `eye_pseudo_eye_quality` | `[B,2]` | 每眼 geometry 最后一维 quality |
| `eye_pseudo_effective_quality` | `[B,2]` | eye quality × eye valid × visible fraction |
| `iris_center_xy` | `[B,2,2]` | 第 10 个点；旧 iris aux 关闭 |
| `iris_center_valid_mask` | `[B,2]` | iris center 有效性 |

构造位置：`modelv1/data/dataset.py:785-800, 876-904`。

关键点伪标签只作为 loss target，不直接拼接到模型输入。

## 6. `ModelV1.forward` 的当前实际输入

入口：`modelv1/model.py:2930-3667 / ModelV1.forward()`。

训练循环调用：

```python
model_output = model(device_batch, return_auxiliary=True)
```

位置：`scripts/train_modelv1.py:2092-2099`。因为 keypoint auxiliary 开启，所以 `return_auxiliary=True`；当前 gate regularization 为 0，所以训练时通常 `return_features=False`。

模型从 batch 实际取出的字段：

| 输入 | 形状 | 读取位置 |
|---|---:|---|
| `face` | `[B,3,160,160]` | `modelv1/model.py:2977-2978` |
| `left_eye` | `[B,3,56,90]` | `:2979-2980` |
| `right_eye` | `[B,3,56,90]` | `:2981-2982` |
| `scene_vec`，实际优先取 `table_frame7_n` | `[B,7]` | `:2985-2997` |
| `virtual_camera_pose_table` | `[B,9]` | `:3008-3015` |

不读取：DECA、crop_cam、raw eye depth/geometry、depth uncertainty、PnP quality、gaze target、独立 virtual scale scalar。

## 7. 视觉主干：逐模块、逐形状

### 7.1 Face：Inception-ResNet V1

模块：`modelv1/model.py:1353-1422 / InceptionResnetFaceEncoder`。

- backbone：`facenet_pytorch.InceptionResnetV1(pretrained="vggface2")`；
- 输入：`face [B,3,160,160]`；
- 执行至 `block8`，不使用分类/最终 embedding head；
- 输出 feature map：`[B,1792,3,3]`；
- flatten spatial：`[B,9,1792]`；
- `Linear(1792→128) + LayerNorm`：`face_tokens [B,9,128]`；
- 加可学习 position embedding `[1,9,128]`。

编码器输出检查在 `modelv1/model.py:1408-1414`；projection 在 `:1644-1647`；forward tokenization 在 `:1819,1834`。

当前 `freeze_face_image_backbone=false`，所以 face backbone 接收主 UV loss 的梯度，学习率属于 base group `3e-4`。

### 7.2 Eye：共享 ResNet18 H5 small stem

模块：`modelv1/model.py:1017-1238 / ResNetEyeImageEncoder`。

左右眼共用同一个 encoder（`share_eye_encoder=true`），每只眼的形状相同：

| 层 | 输出形状 | 说明 |
|---|---:|---|
| input | `[B,3,56,90]` | legacy eye crop |
| 3×3 conv, stride 2 | `[B,64,28,45]` | 替换标准 7×7 stem |
| maxpool | `[B,64,28,45]` | `Identity()` |
| layer1 | `[B,64,28,45]` | ResNet18 stage1 |
| layer2 | `[B,128,14,23]` | landmark decoder 的浅层输入 |
| layer3 | `[B,256,7,12]` | landmark decoder 和 FPN 输入 |
| layer4 | `[B,512,4,6]` | FPN 输入 |

形状意图写在 `modelv1/model.py:1084-1102`，forward 在 `:1161-1207`。

### 7.3 Eye multiscale FPN24

位置：`modelv1/model.py:1084-1125, 1174-1190`。

```text
layer2 [B,128,14,23] --3×3 s2--> [B,128,7,12]
layer3 [B,256,7,12]  --1×1----> [B,128,7,12]
layer4 [B,512,4,6]   --1×1 + bilinear--> [B,128,7,12]
concat(channel)                    [B,384,7,12]
3×3 fusion                         [B,128,7,12]
```

随后每眼：

```text
[B,128,7,12] → flatten → [B,84,128] → Linear(128→128)+LayerNorm
```

左右眼加入各自 eye-type embedding 和共享 spatial position embedding 后 concat：

```text
eye_tokens = [left 84, right 84] = [B,168,128]
```

位置：`modelv1/model.py:1634-1651, 1835-1867`。

当前 binocular self-attention 关闭，因此 `binocular_tokens = eye_tokens`，位置：`modelv1/model.py:1868-1884`。

## 8. 概率 15 点关键点与 landmark-guided fusion

### 8.1 ProbabilisticEyeLandmarkHead

模块：`modelv1/vertical_eye_geometry.py:25-157`，由 `modelv1/model.py:1672-1689` 创建。

每只眼独立执行同一个 head：

```text
layer2 [B,128,14,23] → 1×1 → [B,128,14,23]
layer3 [B,256,7,12]  → 1×1 + upsample → [B,128,14,23]
concat → [B,256,14,23]
3×3 fusion → [B,128,14,23]
bilinear decode → [B,128,28,45]
heatmap predictor → logits [B,15,28,45]
```

每只眼输出：

| 名称后缀 | 形状 | 含义 |
|---|---:|---|
| `logits` | `[B,15,28,45]` | 15 张未归一化 heatmap |
| `probability_maps` | `[B,15,28,45]` | 每点 spatial softmax |
| `mean_xy` | `[B,15,2]` | 概率期望，crop normalized `[0,1]` |
| `covariance_heatmap` | `[B,15,2,2]` | heatmap 二阶矩 |
| `log_std_correction` | `[B,15,2]` | learned bounded variance correction |
| `covariance` | `[B,15,2,2]` | 校正后的 full 2D covariance |
| `variance_xy` | `[B,15,2]` | covariance 对角线 |
| `entropy` | `[B,15]` | 归一化 heatmap entropy |
| `confidence` | `[B,15]` | `1-entropy` |
| `visibility_logits` | `[B,15]` | pooled feature 预测的可见性 |
| `visibility_probability` | `[B,15]` | sigmoid visibility |
| `canonical_shape` | `[B,15,2]` | canthus-normalized 共享形状 |
| `shape_embedding` | `[B,32]` | L2-normalized shape embedding |

模型 mapping 的 key 会加 `left_eye_keypoint_` 或 `right_eye_keypoint_` 前缀。组装位置：`modelv1/model.py:1962-1998`。

### 8.2 ProbabilisticLandmarkTokenizer

模块：`modelv1/vertical_eye_geometry.py:160-393`。

每个点从 layer2 和 layer3 概率加权采样 appearance feature：

```text
probability [B,15,H,W]
× projected feature [B,128,H,W]
→ appearance tokens [B,15,128]
```

再拼接 7D 统计量：

```text
[mean_x, mean_y, cov_xx, cov_xy, cov_yy, visibility, entropy]
```

经 MLP 融合后：

- 每眼 landmark tokens `[B,15,128]`；
- 两眼 concat `[B,30,128]`；
- landmark quality `[B,30]`。

当前三个 detach 开关都为 true：

- sampling probability detach；
- landmark statistics detach；
- quality detach。

因此主 UV 经 landmark-guided 分支不会直接改 landmark posterior 的位置/置信度，但仍可训练被采样的 layer2/layer3 appearance feature 和 fusion 参数。

### 8.3 LandmarkGuidedEyeFusion

模块：`modelv1/vertical_eye_geometry.py:396-537`。

```text
query  = eye tokens      [B,168,128]
key/value = landmark tokens [B,30,128]
heads = 4
attention output         [B,168,128]
attention weights（debug）[B,4,168,30]
```

质量会：

- 作为 key 的 embedding；
- 乘到 value；
- 通过 `log(quality)` 形成 attention bias。

最后：

```text
global_gate = tanh(raw_gate) × 1.0        [scalar]
sample_quality = mean(30 qualities)       [B,1]
applied_gate = global_gate × sample_quality [B,1]
refined_eye_tokens = eye_tokens + applied_gate × delta
                  = [B,168,128]
```

`gate_init=0`，所以初始化时该模块是严格 identity；第一步主 loss 首先能更新 `raw_gate`，而 token delta 分支的主梯度要在 gate 离开 0 后才逐渐出现。

## 9. Eye-to-face CrossAttention 与视觉向量

位置：`modelv1/model.py:1748-1796, 1904-1931`。

```text
query = refined binocular eye tokens [B,168,128]
key   = face tokens                  [B,9,128]
value = face tokens                  [B,9,128]
heads = 8，每头维度 16
```

输出：

1. MHA attended `[B,168,128]`；
2. residual + LayerNorm `[B,168,128]`；
3. FFN `128→256→128` + residual + LayerNorm，得到 `fused_tokens [B,168,128]`；
4. `visual_token_pooling=mean`，沿 token 维取均值：`pooled [B,128]`；
5. output projection 是 Identity：`visual_features [B,128]`。

同时返回：

- `face_image_features = mean(face_tokens) [B,128]`；
- `eye_features = mean(refined_eye_tokens) [B,128]`。

正常训练不返回完整 attention tensors。单模块 debug 时传 `return_features=True` 可得到：

| debug key | 形状 |
|---|---:|
| `face_tokens` | `[B,9,128]` |
| `eye_tokens` | `[B,168,128]` |
| `pre_landmark_eye_tokens` | `[B,168,128]` |
| `cross_attended_tokens` | `[B,168,128]` |
| `cross_attention_weights` | `[B,8,168,9]` |
| `landmark_guided_attention_weights` | `[B,4,168,30]` |

## 10. Scene、pose-FiLM、UV head 与 vertical residual

### 10.1 Scene branch

构造位置：`modelv1/model.py:2696-2705`；执行位置：`:3219-3231`。

```text
scene_vec / TableFrame7_N [B,7]
→ MLP 7→128→64
→ scene_features [B,64]
```

当前 fusion 输入只有：

```text
concat(visual_features [B,128], scene_features [B,64])
= fusion_input [B,192]
```

没有 DECA `[0]`、crop/camera、offline eye geometry。

### 10.2 DirectUVFiLMFusion

模块：`modelv1/model.py:2231-2355`。

当前精确形状：

```text
fusion_input [B,192]
  → pre_film Linear 192→256 + activation/dropout
  → hidden [B,256]

virtual_camera_pose_table [B,9]
  → LayerNorm + MLP 9→16→32
  → condition_embedding [B,32]
  → Linear 32→512
  → split gamma_raw/beta_raw，各 [B,256]
  → gamma,beta = 0.1*tanh(raw)，各 [B,256]

modulated = (1+gamma)*hidden + beta [B,256]
post_film Linear 256→128              [B,128]
```

FiLM 参数层在构造后被 `reset_to_identity()` 置零，因此初始 `gamma=beta=0`。需要检查 reset 调用时可搜索 `direct_uv_film_fusion.reset_to_identity`。

输出 debug keys：

| key | 形状 |
|---|---:|
| `virtual_film_condition` | `[B,9]` |
| `virtual_film_condition_embedding` | `[B,32]` |
| `virtual_pose_embedding` | `[B,32]` |
| `virtual_film_gamma` | `[B,256]` |
| `virtual_film_beta` | `[B,256]` |
| `virtual_film_features` | `[B,256]` |
| `fused_features` | `[B,128]` |

### 10.3 Base UV head

构造：`modelv1/model.py:2778-2782`；执行：`:3555-3556`。

```text
fused_features [B,128]
→ Linear(128,2)
→ uv [B,2]
```

这个 `uv` 是 Dataset13 target normalizer 的 z-score 空间，不是 mm。

### 10.4 VerticalEyeGeometryFeatureExtractor

模块：`modelv1/vertical_eye_geometry.py:1535-1675`。

输入由左右 keypoint 输出 stack：

- mean `[B,2,15,2]`；
- covariance `[B,2,15,2,2]`；
- visibility `[B,2,15]`；
- entropy `[B,2,15]`。

先把 normalized 坐标/协方差换成 90×56 crop 的 pixel scale，再为每只眼形成 10D：

```text
iris_center_y
aperture_height
iris_radius_y
iris_radius_x
iris_log_aspect
iris_visibility
rim_visibility
iris_std
iris_entropy
reference_visibility
```

输出：

- `per_eye_features [B,2,10]`；
- 双眼 mean `[B,10]`；
- left-right difference `[B,10]`；
- geometry confidence `[B,1]`；
- iris vertical disagreement `[B,1]`；
- 最终 `features [B,22]`。

### 10.5 VerticalGeometryVResidual

模块：`modelv1/vertical_eye_geometry.py:1678-1785`。

输入拼接：

```text
geometry features       [B,22]
fused features          [B,128]
scene vec               [B,7]
virtual pose            [B,9]
base_v_normalized.detach [B,1]
--------------------------------
total                    [B,167]
```

网络：`LayerNorm(167) → Linear 167→64 → SiLU → 64→32 → SiLU → 32→2`。

两个输出通道：

```text
raw_delta_v_mm = 250*tanh(channel0) [B,1]
learned_gate = sigmoid(channel1)     [B,1]
```

训练 gate：

```text
training_gate = 0.10 + 0.90*geometry_confidence
applied_gate = training_gate*learned_gate
delta_v_mm = applied_gate*raw_delta_v_mm
```

最后一层初始化为全零，因此初始：

- `raw_delta_v_mm=0`；
- `delta_v_mm=0`；
- `learned_gate=0.5`，但乘零 residual 后不改变预测。

eval 时若 confidence `<0.10`，直接 fallback 到 `delta_v_mm=0`。vertical residual 只改 v，不能改 u。

## 11. 模型最终输出、loss 与反向传播

### 11.1 训练时主要输出

因为 `return_auxiliary=True` 且 vertical residual 开启，`ModelV1.forward()` 返回 mapping，组装位置 `modelv1/model.py:3621-3666`。

最重要的 keys：

| key | 形状 | 用途 |
|---|---:|---|
| `uv` | `[B,2]` | normalized base UV |
| `vertical_delta_v_mm` | `[B,1]` | 物理 mm 的 v 修正 |
| `visual_features` | `[B,128]` | debug |
| `scene_features` | `[B,64]` | debug |
| `fused_features` | `[B,128]` | debug/vertical 输入 |
| left/right keypoint outputs | 见第 8 节 | auxiliary loss/vertical branch |
| landmark-guided gate/quality | `[B,1]` 等 | 监控 |
| vertical geometry outputs | `[B,22]`、`[B,1]` 等 | residual loss/监控 |
| virtual FiLM outputs | `[B,9/32/256]` | 监控 |

### 11.2 DirectTableUVLoss

模块：`modelv1/losses.py:62-114 / DirectTableUVLoss`。

```text
uv_norm [B,2]
→ uv_base_mm = uv_norm*std_mm + mean_mm      [B,2]
→ uv_pred_mm = [u_base, v_base+delta_v_mm]  [B,2]
→ SmoothL1(uv_pred_mm, uv_gt_table_mm, beta=30 mm)
→ scalar L_uv
```

v residual 应用函数：`modelv1/losses.py:214-226`，明确保持 u 不变。

### 11.3 当前总 loss

训练组合位置：

- direct UV + vertical：`scripts/train_modelv1.py:2219-2275`；
- keypoint auxiliary：`:2347-2549`。

当前配置实际总目标：

```text
L_total = L_uv
        + 0.001 * L_vertical_regularization
        + 0.25  * L_vertical_supervision
        + 0.02  * L_keypoint_full_cov_NLL
        + 1.00  * L_landmark_coordinate
        + 0.05  * L_landmark_heatmap
        + 0.10  * L_landmark_visibility
        + 0.10  * L_landmark_equivariance
        + 0.02  * L_mirror_shape
```

各项输入、输出和定义：

| loss | 输入形状 | 主要作用 | 位置 |
|---|---|---|---|
| `L_uv` | pred/gt `[B,2]` mm | 主 table-local UV SmoothL1，beta 30 mm | `modelv1/losses.py:79-114` |
| vertical regularization | delta `[B,1]` | `mean((delta/250)^2)`，限制过大修正 | `modelv1/vertical_eye_geometry.py:2064-2075` |
| vertical supervision | delta/target `[B,1]` | target=`clamp(v_gt-stopgrad(v_base),±250)`；SmoothL1 beta 30 mm；乘 detached training gate | `scripts/train_modelv1.py:2250-2275` |
| full-cov NLL | mean `[B,2,15,2]`，cov `[B,2,15,2,2]` | 质量加权 2D Gaussian NLL，std floor 0.01 norm | `modelv1/losses.py:319-423` |
| coordinate | pred/target `[B,2,15,2]` | quality/mask 加权 SmoothL1，beta 0.02 norm | `modelv1/vertical_eye_geometry.py:630-683` |
| heatmap | logits `[B,2,15,28,45]` | 与质量自适应 sigma 1-3 px 的 Gaussian target 做 cross entropy | `:685-713` |
| visibility | logits/mask `[B,2,15]` | BCEWithLogits | `:715-721` |
| equivariance | 两个 view mean 各 `[B,2,15,2]` | 两次已知仿射增强映回原 crop 后 SmoothL1 | `:750-887` |
| mirror shape | left/right embedding `[B,32]` | canonical shape embedding cosine distance | `:730-747` |

旧 iris heatmap、3D low-DOF template、eye-geometry gate regularization 权重当前都为 0，不进入 `L_total`。

### 11.4 等变 loss 会额外跑一次共享 eye encoder/head

训练循环在 `scripts/train_modelv1.py:2434-2536`：

1. 采样两个 `[B,3,3]` 的 crop-space affine transform；
2. 分别 warp 左右眼，拼成 `2B`；
3. 调 `model.predict_eye_landmarks()`；
4. 得到两个 view 的 `[B,2,15,2]` mean；
5. 用逆变换映回 canonical crop 比较。

因此一个训练 step 中 eye ResNet/keypoint head 的计算量不止主 forward：还会额外处理两组仿射 view。验证时也会用固定正负 transform 计算该 loss/metric。

### 11.5 梯度路由

| 梯度来源 | 主要能更新的模块 | 明确被切断的路径 |
|---|---|---|
| `L_uv` | UV head、vertical residual、FiLM、scene MLP、CrossAttention/FFN、face backbone、eye backbone/FPN、landmark-guided appearance fusion | keypoint-guided sampling probability/statistics/quality 因 detach 不接收该分支梯度 |
| `L_uv` 经 vertical residual | vertical MLP、22D geometry path、keypoint mean/cov/visibility/entropy、共享 eye backbone | residual 输入中的 `base_v_normalized` 被 detach，不通过这条支路回写 base UV |
| vertical supervision | vertical MLP 和其上游 geometry/keypoint features | target 中 `uv_base_table_mm` detach；training gate detach |
| keypoint NLL/coordinate/heatmap/visibility | probabilistic head、shape/variance/visibility 子头、shared eye ResNet layer2/3 | 不更新 face、scene、UV head |
| equivariance | shared eye ResNet + probabilistic head | 不经过主 face/scene/UV fusion |
| mirror shape | canonical shape encoder及其上游 keypoint mean | 不比较原始左右 XY |

### 11.6 backward、clip、optimizer、scheduler

位置：`scripts/train_modelv1.py:2691-2702`。

每个训练 batch：

```text
optimizer.zero_grad(set_to_none=True)
scaler.scale(L_total).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

当前 `amp=false`，所以 GradScaler 是 disabled，调用接口不变，实际使用 FP32。

AdamW：`scripts/train_modelv1.py:3671-3680`。

- base lr：`3e-4`；
- weight decay：`1e-4`；
- eye ResNet backbone lr multiplier：`0.05`，实际初始 lr=`1.5e-5`；
- eye FPN、projection、keypoint head、face backbone、attention、fusion、heads 均在 base group `3e-4`。

参数分组：`scripts/train_modelv1.py:1829-1861`。

`frozen_epochs=0`、`unfreeze_from=stem`，所以 epoch 1 开始 eye ResNet 从 stem 到 layer4 都可训练；逻辑在 `scripts/train_modelv1.py:1804-1826`。

scheduler 是对所有 param group 使用同一 cosine multiplier 的 LambdaLR，保持 0.05 的组间比例，位置 `scripts/train_modelv1.py:1864-1895`。每个 epoch 完成验证后调用一次 `scheduler.step()`，位置 `:3915`。

## 12. 当前主流程中明确不执行的模块

| 模块/路径 | 关闭原因 | 代码仍在何处 |
|---|---|---|
| DECA face feature branch | `deca_feature_representation=none` | `modelv1/model.py:2513-2540` |
| legacy EyeBranch global embedding | `use_face_image=true` 时改走 CrossAttentionVisualEncoder | `modelv1/model.py:2673-2684` |
| crop/camera 36D branch | `use_crop_cam=false` | `modelv1/model.py:2685-2695` |
| offline eye geometry 6D + quality gate | `use_eye_geometry=false` | `modelv1/model.py:2707-2742, 3238-3321` |
| binocular self-attention | `use_binocular_self_attention=false` | `modelv1/model.py:1754-1767, 1868-1884` |
| old one-point iris heatmap head | `use_eye_iris_auxiliary=false` | `modelv1/model.py:1652-1670, 1932-1938` |
| low-DOF 3D eyeball template | `use_low_dof_eye_template=false` | `modelv1/model.py:1691-1714` |
| pitch-to-v mapper | `use_pitch_to_v_residual=false` | `modelv1/model.py:2783-2815, 3558-3593` |
| scalar virtual distance normalizer/FiLM | pose-FiLM 开启时不生成 scalar | `modelv1/data/dataset.py:920-931`; `modelv1/model.py:2312-2331` |
| gaze direction/uncertainty head | `prediction_mode=direct_uv` | `modelv1/model.py:2847-2855, 3333-3553` |
| PnP depth hypothesis/reweight/correction | direct-UV 且 depth correction false | `modelv1/model.py:2857-2928, 3398-3543` |
| ray/table geometry | direct table UV 直接回归，不做 3D ray intersection | `modelv1/model.py:2868-2877, 3431-3500` |

其他 YAML 中 `prediction_mode=gaze_geometry` 的流程与本文不同，会执行 gaze direction → uncertainty sigma points → depth hypotheses → ray/table geometry；不要把那条路径的维度混入当前 direct-UV 调试。

## 13. 建议的逐模块 debug 顺序

每一步先只检查 forward 数值/shape，再检查 loss 和 gradient。不要一开始把整条链路一起判断。

### 阶段 A：数据契约

- [ ] 显式配置确实是 Dataset13 train / Dataset11 val。
- [ ] 两个 training manifest 版本都是 `modelv1_metric_depth_virtual_camera_v2`。
- [ ] merged CSV 中 sample_id 唯一，train 798、val 505。
- [ ] virtual manifest 行数分别 798/505，`normalization_status` 全部有效。
- [ ] 每个 selected sample 能在对应 pseudo NPZ 中 lookup。
- [ ] anatomical left/right 没有交换；legacy eye crop 和 pseudo label 使用同一 bbox 坐标系。

### 阶段 B：单样本 Dataset

- [ ] face 是 `[3,160,160]`、finite、归一化后大致在 `[-1,1]`。
- [ ] 两眼各 `[3,56,90]`、finite；增强前 `[0,1]`，增强后再 ImageNet normalize。
- [ ] `uv_gt_table_mm [2]`、`uv_target [2]` 都 finite。
- [ ] `table_frame7_n [7]` 的 e1/e2 单位长度且正交。
- [ ] `virtual_camera_pose_table [9]` finite。
- [ ] pseudo XY `[2,15,2]` 与 valid `[2,15]`；无遮挡时有效坐标应主要落在 `[0,1]`。
- [ ] appearance augmentation 不改变 XY，只能减少 valid mask。

### 阶段 C：眼 backbone/FPN

- [ ] layer2 `[B,128,14,23]`。
- [ ] layer3 `[B,256,7,12]`。
- [ ] layer4 `[B,512,4,6]`。
- [ ] FPN `[B,128,7,12]`。
- [ ] 每眼 84 token，双眼 168 token。
- [ ] shared encoder 的左右眼输出不同，但参数对象相同。

### 阶段 D：概率关键点

- [ ] logits/probability `[B,15,28,45]`。
- [ ] 每张 probability map 的 H×W 和约等于 1。
- [ ] mean `[B,15,2]` finite；covariance `[B,15,2,2]` 对称、对角非负。
- [ ] visibility/entropy/confidence 均在合理范围。
- [ ] 初始 heatmap 若接近均匀，entropy 应接近 1、confidence 接近 0，这是预期而非 NaN。
- [ ] NLL、coordinate、heatmap、visibility 各自单独 backward 时 keypoint head 和 eye layer2/3 有 finite grad。

### 阶段 E：landmark-guided fusion

- [ ] landmark tokens `[B,30,128]`，quality `[B,30]`。
- [ ] eye query `[B,168,128]`，attention key/value `[B,30,128]`。
- [ ] 初始化 `global_gate=0`，refined tokens 与原 eye tokens严格相同。
- [ ] 第一次 backward 后 `raw_gate.grad` 存在且 finite。
- [ ] 训练若 gate 永远为 0，检查主 loss 是否真正能到达 `raw_gate`。

### 阶段 F：face + cross attention

- [ ] face map `[B,1792,3,3]`，face tokens `[B,9,128]`。
- [ ] CrossAttention weights（debug 模式）`[B,8,168,9]`。
- [ ] attention/FFN 后 `[B,168,128]`。
- [ ] mean pool 后 `[B,128]`，无 NaN/Inf。

### 阶段 G：scene/FiLM/base UV

- [ ] scene `[B,7] → [B,64]`。
- [ ] fusion concat `[B,192]`。
- [ ] FiLM condition 是 `[B,9]`，不是 `[B,1]`。
- [ ] FiLM gamma/beta 各 `[B,256]` 且初始为 0。
- [ ] fused `[B,128]`，uv normalized `[B,2]`。
- [ ] denormalize(normalize(train_uv)) 能近似还原原始 mm。

### 阶段 H：vertical residual

- [ ] geometry input `[B,2,15,*]`，22D feature `[B,22]`。
- [ ] residual MLP input `[B,167]`。
- [ ] 初始 delta `[B,1]` 为 0；任何时候绝对值不超过 250 mm。
- [ ] `uv_pred[:,0]` 与 `uv_base[:,0]` bit-for-bit 相同。
- [ ] train gate 至少 0.10；eval confidence 低于 0.10 时 delta 回到 0。
- [ ] residual target 使用 detached base v，避免目标随 residual 分支反向漂移。

### 阶段 I：loss/backward/optimizer

- [ ] 逐项打印未加权 loss 和加权贡献，确认总和等于 `L_total`。
- [ ] `L_uv` 的单位是 mm，不是 normalized space。
- [ ] total loss finite；backward 后所有预期 trainable 参数 grad finite。
- [ ] clip 前记录 global grad norm，clip 后不超过 1.0（数值误差除外）。
- [ ] eye ResNet param group lr 是 base group 的 0.05。
- [ ] 一个 epoch 实际 optimizer step=13，global_step 正确累加。
- [ ] scheduler 只在 epoch 末 step 一次。

## 14. 最小 shape 断言清单

建议你在每个模块单独 debug 时优先检查以下不变量：

```python
assert batch["face"].shape[1:] == (3, 160, 160)
assert batch["left_eye"].shape[1:] == (3, 56, 90)
assert batch["right_eye"].shape[1:] == (3, 56, 90)
assert batch["scene_vec"].shape == (B, 7)
assert batch["virtual_camera_pose_table"].shape == (B, 9)
assert batch["eye_pseudo_landmarks_xy"].shape == (B, 2, 15, 2)

assert left_layer2.shape == (B, 128, 14, 23)
assert left_layer3.shape == (B, 256, 7, 12)
assert left_layer4.shape == (B, 512, 4, 6)
assert left_fpn.shape == (B, 128, 7, 12)

assert face_tokens.shape == (B, 9, 128)
assert eye_tokens.shape == (B, 168, 128)
assert landmark_tokens.shape == (B, 30, 128)
assert visual_features.shape == (B, 128)
assert scene_features.shape == (B, 64)
assert fusion_input.shape == (B, 192)
assert fused_features.shape == (B, 128)

assert output["uv"].shape == (B, 2)
assert output["vertical_delta_v_mm"].shape == (B, 1)
assert output["left_eye_keypoint_logits"].shape == (B, 15, 28, 45)
assert output["left_eye_keypoint_mean_xy"].shape == (B, 15, 2)
assert output["left_eye_keypoint_covariance"].shape == (B, 15, 2, 2)
```

最后不要只检查 shape。每个阶段至少同时检查：

```text
dtype、device、isfinite、数值范围、左右眼语义、坐标系、单位、requires_grad、grad 是否到达。
```

这条 direct-UV 路径中最容易造成“shape 正确但语义错误”的三处是：

1. legacy eye crop 与 pseudo-label bbox 是否完全一致；
2. `TableFrame7_C` 是否正确旋转成 `TableFrame7_N`；
3. 模型 `[B,2]` 输出是 normalized UV，而 vertical delta 是 mm，两者只能在 `DirectTableUVLoss` 中完成单位合并。
