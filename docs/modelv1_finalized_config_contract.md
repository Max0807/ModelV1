# ModelV1 最终训练配置约定

适用入口：`scripts/train_modelv1.py`。当前主配置通过顶层
`defaults: defaults_direct_uv.yaml` 继承稳定约定，只保留仍可能调节的实验项。

## 由默认文件提供

| 参数 | 固定值 | 含义 / 所属模块 |
|---|---|---|
| `data.require_eye_geometry_pseudo_labels` | `true` | 数据完整性：训练样本必须存在 15 点眼部伪标签。 |
| `data.direct_uv_target_frame` | `table_local` | 监督目标：直接回归桌面局部坐标 `(u,v)`。 |
| `data.skip_invalid_virtual_camera_samples` | `true` | 虚拟相机数据过滤：丢弃清单中无效的样本。 |
| `data.split_mode` | `explicit_datasets` | 数据划分：仅接受显式且互斥的训练/验证数据集。 |

默认值文件：`configs/modelv1/defaults_direct_uv.yaml`。

眼图像和人脸图像训练增强分别只暴露命名策略
`data.eye_augmentation` 与 `data.face_augmentation`。当前正式配置均使用
`appearance_mild_v1`，数值固定在 `modelv1/data/augmentation.py`，配置文件
不再逐项覆盖。Face 策略只改变亮度、对比度、gamma、模糊和噪声，不包含
翻转、旋转、裁剪、平移、缩放或其他空间变换，并且只应用于训练集。

当前正式训练配置启用 CUDA BF16 autocast 和 fused AdamW。BF16 路径不使用
GradScaler；若 CUDA 设备不支持 BF16，训练会在启动时给出明确错误。数据加载
使用 2 个 persistent worker，训练和验证迭代器在多个 epoch 间复用工作进程。

## 已固定在代码中的训练契约

| 原参数 | 固定值 | 处理 |
|---|---|---|
| `data.depth_method` | `iris65` | 从训练配置删除；训练端始终读取 Iris65 的 preprocessed-v2 manifest。离线预处理仍保留 PnP/Iris 产物生成与核验能力。 |
| `data.eye_image_source` | `legacy` | 从配置和数据集 API 删除；虚拟相机只替换人脸图，双眼始终读取原始眼部裁剪。 |
| `data.eye_image_size` | `[90,56]` | 从配置和数据集 API 删除；数据加载固定为 90×56。 |
| `data.load_face_image` | `true` | 从配置和数据集 API 删除；最终模型始终加载 160×160 人脸图。 |
| Eye spatial-token topology | 7×12 | 固定使用 ResNet18 small stride-2 stem 与 layer2/3/4 FPN；已删除 4×4 face padding、learned token pooling、binocular self-attention 等淘汰实验开关。 |
| Direct-UV FiLM 条件 | `table_frame7_n` 7D | 删除冗余的 9D 派生特征；FiLM 直接使用虚拟相机坐标系下的 TableFrame7。 |

## 已删除的实验路线

| 参数 / 模块 | 删除原因与结果 |
|---|---|
| `experiment.description` | 不再手写；实验日记从模型配置自动生成摘要。 |
| `data.iris_supervision_csv_paths` | 单虹膜中心 H7 路线已淘汰；CSV 读取、样本字段、单点热图头、MSE 损失、指标与测试一并删除。当前只保留 15 点概率眼部关键点监督。 |
| `data.all_datasets`、`data.val_ratio`、`data.split_seed` | 随机 80/20 和 Dataset-5 划分路线已删除；只保留显式数据集划分。 |
| 离线 DECA 236D/Geometry156 特征分支 | 已不再需要；删除 `deca_cache` 读取与合并、特征表示选择、flat/factorized MLP、模型输入输出字段、缓存生成脚本及相关配置和测试。独立的离线深度先验/DECA-FLAME 几何工具不属于该训练分支。 |
| 旧单点 iris 辅助头 | 位于每眼 ResNet 的 layer2/layer3 特征之后、eye-face CrossAttention 之前；该路线已弃用，热图头、CSV 监督、损失和指标代码已删除。 |
| `configs/modelv1/train_random_80_20_100_v3.yaml` | 对应已删除的陈旧划分/模型实验，配置文件一并删除。 |
| low-DOF eye template、pitch-to-v、vertical-geometry residual | 模型、推理、训练 loss、指标、参数和测试全部删除。 |
| mixture NLL、gaze-direction sign loss、depth-prior KL、depth-correction prior | 所有正式配置均未启用；对应 loss 参数和实现已经删除。gaze–平面相交路径继续使用 UV Huber、UV Gaussian NLL、gaze angular 与 ray-validity loss。 |

## `auto_experiment_identity`

保持 `false` 是合理的：此时 `run_name: null` 会在启动时使用时间戳，W&B project/tags 保持配置中的值。
改成 `true` 也可以；对于带 `dataset_ids` 的配置，程序会根据训练集、验证集、预测模式、固定的 Iris65 契约和深度分布模式自动生成 run name，并重写 W&B project/tags。它不再生成 `description`。
