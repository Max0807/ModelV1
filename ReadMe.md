# ModelV1

Basic project skeleton for model experiments.

## Debug

1. Open this folder in VS Code.
2. Install the recommended Python extensions when prompted.
3. Copy `.env.example` to `.env` if local environment variables are needed.
4. Open a Python file and run `Python: Current File` from the Run and Debug panel.

To debug how raw collection data becomes the virtual face image and
`table_frame7_n[7]`, use the `Table7 00`–`Table7 05` entries in
`.vscode/launch.json` and follow
[`docs/table_frame7_stage1_3_debug_guide.md`](docs/table_frame7_stage1_3_debug_guide.md).
The separate `modelv1_pre_model_debug_guide.md` covers the later DataLoader-to-model boundary.

## Data Preparation

Build the unified ModelV1 CSV from the CrossGaze collection:

```powershell
python scripts\build_modelv1_dataset.py
```

Default outputs:

- `data/processed/modelv1_dataset.csv`
- `data/processed/modelv1_dataset_report.json`

The script uses only the Python standard library by default. If `numpy` is installed, add `--write-npz` to also create `modelv1_dataset.npz`.

### 清理低质量图像

`scripts/delete_low_quality_images.py` 会依据帧号，同时清理原图、人脸图、左眼图和右眼图。先在脚本顶部修改 `DATASET` 和 `INDICES`；索引支持单个数字、范围，以及二者混用。默认仅预览；确认输出无误后再加 `--delete` 真正删除：

```powershell
# 在脚本中设置：DATASET = "3"、INDICES = "1-5,8,10-12"
# 然后预览对应的四类图像
python scripts\delete_low_quality_images.py

# 确认预览后执行删除
python scripts\delete_low_quality_images.py --delete
```

`DATASET` 也可写完整目录名，例如 `dataset_dual_rigid_body_4`。若数据集根目录不在默认位置，则修改脚本顶部的 `DEFAULT_SOURCE_ROOT`。

## DataLoader

Install the runtime dependencies first:

```powershell
pip install -r requirements.txt
```

Then check one training batch and normalized UV targets:

```powershell
python scripts\check_dataloader.py
```

Training uses an explicit, disjoint session/camera holdout selected by
`train_datasets` and `val_datasets`.

Examples:

```python
from modelv1.data import build_modelv1_dataloaders

train_loader, val_loader = build_modelv1_dataloaders(
    split_mode="explicit_datasets",
    train_datasets=("3",),
    val_datasets=("4",),
)
```

The smoke-test script supports the same selection:

```powershell
python scripts\check_dataloader.py --split-mode explicit_datasets --train-datasets 3 --val-datasets 4
```

## Data Validation Notebook

Open [validate_modelv1_dataset.ipynb](notebooks/validate_modelv1_dataset.ipynb) to randomly inspect one sample, including original-image bboxes, resized crops, key vectors, and the ground-truth table-local gaze point.

## Model

The V1 model is a multi-branch PyTorch regressor:

- `visual_encoder` (V3 optional): extracts a 3x3 face feature map with a
  VGGFace2-pretrained Inception-ResNet V1 and two 2x2 eye feature maps with a
  shared ResNet-18, then fuses them with eye-to-face cross-attention.
- `eye_branch`: encodes left/right eye crops when the RGB face branch is disabled.
- `crop_cam_branch`: embeds the 36D crop/camera vector.
- `scene_branch`: embeds the 25D scene/table vector.
- `fusion_mlp` + `uv_head`: predicts normalized table-local `(u, v)`.

The eye backbone is selected in the training YAML:

```yaml
model:
  eye_backbone: resnet18  # cnn, resnet18, resnet18_3x3, resnet34, resnet50, resnet101, resnet152
  eye_backbone_weights: null  # DEFAULT enables ImageNet weights for torchvision ResNets only
```

The torchvision ResNets can optionally use ImageNet weights when
`eye_backbone_weights: DEFAULT` and `data.normalize_images: true`.
`resnet18_3x3` uses a small-image stem (3x3, stride 1, no max-pool) for the
60x36 eye crops; with pretrained weights, its new first convolution is initialized
from the center of the original 7x7 filters. Use `cnn` to select the project's
compact custom eye encoder.

Smoke-test the network shape with synthetic inputs:

```powershell
python scripts\check_model.py
```

Minimal use:

```python
from modelv1 import ModelV1

model = ModelV1()
uv_pred = model(batch)
```

### Optional V2.1/V2.2 binocular geometry

V2.1 can append a normalized 6D binocular camera-geometry vector directly to
the four existing branch embeddings. V2.2 optionally gates that vector with
offline PnP quality:

```yaml
data:
  depth_prior_csv_path: data/processed/depth_priors_deca_crop_v1.csv

model:
  scene_dim: 6
  scene_representation: orientation6d
  eye_geometry_dim: 6
  use_crop_cam: true
  use_eye_geometry: true
  eye_geometry_gate_mode: learned_residual
```

The DataLoader joins the depth-prior table by `sample_id`, derives
`[cx/cz, cy/cz, log(cz), (right-left)/cz]`, and fits its z-score normalization
on training samples only. The three gate modes are:

- `none`: exact V2.1 direct concatenation; fusion grows from 384D to 390D.
- `fixed`: use `clip(pnp_confidence ** (1/3), min, max)` as a non-trainable
  scalar gate.
- `learned_residual`: use a 4D normalized PnP-quality vector to learn a bounded
  logit-space correction around the fixed gate.

Gated modes append both the gated 6D vector and its scalar gate, producing a
391D fusion input. The gate affects only binocular geometry; the four visual
and metadata branches remain unchanged. Invalid priors are represented by a
zero geometry vector, zero gate, and `eye_geometry_valid_mask=0`.

Scene ablations support three representations:

- `full25`: the original 25D scene vector.
- `table_frame7`: `[e1_c(3), e2_c(3), d_c / 1000]`, combining table
  orientation with signed plane distance in meters.
- `orientation6d`: `[e1_c(3), e2_c(3)]`, using only the continuous 6D table
  orientation in camera coordinates. The third axis is recoverable by cross
  product.

The scene embedding width remains 64D, so switching representations changes
only the physical input to the scene branch.

Use the ready configuration or smoke-test the individual paths:

```powershell
python scripts\train_modelv1.py --config configs\modelv1\train_random_80_20_100_v2_1.yaml
python scripts\train_modelv1.py --config configs\modelv1\train_random_80_20_100_v2_2.yaml
python scripts\train_modelv1.py --config configs\modelv1\train_random_80_20_100_v3.yaml
python scripts\check_model.py --use-eye-geometry --eye-geometry-gate-mode none
python scripts\check_model.py --eye-geometry-gate-mode fixed
python scripts\check_model.py --eye-geometry-gate-mode learned_residual
python scripts\check_dataloader.py --split-mode random_80_20 --eye-geometry-gate-mode learned_residual
python scripts\check_dataloader.py --split-mode random_80_20 --scene-representation table_frame7
python scripts\check_dataloader.py --split-mode random_80_20 --scene-representation orientation6d
```

### V3 RGB face encoder and cross-attention fusion

The RGB face/eye path computes:

```text
face [B,3,160,160] -> Inception-ResNet V1 -> [B,1792,3,3] -> 9x128 K/V tokens
each eye [B,3,36,60] -> shared ResNet18   -> [B,512,2,2]  -> 4x128 Q tokens
CrossAttention(Q=8 eye tokens, K/V=9 face tokens) -> mean pool -> f_visual [B,128]
```

The final fusion consumes `f_visual + scene` plus any enabled geometry
branches; it does not append separate global RGB face and eye embeddings.
Enable it with:

```yaml
model:
  use_face_image: true
  freeze_face_image_backbone: false
  visual_embedding_dim: 128
  visual_attention_dim: 128
  visual_attention_heads: 8
  visual_attention_ffn_dim: 256
  visual_attention_dropout: 0.1
  eye_backbone: resnet18
  eye_backbone_weights: DEFAULT
  share_eye_encoder: true
```

The face backbone is always initialized with VGGFace2 weights. Set
`freeze_face_image_backbone: true` to freeze the complete Inception-ResNet
backbone (including BatchNorm statistics) while training the 128D projection
and cross-attention layers. Face images are resized to 160x160 and
fixed-standardized to `[-1, 1]`;
eye images keep their existing ImageNet normalization.

## UV Target Normalization

`uv_gt` always remains the physical table-local target in millimeters. The
DataLoader additionally returns `uv_target`, a per-axis z-score target. Its
mean and standard deviation are fitted on training samples only, then shared
with validation samples to prevent validation-set leakage.

The UV head predicts this normalized space. Retrieve the normalizer from the
training dataset to convert a prediction back to millimeters and to build the
per-axis normalized Smooth L1 loss whose robust transition remains 30 mm:

```python
from modelv1 import UVRegressionLoss
from modelv1.data import build_modelv1_dataloaders, get_uv_target_normalizer

train_loader, val_loader = build_modelv1_dataloaders()
normalizer = get_uv_target_normalizer(train_loader.dataset)
criterion = UVRegressionLoss(normalizer)

uv_pred_normalized = model(batch)
loss = criterion(uv_pred_normalized, batch["uv_target"])
uv_pred_mm = normalizer.denormalize(uv_pred_normalized)
```

Smoke-test the complete non-training path:

```powershell
python scripts\check_loss.py
```

## Training

The finalized training configuration uses explicit training/validation datasets
and selects the best checkpoint by validation EPE in millimeters. Training uses
the RGB face and eye inputs directly and has no DECA feature-cache dependency.

All W&B naming and routing settings live in the selected config file:

```yaml
experiment:
  run_name: null  # null generates a timestamp; set a string to name the run
  diary_enabled: true
  diary_path: experiments/experiment_diary.md

logging:
  wandb:
    enabled: true
    project: My_ModelV1_Project
    entity: null
    mode: online
    tags: [ModelV1, experiment_name]
```

The training code does not provide a hard-coded project-name fallback. Change
`logging.wandb.project` in the config, then run:

```powershell
python scripts\train_modelv1.py
```

After each successfully completed non-dry-run training, the script appends the
experiment description, key data/model/optimizer settings, best validation
metrics, final metrics, and comparison with the previous recorded run to
`experiments/experiment_diary.md`. Runs are keyed by output directory so resume
or repeated completion does not create duplicate entries. Recording starts
with the next experiment; existing runs are not backfilled.

Every epoch is written to `outputs/<project>/<run>/train.log` and
`metrics.csv`; W&B receives the training/validation losses, EPE/median EPE and
per-axis MAE in millimeters, learning rate, throughput, and elapsed time. The terminal prints the first,
every tenth, and final epoch. `checkpoints/best.pt` tracks lowest validation
EPE, while `checkpoints/last.pt` supports resuming:

When `training.ema.enabled=true`, EMA averaging starts after warmup. Validation and best-checkpoint
selection then use EMA weights. Checkpoints keep both the raw model (for exact
optimizer resume) and EMA state; analysis prefers EMA weights when available.

```powershell
python scripts\train_modelv1.py --resume outputs\<project>\<run>\checkpoints\last.pt
```

## DECA Offline Features

The former offline 236-D DECA `E_flame` cache and its train-time MLP branch
have been removed. Current preprocessing and training do not generate or load
`deca_features_*.npz`. DECA/FLAME utilities that support separate offline
geometry or depth-prior research are independent of this removed model branch.
