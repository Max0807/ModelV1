# ModelV1

Basic project skeleton for model experiments.

## Debug

1. Open this folder in VS Code.
2. Install the recommended Python extensions when prompted.
3. Copy `.env.example` to `.env` if local environment variables are needed.
4. Open a Python file and run `Python: Current File` from the Run and Debug panel.

## Data Preparation

Build the unified ModelV1 CSV from the CrossGaze collection:

```powershell
python scripts\build_modelv1_dataset.py
```

Default outputs:

- `data/processed/modelv1_dataset.csv`
- `data/processed/modelv1_dataset_report.json`

The script uses only the Python standard library by default. If `numpy` is installed, add `--write-npz` to also create `modelv1_dataset.npz`.

## DataLoader

Install the runtime dependencies first:

```powershell
pip install -r requirements.txt
```

Then check one training batch, including frozen DECA features and normalized UV targets:

```powershell
python scripts\check_dataloader.py
```

The default loader requires and reads
`data/processed/deca_features_deca_crop_v1.npz`.
Use `deca_cache_path` only when intentionally selecting a different cache.

The split strategy is selected with `split_mode`:

- `dataset_5` (default): train on `dataset_dual_rigid_body_3` + `dataset_dual_rigid_body_4`; use `dataset_dual_rigid_body_5` as validation/test.
- `random_80_20`: merge datasets 3, 4, and 5, then split them into train/validation with a deterministic 4:1 ratio.

Examples:

```python
from modelv1.data import build_modelv1_dataloaders

train_loader, val_loader = build_modelv1_dataloaders(
    split_mode="random_80_20",
    val_ratio=0.2,
    split_seed=42,
)
```

The smoke-test script supports the same selection:

```powershell
python scripts\check_dataloader.py --split-mode random_80_20
python scripts\check_dataloader.py --split-mode dataset_5
```

## Data Validation Notebook

Open [validate_modelv1_dataset.ipynb](notebooks/validate_modelv1_dataset.ipynb) to randomly inspect one sample, including original-image bboxes, resized crops, key vectors, and the ground-truth table-local gaze point.

## Model

The V1 model is a multi-branch PyTorch regressor:

- `face_branch`: consumes frozen/offline DECA features from `deca_feat`.
- `face_image_encoder` (V3 optional): extracts a 128D appearance feature from
  the RGB 224x224 face crop.
- `eye_branch`: encodes left/right eye crops with a configurable image backbone.
- `visual_fusion` (V3 optional): fuses the RGB face feature and binocular eye
  feature before they are concatenated with structured branches.
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
uv_pred = model(batch)  # batch must include deca_feat, left_eye, right_eye, crop_cam_vec, scene_vec
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

### V3 RGB face encoder and visual fusion

V3 retains the V2.2 DECA, TableFrame, binocular geometry, and quality-gate
branches. It additionally computes:

```text
face [B,3,224,224] -> pretrained ResNet18 -> f_face_image [B,128]
left/right eye     -> shared ResNet18     -> f_eye        [B,128]
concat(f_face_image, f_eye) -> MLP        -> f_visual     [B,128]
```

The final fusion consumes `DECA feature + f_visual + scene + gated eye
geometry`; it does not append the RGB face and eye embeddings independently.
Enable it with:

```yaml
data:
  load_face_image: true

model:
  use_face_image: true
  face_image_embedding_dim: 128
  face_image_backbone: resnet18
  face_image_backbone_weights: DEFAULT
  face_image_freeze_until: layer3
  visual_embedding_dim: 128
  visual_fusion_hidden_dims: [256]
```

`face_image_freeze_until: layer3` freezes the pretrained stem and layers 1-3,
including their BatchNorm statistics, while leaving layer4 and the projection
head trainable.

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

The default training configuration uses datasets 3, 4, and 5 in a fixed 80/20
random split, trains for 300 epochs, and selects the best checkpoint by
validation EPE in millimeters. Training reads only cached DECA features and eye
crops; it does not load the unused face image tensor.

All W&B naming and routing settings live in the selected config file:

```yaml
experiment:
  run_name: null  # null generates a timestamp; set a string to name the run
  description: "Describe the hypothesis or model change for this run."
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

```powershell
python scripts\train_modelv1.py --resume outputs\<project>\<run>\checkpoints\last.pt
```

## DECA Offline Features

The face branch uses DECA's frozen 236-D coarse `E_flame` output. By default,
DECA reads `source_image_path`, forms the official-style square bbox crop with
a downward 0.12 bbox-size center offset and a 1.25 scale, pads outside pixels
with black, and resizes to `224x224`. Create the default cache before training:

```powershell
python scripts\cache_deca_features.py --device cuda
```

This writes `data/processed/deca_features_deca_crop_v1.npz`. It stores one
float32 `deca_feat` vector per `sample_id`, plus face-image and checkpoint
SHA-256 digests and JSON metadata. Re-running the command reuses entries whose
sample id and image digest have not changed.

```powershell
python scripts\cache_deca_features.py --verify-cache
python scripts\train_modelv1.py --config configs\modelv1\train_random_80_20_100_deca_crop.yaml
```

`data.deca_face_preprocess` and `data.deca_crop_scale` are recorded in the
experiment configuration and checked against cache metadata before training.
The ModelV1 network reads the resulting cached DECA feature, so this crop is
performed once during cache generation rather than repeatedly in every epoch.

The checked-in `DECA-master` directory must be a complete DECA checkout and
contain `data/deca_model.tar`; see the official DECA README for its model/data
download instructions. The cache script only loads the `E_flame` encoder, so
it does not run DECA rendering or mesh decoding.

The default loader uses this result automatically. To override its location:

```python
from modelv1.data import build_modelv1_dataloaders

train_loader, val_loader = build_modelv1_dataloaders(
    deca_cache_path="data/processed/deca_features_deca_crop_v1.npz",
    require_deca_features=True,
)
```
