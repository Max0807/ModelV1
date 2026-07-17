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

The default loader requires and reads `data/processed/deca_features_v1.npz`.
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
- `eye_branch`: encodes left/right eye crops.
- `crop_cam_branch`: embeds the 36D crop/camera vector.
- `scene_branch`: embeds the 25D scene/table vector.
- `fusion_mlp` + `uv_head`: predicts normalized table-local `(u, v)`.

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
random split, trains for 100 epochs, and selects the best checkpoint by
validation EPE in millimeters. Training reads only cached DECA features and eye
crops; it does not load the unused face image tensor.

Set `logging.wandb.entity` and `logging.wandb.mode` in
`configs/modelv1/train_random_80_20_100.yaml`, then run:

```powershell
python scripts\train_modelv1.py
```

Every epoch is written to `outputs/<project>/<run>/train.log` and
`metrics.csv`; W&B receives the training/validation losses, EPE/median EPE and
per-axis MAE in millimeters, learning rate, throughput, and elapsed time. The terminal prints the first,
every tenth, and final epoch. `checkpoints/best.pt` tracks lowest validation
EPE, while `checkpoints/last.pt` supports resuming:

```powershell
python scripts\train_modelv1.py --resume outputs\<project>\<run>\checkpoints\last.pt
```

## DECA Offline Features

The face branch uses DECA's frozen 236-D coarse `E_flame` output. Create the
cache before training:

```powershell
python scripts\cache_deca_features.py --device cuda
```

This writes `data/processed/deca_features_v1.npz`. It stores one float32
`deca_feat` vector per `sample_id`, plus face-image and checkpoint SHA-256
digests and JSON metadata. Re-running the command reuses entries whose sample
id and image digest have not changed.

The checked-in `DECA-master` directory must be a complete DECA checkout and
contain `data/deca_model.tar`; see the official DECA README for its model/data
download instructions. The cache script only loads the `E_flame` encoder, so
it does not run DECA rendering or mesh decoding.

The default loader uses this result automatically. To override its location:

```python
from modelv1.data import build_modelv1_dataloaders

train_loader, val_loader = build_modelv1_dataloaders(
    deca_cache_path="data/processed/deca_features_v1.npz",
    require_deca_features=True,
)
```
