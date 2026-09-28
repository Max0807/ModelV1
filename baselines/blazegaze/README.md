# Native BlazeGaze baseline

This package is kept independent of `modelv1`. It reproduces the released
BlazeGaze input contract while retaining the TableFrame7 metric pose for audit:

- `image`: float32 `[128, 512, 3]` in `[0, 1]` (HWC by default);
- `head_vector`: float32 `[3]`, derived with the released WebEyeTrack angle
  convention;
- `face_origin_3d`: float32 `[3]` in centimetres, taken from the Stage 2
  iris/eye midpoint proxy;
- optional `pog_norm`: zero-centred normalized PoG label;
- optional `screen_info`: `[height_cm, width_cm]`;
- `sample_weight`: scalar.

For auditability the adapter also returns `metric_rt_mm`, `metric_rt_cm`, and
`metric_rt_flat_mm`. `R` is `R_C_H`; `t` is the head-frame origin expressed in
camera coordinates. The released runtime constructs `face_origin_3d` from the
two reconstructed eye origins, so it is intentionally **not** set to `t_C_H`.

## Install preprocessing dependencies

The exact eye-patch builder requires MediaPipe Tasks and OpenCV:

```powershell
python -m pip install -r baselines/blazegaze/requirements.txt
```

The default face-landmarker path points to the model shipped with the local
WebEyeTrack checkout:

```text
D:\GithubCode\WebEyeTrack-main\python\webeyetrack\model_weights\face_landmarker_v2_with_blendshapes.task
```

Override it with `--face-landmarker-model` when the checkout lives elsewhere.

## Step 1: build exact MediaPipe478 homography eye patches

Run from the repository root:

```powershell
python scripts/build_blazegaze_eye_patches.py `
  --stage3-csv data/preprocessed_table7/dataset11/stage3_virtual_table7/training_table7_dataset11.csv `
  --stage3-csv data/preprocessed_table7/dataset13/stage3_virtual_table7/training_table7_dataset13.csv `
  --output-dir data/preprocessed_blazegaze/table7/eye_patches `
  --manifest data/preprocessed_blazegaze/table7/blazegaze_eye_patches.csv `
  --report data/preprocessed_blazegaze/table7/blazegaze_eye_patches.report.json
```

The implementation follows
`WebEyeTrack/python/webeyetrack/model_based.py::obtain_eyepatch`: it detects the
478-point MediaPipe face mesh, computes the face homography from landmarks
`103, 150, 379, 332`, crops vertically using landmarks `151` and `195`, and
resizes the result to `512x128` RGB. The manifest records the homography,
selected landmark coordinates, crop bounds, condition number, and black-border
fraction for every sample. Use `--overwrite` only when intentionally rebuilding
existing outputs; `--limit N` is useful for a smoke test.

## Step 2: build the joined BlazeGaze manifest

```powershell
python scripts/build_blazegaze_data_manifest.py `
  --stage3-csv data/preprocessed_table7/dataset11/stage3_virtual_table7/training_table7_dataset11.csv `
  --stage3-csv data/preprocessed_table7/dataset13/stage3_virtual_table7/training_table7_dataset13.csv `
  --metric-pose-csv data/preprocessed_table7/dataset11/stage2_metric_head_pose/iris65_metric_head_pose_dataset11.csv `
  --metric-pose-csv data/preprocessed_table7/dataset13/stage2_metric_head_pose/iris65_metric_head_pose_dataset13.csv `
  --eye-patch-manifest data/preprocessed_blazegaze/table7/blazegaze_eye_patches.csv `
  --require-exact-eye-patches `
  --output-csv data/preprocessed_blazegaze/table7/blazegaze_table7.csv
```

`--require-exact-eye-patches` is recommended for BlazeGaze comparison runs. It
fails the build if even one Stage 3 sample lacks a successful exact eye patch,
preventing a silent fallback to the legacy paired-eye crops. Omitting the exact
manifest remains supported only for explicit compatibility experiments; those
rows are labelled `paired_legacy_eye_crops` rather than
`mediapipe478_homography`.

Add `--pog-bounds-mm U_MIN U_MAX V_MIN V_MAX` only when those bounds have been
fixed by the physical table or evaluation protocol. The released loader maps
those bounds to `[0, 1]` and subtracts `0.5`. Estimating bounds from a test split
would leak test information, so the adapter does not invent them.

## Current TableFrame7 output

The generated report records 1,465 successful patches, zero failures, and 478
landmarks for every sample. The joined manifest contains 1,465 strict exact
homography rows. See:

- `data/preprocessed_blazegaze/table7/blazegaze_eye_patches.report.json`
- `data/preprocessed_blazegaze/table7/blazegaze_table7.report.json`

## Step 3: train supervised Native BlazeGaze

The native baseline is separate from the Blaze-style ModelV1 ablations. It
uses exactly one `128x512` eye patch, a 512D released Blaze encoder,
`head_vector[3]`, `face_origin_3d[3]`, and the released `16-16-2` gaze MLP. It
does not use the ModelV1 face/eye branches, TableFrame7 input, or FiLM fusion.
The Stage-1 decoder and all three released objectives are enabled:

```text
loss = 1.0 * weighted_gaze_L2
     + 1.0 * reconstruction_MSE
     + 0.2 * embedding_consistency
```

First confirm the fixed physical or protocol PoG rectangle in millimetres.
Then run a one-batch end-to-end check, replacing the four placeholders:

```powershell
python scripts/train_native_blazegaze.py `
  --config configs/blazegaze/native_supervised_table7.yaml `
  --pog-bounds-mm U_MIN U_MAX V_MIN V_MAX `
  --dry-run
```

Start the supervised comparison after the dry run passes. The epoch count is
read from `training.epochs` in the YAML (currently 80):

```powershell
python scripts/train_native_blazegaze.py `
  --config configs/blazegaze/native_supervised_table7.yaml `
  --pog-bounds-mm U_MIN U_MAX V_MIN V_MAX
```

The same values may instead be written to `data.pog_bounds_mm` in the YAML.
The trainer rejects labels outside the configured rectangle and never infers
bounds from validation data. For orientation only, the current combined rows
span approximately `u=[-501.32, -82.33] mm` and
`v=[-255.81, 317.95] mm`; those observed extrema are not a substitute for a
physical/protocol definition.

The default split trains on dataset 13 and validates on dataset 11. Each run is
written to a new timestamped directory under `outputs/blazegaze_native` with:

- `checkpoints/best.pt` and `checkpoints/last.pt`;
- `metrics.csv` with total loss, all three component losses, and EPE in mm;
- `config.resolved.yaml` and `summary.json`.

For non-dry runs, W&B is enabled by the YAML and logs the train/validation
total loss, gaze L2, reconstruction MSE, embedding consistency, EPE in mm,
learning rate, epoch time, and running best validation values. The default
project is `ModelV1_virtual_camera_cross_camera`, with group
`native_blazegaze_supervised`. Log in once before training if necessary:

```powershell
wandb login
```

Set `logging.wandb.enabled: false` to keep only the local files, or set
`logging.wandb.mode: offline` when the training machine cannot access the
network. A `--dry-run` deliberately does not create a W&B run.

The inference graph contains 155,826 parameters (about 0.156M); the Stage-1
training graph including the discarded decoder contains 564,573. The default
uniform weights match the released MPIIFaceGaze training configuration. To use
the released optional 30x30 inverse-frequency grid, set
`data.sample_weighting: official_inverse_frequency_30x30`; the grid is fitted
on dataset 13 only and then applied to both splits.
