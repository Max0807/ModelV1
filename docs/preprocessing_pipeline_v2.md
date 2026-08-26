# ModelV1 versioned preprocessing pipeline v2

Entry point: `scripts/prepare_modelv1_preprocessing.py`.

The pipeline rebuilds the unified dataset table from the current `data_log_*.csv`
and `insightface_coordinates.csv`, regenerates the selected metric depth prior,
audits PnP v2 geometry, creates valid sample subsets, and writes the final
virtual-camera face/eye images and manifest.

## Depth-method contract

- `--depth-method pnp1010`: training consumes the fixed 1010 mm/FLAME-unit prior.
- `--depth-method iris65`: training consumes the 65 mm Iris-IPD prior.
- `--depth-method both`: both priors are produced and both training manifests use
  the same paired sample IDs for a controlled comparison.

All three modes use the verified PnP1010 `rotation_*` and `tvec_*` as the virtual
camera pose and centre. Iris65 changes only the training depth prior.

## PowerShell dataset variable

```powershell
$Dataset = "11"
```

Only the dataset ID is stored in a PowerShell variable. Every other setting is
written explicitly as a command-line option after the script name.

## Inspect commands without writing files

```powershell
& "D:\software\Anaconda\envs\cross_gaze\python.exe" scripts\prepare_modelv1_preprocessing.py `
  --dataset-id $Dataset `
  --depth-method both `
  --source-root "D:\GithubCode\CrossGaze-main\CrossGaze-main\data_collection" `
  --output-root "D:\GithubCode\ModelV1\data\preprocessed_v2\dataset$Dataset" `
  --device cuda `
  --deca-batch-size 8 `
  --ipd-mm 65 `
  --calibration-start 70 `
  --calibration-stop 170 `
  --calibration-index-mode frame_idx `
  --dry-run
```

## Formal PnP1010-only generation

```powershell
& "D:\software\Anaconda\envs\cross_gaze\python.exe" scripts\prepare_modelv1_preprocessing.py `
  --dataset-id $Dataset `
  --depth-method pnp1010 `
  --source-root "D:\GithubCode\CrossGaze-main\CrossGaze-main\data_collection" `
  --output-root "D:\GithubCode\ModelV1\data\preprocessed_v2\dataset$Dataset" `
  --device cuda `
  --deca-batch-size 8
```

## Formal Iris65-only generation

Iris65 still generates PnP1010 geometry because virtual-camera normalization
must use the PnP face origin and head rotation.

```powershell
& "D:\software\Anaconda\envs\cross_gaze\python.exe" scripts\prepare_modelv1_preprocessing.py `
  --dataset-id $Dataset `
  --depth-method iris65 `
  --source-root "D:\GithubCode\CrossGaze-main\CrossGaze-main\data_collection" `
  --output-root "D:\GithubCode\ModelV1\data\preprocessed_v2\dataset$Dataset" `
  --device cuda `
  --deca-batch-size 8 `
  --ipd-mm 65 `
  --calibration-start 70 `
  --calibration-stop 170 `
  --calibration-index-mode frame_idx
```

Add `--estimate-depth-uncertainty` when the model uses `fixed_prior` or
`learned_reweight` depth-distribution training.

## Recommended controlled comparison

```powershell
& "D:\software\Anaconda\envs\cross_gaze\python.exe" scripts\prepare_modelv1_preprocessing.py `
  --dataset-id $Dataset `
  --depth-method both `
  --source-root "D:\GithubCode\CrossGaze-main\CrossGaze-main\data_collection" `
  --output-root "D:\GithubCode\ModelV1\data\preprocessed_v2\dataset$Dataset" `
  --device cuda `
  --deca-batch-size 8 `
  --ipd-mm 65 `
  --calibration-start 70 `
  --calibration-stop 170 `
  --calibration-index-mode frame_idx
```

Use a new `--output-root` when the raw dataset changes. `--overwrite` replaces
derived files but deliberately does not delete unreferenced images left by an
older run.

## Final outputs

```text
data/preprocessed_v2/dataset<ID>/
├── shared/
│   ├── modelv1_dataset<ID>.csv
│   ├── mediapipe_pnp_landmarks_v2_dataset<ID>.csv
│   ├── mediapipe_iris_centres_v2_dataset<ID>.csv
│   └── deca_features_deca_crop_v1_dataset<ID>.npz
├── pnp1010/
│   ├── depth_priors_pnp1010_v2_dataset<ID>.csv
│   ├── pnp_geometry_validation_dataset<ID>.json
│   └── modelv1_dataset<ID>_pnp1010.csv
├── iris65/
│   ├── depth_priors_iris65_pnp_v2_dataset<ID>.csv
│   ├── iris65_pnp_geometry_validation_dataset<ID>.json
│   └── modelv1_dataset<ID>_iris65.csv
├── comparison/
│   └── modelv1_dataset<ID>_common_pnp1010_iris65.csv
├── virtual_camera/<selected-method>/
│   ├── face/
│   ├── left_eye/
│   ├── right_eye/
│   ├── manifest.csv
│   └── report.json
└── manifests/
    ├── training_inputs_pnp1010_dataset<ID>.json
    ├── training_inputs_iris65_dataset<ID>.json
    └── pipeline_<selected-method>_dataset<ID>.json
```

The `training_inputs_*.json` files are the final hand-off contracts for model
configuration. In `both` mode, they differ in `depth_prior_csv` but share the
same `dataset_csv` and `virtual_camera_manifest`.
