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

Then check one training batch:

```powershell
python scripts\check_dataloader.py
```

Default split:

- train: `dataset_dual_rigid_body_3` + `dataset_dual_rigid_body_4`
- val: `dataset_dual_rigid_body_5`
