# New User Guide

This guide explains how a new user should set up and run this project after it has been migrated to another Windows computer.

The project contains Python code, local model files, optional data batches, and a static web demo. The GitHub repository does not contain large raw data, model checkpoints, or historical results.

## 1. What You Need

For a usable test environment, prepare these items:

```text
FDFLPredictions/                       project code
models/offline_tabpfn_2_5/             offline TabPFN checkpoint
data/                                  only the batches needed for the task
FDFL_offline_kit/wheelhouse/           offline Python packages, recommended on slow networks
FDFL_offline_kit/requirements-freeze.txt
```

The required offline model directory is:

```text
models/offline_tabpfn_2_5/
  config.json
  tabpfn-v2.5-regressor-v2.5_default.ckpt
```

Do not copy `.venv/` unless you are doing a temporary same-machine-style transfer. A fresh virtual environment is more reliable.

## 2. Recommended Folder Layout

Use a short ASCII path when possible:

```text
D:\2026\tab\FDFLPredictions
E:\TEMP\FDFL_offline_kit
```

The project should look like:

```text
D:\2026\tab\FDFLPredictions
  src/
  utils/
  docs/
  web-demo/
  tabpfn-extensions-main/
  models/
    offline_tabpfn_2_5/
  data/
```

## 3. Python Version

Use Python 3.12.

Python 3.9 is too old for the current dependency set. Python 3.14 may run `main.py`, but the prepared offline wheels are built for Python 3.12, so 3.14 is not recommended.

Check Python 3.12:

```powershell
py -3.12 --version
```

Create the project virtual environment:

```powershell
cd D:\2026\tab\FDFLPredictions
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe --version
```

Always run project commands through:

```powershell
.\.venv\Scripts\python.exe
```

It is acceptable if plain `python --version` still shows Python 3.9. What matters is that `.\.venv\Scripts\python.exe --version` shows Python 3.12.

## 4. Install Dependencies Offline

On a slow network, do not use `uv sync` or online `pip install`.

Install from the offline wheelhouse:

```powershell
cd D:\2026\tab\FDFLPredictions
.\.venv\Scripts\python.exe -m pip install --no-index --find-links E:\TEMP\FDFL_offline_kit\wheelhouse -r E:\TEMP\FDFL_offline_kit\requirements-freeze.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .\tabpfn-extensions-main
```

Important: the current `pyproject.toml` uses `name = "tabpfn"`, which collides with the real third-party `tabpfn` package. Until the project is renamed, avoid running:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Run project modules from the repository root instead, for example:

```powershell
.\.venv\Scripts\python.exe -m src.experiments.predict_fl_positions_from_matching_fd_tabpfn
```

If `tabpfn` was accidentally replaced by the local project package, reinstall the real package:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y tabpfn
.\.venv\Scripts\python.exe -m pip install --no-index --find-links E:\TEMP\FDFL_offline_kit\wheelhouse tabpfn==8.0.3 tabpfn-common-utils==0.2.21
```

## 5. Verify the Environment

Run these checks from the project root:

```powershell
cd D:\2026\tab\FDFLPredictions

.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.venv\Scripts\python.exe -c "import numpy, pandas, sklearn, torch, tabpfn; print('imports ok'); print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('tabpfn', tabpfn.__file__)"
.\.venv\Scripts\python.exe -m compileall src utils tests
.\.venv\Scripts\python.exe main.py
```

For a GPU workstation, confirm CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If CUDA is false, the code can still run on CPU, but TabPFN jobs will be much slower.

## 6. Run the Static Web Demo

The demo does not require model files. It is a static front-end surrogate, not the real TabPFN backend.

Open directly:

```text
web-demo/index.html
```

Or run a local server:

```powershell
cd D:\2026\tab\FDFLPredictions
.\.venv\Scripts\python.exe -m http.server 8000 -d web-demo
```

Then open:

```text
http://localhost:8000
```

## 7. Run the Existing FD-to-FL Prediction Pipeline

This is the existing pipeline:

```text
matching FD file trains TabPFN
-> predict u_pred at original FL positions
-> interpolate fl and u_pred inside each FL file
```

Required data:

```text
data/rawdata/2026-06-20/FD/
data/rawdata/2026-06-20/FL/
models/offline_tabpfn_2_5/
```

First check the folders:

```powershell
dir data\rawdata\2026-06-20
dir data\rawdata\2026-06-20\FD
dir data\rawdata\2026-06-20\FL
dir models\offline_tabpfn_2_5
```

### 7.1 Smoke Test One Pair

Use this first to confirm the machine is configured correctly. Replace `750-320-10` with a key that exists in the batch if needed.

```powershell
.\.venv\Scripts\python.exe -m src.experiments.predict_fl_positions_from_matching_fd_tabpfn `
  --batch-dir data\rawdata\2026-06-20 `
  --output-dir results\fd_to_matching_fl_2026-06-20-test `
  --device cuda `
  --n-estimators 1 `
  --max-train-rows 3000 `
  --chunk-size 5000 `
  --only 750-320-10
```

If the machine has no CUDA GPU, use:

```powershell
--device cpu
```

### 7.2 Full Recommended GPU Run

For a strong GPU machine such as an RTX A4000:

```powershell
.\.venv\Scripts\python.exe -m src.experiments.predict_fl_positions_from_matching_fd_tabpfn `
  --batch-dir data\rawdata\2026-06-20 `
  --output-dir results\fd_to_matching_fl_2026-06-20 `
  --device cuda `
  --n-estimators 1 `
  --max-train-rows 10000 `
  --chunk-size 10000
```

If the GPU runs out of memory, reduce `--chunk-size` to `5000` or `2000`.

Expected outputs:

```text
results/fd_to_matching_fl_2026-06-20/summary.csv
results/fd_to_matching_fl_2026-06-20/run_manifest.json
results/fd_to_matching_fl_2026-06-20/*-FL-with-u-pred.xlsx
```

### 7.3 Interpolate Predicted FL Files

After the first step succeeds:

```powershell
.\.venv\Scripts\python.exe -m src.experiments.interpolate_predicted_fl_files `
  --input-dir results\fd_to_matching_fl_2026-06-20 `
  --output-dir results\fd_to_matching_fl_2026-06-20-interpolated
```

Expected outputs:

```text
results/fd_to_matching_fl_2026-06-20-interpolated/summary.csv
results/fd_to_matching_fl_2026-06-20-interpolated/run_manifest.json
results/fd_to_matching_fl_2026-06-20-interpolated/*-FL-with-u-pred-interpolated.xlsx
```

## 8. Run `src.inference`

`src.inference` does not read raw Excel files directly. It expects a pre-split fold directory:

```text
data/triangle_groups_fixed_v2_focus/combined/
  <shape>/
    <R>/
      <Re>/
        fold1.csv
        fold2.csv
        fold3.csv
        fold4.csv
        fold5.csv
```

Only run it if this directory exists:

```powershell
dir data\triangle_groups_fixed_v2_focus\combined
```

Run:

```powershell
.\.venv\Scripts\python.exe -m src.inference --mode fd
.\.venv\Scripts\python.exe -m src.inference --mode fl
.\.venv\Scripts\python.exe -m src.inference --mode focus
```

Do not start with `--mode focus` on a slow machine. It is the heaviest path.

## 9. Data and Result Policy

GitHub contains code and lightweight metadata. Large or private files should remain outside Git:

```text
data/rawdata/
data/interim/
data/processed/
models/
results/
outputs/
dataAnalysis/
.venv/
```

When giving this project to a new user, send:

```text
project code
offline wheelhouse
requirements-freeze.txt
models/offline_tabpfn_2_5
only the needed data batch
this guide
```

## 10. Common Problems

### `python --version` shows Python 3.9

This is acceptable if the project virtual environment is 3.12:

```powershell
.\.venv\Scripts\python.exe --version
```

Use `.\.venv\Scripts\python.exe` for all project commands.

### `No module named tabpfn`

Install the real package from the offline wheelhouse:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links E:\TEMP\FDFL_offline_kit\wheelhouse tabpfn==8.0.3 tabpfn-common-utils==0.2.21
```

### `pip show tabpfn` says version `0.1.0`

That is the local project package name colliding with the real TabPFN dependency. Reinstall real TabPFN and avoid `pip install -e .` until the project package is renamed.

### CUDA is false

Check that the machine has an NVIDIA GPU and compatible driver. CPU execution still works, but reduce workload:

```powershell
--device cpu --max-train-rows 3000 --chunk-size 500
```

### `--only` key is unknown

The key is derived from matched FD/FL filenames. Run without `--only`, or inspect the error message for valid keys.

### Excel files are missing

The pipeline cannot run from GitHub alone. Copy the required data batch, for example:

```text
data/rawdata/2026-06-20/FD/
data/rawdata/2026-06-20/FL/
```

