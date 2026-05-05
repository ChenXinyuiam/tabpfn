# TabPFN FD/FL Prediction Project

This repository contains a TabPFN-based workflow for tabular regression and focus classification experiments. It includes data preprocessing scripts, feature engineering utilities, cross-validation training, inference, and analysis tools for FD/FL prediction.

## Project Structure

```text
.
├── main.py                  # Prints the main workflow commands
├── pyproject.toml           # Python project metadata and dependencies
├── src/
│   ├── config.py            # Central paths, column names, and experiment settings
│   ├── train.py             # TabPFN FD/FL training and cross-validation entry points
│   ├── inference.py         # FD, FL, and focus prediction entry points
│   ├── data/                # Cleaning, scaling, feature calculation, and generation scripts
│   ├── split/               # Triangle split and fold-generation scripts
│   ├── experiments/         # Experiment-specific runners
│   └── analysis/            # Plotting and feature-analysis scripts
├── utils/                   # Shared metrics, plotting, feature, path, and TabPFN helpers
├── templates/               # Manifest templates
└── data/                    # Local data workspace, ignored except lightweight docs/manifests
```

Large local artifacts such as raw data, processed data, model checkpoints, trained `.pkl` files, and experiment outputs are intentionally excluded from Git. Keep those files locally or publish them separately through GitHub Releases, cloud storage, or another data registry.

## Requirements

- Python 3.12+
- `uv` is recommended because this repository includes `uv.lock`

Install dependencies:

```bash
uv sync
```

Or with pip:

```bash
python -m pip install -e .
```

## Usage

Print the available workflow commands:

```bash
python main.py
```

Run the data pipeline in order:

```bash
python -m src.data.clean --batch 2026-04-26
python -m src.data.Feature_scale --batch 2026-04-26
python -m src.data.calculate --batch 2026-04-26
python -m src.data.Reverse_FL --batch 2026-04-26
```

If `--batch` is omitted, scripts that support it use the latest batch directory.

Train FD/FL models:

```bash
python -m src.train --target FD
python -m src.train --target FL
python -m src.train --target both
```

Run inference/evaluation:

```bash
python -m src.inference --mode focus
python -m src.inference --mode fd
python -m src.inference --mode fl
```

Run exploratory data analysis:

```bash
python -m src.data.explore --dir data/processedv2 --verbose
```

## Data And Models

Expected local directories are configured in `src/config.py`, including:

- `data/rawdata/`
- `data/cleaned/`
- `data/scaled/`
- `data/processed/`
- `data/processedv2/`
- `models/`
- `results/`
- `outputs/`

These folders may contain large or private research artifacts, so they are ignored by default. Add only small documentation files, manifests, or reproducible examples to Git.

## GitHub Upload Notes

Before publishing, review:

- `.gitignore` to confirm large data/model files are excluded.
- `docs/GITHUB_UPLOAD.md` for the exact upload checklist.
- `git status --short` to confirm the staged files are intentional.

## License

No license has been selected yet. Add a `LICENSE` file before making the repository public if others should be allowed to reuse the code.
