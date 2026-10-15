# FD/FL Prediction and Focus Classification

TabPFN workflows for FD/FL regression and F/UF focus classification from
triangle point distributions. Maintained workflows include native FD/FL
prediction, geometry-held-out FD transfer and labeled F/UF classification.

The October 2026 cleanup removed deprecated runners and forwarding imports.
Use the commands below; see [migration and deletion notes](docs/CODE_CLEANUP_20261015.md).

## Start Here

The redesigned, versioned system is available through `python -m src system`:

```powershell
.\.venv\Scripts\python.exe -m src system demo
.\.venv\Scripts\python.exe -m src system --help
```

`demo` runs an explicitly synthetic audit -> five-fold cascade evaluation ->
model-context bundle -> new-triangle prediction workflow. It verifies execution,
not scientific accuracy. The real-data workflow uses explicit source manifests,
units, target definitions and independent labels. See the
[new system guide](docs/SYSTEM_WORKFLOW.md) and
[architecture design](docs/ML_SYSTEM_REDESIGN.html). Existing commands below
remain available. Fzong composition requires a confirmed physical configuration;
unknown native FD-u and FL are returned separately.

For real native FD-u unseen-geometry validation, run
`python -m src fd-optimize`. This compares frozen candidates using nested
geometry holdouts, without changing production defaults. The June 2026 batch
has four independent shapes, so this benchmark uses four outer folds rather
than five random point folds. See [FD extrapolation guide](docs/FD_EXTRAPOLATION.md).

`python -m src fd-tabpfn-tune` runs the TabPFN-only follow-up with a frozen menu
of ensemble, numeric-feature and target-scaling candidates. See the
[TabPFN FD tuning guide](docs/FD_TABPFN_TUNING.md).

For the updated `2026-09-29` rotation dataset, use `fd-rotation-audit` followed
by `fd-rotation-tune --snapshot <completed-audit-run>`. The workflow verifies
all rotation copies and keeps them in the same geometry fold. See the
[rotation dataset guide](docs/FD_ROTATION_20260929.md).

The subsequent FD transfer study uses `fd-transfer-tune`,
`fd-transfer-ensemble`, `fd-transfer-audit`, `fd-transfer-report` and
`fd-zero-audit`. These preserve the nested geometry validation protocol.
See [transfer results and zero audit](docs/FD_R2_09_AND_ZERO_AUDIT.md).

Run from the repository root with the project environment:

```powershell
.\.venv\Scripts\python.exe -m src doctor
.\.venv\Scripts\python.exe -m src focus-audit
.\.venv\Scripts\python.exe -m src focus-ablate
.\.venv\Scripts\python.exe -m src focus-optimize
```

`python main.py`, `python -m src`, and the installed `fdf` command share one
dispatcher. Use `--help` after any command for its options.

The optimization command rebuilds sample features, runs nested grouped TabPFN
evaluation and saves its report under:

```text
results/focus_inspection/2026-09-04/focus_tabpfn_optimized/runs/<run-id>/
```

Open `gallery.html` or `report.md` for results. The `nested` row in
`metrics.csv` evaluates candidate/threshold selection using held-out geometries.
`oof_predictions.csv` contains one test prediction per sample.
The exact output path is printed; `latest.json` points to the last completed
run. Earlier reports are preserved. `focus-ablate` provides an equal-budget,
three-seed comparison under `focus_ablation/runs/<run-id>/`.

## Structure

```text
src/
  cli.py                       Unified workflow commands
  doctor.py                    Import, dependency and checkpoint checks
  data/                        Workbook schemas, units and preprocessing
  geometry/                    Shared triangle geometry, query grids and features
  physics/                     Explicit dimensional validation and Fzong adapters
  evaluation/                  Shared folds and cross-fitted cascade evaluation
  workflows/                   Audit, evaluation, model bundles and inference
  regression/                  Maintained FD/FL prediction and evaluation
  focus_inspection/
    settings.py                Paths and baseline feature sets
    dataset.py                 Sample keys, source hashes, geometry and labels
    focus_boundary_feature.py  Smoothing and spatial feature calculations
    modeling.py                TabPFN fitting and probability prediction
    evaluation.py              Grouped splits and classification metrics
    train_focus_tabpfn.py       Original baseline reproduction
    optimize_focus.py          Nested feature/model/threshold selection
    ablate_focus.py            Equal-budget feature comparison across fixed seeds
    quality.py                 Units, feature settings and input checks
    geometry_integrity.py      Physical grouping and geometry consistency
    calibrate_focus.py         Nested grouped probability calibration
    audit_focus.py             Model-free input and label audit
    cache.py                   Content-validated inference cache for resume
    prediction_safety.py       Review flags and training identity checks
    predict_focus.py           Saved-recipe prediction on new batches
    plotting.py                Shared scientific plotting style
    reporting.py               Baseline figures and gallery
    optimization_report.py     Nested benchmark report
utils/                         Shared metrics, paths, transforms, provenance
tests/                         Regression, contract and leakage tests
docs/                          Data contracts, architecture and user guides
web-demo/                      Existing application and prediction service
data/, models/, results/       Local research data, checkpoints and outputs
```

Data processing lives in `src/data`; prediction and interpolation workflows live
in `src/regression`. Removed module paths no longer forward to these packages.
Source data, checkpoint files and existing experiment results retain their paths.

## Environment

Python 3.12 or newer is required. The lockfile pins TabPFN 8.0.3 and PyTorch
2.11.0, with CUDA 12.8 wheels for Windows/Linux.

```powershell
uv sync --locked
python -m src doctor
```

For the existing environment, install just this repository without changing
installed model dependencies:

```powershell
python -m pip install -e . --no-deps --no-build-isolation
```

The distribution name is `fdf-predictions`. It does not shadow the external
`tabpfn` package. The current environment and CUDA availability are reported by
`doctor`. Pass `--device cpu` to focus evaluation/prediction for CPU execution.

## Focus Workflow

One sample is **Excel file x R x Re**. The current labels are binary:
F=focused and UF=unfocused, read from `三角形聚焦汇总.xlsx`.
Each raw workbook provides `x,z,R,re,Fzong2` (`Fzong` is also supported) and
the geometry columns listed in the [focus guide](src/focus_inspection/README.md).

```powershell
python -m src focus-plots
python -m src focus-features
python -m src focus-baseline
python -m src focus-optimize
python -m src focus-ablate
```

The feature table includes the smoothed boundary width, its midpoint x,
normalized location/gap and row-wise spatial descriptors. Add `--no-plots` to
`focus-features` when only the numerical table is needed.

The optimized workflow leaves out one complete geometry at a time. It selects
features and probability thresholds only within inner training folds. Source
hashes detect stale feature tables after a workbook is replaced.

For a new batch:

```powershell
python -m src focus-predict `
  --recipe results/focus_inspection/2026-09-04/focus_tabpfn_optimized/selected_recipe.json `
  --raw-dir data/rawdata/NEW_BATCH `
  --out-dir results/focus_predictions/NEW_BATCH
```

The saved recipe references the exact training feature snapshot and checkpoint.
Predictions include probabilities, F/UF labels, REVIEW decisions and
content-aware training-overlap flags. For new runs, pass a recipe from the
exact completed `runs/<run-id>/` directory. Legacy recipes are explicitly flagged.
See [reliability controls and run management](docs/FOCUS_RELIABILITY.md) for
input units, validation, resume and human review.
See the [evaluation guide](docs/REPOSITORY_OPTIMIZATION.md) for statistical
limits, feature definitions and all output files.

## FD/FL Workflows

```powershell
python -m src data show
python -m src data run --batch 2026-04-26
python -m src regression-evaluate --help
python -m src fd-to-fl --help
python -m src fl-at-fd --help
python -m src fd-from-fl --help
python -m src interpolate --help
```

The [FD/FL closed-loop guide](docs/FD_FL_CLOSED_LOOP.md) describes native
workbook contracts and physical units. Prepared-fold AutoTabPFN runners have
been removed; use `regression-evaluate` or `system evaluate` for grouped tests.

```powershell
python -m src interpolate-raw-fl --help
python -m src interpolate-raw-fl-tabpfn --help
python -m src interpolate-fly --help
python -m src fl-lines --help
```

`interpolate-fly` uses TabPFN for added FLy values; the separate
`interpolate-fly-linear` command is an arithmetic reference, not the default.

## Validation and References

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src utils tests
uv lock --check --offline
```

- [Architecture and evaluation](docs/REPOSITORY_OPTIMIZATION.md)
- [Reliability controls and controlled experiments](docs/FOCUS_RELIABILITY.md)
- [Probability calibration and physical grouping](docs/FOCUS_CALIBRATION.md)
- [Unique-sample error review and geometry uncertainty](docs/FOCUS_REVIEW.md)
- [Batch prediction, statistics and visualization](docs/FOCUS_PREDICTION.md)
- [Native FD-u/FL prediction and service safeguards](docs/FD_FL_PREDICTION_IMPROVEMENTS.md)
- [Known data problems, original values retained](docs/FOCUS_DATA_ISSUES_2026-09-08.md)
- [Raw data standard](docs/RAWDATA_STANDARD.md)
- [Application file interface](docs/APPLICATION_FILE_INTERFACE.md)
- [Current user guide](docs/USER_GUIDE.md)
- [Code cleanup and command migration](docs/CODE_CLEANUP_20261015.md)
- [Web service](web-demo/README.md)
- [Publishing checklist](docs/GITHUB_UPLOAD.md)

The current F/UF workflow uses the binary learning signal supplied for the
September batch. Removing historical classifiers does not add WF/F/UF
three-class capability; that requires independent labels for all three classes.

Large datasets, generated results, checkpoints and archives are excluded from
Git. No license has been selected; add a license before public reuse.
