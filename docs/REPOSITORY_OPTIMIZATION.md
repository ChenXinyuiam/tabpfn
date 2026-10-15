# Repository structure and focus evaluation

## Ownership

| Area | Responsibility | Main entry |
|---|---|---|
| `src/data` | Workbook schemas, units and preprocessing | `python -m src data` |
| `src/regression` | FD/FL position prediction and grouped evaluation | `python -m src regression-evaluate` |
| `src/focus_inspection` | F/UF sample features, evaluation and prediction | `python -m src focus-optimize` |
| `utils` | Shared small helpers | Imported by maintained packages |
| `web-demo` | Existing web service and application | Its own README |

Data processing is owned by `src/data`; FD/FL prediction and interpolation are
owned by `src/regression`. The former experiment namespace and compatibility
wrappers were removed in October 2026. See [migration notes](CODE_CLEANUP_20261015.md).
The source datasets, model checkpoints, historical results and existing web
application are kept in their original locations.

## Environment

The project distribution is `fdf-predictions`, distinct from its `tabpfn`
dependency. The lockfile includes TabPFN 8.0.3 and PyTorch 2.11.0, using CUDA
12.8 wheels on Windows/Linux. CPU inference is available with `--device cpu`.
The editable installation keeps repository-relative data/model defaults valid.
For installed wheels, pass explicit data, output and checkpoint paths.

```powershell
uv sync --locked
python -m src doctor
python -m unittest discover -s tests -v
```

The legacy prepared-fold AutoTabPFN runners and vendored extension source have
been removed. Use `src/regression/evaluation.py` for native FD/FL evaluation,
or the explicit-manifest system workflow. Existing checkpoint binaries and
historical result snapshots have not been deleted.

## Features

One sample is `(source_file, R, re)`. Equal `(x,z)` coordinates are averaged,
then each horizontal row receives a centered median filter of size three.
Endpoints keep their original force values. This operation does not access
labels or combine different samples.

| Feature | Meaning |
|---|---|
| `boundary_distance_abs` | Absolute distance between the global rightmost red point and leftmost blue point |
| `boundary_midpoint_x` | Arithmetic mean of these two x coordinates |
| `boundary_midpoint_norm` | Midpoint measured from the sample minimum x, divided by its x span |
| `boundary_gap_norm` | Signed blue-minus-red boundary gap divided by x span |
| `red_fraction` | Fraction of unique coordinates with positive smoothed force |
| `row_midpoint_iqr_norm` | Interquartile spread of row midpoints divided by x span |
| `row_gap_median_norm` | Median signed row boundary gap divided by x span |
| `row_gap_q90_norm` | 90th percentile of signed row gaps divided by x span |

Rows without both signs do not contribute to the row-boundary statistics.
Undefined boundaries or zero-span normalizations remain missing. Imputation
uses only the relevant training fold. Coordinates and geometric lengths are
normalized to micrometers using the declared source unit; R retains its source scale.

`focus-features --no-plots` skips the expensive rendering stage. Current
feature tables include `source_sha256`; the dataset loader rejects a changed
workbook when its saved feature hash is stale. `focus-optimize` rebuilds its
own feature snapshot on every run. Older unhashed tables remain readable for
baseline compatibility; their audit explicitly reports hashes as unverified.

## Evaluation

The original baseline has one TabPFN estimator and geometry/Re plus boundary
width. Its seed and outer folds are reproduced in the optimization benchmark.

Three candidate feature sets are fixed in code before evaluation: geometry/Re,
geometry/Re plus width and midpoint, and compact geometry plus normalized
spatial descriptors. Each uses four TabPFN estimators. A fourth candidate is
their equal-weight mean probability. Seven fixed probability thresholds range
from 0.35 to 0.65.

For each outer geometry holdout, three inner grouped folds use only the outer
training samples. Candidate and threshold selection maximizes inner OOF
accuracy, breaking ties by balanced accuracy, then closeness to 0.5. The chosen
recipe is refitted on all outer training samples before scoring the held-out
geometry. Every sample receives exactly one outer prediction. Model interfaces
receive no test labels. All radii, orientations and Re values of a named
geometry family remain together.

The `nested` row in `metrics.csv` is the evaluation of that complete selection
procedure. Other candidate rows are fixed-threshold diagnostics. Do not use the
best outer diagnostic row as evidence of an independently selected winner.
The paired bootstrap resamples whole geometry groups. With only eight groups,
its confidence interval is descriptive, not strong evidence about all future
geometries. An independently collected batch is needed for final validation.

The data have already been inspected in prior research. Nested validation
prevents fold-level fitting/selection leakage in this implementation, but does
not turn an already studied dataset into a new independent test set.

`selected_recipe.json` selects a future prediction recipe using grouped OOF
predictions on all available training data. It references the exact saved
training feature table and checkpoint hashes. Its selection score is never
presented as test accuracy. `focus-predict` refits TabPFN using that small saved
training table and returns probabilities, F/UF labels and training-overlap flags.

`fixed_baseline_recipe.json` and `fixed_midpoint_recipe.json` also export the
fixed benchmark recipes. They can be supplied to the same prediction command
for prospective comparison. An improvement in a fixed candidate's grouped CV
score is not evidence that the full automatic selection procedure improved.

## Verification

Tests cover restored FD/FL imports and unit conversions, grouping and label
alignment, original smoothing equivalence, spatial feature invariance, stale
file rejection and nested isolation. The original 200 sample widths, midpoints,
sign-change counts and coordinate counts are compared with regenerated values.
Results and protocol files stay under the dedicated optimization directory.

## Reliability Update

New command-line runs are isolated under `runs/<run-id>/`, with the exact
output directory printed at startup and a `latest.json` pointer on completion.
The equal-budget `focus-ablate` workflow separates midpoint contribution from
ensemble-size changes. See [FOCUS_RELIABILITY.md](FOCUS_RELIABILITY.md) for
strict input checks, single-class holdouts, prediction review, source snapshots
and resumable inference caches. Historical results retain their original paths.
