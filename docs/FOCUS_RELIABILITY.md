# Focus Workflow Reliability

## Implemented Controls

The next round adds physical-geometry grouping and training-only probability
calibration. See [FOCUS_CALIBRATION.md](FOCUS_CALIBRATION.md). Original numeric
values are retained; confirmed contradictions are listed in
[the data issue log](FOCUS_DATA_ISSUES_2026-09-08.md). New recipes use schema 3.

| Area | Change | Status |
|---|---|---|
| Invalid point rows | Fail by default; explicit dropping produces counts | Implemented and tested |
| Physical units | Explicit um/mm/m source length setting; normalized to um | Implemented and tested |
| Geometry consistency | Reject missing, nonpositive or varying geometry values within samples | Implemented and tested |
| Multiple radii | Label keys use the actual row R, not the radius in the filename | Implemented and tested |
| Point coverage | Report the fraction of the union of coordinates within each workbook | Implemented and tested |
| Smoothing | Retain original feature semantics; flag residual multiple transitions | Implemented and tested |
| Single-class holdout | Allow one class in test; require both classes in training | Implemented and tested |
| Controlled ablation | Identical estimator count, folds and seeds across feature sets | Completed: 72 model fits |
| Prediction identity | Distinguish a seen sample key from identical workbook bytes | Implemented and tested |
| Prediction review | Flag out-of-range values, new geometries, missing features and ambiguous probabilities | Implemented and tested |
| Feature compatibility | Persist settings, schema version and implementation hashes in recipes | Implemented and tested |
| Output integrity | Unique run directories and running/failed/complete status | Implemented and tested |
| Resume | Exact-content probability cache and protocol validation | Implemented and tested |
| Error review | Per-sample plots, rule conflicts and a human review queue | Completed: 110 review samples |
| Continuous checks | Windows/Linux CPU-only unit-test workflow, no private data or model download | Added; remote execution pending |

## Run and Locate Results

```powershell
python -m src focus-ablate
python -m src focus-optimize
python -m src focus-features --no-plots
```

These commands and `focus-baseline` / `focus-predict` create a unique directory
under `<out-dir>/runs/<UTC timestamp>-<id>/`. The exact directory is printed
at startup. `<out-dir>/latest.json` points only to the last completed run.
Existing reports are never overwritten. A source snapshot records the local
code, including uncommitted edits; it is not a substitute for version control.
`focus-baseline` automatically resolves the latest completed feature run unless
an explicit `--boundary-features` file is provided.

Long experiments can continue from the exact interrupted directory:

```powershell
python -m src focus-ablate --out-dir <interrupted-run-directory> --resume
python -m src focus-optimize --out-dir <interrupted-run-directory> --resume
```

Repeat the original parameters when resuming. Input bytes, feature settings,
code, model checkpoint or package version changes invalidate resume. The
cache also checks the exact ordered training features/labels and prediction
features, seed, device and estimator count. Completed runs cannot be resumed
or overwritten. A failed run cannot become `latest.json`.

The standalone historical sign-plot command retains its existing output layout.
Library-level feature extraction writes to its caller-provided directory so
that a complete experiment owns one coherent feature snapshot.

## Input Contract

- Samples remain Excel file x R x Re. R/Re must be positive finite numbers.
- Labels remain F/UF. No automated label correction is performed.
- Names still identify geometry and orientation using the established naming
  convention. The radius used for label matching comes from each data row.
- `--length-unit um` is the default for x/z and geometric lengths. `mm` and
  `m` convert these values to micrometers and area to square micrometers.
  Angles and the curvature radius R retain their original conventions.
- Units cannot be proven from bare numbers; declaring the correct unit remains
  the data producer's responsibility. Mixed units within a workbook are not
  supported. This setting is saved in recipes and reused for prediction.
- Invalid point rows fail by default. `--invalid-rows drop` is an explicit
  exception and records the number of rows lost. Invalid geometry is never
  silently imputed or averaged across conflicting values.
- Coverage is relative to observed coordinates in the same workbook, not proof
  that the physical triangle was fully sampled. Changing grids may be valid,
  but require inspection.
- Multi-transition rows and conflicting duplicate signs are warnings, not
  instructions to force a monotone sign field.

The batch audit is in `features/input_quality.csv`, `quality_warnings.csv`
and `run_summary.json`. Coordinates used for calculations remain separate
from any visual jitter used by historical raw-point plots.

## Controlled Feature Evaluation

`focus-ablate` compares geometry/Re, geometry/Re + width, and geometry/Re +
width + midpoint. All use four TabPFN estimators, fixed threshold 0.5 and the
predeclared seeds 42, 2026 and 3407. Canonically ordered leave-one-geometry-out
folds are the same for every seed and feature set. Preprocessing is fit on
training samples only. Models never receive held-out labels.

The fold-training majority predictor and fixed 100-um rule provide reference
scores. A rule with an undefined width has no prediction for that sample;
coverage must be reported rather than treating a missing width as evidence of UF.
Balanced accuracy, F recall, UF specificity, Brier score and log loss accompany
accuracy so different error types and probability quality remain visible.

Results are averaged across the predeclared seeds, never reported using the
best seed. Paired geometry bootstrap keeps all seeds and all samples of a
geometry together. Repeated seeds do not increase the independent sample count.
Fixed recipe files use the first predeclared seed, not a test-selected winner.

These data have already guided development. No cross-validation procedure can
make them a fresh external test. Collect a separate, unseen geometry batch
before claiming a reliable deployment improvement. Do not tune repeatedly on
the new batch after observing its labels.

## Prediction Output

```powershell
python -m src focus-predict --recipe <run>/fixed_midpoint_recipe.json `
  --raw-dir <new-input-directory> --out-dir results/focus_predictions
```

`focus_prediction` always preserves the model's F/UF classification.
`decision` is REVIEW when `needs_review` is true; `review_reasons` explains why.
This avoids silently treating a model output as an approved scientific decision.

`seen_sample_key` matches name/R/Re. `known_training_sample` additionally
requires the same workbook SHA-256. A changed same-name file is not declared
identical. `seen_geometry` is distinct from both. Original and replaced bytes
may still describe the same physical case; the flags do not establish a new
independent evaluation dataset.

Probability is not presented as calibrated confidence. A fixed +/-0.1 margin
around the decision threshold is only a review heuristic. High-probability
errors remain possible. No guarantee about the accuracy of unflagged samples
is implied. Reliability curves use out-of-fold predictions for diagnosis;
any future calibrator must itself be fit within training-only grouped folds.

Legacy recipes without feature settings remain readable for compatibility,
but every prediction is flagged as having unverified legacy configuration.
New recipes reject changes in feature schema/implementation. Training snapshot
and checkpoint hashes are also checked before fitting.

## September 7 Controlled Results

The completed three-seed run is
`results/focus_inspection/2026-09-04/focus_ablation/runs/20260907T111143Z-6b65a27c/`.

| Feature set | Mean accuracy | Individual seed accuracies |
|---|---:|---|
| Geometry/Re | 92.00% | 93.5%, 91.5%, 91.0% |
| Geometry/Re + width | 92.33% | 93.0%, 92.0%, 92.0% |
| Geometry/Re + width + midpoint | 91.50% | 92.0%, 90.5%, 92.0% |
| Fold-training majority | 69.50% | 69.5% for each seed |
| Fixed width <100 um | 62.50% | 62.5% for each seed |

Under equal settings, adding midpoint did not improve accuracy. Its paired
mean difference relative to width was -0.83 percentage points. This corrects
the earlier interpretation of the unequal-budget 90.5% versus 92.0% comparison.
It does not invalidate the midpoint's physical meaning or its use for inspection.
No recipe is promoted automatically. These remain development comparisons,
not evidence from new independent geometries.

All 200 original widths, midpoints, extreme positions, point counts and smoothing
change counts were unchanged by the reliability refactor. There were no invalid
input rows. Quality flags cover 55 samples: 36 with multiple sign transitions
and 20 with within-file coordinate-grid differences, with one sample overlapping.
The broader 110-sample review queue also includes model errors across seeds and
disagreement with the fixed 100-um rule; a flag does not establish a bad label.

## Human Review and Remaining Decisions

`review.html` links point-distribution comparisons to sample identities;
`review_queue.csv` contains blank `reviewed_label` and `review_note` columns.
These are suggestions for human review, not an automatically consumed relabeling
file. Reconcile annotations against the original experimental criteria, not
against whichever label makes model accuracy higher.

The spatial definition of focus, acceptable false-F versus false-UF costs,
and new independent experimental data require scientific input. Code cannot
establish these facts or fabricate additional independent observations.
