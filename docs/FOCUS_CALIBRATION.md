# Grouped Probability Calibration

## This Round

- Physical triangle identity supplements filename-based grouping.
- Names or matching sides connect related samples into one holdout group.
- Redundant geometry checks expose contradictions without changing inputs.
- A fixed sigmoid calibration is evaluated strictly inside geometry holdouts.
- A read-only cache from an earlier run can avoid identical repeated inference.
- Both raw and transformed probabilities remain available in predictions.

The original values are preserved as explicitly requested. Known data problems
are recorded in [the data issue log](FOCUS_DATA_ISSUES_2026-09-08.md).

## Commands

```powershell
python -m src focus-audit
python -m src focus-calibrate
```

The audit does not load a model or require a GPU. It writes `issues.md`,
`audit.json`, `geometry_audit.csv` and `audited_samples.csv` to a unique run.
`--skip-labels` checks only input features; otherwise label matching is checked.
`--strict` makes any review flag a failed audit while preserving the report.

Calibration defaults to three fixed seeds (42, 2026, 3407), four TabPFN
estimators and geometry/Re + width features. The prediction threshold is 0.5.
No threshold, seed or candidate is selected from these results.

`--cache-from <earlier-run>/prediction_cache` enables read-through reuse.
Reuse requires exactly matching ordered features, training labels, checkpoint,
model code, package versions, seed, estimator count and device. Old cache files
are never modified. `--resume --out-dir <interrupted-run>` remains available.

## No Fold-Level Leakage

For each outer held-out geometry, split only the outer training groups into
three inner folds. Base-model predictions for each inner validation partition
come from a model fitted without those samples or geometries. Fit a logistic
sigmoid to these combined inner OOF probabilities and outer-training labels.
Then fit the base model on all outer-training samples and apply that sigmoid
to the untouched outer-test predictions.

The sigmoid uses clipped logit probabilities as its single input, fixed
regularization C=1 and no class reweighting. This is regularized logistic
recalibration, not an exact implementation of Platt's smoothed-target procedure.
Training-only calibration follows the principle described by the
[scikit-learn 1.8 calibration guide](https://scikit-learn.org/1.8/modules/calibration.html).

Evaluation reports Brier score, log loss, reliability bins, both types of
classification error and high-probability mistakes. Brier and log loss assess
overall probability quality; neither isolates calibration by itself. Sparse
bins and only eight geometry groups limit conclusions. Bootstrap resampling
keeps all samples and seeds from a geometry together.

## Geometry Checks

Identity sorts the three side lengths rounded to 0.001 um. Mirrored files with
swapped ab/ac share an identity. Shared names also keep samples together, so
small differences or bad supplied fields do not split one named family.
The grouping is conservative and does not use labels. It cannot prove the
independence of two physically related experiments from metadata alone.

Checks include triangle inequality, height versus side length, angle sum,
area versus base/height, perimeter versus sides, and side/angle relationships.
Default behavior preserves conflicting inputs and records warnings.
`focus-calibrate --strict-geometry` stops before training if contradictions
exist. It is deliberately not the default for the current user-approved run.

## Prediction Recipes and Compatibility

This round uses feature schema version 3. Versioned older recipes will fail
the strict schema/implementation check instead of silently changing semantics.
Their old reports and exact source snapshots remain intact. Use newly generated
`fixed_width_recipe.json` for the unchanged raw-probability width model or
`sigmoid_width_recipe.json` for the explicitly transformed probability option.
No model is promoted automatically based on the development score.

The final sigmoid for future inference is fit on full-data grouped OOF
probabilities from the first predeclared seed. Its fitting score is not used as
the reported calibration evaluation. The untouched outer predictions are.

Predictions include `raw_focus_probability`, `focus_probability` and
`calibration_applied`. Applying calibration does not guarantee reliable
probabilities on new geometries, so `calibration_guarantee` remains false.
The legacy `probability_is_calibrated` field is an alias for whether the
transformation was applied, not evidence that it improved probability quality.
Review flags and physical-geometry identity checks still apply.

## Verified 2026-09-08 Result

Run: `results/focus_inspection/2026-09-04/focus_calibration/runs/20260907T163200Z-fd5389fb`.
All 200 samples and their original labels/geometry values were retained.

| Three-seed mean | Raw width model | Sigmoid recalibration |
|---|---:|---:|
| Accuracy | 92.33% | 90.00% |
| Balanced accuracy | 90.04% | 86.83% |
| Brier score (lower is better) | 0.06701 | 0.07947 |
| Log loss (lower is better) | 0.21813 | 0.28458 |

Calibration worsened the predeclared primary score in all three seeds.
The paired geometry-bootstrap Brier difference was +0.01245 with a descriptive
95% interval of [+0.00360, +0.01785]. High-confidence mistakes increased from
20 to 36 across 600 repeated-seed predictions; these are not 600 independent
samples. The raw model remains the working choice. The sigmoid recipe is an
experimental output, not a replacement or a new accuracy claim.

The 600 raw predictions exactly match the preceding width ablation. Source
and label hashes, all shared numerical features, and fold assignments also
match. The 24 saved outer calibrators reproduce the calibrated predictions;
all 72 inner-fold audits keep fit, validation and outer-test geometries disjoint.
The run used 24 identical cached predictions and 72 new fits (about 13 minutes).

The model-free audit found 30 geometry-warning samples in three files and 78
samples with at least one review flag. All 200 labels matched. This round adds
data and evaluation safeguards; it does not demonstrate higher accuracy.

Verification artifacts are under `outputs/repository_optimization/calibration_round`.
Desktop (1440 x 1000) and mobile (390 x 844) report checks passed, with no page
overflow, failed images or browser errors. The local test suite has 59 passing
tests. Hosted CI has not been executed in this local verification.

## Remaining Limits

The same data have already informed development. Cross-validation is not a
fresh external test. The supplied geometric contradictions remain in training
and evaluation as requested. External geometry data, confirmed measurement
definitions and decisions about label corrections cannot be supplied by code.
