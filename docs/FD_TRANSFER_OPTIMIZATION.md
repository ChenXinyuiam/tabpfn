# FD transfer optimization protocol

## Objective and interpretation

The requested goal is R2 >= 0.9 for previously unseen triangles. The primary
reported score remains arithmetic mean of per-geometry R2 over all 24 angles.
Original 0-degree, 90-degree, pooled, and worst-geometry scores must also be shown.
The 24 deterministic rotations are not 24 independent physical experiments.

This is iterative research on the same eight geometries used in the earlier
benchmark. Nested validation prevents direct per-fold test-label selection, but
does not turn repeatedly studied data into an untouched external blind test.
A new independently supplied triangle dataset is needed for that confirmation.

## New experiment

`python -m src fd-transfer-tune --snapshot <completed-rotation-audit-run>`

Implementation: `src/regression/transfer_fd.py`.

- Five outer geometry folds, with the same seed and test queries as the prior run.
- Three inner folds, containing only the outer-training geometries.
- Select u0 and u90 recipes separately by inner geometry-equal native-target R2.
- Complete every inner search before calculating any new outer score.
- Predict the two components and combine them with the previously verified
  `u(theta) = u0*cos(theta) + u90*sin(theta)` identity.
- Evaluate against actual workbook labels in the trusted audit snapshot.
- Retain all previously accepted test points, including suspicious Re conditions.
- Save inner scores, actual training point IDs, selected recipes, source code,
  checkpoint hashes, native-scale predictions, and per-geometry metrics.

The first study freezes eight candidates: provided barycentric features with raw
or scaled targets; mesh-derived features with raw or scaled targets; 1500/4000
context rows; per-component Re powers (2,1) or (2,2); an empirical boundary-product
normalization; and the already installed TabPFN-3 checkpoint.

Before examining the first study's outer results, a second menu was added:
mesh raw targets with 4000/8000 rows, TabPFN-3 raw targets with 4000 rows,
reflection-aligned mesh features with raw/scaled targets, and conditional median
prediction with scaled targets. These test whether inconsistent Re labels make
raw targets preferable, and whether a common geometric representation helps.
An additional Re-local candidate splits a 24000-row training pool into separate
same-Re contexts. For an unseen Re it uses the nearest two training Re levels.
This increases total training-label coverage while limiting each individual
TabPFN context. It does not admit any held-out geometry labels.

Example second-study command:

```powershell
.\.venv\Scripts\python.exe -m src fd-transfer-tune `
  --snapshot results/fd_rotation_audit/2026-09-29/runs/20261007T121511Z-875c23be `
  --recipes provided_raw_1500 mesh_raw_1500 mesh_raw_4000 mesh_raw_8000 mesh_v3_raw_4000 mesh_symmetric_raw_4000 mesh_symmetric_scaled_4000 mesh_scaled_median_4000 mesh_local_re_24000
```

Ensemble count is not training epochs. These experiments condition fixed local
TabPFN weights; they do not fine-tune pretrained weights. Different context
budgets and checkpoint sizes also use different amounts of data and computation.

## Mesh-coordinate contract

Some nominal dimensions and provider barycentric coordinates disagree with the
actual coordinate mesh. The experimental alternative infers the left/right
base vertices and apex from the complete source x/z mesh without reading labels.
It uses a size-relative tolerance for bottom-edge numerical roundoff and rejects
meshes that do not fit the assumed horizontal-base triangle.

This uses the full unlabeled mesh of a held-out geometry, not just its sampled
evaluation points. This input requirement is explicit: future new-triangle
inference must supply reliable vertices or the complete unlabeled mesh. It is
not valid to estimate a triangle from extrema of arbitrary interior query points.
Inferred vertices are a research hypothesis, not provider-confirmed metadata.

Mesh barycentric coordinates, edge interactions, shape ratios, absolute size,
Re, and the file's radius label form model inputs. Reflection-aligned variants
keep an orientation feature; they do not assume or modify output vector signs.
No derived feature reads u0, u90, rotated truth, or another query label.

## Empirical normalization

The scaling candidates train on `u / ((Re/30)^p * 10/radius)` and invert exactly
at prediction time. Candidate powers are hypotheses selected on inner folds,
not established physical laws. Boundary-product normalization also has a
strict positive floor and does not force test labels or predictions to zero.

## Non-destructive diagnostics

`python -m src.regression.transfer_diagnostics --snapshot <audit-run>`

The diagnostic joins exact x/z positions across Re conditions and reports almost
identical nonzero response fields. It never determines which condition is right,
changes source values, or drops validation points. The current source has an
exact duplicate of both u0/u90 at Re=12 and Re=60 for all 1601 points in 800-333.
The source owner must confirm whether this is intentional or mislabeled.
In addition, all 1830 positions of 750-400-reflected at Re=30 satisfy
`u90 = -u0` to numerical precision. This is flagged for confirmation, not
automatically declared impossible.

A third experimental menu tests a training-only quarantine: when those exact
patterns occur within the training geometries, both ambiguous Re copies or
the sign-identical component condition are excluded from training context
sampling. No choice of a "correct" copy is made. All outer and inner evaluation
labels remain unchanged, and the removed source/Re conditions are recorded
per split. This is a validation-selected robustness hypothesis, not a silent
rewrite of the source dataset.

`src/regression/transfer_ensemble.py` combines compatible inner-score tables.
It evaluates both the best inner-selected recipe per component and a fixed
top-three softmax-weighted average (temperature 0.1). Weights use only the
outer-training geometries' inner validation scores. This comparison also
remains exploratory; its outer scores must not be used as training weights.

## References

- Prior Labs model report: https://priorlabs.ai/technical-reports/tabpfn-3
- Official inference configuration:
  https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/inference_config.py

Actual local results, not general benchmark claims, determine the selected
recipes. Reaching 0.9 on this exploratory dataset alone is not a deployment or
arbitrary-distance extrapolation guarantee.
