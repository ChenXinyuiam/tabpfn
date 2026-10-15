# Fzong focus inspection

This package is for visual and rule-based inspection of focus state from raw
point-level Fzong distributions.

The package also provides supervised F/UF classification. The unified workflow
is `python -m src focus-optimize`, which rebuilds features and runs nested
grouped TabPFN evaluation. See `docs/REPOSITORY_OPTIMIZATION.md` for the full
protocol and the split between dataset, features, modeling and reports.

For a controlled feature comparison use `python -m src focus-ablate`.
New feature/classifier command-line runs save to `<out-dir>/runs/<run-id>/`;
the path is printed and the last completed run is recorded in `latest.json`.
Output trees below describe the contents within that run directory. Historical
outputs and standalone sign-plot paths are unchanged. See
`docs/FOCUS_RELIABILITY.md` for units, strict validation, resume and REVIEW flags.

`python -m src focus-audit` checks inputs without a model. `python -m src
focus-calibrate` evaluates probability recalibration inside geometry holdouts.
Physical side-length identity now supplements filename-based groups. See
`docs/FOCUS_CALIBRATION.md` and `docs/FOCUS_DATA_ISSUES_2026-09-08.md` for the
schema-3 recipe transition and the retained source-data contradictions.

`python -m src focus-review --evaluation-dir <completed-run> --model raw`
creates a separate sample-level error review without retraining. Use `width`
for a controlled ablation run. See `docs/FOCUS_REVIEW.md` for counting, queue
priorities and geometry-level uncertainty.

`focus-predict` now creates grouped statistics, probability/boundary figures
and `gallery.html` automatically. Optional exact-content inference caching uses
`--cache` or `--cache-from <run>/prediction_cache`. See
`docs/FOCUS_PREDICTION.md` for overlap detection and statistical caveats.

Current sample unit:

```text
one Excel file x one curvature radius R x one Reynolds number re
```

Current entry point:

```powershell
python -m src.focus_inspection.plot_fzong_sign_by_re `
  --raw-dir data\rawdata\2026-09-04 `
  --out-dir results\focus_inspection\2026-09-04\fzong_sign_triangles_by_re
```

Smoothed boundary feature extraction:

```powershell
python -m src.focus_inspection.build_focus_boundary_features `
  --raw-dir data\rawdata\2026-09-04 `
  --out-dir results\focus_inspection\2026-09-04\focus_boundary_features
```

Boundary feature definition:

- Group samples by source Excel file, `R`, and `re`.
- Treat equal `z` values as one horizontal row and sort that row by `x`.
- Average duplicate `(x, z)` coordinates inside the sample.
- Apply a centered three-point median to the numeric Fzong values in each row.
  Row endpoints retain their original values.
- Recompute red/blue signs from the smoothed values.
- `boundary_distance_abs` is the absolute difference between the global
  rightmost red `x` and global leftmost blue `x` in the sample.
- `boundary_midpoint_x = (rightmost_red_x + leftmost_blue_x) / 2` is the x
  coordinate at the center of that interval. The raw value is retained as
  `boundary_midpoint_x_raw` for audit.
- `boundary_gap_signed = leftmost_blue_x - rightmost_red_x` is retained for
  audit. A negative value means the red and blue x-ranges overlap.
- `boundary_distance_norm` divides the absolute distance by the sample x-span.
- Normalized midpoint, signed gap, red fraction and row-boundary spread are
  available to the optimized classifier. They are computed within each sample.
- Samples without both red and blue points receive a missing boundary value,
  including a missing midpoint, not zero.

Default feature outputs:

```text
results/focus_inspection/2026-09-04/focus_boundary_features/
  features.csv
  smoothed_points.csv
  run_summary.json
  gallery.html
  figures/<excel_stem>/R<radius>/*.png
```

Leakage-resistant TabPFN classification:

```powershell
python -m src.focus_inspection.train_focus_tabpfn
```

The classifier reads labels from `三角形聚焦汇总.xlsx`, where `F` is the
focused class and `UF` is the unfocused class. Evaluation uses eight grouped
out-of-fold splits, leaving out one geometry size per fold for this batch. A
geometry size and all of its radii, orientations, and Reynolds numbers remain
in one fold. Missing-value imputation and scaling are fitted separately inside
each training fold.

The report compares the fixed `boundary_distance_abs < 100` rule with three
TabPFN feature sets: boundary only, geometry/Re context only, and context plus
the smoothed boundary feature. The context-plus-boundary model is designated as
the main model before evaluation. `boundary_midpoint_x` is carried into the
classification output tables for future feature selection, but it is not added
to this original baseline automatically. The separate `focus-optimize` command
evaluates midpoint and spatial features with inner-fold selection.

Default classification outputs:

```text
results/focus_inspection/2026-09-04/focus_tabpfn_classification/
  labeled_features.csv
  oof_predictions.csv
  metrics.csv
  evaluation.json
  main_model_metrics_by_fold.csv
  main_model_metrics_by_geometry.csv
  main_model_errors.csv
  gallery.html
  report.md
  figures/*.png
```

Default assumptions:

- The target column is `Fzong` if present, otherwise `Fzong2`.
- Red points mean target value `> 0`.
- Blue points mean target value `< 0`.
- Gray points mean target value `= 0`.
- Every image corresponds to exactly one `source Excel + R + re` sample.
- `--duplicate-mode raw` is the default. It plots every row inside the same
  sample directly.
- In raw mode, exact duplicate `x,z` rows are slightly spread in display
  coordinates so overlapping points remain visible. The source coordinates are
  not changed.
- `--duplicate-mode mean` merges duplicate `x,z` rows inside the same sample.

Default outputs:

```text
results/focus_inspection/2026-09-04/fzong_sign_triangles_by_re/
  figures/<excel_stem>/R<radius>/*.png
  summary.csv
  gallery.html
```
