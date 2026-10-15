# Focus Prediction and Batch Statistics

The prediction workflow uses the saved training snapshot, preprocessing,
feature configuration and fixed decision threshold. It does not read labels
for the prediction batch, tune thresholds or select a new model. The retained
raw width recipe is the working model; sigmoid calibration is not promoted.

## Run

```powershell
python -m src focus-predict --recipe results/focus_inspection/2026-09-04/focus_calibration/runs/20260907T163200Z-fd5389fb/fixed_width_recipe.json --raw-dir data/rawdata/2026-09-04 --out-dir results/focus_inspection/2026-09-04/focus_prediction --cache
```

For an actual new batch, change `--raw-dir` to its directory. Input workbooks
must follow the existing Fzong geometry/coordinate naming and column contract.
Each run has a unique output directory and source-code snapshot. Completed
outputs are not overwritten. `latest.json` only points to completed runs.

`--cache` saves exact inference results. `--cache-from <prior-run>/prediction_cache`
enables read-through reuse and a local cache. Reuse requires matching training
features and labels, ordered test features, model identity, versions, device,
seed and estimator count. It does not bypass recipe validation. Invalid model
outputs are rejected before caching. Cold and warm outputs use the same float
precision. Feature extraction and report generation still run.

## Safety and Interpretation

- Renaming an identical workbook does not make it new data. Training overlap
  uses source-content hash x R x Re, independently of filename.
- Repeated content inside the batch is flagged, not silently dropped. Main
  counts retain the Excel x R x Re unit; a separate content-unit count exposes
  duplicate contributions. A byte hash does not detect every semantically
  equivalent re-saved workbook, so physical-geometry checks also remain.
- `focus_prediction` is the model's F/UF class. `decision=REVIEW` is a request
  for inspection, not a third learned class. Review reasons may overlap.
- Data-quality flags are counted separately from training overlap. All input
  geometry contradictions and original labels remain unchanged.
- Probabilities are not a correctness guarantee. Neither predicted F fraction
  nor agreement with the 100 um rule is an accuracy estimate.
- No external test labels are supplied to this report. `evaluation=false` and
  `accuracy=null` are explicit; training-content repeats are not independent
  evaluation samples. Same-geometry samples are also correlated.

## Outputs

- `predictions.csv`: per-sample probabilities, raw probabilities, F/UF class,
  review decision, width, midpoint, source identity, overlap, missing/range
  flags and 100 um rule agreement with the model.
- `by_file.csv`, `by_radius.csv`, `by_reynolds.csv`, `by_geometry.csv`,
  `by_file_radius.csv`, `by_overlap.csv`: counts with explicit denominators,
  predicted F fraction, review fraction and available-feature medians.
- `feature_statistics.csv`: available/missing counts, min, quartiles and max;
  genuine zero is retained and missing is not replaced with zero.
- `review_reasons.csv`: each reason counted once per sample, divided by all
  batch samples. Reason counts are not mutually exclusive.
- `boundary_plot_counts.csv`: coincident width/midpoint samples counted per
  predicted class; marker area is proportional to this count. Class-specific
  circles/triangles distinguish overlapping F/UF groups.
- `prediction_statistics.json`, `prediction_run.json`: statistics, recipe
  identity, overlap warnings, cache use and timing.
- `gallery.html` and `figures/`: searchable Chinese report, probability
  histogram, boundary plot, Excel/R x Re probability heatmaps and counts by Re.

Heatmap cells are exact sample values, without averaging or interpolation;
missing combinations are gray. Heatmaps paginate by 30 Excel/R rows. Re
counts describe the observed sampling mix, not a causal Re effect or a
balanced comparison across geometries. These statistics do not modify data.

## Verified 2026-09-10

The existing batch has 200 samples: predicted F 139, UF 61. All 200 match
training content and require that caveat; 78 separately have data-quality
flags. Width median is 80 um and midpoint-x median is 502.5 um, with all 200
values available. The rule and model disagree in 75/200 cases; this is not a
ground-truth comparison.

Cold and warm full-batch predictions matched exactly. In this local run,
inference took 14.93 seconds cold versus 0.02 seconds from cache. Time before
report generation was 52.03 versus 27.68 seconds, so this is not a claim of
the same speedup for the whole pipeline. Hardware and data size affect timing.

Raw workbook and label hashes were unchanged. Group totals reconciled against
200 detailed rows. The local suite has 76 passing tests, including renamed
content detection, duplicate flags, missing-versus-zero handling and cache
precision. Hosted CI has not been executed by this local verification.
