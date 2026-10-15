# Sample-Level Focus Review

`focus-review` summarizes saved repeated-seed predictions without loading a
model, changing labels, excluding samples, or overwriting the source run.
Select the saved model explicitly; the report does not search for a winner.

```powershell
python -m src focus-review --evaluation-dir results/focus_inspection/2026-09-04/focus_calibration/runs/20260907T163200Z-fd5389fb --model raw
```

Completed `focus-ablate` runs also work with `--model width`, `context` or
`midpoint`. A missing historical geometry audit is reported as unavailable,
not as proof that geometry is valid. The report does not reinterpret original
folds or update their feature implementation.

## Counting and Triage

One unique sample is Excel x R x Re. Repeated seeds do not create new samples.
Each sample must have exactly one prediction for every seed, unchanged label,
geometry group, fold and available boundary metadata. Incomplete, duplicated,
nonbinary or nonfinite records fail explicitly.

The queue prioritizes persistent errors, error count, high-confidence errors,
decision instability and geometry warnings. This ordering is for inspection,
not evidence that a supplied label is wrong. Rule disagreements at 100 um are
also flagged but never used to overwrite a label. All flagged sets can overlap.

- `sample_summary.csv`: every sample, width, midpoint x, probability range,
  repeated-run errors and review reasons.
- `review_queue.csv`: flagged samples with blank `reviewed_label` and
  `review_note` fields. These are not automatically read by training.
- `geometry_summary.csv`: per-geometry accuracy, unique samples, both error
  directions, persistent errors and warning counts.
- `review_summary.json`: source hash, counts, method and uncertainty summary.
- `gallery.html`: Chinese report, searchable sample list and category filter.

High-confidence mistakes use saved probability >=0.9 or <=0.1. They are a
descriptive diagnostic, not a guarantee that other predictions are safe.
Probability min/max describes seed variation, not a prediction interval.

## Uncertainty

Resample entire geometry groups 5,000 times with fixed random seed 42. All
samples and all seed predictions in a selected group stay together. Report
the 2.5% and 97.5% percentile limits for sample-weighted accuracy and the
equal-geometry mean. These are different estimands, not competing models.
Replicating identical seeds cannot tighten these intervals.

Only eight geometries are available and the data have already informed
development. The intervals are descriptive, not fresh external validation,
and do not guarantee an accuracy range for future geometries.

## Verified Current Results

For the retained raw width model: 200 unique samples, 600 predictions, 8 groups.
19 samples were wrong in at least one seed; 13 were wrong in every seed and
6 changed their predicted class. 12 samples had at least one high-confidence
mistake. The combined queue contains 123 samples, including input-quality and
100 um rule disagreements; this does not mean 123 labels are wrong.

Sample-weighted accuracy remains 92.33%, with a descriptive geometry-bootstrap
95% interval of 88.55% to 97.50%. Equal-geometry accuracy is 95.14%, reflecting
different weighting, not an accuracy improvement. Original data problems
still affect 30 samples in three files; see the
[preserved-value issue log](FOCUS_DATA_ISSUES_2026-09-08.md).

The local suite now has 67 passing tests, including duplicate/missing seed
rejection, unchanged inputs, HTML escaping, geometry weighting and invariance
to repeated copies of the same seed predictions. Hosted CI is not locally run.
