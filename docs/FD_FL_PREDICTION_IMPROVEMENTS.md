# FD-u / FL Prediction Improvements

## Scope and Physical Meaning

This round covers paired native-workbook prediction and the existing
`web-demo/tabpfn_service.py` API. It does not change the focus classifier.
Original workbooks, target values and saved model files are not overwritten.

The current FD workbook target is `u`. Its physical meaning and units have not
been confirmed by the user. The existing training builder maps `u` directly to
`fd_x_pred`; this is not evidence that velocity has been converted to force.
The new batch workflow therefore reports `FD_u` and `FL` separately and never
adds them. It keeps native target scales. Coordinate normalization still uses
the established template adapter and records every unit multiplier.

The filename suffix 10/20/30 is recorded as `radius_label`; its unit and physical
interpretation are not silently equated with the service's 300-5000 um radius.

## Prediction Entry

```powershell
python -m src regression-predict --only 750-320-10 --max-train-rows 500 --chunk-size 1024 --n-estimators 1
```

For each selected pair:

1. Fit FD `u` on native FD rows and predict at every native FL position.
2. Fit FL `fl` on native FL rows and predict at every native FD position.
3. Preserve invalid query rows with missing prediction and an explicit status.
4. Record actual training coverage, same-Re nearest distance, spatial convex
   hull relation, range warnings and any training-position overlap.

The preprocessing is fitted only on training rows; FL/u targets never enter
the feature matrix. No interpolated pseudo-labels are generated. A training
cap keeps every available Re represented and is deterministic. Prediction
positions are not sampled down. The cap is a computational tradeoff, not a
validated optimum or an accuracy claim. `--max-train-rows` defaults to 1000.

`--batch-dir` and `--out-dir` accept alternative roots. Native coordinate units
can be specified using `--length-unit um` or `m`; `auto` retains the existing
adapter's inference and records it. It never infers or converts target units.

## Pairing Corrections

Reflection markers before the radius, before FL/FD, or after FL/FD now map to
the same reflected pair key. Reflected and original workbooks remain distinct.
Two files mapping to one key raise an ambiguity error instead of overwriting
one another. Pairing is based on naming, not proof that supplied geometry is
correct; source geometry remains subject to scientific review.

The current directory has 16 FL and 14 FD files, forming 14 pairs. There is no
FD counterpart for `750-400-10-FL.xlsx` or `750-530-10-FL.xlsx`. Default full
batch execution rejects missing pairs. An explicit `--only` selection or
`--allow-unpaired` permits partial processing, with all omissions recorded in
`pair_inventory.csv`.

The older `fd-to-fl` / `fl-at-fd` paths also received the nonfinite-row,
same-Re nearest-distance and filtered-index alignment fixes. FL augmentation
in the legacy path remains available but is not used by the new native path.

## Statistics and Visualization

Every run is versioned, with source hashes, checkpoint hash, settings and an
exact code snapshot. Outputs include:

- `predictions.csv` and per-pair/per-target CSV files: all requested rows,
  original workbook row references, predictions and diagnostic flags.
- `by_field.csv`, `by_pair.csv`, `by_re.csv`: coverage, mean, population standard
  deviation, quartiles, sign fractions, distances and spatial-risk counts.
- `prediction_audit.json`: native/used/invalid training counts, each Re's
  training sample count, selected features and normalization details.
- `pair_inventory.csv`: all recognized pairs, missing counterparts and actual
  selection. No silent reassignment of reflected data.
- `gallery.html`: searchable per-Re maps and separate FD-u/FL statistics.

Field values are not pooled across FD-u and FL. Missing values are not zero.
Sign fractions use finite predicted rows as the denominator. Same-geometry
and same-position records are correlated; row counts are not independent
sample counts. No MAE, RMSE or R2 is produced without a legitimate held-out
reference. This workflow is prediction, not an accuracy evaluation.

The spatial maps plot all valid predicted query rows without interpolation.
Outside or unsupported training coverage uses cross markers. Within each
pair and target the color scale is shared over Re; color magnitudes cannot be
compared across targets with unconfirmed units. Exact-coordinate points may
overlap visually; titles expose row counts.

Rebuild reports without retraining:

```powershell
python -m src regression-report --prediction-run results/regression_prediction/2026-06-20/runs/20260910T145117Z-ea34c70c
```

This creates another versioned report and copies the saved numerical outputs,
preserving the original prediction run. Source hashes and zero new fits are
recorded in `report_provenance.json`.

## Service Safety

- Prediction arrays must contain all four aligned, finite output components.
- Invalid or transient model results do not poison the prediction cache.
- Cache identity includes exact numeric inputs, positions and model revision;
  reload invalidates old entries. The bounded LRU defaults to eight entries.
- `TRIFOCUS_CACHE_MAX_ENTRIES=0` disables and clears stored entries.
- Model loading resets stale state; model calls are serialized for thread safety.
- `require_model: true` rejects unavailable/failed inference instead of silently
  substituting deterministic rules. Default fallback remains for compatibility
  and is explicitly labeled.
- Actual TabPFN identity is distinguished from an arbitrary ready joblib model.
- Focus scores are labeled heuristic, not learned or calibrated probabilities.
- Radius correction is labeled an empirical clipped ratio, not a learned
  radius feature. Existing numerical behavior is retained.
- Legacy force-sum and transverse-component fields remain API-compatible but
  carry `physical_units_verified=false` and `fnet_physically_validated=false`.

The old builder uses zero transverse labels and a centroid approximation;
this round does not silently retrain or replace those saved models. Corrected
training data, confirmed target units, and independent geometry holdouts are
prerequisites for a scientific claim about forces on a new triangle.

## Verification Scope

The real TabPFN run above processed one pair, not all 14 pairs: FD-u at 490 FL
positions and FL at 15,900 FD positions, totaling 16,390 finite predictions.
FD used 500 of 15,900 native training rows; FL used all 490 native rows. Ten Re
slices were visualized. There were four FD-u query positions outside the
same-Re training hull and 9,630 FL positions outside it. The latter is about
60.6% of FL queries and requires particular caution about spatial extrapolation.

A separate strict service smoke test loaded the real `tabpfn_server_500`
artifacts and produced 20 model rows from two positions across ten Re values;
`uses_tabpfn=true`. This checks execution, not physical accuracy.

Remaining work includes confirmed units/target meaning, full-batch model runs
when desired, and independent evaluation. Offline/server export packages were
not regenerated by this local change. Hosted CI was not run locally.
