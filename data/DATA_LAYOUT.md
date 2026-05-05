# Data Layout

The active data pipeline is organized by batch date. A batch name is a folder
such as `2026-04-26`.

## Active Pipeline

Run each command from the repository root.

```powershell
.\.venv\Scripts\python.exe -m src.data.clean --batch 2026-04-26
.\.venv\Scripts\python.exe -m src.data.Feature_scale --batch 2026-04-26
.\.venv\Scripts\python.exe -m src.data.calculate --batch 2026-04-26
.\.venv\Scripts\python.exe -m src.data.Reverse_FL --batch 2026-04-26
```

Stage directories:

```text
rawdata/<batch>/      raw Excel workbooks
cleaned/<batch>/      cleaned workbooks
scaled/<batch>/       scaled workbooks
processedv2/<batch>/  geometry-feature workbooks
processed/<batch>/    final model/plot-ready workbooks
```

`processed/<batch>/` is the main working dataset. It contains:

```text
*-FL.xlsx                original FL workbooks copied from processedv2
*-sym.xlsx               symmetric workbooks after FL sign reversal
*_generated.xlsx         dense-grid augmented workbooks
figures/original/        heatmaps before augmentation
figures/generated/       heatmaps after augmentation
manifest.json            manifest for this batch
```

## Manifests

Every discovered batch has a manifest at:

```text
manifests/<batch>/manifest.json
```

For active processed batches, the same manifest is also copied to:

```text
processed/<batch>/manifest.json
```

Regenerate manifests with:

```powershell
.\.venv\Scripts\python.exe -m src.data.manifest
```

or for one batch:

```powershell
.\.venv\Scripts\python.exe -m src.data.manifest --batch 2026-04-26
```

## Plotting And Augmentation

Original FL heatmaps:

```powershell
.\.venv\Scripts\python.exe -m src.analysis.fl_plot --data-dir data\processed\2026-04-26 --out-dir data\processed\2026-04-26\figures\original --pattern "*-FL.xlsx" --suffix=-FL.xlsx --summary-name ALL_original_FL_summary.png --output-label-suffix _original_FL
```

Generate dense-grid augmented workbooks:

```powershell
.\.venv\Scripts\python.exe -m src.data.generate --batch 2026-04-26
```

Generated heatmaps:

```powershell
.\.venv\Scripts\python.exe -m src.analysis.fl_plot_generated --batch 2026-04-26
```

Per-Re augmentation and plots:

```powershell
.\.venv\Scripts\python.exe -m src.data.generate_per_re --batch 2026-04-26
.\.venv\Scripts\python.exe -m src.analysis.fl_plot_per_re --batch 2026-04-26
```

## Legacy Data

Older folders such as `processed/2026-04-21`, `processed/2026-04-22`,
`processed/FL-Files`, and root-level files under `processedv2/` are historical
outputs. Keep them for reference, but new work should use the batch-date layout
above.
