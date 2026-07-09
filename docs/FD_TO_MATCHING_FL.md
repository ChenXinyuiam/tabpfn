# 使用同名 FD 文件预测 FL 位置的 u

本流程对 `data/rawdata/2026-06-20` 中的文件严格按名称配对。例如：

```text
FD/750-320-10-FD-结果.xlsx -> FL/750-320-10-FL.xlsx
```

每个 FD 文件独立拟合 TabPFN，以 `u` 为目标，仅预测其同名 FL 文件的原始位置。FL 不做补点，原始 `fl` 列保留，输出新增：

- `u_pred`
- `nearest_fd_distance_um`
- `prediction_status`
- `fd_training_file`
- `fl_source_file`

运行命令：

```powershell
.\.venv\Scripts\python.exe -m src.experiments.predict_fl_positions_from_matching_fd_tabpfn `
  --batch-dir data\rawdata\2026-06-20 `
  --output-dir results\fd_to_matching_fl_2026-06-20 `
  --n-estimators 1 `
  --max-train-rows 10000
```

FL 文件中的无效占位行会保留，`prediction_status=invalid_input` 且 `u_pred` 为空；所有具有完整几何、`Re`、`x/z` 的位置均进行预测。
