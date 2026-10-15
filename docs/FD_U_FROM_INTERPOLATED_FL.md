# 使用插值 FL 预测 FD 的 u

适用数据目录：`data/rawdata/2026-06-20`。

处理流程：

1. 按文件名中的几何、方向和半径配对 FL/FD 文件。
2. 在每个 `Re` 内对 FL 坐标加入相邻中点，并在原始点凸包内插值 `fl` 和距离特征。
3. 将补点后的 FL 映射到对应 FD 坐标，生成 `fl_interp`。
4. 以同一几何和方向的其他半径 FD 文件作为训练集，用 TabPFN 预测当前文件的 `u`。
5. 保留原始 `u`，新增 `u_pred`、`u_residual` 和 `u_abs_error`。

这种留一半径文件预测避免模型使用当前文件的 `u` 训练后再预测自身。半径 30 属于从半径 10/20 向外推，指标可能明显弱于半径 10/20，应结合 `metrics.csv` 使用。

运行命令：

```powershell
.\.venv\Scripts\python.exe -m src fd-from-fl `
  --batch-dir data\rawdata\2026-06-20 `
  --output-dir results\fd_u_from_fl_2026-06-20 `
  --n-estimators 1 `
  --max-train-rows 10000 `
  --chunk-size 2000
```

输出目录：

- `fl_augmented/`：13 份 FL 补点结果。
- `fd_u_predictions/`：13 份 FD 全点 `u_pred` 结果。
- `metrics.csv`：逐文件指标。
- `run_manifest.json`：配对、清洗、特征和模型参数。

已知数据质量记录：部分 `750-530` FL 文件末尾含 `h=0`、`re/angle` 为空的占位行。程序只删除这些无效行，并在 manifest 的 `fl_invalid_rows_dropped` 中记录数量，原文件不修改。
