# 对预测后的 FL 文件逐文件插值

输入目录：`results/fd_to_matching_fl_2026-06-20`。

每个 `*-FL-with-u-pred.xlsx` 文件独立处理，文件之间不共享插值点或数值；同一文件中不同 `Re` 也分别处理。

处理内容：

1. 在相邻 `x/z` 坐标之间加入中点。
2. 只保留原始采样点凸包内的新增点。
3. 分别插值原始 `fl`、预测 `u_pred` 和距离特征。
4. 原始点的 `fl/u_pred` 精确恢复，不使用插值值覆盖。
5. 使用 `generated`、`fl_fill_method`、`u_pred_fill_method` 标记来源。

如果同一文件内存在重复的 `Re/x/z` 位置，先分别对 `fl` 和 `u_pred` 取均值，再将其作为一个唯一物理位置参与插值；聚合行数记录在 manifest 的 `duplicate_position_rows_aggregated`。

运行命令：

```powershell
.\.venv\Scripts\python.exe -m src interpolate
```

输出目录：`results/fd_to_matching_fl_2026-06-20-interpolated`。
