# 用户使用指南

## 8. 运行现有 FD-to-FL 预测流程

当前已有流程是：

```text
匹配 FD 文件训练 TabPFN
-> 在原始 FL 坐标位置预测 u_pred
-> 对 FL 文件里的 fl 和 u_pred 一起做表内插值
```

这个流程不是“先插值 FL，再预测插值后全部 FL 坐标”的严格新流程，而是项目里已经存在、可以直接运行的旧流程。

需要数据：

```text
data/rawdata/2026-06-20/FD/
data/rawdata/2026-06-20/FL/
models/offline_tabpfn_2_5/
```

先检查目录：

```powershell
dir data\rawdata\2026-06-20
dir data\rawdata\2026-06-20\FD
dir data\rawdata\2026-06-20\FL
dir models\offline_tabpfn_2_5
```

模型目录中至少应包含：

```text
config.json
tabpfn-v2.5-regressor-v2.5_default.ckpt
```

### 8.1 先跑单文件烟雾测试

先用一个样本确认环境、模型、数据都能跑通。`750-320-10` 需要替换成数据中真实存在的 key。

```powershell
.\.venv\Scripts\python.exe -m src.experiments.predict_fl_positions_from_matching_fd_tabpfn `
  --batch-dir data\rawdata\2026-06-20 `
  --output-dir results\fd_to_matching_fl_2026-06-20-test `
  --device cuda `
  --n-estimators 1 `
  --max-train-rows 3000 `
  --chunk-size 5000 `
  --only 750-320-10
```

如果没有 CUDA GPU，改成：

```powershell
--device cpu
```

### 8.2 GPU 正式运行命令

如果机器是 i9-12900K + RTX A4000 这类配置，推荐：

```powershell
.\.venv\Scripts\python.exe -m src.experiments.predict_fl_positions_from_matching_fd_tabpfn `
  --batch-dir data\rawdata\2026-06-20 `
  --output-dir results\fd_to_matching_fl_2026-06-20 `
  --device cuda `
  --n-estimators 1 `
  --max-train-rows 10000 `
  --chunk-size 10000
```

如果显存不足，把 `--chunk-size` 降到：

```text
5000
```

或：

```text
2000
```

第一步成功后会生成：

```text
results/fd_to_matching_fl_2026-06-20/summary.csv
results/fd_to_matching_fl_2026-06-20/run_manifest.json
results/fd_to_matching_fl_2026-06-20/*-FL-with-u-pred.xlsx
```

### 8.3 对预测后的 FL 文件做插值

第一步成功后运行：

```powershell
.\.venv\Scripts\python.exe -m src.experiments.interpolate_predicted_fl_files `
  --input-dir results\fd_to_matching_fl_2026-06-20 `
  --output-dir results\fd_to_matching_fl_2026-06-20-interpolated
```

输出：

```text
results/fd_to_matching_fl_2026-06-20-interpolated/summary.csv
results/fd_to_matching_fl_2026-06-20-interpolated/run_manifest.json
results/fd_to_matching_fl_2026-06-20-interpolated/*-FL-with-u-pred-interpolated.xlsx
```

### 8.4 结果检查

第一步重点看：

```text
results/fd_to_matching_fl_2026-06-20/summary.csv
```

主要字段：

```text
fd_rows_available
fd_rows_used
fl_rows
fl_rows_predicted
fl_invalid_rows
u_pred_min
u_pred_mean
u_pred_max
nearest_fd_distance_um_mean
nearest_fd_distance_um_max
```

第二步重点看：

```text
results/fd_to_matching_fl_2026-06-20-interpolated/summary.csv
```

主要字段：

```text
rows_in
valid_rows_in
rows_out
rows_added
original_rows_out
fl_method_counts
u_pred_method_counts
u_pred_min
u_pred_mean
u_pred_max
```

## 9. 运行 `src.inference`

`src.inference` 不直接读取 raw Excel。它默认读取已经切好 5 折的 CSV 目录：

```text
data/triangle_groups_fixed_v2_focus/combined/
  <shape>/
    <R>/
      <Re>/
        fold1.csv
        fold2.csv
        fold3.csv
        fold4.csv
        fold5.csv
```

只有这个目录存在时才运行：

```powershell
dir data\triangle_groups_fixed_v2_focus\combined
```

运行 FD 或 FL 评估：

```powershell
.\.venv\Scripts\python.exe -m src.inference --mode fd
.\.venv\Scripts\python.exe -m src.inference --mode fl
```

聚焦模式最重：

```powershell
.\.venv\Scripts\python.exe -m src.inference --mode focus
```

不要在慢机器上第一步就跑 `focus`。

## 10. 数据和结果管理

GitHub 只保存代码、轻量文档和少量元数据。下面这些不建议进 Git：

```text
data/rawdata/
data/interim/
data/processed/
models/
results/
outputs/
dataAnalysis/
.venv/
```

给新用户交付项目时，建议发送：

```text
项目代码
offline wheelhouse
requirements-freeze.txt
models/offline_tabpfn_2_5
当前任务需要的数据批次
本指南
```

如果只让新用户运行现有 FD-to-FL 预测流程，最少需要：

```text
项目代码
Python 3.12 虚拟环境或离线依赖包
models/offline_tabpfn_2_5/
data/rawdata/2026-06-20/FD/
data/rawdata/2026-06-20/FL/
```

## 11. 常见问题

### `python --version` 显示 Python 3.9

只要下面命令显示 Python 3.12 就可以：

```powershell
.\.venv\Scripts\python.exe --version
```

运行项目时不要直接用 `python`，用：

```powershell
.\.venv\Scripts\python.exe
```

### `No module named tabpfn`

安装真正的 TabPFN 包：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links E:\TEMP\FDFL_offline_kit\wheelhouse tabpfn==8.0.3 tabpfn-common-utils==0.2.21
```

### `pip show tabpfn` 显示版本 0.1.0

这是项目本身的包名和第三方 `tabpfn` 重名造成的。重新安装真正的 TabPFN，并暂时不要执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

### CUDA 是 False

检查 NVIDIA 驱动和 CUDA 版 PyTorch 是否正确。短期可以用 CPU 跑小任务：

```powershell
--device cpu --max-train-rows 3000 --chunk-size 500
```

### `--only` key 不存在

`--only` 的 key 来自 FD/FL 文件名配对。如果不确定 key，先去掉 `--only`，或根据报错提示查看可用 key。

### 找不到 Excel 数据

GitHub 代码本身不能跑完整预测。需要另外复制数据批次，例如：

```text
data/rawdata/2026-06-20/FD/
data/rawdata/2026-06-20/FL/
```

### Git 提示 dubious ownership

Windows 上如果仓库所有者不一致，Git 可能提示安全目录问题。临时查询可用：

```powershell
git -c safe.directory=D:/2026/tab/FDFLPredictions status
```

