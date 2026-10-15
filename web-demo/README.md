# TriFocus ForceLab

## 两种后端

- `tabpfn_service.py`：真实加载 TabPFN joblib 模型，支持指定位置或自动采样，并输出 FD、FL 与 Fnet。
- `server.py` / `server.js`：网页静态服务与规则回退。只有接口返回的 `prediction_meta.uses_tabpfn=true` 才能称为 TabPFN 预测。

## 单次 TabPFN 推理

提供位置：

```powershell
cd D:\2026\tab\FDFLPredictions
.\.venv\Scripts\python.exe web-demo\tabpfn_service.py `
  --model-dir models\tabpfn_server_500 `
  --predict-json web-demo\examples\triangle_with_positions.json `
  --output-json results\triangle_force_with_positions.json
```

不提供位置时，服务自动为每个 Re 使用 400 个 Halton 采样点：

```powershell
.\.venv\Scripts\python.exe web-demo\tabpfn_service.py `
  --model-dir models\tabpfn_server_500 `
  --predict-json web-demo\examples\triangle_auto_sample.json `
  --output-json results\triangle_force_auto_sample.json
```

## 启动 TabPFN API

PowerShell 中先配置模型，再启动服务：

```powershell
$env:TRIFOCUS_MODEL_MODE='joblib'
$env:TRIFOCUS_FLX_MODEL='models/tabpfn_server_500/fl_x.joblib'
$env:TRIFOCUS_FLY_MODEL='models/tabpfn_server_500/fl_y.joblib'
$env:TRIFOCUS_FDX_MODEL='models/tabpfn_server_500/fd_x.joblib'
$env:TRIFOCUS_FDY_MODEL='models/tabpfn_server_500/fd_y.joblib'
.\.venv\Scripts\python.exe web-demo\tabpfn_service.py --host 127.0.0.1 --port 8100
```

接口：

```text
GET  http://127.0.0.1:8100/api/health
GET  http://127.0.0.1:8100/api/schema
POST http://127.0.0.1:8100/api/predict-triangle
```

请求格式与字段见 `web-demo/API_CONTRACT.md`；模型验证边界见 `docs/FD_FL_CLOSED_LOOP.md`。

## 当前模型边界

- 当前训练表为 `models/tabpfn_server/training.csv`，共 7829 行，来自 14 个匹配 FD/FL 文件。
- 真实标签只有单方向投影：`fl_x_pred` 和 `fd_x_pred`。
- `fl_y_pred`、`fd_y_pred` 是 0 占位，因此当前主要物理输出是 `fnet_pred = fl_pred + fd_pred`。
- 未知三角形的严格分组验证尚未通过，不能把能运行的外推结果等同于已证明准确。

## 网页演示

直接打开 `web-demo/index.html` 可使用浏览器规则回退。网页展示可以演示交互流程，但不能作为 TabPFN 模型效果证据。
