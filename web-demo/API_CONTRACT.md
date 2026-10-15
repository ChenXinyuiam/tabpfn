# TriFocus ForceLab API Contract

## `GET /api/health`

返回模型是否加载、是否使用 TabPFN、默认采样点数、Re 列表和曲率半径范围。

## `POST /api/predict-triangle`

### 自动采样请求

```json
{
  "side_ab_um": 500,
  "side_ac_um": 500,
  "side_bc_um": 600,
  "curvature_radius_um": 1200
}
```

服务会在三角形内部生成 400 个 Halton 点，并对 `Re = 6, 12, ..., 60` 预测。

### 指定位置请求

```json
{
  "side_ab_um": 500,
  "side_ac_um": 500,
  "side_bc_um": 600,
  "curvature_radius_um": 1200,
  "positions": [
    {"point_id": "p1", "x_um": 300, "z_um": 100},
    {"point_id": "p2", "x_um": 300, "z_um": 20}
  ]
}
```

指定点必须位于三角形内部。`x_um`、`z_um` 单位均为微米。

### 逐点输出

```text
fl_pred
fd_pred
fnet_pred = fl_pred + fd_pred
fnet_x
fnet_z
fnet_magnitude
```

兼容字段仍包括 `fl_x_pred`、`fl_y_pred`、`fd_x_pred`、`fd_y_pred`、`total_force_x`、`total_force_y`、`total_force_magnitude`。

`prediction_meta.data_source` 为：

- `provided_positions`：使用请求提供的位置。
- `halton_sampled_positions`：服务自动采样。

必须检查 `prediction_meta.uses_tabpfn`。只有它为 `true` 时，输出才来自已加载的 TabPFN 模型。

## 单次文件预测

```powershell
.\.venv\Scripts\python.exe web-demo\tabpfn_service.py `
  --model-dir models\tabpfn_server_500 `
  --predict-json web-demo\examples\triangle_with_positions.json `
  --output-json results\triangle_force_with_positions.json
```

完整验证边界见 `docs/FD_FL_CLOSED_LOOP.md`。
