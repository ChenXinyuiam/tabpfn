# FL 加点并在 FD 坐标预测

新入口支持 `docs/templates` 中的英文列名，也兼容历史中文列名。内部统一把长度转换为 `μm`，执行顺序为：

1. 读取并校验 FL、FD 文件。
2. 对每个 `Re` 的 FL 截面分别加入相邻坐标中点，仅保留原采样点凸包内的点。
3. 用 cubic、linear、nearest 依次回退插值得到新增点的 FL。
4. 使用原始点和新增点共同 `fit` TabPFN。
5. 预测 FD 文件中全部点的 `fl_pred`。

运行示例：

```powershell
.\.venv\Scripts\python.exe -m src fl-at-fd `
  --fl path\to\FL数据.xlsx `
  --fd path\to\迪恩速度数据.xlsx `
  --output-dir results\fl_at_fd_points_tabpfn
```

新模板默认长度单位为 `μm`。历史文件自动判断单位；如果自动判断不合适，明确指定：

```powershell
--fl-length-unit m --fd-length-unit m
```

输出包括：

- `*_augmented.xlsx`：FL 原始点和新增点，`generated` 标记新增点。
- `*_with_fl_pred.xlsx`：标准化后的全部 FD 点和 `fl_pred`。
- `run_manifest.json`：字段迁移、单位转换、特征、模型参数和行数。

模板文件本身只有“列名/单位/解释”三行，不能直接作为数据运行。实际数据文件第一行必须是模板中的英文列名，第二行开始放逐点数据。
