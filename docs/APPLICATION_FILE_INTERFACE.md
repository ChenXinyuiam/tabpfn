# 应用者接口文件清单

当前统一使用 `python -m src <command>`。原有实验路径和兼容包装已删除。
物理合成与原生回归输出必须区分，不把未确认单位的 FD-u 当作力。

| 命令 | 实现 | 输入与用途 | 主要输出 |
| --- | --- | --- | --- |
| system audit | src/data/audit.py | 显式源 manifest、单位、目标与几何 | 审核报告、数据快照 |
| system evaluate | src/evaluation/system.py | 审核快照和模型/物理配置；按几何五折 | 逐折预测、指标、划分 |
| system train / predict | src/workflows/system.py | 训练快照或模型包和新几何请求 | 模型上下文包、逐点 FD/FL；满足物理条件才合成与分类 |
| fd-to-fl | src/regression/fd_to_fl.py | 同批次 FD/FL 配对，FD 目标 u；在 FL 原始位置预测 u | *-FL-with-u-pred.xlsx、summary.csv、run_manifest.json |
| interpolate | src/regression/interpolation.py | 上一步含 fl/u_pred 的文件 | *-FL-with-u-pred-interpolated.xlsx、汇总、manifest |
| fl-at-fd | src/regression/fl_at_fd.py | 单个 FL + FD，预测 FD 位置的 FL | 增强 FL、含 fl_pred 的 FD、manifest |
| fd-from-fl | src/regression/fd_from_fl.py | 同几何不同半径 FD/FL，辅助 FL 映射 | 增强 FL、FD-u 预测和指标 |
| regression-evaluate | src/regression/evaluation.py | 标准 FD/FL 文件，空间留出或新几何留出 | 逐点预测与回归指标 |
| regression-predict | src/regression/predict_fields.py | 原生 FD/FL 数据与查询位置 | 原生 u/FL、统计与空间诊断 |
| interpolate-raw-fl | src/regression/raw_fl_interpolation.py | 原始 re/x/z/fl 或 flx | 每文件数学插值工作簿及汇总 |
| interpolate-raw-fl-tabpfn | src/regression/raw_fl_tabpfn.py | 同上，使用 TabPFN 补点 | 每文件 TabPFN 插值结果及汇总 |
| interpolate-fly | src/regression/fly_tabpfn.py | re/x/z/fly；邻点与完整四角单元 | 每文件独立 FLy 预测工作簿，不合并汇总工作簿 |
| fl-lines | src/regression/fl_lines.py | 默认两条 750-400/750-530 反向 FL 与对应 FD 文件 | 数学插值及 FD 位置 FL 映射 |
| focus-predict | src/focus_inspection/predict_focus.py | 保存的配方、新批次截面数据 | F/UF 概率、判定及 REVIEW 标记 |

每个命令的参数以 `--help` 为准。模板适配层为 `src/data/template_schema.py`。
网页服务由 `web-demo/tabpfn_service.py` 提供；请按其 README/API_CONTRACT 配置模型和物理转换，
不能仅凭 API 返回数值就认定是真实 FD/FL/Fzong 或已通过新几何验证。

## 最小位置匹配流程

```powershell
python -m src fd-to-fl --help
python -m src interpolate --help
```

批次目录：`data/rawdata/<batch>/FD/*.xlsx` 和 `FL/*.xlsx`。
FD 至少提供 u，FL 至少提供 fl；还需要模板要求的几何、Re 和 x/z。
第一步产生 u_pred，第二步加密 fl 和 u_pred。模型文件路径可以通过命令参数指定。

## 研究流程不是部署模型

`fd-transfer-tune`、`fd-transfer-ensemble` 是嵌套验证与组合实验；
`fd-transfer-audit`、`fd-zero-audit` 是数据诊断。
这些命令保留原始标签和划分记录，不会自动替换网页生产模型。
详见 [FD 优化报告](FD_R2_09_AND_ZERO_AUDIT.md)。
