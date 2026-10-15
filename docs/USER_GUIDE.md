# 当前代码使用指南

维护入口统一为 `python -m src <command>`；根目录 `main.py` 和安装后的 `fdf` 使用相同分发器。
旧研究脚本及兼容转发入口已删除，命令迁移见 [清理记录](CODE_CLEANUP_20261015.md)。

## 运行环境

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m src --help
.\.venv\Scripts\python.exe -m src doctor
```

新环境按 README 使用 `uv sync --locked`。不需要安装本地旧 AutoTabPFN 扩展。

## 审核与完整系统

```powershell
python -m src system audit --help
python -m src system evaluate --help
python -m src system train --help
python -m src system predict --help
python -m src system demo --out-dir results/system/demo
```

真实数据需要显式 manifest、单位及目标定义，参见 [系统流程](SYSTEM_WORKFLOW.md)。
demo 仅验证合成数据上的执行流程，不代表真实数据精度。
没有确认 FD/FL 量纲及转换关系时，不把原生 u 直接作为力合成 Fzong。

## FD 与 FL 位置匹配

批次目录包含 `FD/*.xlsx`、`FL/*.xlsx`，同名几何/半径文件配对。
标准化字段由 `src/data/template_schema.py` 管理：
FD 需要目标 `u`，FL 需要 `fl`，两者需要几何、Re 和 x/z。
这与未知三角形外推是不同任务。

```powershell
python -m src fd-to-fl --help
python -m src fl-at-fd --help
python -m src fd-from-fl --help
python -m src interpolate --help
```

- `fd-to-fl`：学习 FD 的 u，在匹配 FL 的位置产生 u_pred，不是用 FD 值直接学习 FL。
- `fl-at-fd`：学习 FL，在 FD 位置产生 fl_pred。
- `fd-from-fl`：同几何跨半径验证，将插值 FL 映射到 FD 位置作为辅助特征。
- `interpolate`：对已含 fl/u_pred 的预测工作簿按 Re 加点。
- `fl-lines`：复现两条指定反向 FL 线到 FD 位置的数学插值；默认源文件固定，不是新三角形模型。

具体字段和产物见 [应用接口](APPLICATION_FILE_INTERFACE.md)。

## 原始 FL/FLy 加点

```powershell
python -m src interpolate-raw-fl --help
python -m src interpolate-raw-fl-tabpfn --help
python -m src interpolate-fly --help
```

FLy 的默认命令使用 TabPFN 预测新增点的 fly；只有四角均存在时才生成单元中心。
原始行保留，每个输入工作簿独立输出。另有 `interpolate-fly-linear` 数学插值参考，
不能把该参考的算术平均当成 TabPFN 预测。

## 新几何验证与 FD 优化

```powershell
python -m src regression-evaluate --help
python -m src fd-rotation-audit --help
python -m src fd-transfer-tune --help
python -m src fd-transfer-ensemble --help
python -m src fd-transfer-audit --help
python -m src fd-zero-audit --help
```

FD transfer 使用外层五折几何留出、内层三折选型，分别预测 u0/u90，再合成旋转角度。
保留旧基线实现用于对照，不把较低成绩的候选当作废代码删除。
数据快照、训练点清单、内层指标、选型权重和逐点预测均保存在独立 run 中。
原有 0.9303 是特定实验的全角度几何等权 R²，不是本次清理重新训练得到的结果。

## 聚焦分类

```powershell
python -m src focus-audit --help
python -m src focus-optimize --help
python -m src focus-predict --help
```

当前 September 标签驱动的流程为 F/UF 二分类，不自动区分 WF/F/UF。
独立 WF/F/UF 标签及显式类别配置是三分类系统的前提。

## 数据处理与测试

`python -m src data show` 展示仍维护的中文表预处理管线。
标准英文表可直接通过显式审核或模板标准化输入，不必再跑一遍历史清洗与缩放。
保留已是 um 的数值，单位不清楚时应确认来源，不靠修改目标值提升分数。

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src utils tests
```

