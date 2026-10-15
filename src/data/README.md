# 数据模块

## 两类入口

- `audit.py` / `contracts.py`：显式 manifest、单位、目标来源和几何审核，是新系统的数据入口。
- `template_schema.py`：标准或历史 FD/FL 列名适配、单位处理及基本校验，供原生位置匹配使用。

不要把“输入是旧中文表”与“代码已过时”混为一谈。以下预处理模块仍被当前管线调用，
因此保留；标准化数据不需要重复执行每个历史阶段。

## 中文表处理管线

```text
data/rawdata/<batch>
  -> clean       -> data/interim/cleaned/<batch>
  -> scale       -> data/interim/scaled/<batch>
  -> calculate   -> data/interim/features/<batch>
  -> publish     -> data/processed/<batch>
  -> generate / generate_per_re
```

| 阶段 | 模块 | 作用 |
| --- | --- | --- |
| clean | clean.py | 列名规范化、记录与处理无效行 |
| scale | Feature_scale.py / units.py | 按列判断并归一化长度到 um、面积到 um²；不重复放大已转换数据 |
| calculate | calculate.py | 按当前输入几何计算派生字段 |
| publish | stages.py / Reverse_FL.py | 发布 FL；sym 文件反转 FL 符号 |
| generate | generate.py | 共享 Re 稠密网格 |
| generate_per_re | generate_per_re.py | 每个 Re 独立插值 |

这些阶段有各自数据假设，不能在未知源格式上盲目整条执行。
网格重心坐标审核是 `src/regression/transfer_fd.py` 的实验特征步骤，不会覆盖原始工作簿。

## 命令

```powershell
python -m src data show
python -m src data list
python -m src data run --batch 2026-04-26 --until publish
python -m src system audit --help
```

单阶段模块入口仍支持 `python -m src.data.clean` 等命令。
`fd_data.py`、`fl_data.py` 提供专用处理；`manifest.py` 写批次清单；
`explore.py` 仅用于探索统计。

旧的自动 Re-FL 修复脚本与特定五月批次距离生成脚本已移除。
不默认把 u=0 改成正数，也不根据测试误差修改标签。详见仓库 docs 中的零值审核报告。
