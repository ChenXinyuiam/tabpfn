# Data Pipeline

三角形几何 / 流力（FL）数据处理管线。

## 数据流

```
原始 Excel (.xlsx)
  data/rawdata/<batch>/
       │
       │  [clean]      列名规范化、删空列/骨架行、
       │               插补缺失距离、删无效 FL
       ▼
  data/cleaned/<batch>/
       │
       │  [scale]      面积 ×1e12，其他物理列 ×1e6
       ▼
  data/scaled/<batch>/
       │
       │  [calculate]  面积_calc、角和、边长、
       │               重心坐标 (lambda_*)
       ▼
  data/processedv2/<batch>/
       │
       │  [publish]    FL 文件直接复制，
       │               sym 文件 FL 取反
       ▼
  data/processed/<batch>/
       │
       ├── [generate]         ──► {label}_generated.xlsx
       │                         (所有 Re 共享稠密网格)
       │
       └── [generate_per_re]  ──► {label}_generated_per_re.xlsx
                                  (逐 Re 独立稠密网格)
```

## 使用方式

### 统一入口（推荐）

```bash
# 运行完整管线
python -m src.data.pipeline run --batch 2026-04-26

# 运行单个阶段
python -m src.data.pipeline run --stage clean --batch 2026-04-26

# 从指定阶段开始运行到末尾
python -m src.data.pipeline run --from calculate

# 运行到指定阶段为止
python -m src.data.pipeline run --until publish

# 查看管线结构
python -m src.data.pipeline show

# 列出所有阶段
python -m src.data.pipeline list
```

### 传统单阶段调用（仍然可用）

```bash
python -m src.data.clean --batch 2026-04-26
python -m src.data.Feature_scale --batch 2026-04-26
python -m src.data.calculate --batch 2026-04-26
python -m src.data.Reverse_FL --batch 2026-04-26
python -m src.data.generate --data-dir data/processed/2026-04-26
python -m src.data.generate_per_re --data-dir data/processed/2026-04-26
```

### 编程接口

```python
from src.data import main_pipeline

# 运行完整管线
main_pipeline.run(batch="2026-04-26")

# 运行到 publish 阶段
main_pipeline.run(batch="2026-04-26", until="publish")

# 查看 DAG
print(main_pipeline.show())
```

## 阶段说明

| 阶段 | 模块 | 输入 | 输出 |
|------|------|------|------|
| clean | `clean.py` | `data/rawdata/` | `data/cleaned/` |
| scale | `Feature_scale.py` | `data/cleaned/` | `data/scaled/` |
| calculate | `calculate.py` | `data/scaled/` | `data/processedv2/` |
| publish | (内联) | `data/processedv2/` | `data/processed/` |
| generate | `generate.py` | `data/processed/` | `data/processed/` |
| generate_per_re | `generate_per_re.py` | `data/processed/` | `data/processed/` |

### 独立工具

| 模块 | 用途 |
|------|------|
| `Repair_base_on_Re.py` | 基于 Re-FL 回归修复异常 FL 值（独立分支） |
| `explore.py` | 任意阶段 Excel 文件的探索性数据分析 |
| `manifest.py` | 生成全管线文件清单 JSON |

## 变更说明

本次重构 (2026-05):

### 改动
- **新增** `pipeline.py` — `PipelineStage` / `Pipeline` 核心编排抽象
- **新增** `stages.py` — 6 个具体阶段定义，委托给已有模块的核心函数
- **新增** `README.md` — 本文件
- **修改** `__init__.py` — 导出 `Pipeline`, `PipelineStage`, `main_pipeline`
- **修复** `generate.py` — 删除无法运行的死代码 `main()`（引用了未导入的 `os`）
- **修复** `generate_per_re.py` — 删除过时的死代码 `main()`，移除 `DATA_DIR` 全局变量，签名改为接受 `Path`

### 未改动
`clean.py`, `Feature_scale.py`, `calculate.py`, `Reverse_FL.py`, `Repair_base_on_Re.py`, `explore.py`, `manifest.py` 一字未改。所有 `python -m src.data.xxx` 独立调用保持可用。

## 回滚

所有修改是增量叠加：

- **删除新文件回退**：`rm src/data/{pipeline,stages,README}.py.md`
- **恢复被修改文件**：`git checkout -- src/data/{generate,generate_per_re,__init__}.py`
