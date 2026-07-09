# Data Folder Standard

本文档定义 `data/` 目录的职责边界。简单说：

- `rawdata/` 回答“数据从哪里来”
- `interim/` 回答“中间处理到了哪一步”
- `processed/` 回答“模型最终读哪份数据”
- `manifests/` 回答“这批数据的账本是什么”
- `legacy/` 回答“旧数据先放哪里不打扰主流程”

## 推荐主结构

```text
data/
  rawdata/        原始交付数据 + 标准化入口
  interim/        中间产物，可删除后重跑
  processed/      最终训练/推理入口
  manifests/      数据清单、校验结果、来源记录
  legacy/         历史旧数据归档
```

## 当前目录状态

| 目录 | 状态 | 说明 |
| --- | --- | --- |
| `rawdata/` | 保留 | 放数据提供方交付文件，以及标准化后的 `standard/samples.csv` |
| `interim/` | 保留 | 新的清洗、缩放、特征工程中间结果统一放这里 |
| `processed/` | 保留 | 模型、实验、推理默认只读这里 |
| `manifests/` | 保留 | 每个批次的数据账本 |
| `legacy/` | 保留 | 用于逐步收纳历史旧结构，不参与默认训练 |

原顶层 `cleaned/`、`scaled/`、`processedv2/` 已经收拢到 `interim/`：

```text
cleaned/      -> interim/cleaned/
scaled/       -> interim/scaled/
processedv2/  -> interim/features/
```

少量同名旧副本放在 `legacy/top_level_duplicates/`，仅用于回溯。

## 新批次标准流水线

```text
rawdata/<batch>/standard/samples.csv
        ↓
interim/<batch>/cleaned/
        ↓
interim/<batch>/scaled/
        ↓
interim/<batch>/features/
        ↓
processed/<batch>/
```

`<batch>` 使用 `YYYY-MM-DD`。同一天多批次使用 `YYYY-MM-DD-batch1`、`YYYY-MM-DD-batch2`。

## rawdata 规范

新数据推荐这样放：

```text
data/rawdata/<batch>/
  originals/                 原始 Excel/CSV，保持数据提供方交付状态
  standard/
    samples.csv              项目内部统一入口
    manifest.json            批次说明和质量摘要
    files.csv                可选，原始文件到标准数据的映射
```

数据提供方不需要手写所有标准列。标准化脚本负责补充：

- `sample_id`
- `batch`
- `source_file`
- `point_index`
- 可由几何计算出的派生列

更完整的 rawdata 标准见 `docs/RAWDATA_STANDARD.md`。

## interim 规范

`interim/` 只放可重跑的中间数据。这里的数据可以被删除，但删除后需要能从 `rawdata/` 重新生成。

推荐子目录：

```text
data/interim/<batch>/
  cleaned/
  scaled/
  features/
  augmented/
  reports/
```

不要把人工整理后唯一存在的数据只放在 `interim/`。如果无法重跑，它应该放回 `rawdata/originals/` 或记录到 manifest。

## processed 规范

`processed/<batch>/` 是唯一稳定建模入口。训练、实验、推理代码原则上只读这里。

推荐结构：

```text
data/processed/<batch>/
  samples.parquet or samples.csv
  train.csv
  valid.csv
  test.csv
  manifest.json
  figures/
```

如果一个批次同时支持 FL、FD、focus，可以放在同一个点级 `samples` 表里，通过目标列区分：

- `target_fl_x`
- `target_fl_y`
- `target_fd_x`
- `target_fd_y`
- `focus_label`

## manifests 规范

每个批次至少有：

```text
data/manifests/<batch>/manifest.json
```

manifest 应记录：

- 批次名和 schema 版本
- 原始文件列表
- 标准数据文件路径
- 行数、列数、缺失值、重复 `sample_id` 数
- 从 rawdata 到 processed 的处理步骤
- 当前是否可用于训练

## 迁移规则

1. 新数据先按 `rawdata/<batch>/originals + standard` 放。
2. 新中间产物写入 `interim/<batch>/...`。
3. 新训练入口写入 `processed/<batch>/...`。
4. 历史副本只读保存在 `legacy/`，不再作为新实验入口。

## 当前旧流水线

以下命令仍描述当前历史脚本的处理顺序：

```powershell
.\.venv\Scripts\python.exe -m src.data.clean --batch 2026-04-26
.\.venv\Scripts\python.exe -m src.data.Feature_scale --batch 2026-04-26
.\.venv\Scripts\python.exe -m src.data.calculate --batch 2026-04-26
.\.venv\Scripts\python.exe -m src.data.Reverse_FL --batch 2026-04-26
```

后续建议把这些脚本的输出逐步改到 `interim/<batch>/...`，最终只让训练代码读取 `processed/<batch>/...`。
