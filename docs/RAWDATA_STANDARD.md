# FL/FD Rawdata Standard

本文档定义项目后续新增 FL、FD rawdata 的标准入口格式。目标是把“实验原始文件可追溯”和“模型流水线可稳定读取”分开：原始文件可以保留，但每个批次必须提供一份机器可读的标准数据和清单。

## 设计原则

1. rawdata 不再依赖文件名猜字段。几何、半径、方向、目标类型等关键信息必须进入表格列或 manifest。
2. 同一物理量只保留一个标准列名。旧列名通过导入脚本映射，不继续扩散到训练代码。
3. 单位固定写入列名或 manifest，避免同一列有时是 m、有时是 um。
4. 每一行代表一个采样点。FL、FD、focus label 都可以挂在同一行上，缺失目标允许为空。
5. 后续新任务通过新增列扩展，不改变已有列语义。

## 推荐目录结构

```text
data/rawdata/<batch>/
  originals/                 # 可选，实验交付的原始 Excel，保持原样
  standard/
    samples.csv              # 必需，统一后的点级数据主表
    manifest.json            # 必需，批次级元数据和数据质量摘要
    files.csv                # 可选，原始文件到标准样本的映射
```

`<batch>` 使用 `YYYY-MM-DD`，同一天多批次用 `YYYY-MM-DD-batch1`。不建议再使用 `2026-5-31` 这类非零填充日期。

保留 `data/rawdata/<batch>/originals/` 的价值是审计和回溯；训练、清洗、特征工程默认只读 `standard/samples.csv` 和 `standard/manifest.json`。

## samples.csv 标准列

### 标识列

| 列名 | 必需 | 类型 | 说明 |
| --- | --- | --- | --- |
| `sample_id` | 是 | string | 行级唯一 ID，建议为 `<batch>_<geometry_id>_<radius>_<direction>_<re>_<point_index>` |
| `batch` | 是 | string | 批次名 |
| `source_file` | 是 | string | 原始文件相对路径 |
| `geometry_id` | 是 | string | 几何编号，如 `690-340` |
| `direction` | 是 | string | `normal` 或 `reverse`；不确定时填 `unknown` |
| `radius_group` | FD 必需 | string/int | FD 目录中的 `10/20/30`；FL 没有该概念时为空 |
| `point_index` | 是 | int | 同一文件内的点序号，从 0 开始 |

### 几何与位置列

统一使用 SI 单位 m。已有 FL 历史文件里 `BC/AB/AC/X/Z` 若是 m，直接映射；若确认为 um，导入时除以 `1e6`。

| 标准列名 | 必需 | 旧格式常见来源 |
| --- | --- | --- |
| `base_m` | 是 | `BC`, `三角形底边` |
| `height_m` | 是 | `高` |
| `side_ab_m` | 可选 | `AB` |
| `side_ac_m` | 可选 | `AC` |
| `perimeter_m` | 可选 | `周长` |
| `area_m2` | 可选 | `面积` |
| `angle_a_deg` | 是 | `角度A`, `角度1` |
| `angle_b_deg` | 是 | `角度B`, `角度2` |
| `angle_c_deg` | 是 | `角度C`, `角度3` |
| `x_m` | 是 | `X`, `X [ m ]`, `point_x` |
| `y_m` | FD 必需 | `Y [ m ]`；FL 平面数据可填 `0` 或空，manifest 说明 |
| `z_m` | 是 | `Z`, `Z [ m ]`, `point_y` |
| `centroid_x_m` | 可选 | `重心坐标X` |
| `centroid_z_m` | 可选 | `重心坐标Y` |

### 距离特征列

| 标准列名 | 必需 | 旧格式常见来源 |
| --- | --- | --- |
| `dist_to_base_m` | FL 推荐 | `点到底边距离`, `到BC垂直距离`, `dist_BC_vertical` |
| `dist_to_left_m` | FL 推荐 | `点到左边距离`, `到AB垂直距离` |
| `dist_to_right_m` | FL 推荐 | `点到右边距离`, `到AC垂直距离` |
| `dist_to_centroid_m` | 可选 | `点到重心距离` |
| `dist_ab_horizontal_m` | FD 推荐 | `dist_AB_horizontal` |
| `dist_ac_horizontal_m` | FD 推荐 | `dist_AC_horizontal` |

### 实验条件与目标列

| 标准列名 | 必需 | 类型 | 说明 |
| --- | --- | --- | --- |
| `re` | 是 | float | 雷诺数 |
| `target_fl_x` | 可选 | float | FLx；未知方向时不要强行填 |
| `target_fl_y` | 可选 | float | Fly/FL 的标准目标列 |
| `target_fd_x` | 可选 | float | `dean_velocity_x` |
| `target_fd_y` | 可选 | float | `dean_velocity_y` |
| `focus_label` | 分类任务必需 | string | 3 分类标签，建议固定为 `under_focus`, `focused`, `over_focus` |
| `label_source` | 标签存在时必需 | string | `manual`, `rule`, `simulation`, `model` |

当前项目第一阶段预测 FL、FD 时使用 `target_fl_y`、`target_fd_x`、`target_fd_y`。第二阶段聚焦状态分类优先读取 `focus_label`；如果 label 尚未标注，该列可存在但为空。

## manifest.json 标准字段

```json
{
  "schema_version": "rawdata-standard-v1",
  "batch": "2026-06-01",
  "created_at": "2026-06-01T00:00:00+08:00",
  "description": "",
  "coordinate_system": {
    "x": "horizontal coordinate in triangle plane",
    "y": "out-of-plane coordinate for FD velocity field",
    "z": "vertical coordinate in triangle plane",
    "unit": "m"
  },
  "targets": ["FL", "FD", "focus"],
  "standard_file": "standard/samples.csv",
  "source_units": {
    "length": "m",
    "area": "m2",
    "fl": "nF or original simulation unit",
    "fd_velocity": "m/s"
  },
  "quality": {
    "rows": 0,
    "files": 0,
    "missing_required_values": 0,
    "duplicate_sample_id": 0
  },
  "notes": []
}
```

## 旧 rawdata 迁移规则

1. FL 历史批次如 `2026-04-26batch1`、`2026-05-06`：导入到 `standard/samples.csv`，把 `Fly` 和 `FL` 统一映射到 `target_fl_y`，把 `FLx` 映射到 `target_fl_x`。
2. FD 历史批次如 `2026-05-21/FD数据/10/*.xlsx`：目录名 `10/20/30` 映射到 `radius_group`，`dean_velocity_x/y` 映射到 `target_fd_x/y`。
3. 带距离特征的 FD 文件如 `2026-05-24/*distance.xlsx`：只作为距离特征来源，不直接作为训练目标文件；导入时按 `geometry_id + x_m + z_m` 合并。
4. 文件名中的 `反`、`reverse` 映射为 `direction=reverse`；普通文件为 `direction=normal`。
5. 文件名中的 `690-340` 只作为 `geometry_id` 的 fallback。标准表内仍必须有 `base_m` 和 `height_m`。

## 校验规则

导入脚本写出标准数据后，至少检查：

1. `sample_id` 不重复。
2. 必需列存在。
3. `base_m > 0`、`height_m > 0`、`re > 0`。
4. `angle_a_deg + angle_b_deg + angle_c_deg` 接近 180，建议误差小于 1 度。
5. 每行至少有一个目标：`target_fl_x/target_fl_y/target_fd_x/target_fd_y/focus_label` 之一非空。
6. 若 `focus_label` 非空，只能属于 `under_focus/focused/over_focus`。

## 后续代码落点

建议新增一个导入与校验入口：

```text
src/data/standardize_rawdata.py
```

职责：

1. 读取 `data/rawdata/<batch>/originals/` 或旧式批次目录。
2. 根据列名和文件路径映射到标准列。
3. 写出 `data/rawdata/<batch>/standard/samples.csv`、`manifest.json`、`files.csv`。
4. 输出质量报告，失败时阻止进入 `cleaned/scaled/processed`。

这样 `src/data/clean.py` 后续可以从标准入口读取数据，而不是继续兼容越来越多的旧 Excel 变体。
