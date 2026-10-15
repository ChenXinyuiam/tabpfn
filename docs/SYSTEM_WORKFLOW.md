# 统一机器学习工作流

新的入口是 `python -m src system`。它落实了 [重设计方案](ML_SYSTEM_REDESIGN.html) 中的数据审核、共享几何特征、分组评估、模型包和新三角形推理。原有 `data`、`regression-*`、`focus-*` 命令仍然保留，历史脚本的输入和输出不改变。

## 先跑一个完整例子

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m src system demo
```

这会创建 6 个**合成测试几何**、18 个截面工况、独立 FD / FL / Fzong 点表和 UF/F/WF 测试标签，实际执行审核、外层五折、内层交叉拟合、模型包创建和新三角形推理。最终 `results/system/demo/runs/<run-id>/demo.json` 列出所有运行路径。

合成数据用于检验程序、三分类和防泄漏，不是物理仿真，也不能用其指标声称真实数据精度。示例中的 Fzong 求和定义仅适用于该合成例子。

## 四个主要命令

```powershell
# 1. 审核来源清单，输出不可静默修改的数据快照
.\.venv\Scripts\python.exe -m src system audit --manifest path\manifest.json

# 2. 使用审核运行的精确目录进行五折评估
.\.venv\Scripts\python.exe -m src system evaluate `
  --snapshot results\system\audit\runs\<run-id> `
  --config examples\system\model.json --task fields --folds 5 --protocol geometry

# 3. 按相同配置拟合完整训练上下文，关联已经完成的评估
.\.venv\Scripts\python.exe -m src system train `
  --snapshot results\system\audit\runs\<run-id> `
  --config examples\system\model.json --mode fields `
  --evaluation results\system\evaluate\runs\<evaluation-run-id>

# 4. 新几何生成格点并预测
.\.venv\Scripts\python.exe -m src system predict `
  --bundle results\system\train\runs\<model-run-id> `
  --request examples\system\query.json
```

命令中的 `<...>` 需换成前一步打印的实际目录。所有步骤支持 `--out-dir`。不同运行写入不同目录，并保存运行状态、精确源代码快照和 `latest.json`。审核发现 REJECT 时生成报告并返回退出码 2，下游拒绝该快照；REVIEW 允许原生目标回归，但不会解除物理转换限制。

`evaluate --task` 支持 `FD`、`FL`、`fields`、`focus`、`cascade`。`train --mode` 支持 `fields`、`focus`、`cascade`。仅针对单个 FD 或 FL 的评估不会被当作整个 fields 包的已完成评估。省略 `--evaluation` 可以生成研发模型包，但明确标为 `unvalidated`。

`system report --run <run-dir>` 可从保存的表格重建 HTML，不重新拟合。

## 数据清单

模板位于 `examples/system/manifest.template.json`。这是占位模板，不能直接代表现有数据的物理定义。

每个 source 显式指定：

- `path`：相对于清单文件的 CSV、TSV 或 XLSX 路径。
- `task`：FD、FL 或 Fzong。两种 FL 分量应分别建数据集 / 模型包，不能在同一 task 中混合。
- `experiment_id`：同一朝向、几何和实验系列在 FD、FL、Fzong 文件间共享的 ID。不同重复实验可有不同 ID。
- `geometry.vertices`：按 A/B/C 顺序给出 `[[Ax,Az],[Bx,Bz],[Cx,Cz]]`；顶点和点坐标使用同一个 `global_xz` 坐标系。
- `length_unit` / `geometry.unit` / `radius_unit`：分别明确 m、mm 或 um。已经是 um 时倍率为 1，没有 auto 推测。
- `radius`：显式曲率半径；省略时从 `columns.radius` 指定的列读取，默认列 R。文件名里的数字不再被自动当作物理半径。
- `columns`：规范名到来源列名的映射，例如 `{"x":"X [ m ]","z":"Z [ m ]","re":"Re"}`；映射不隐含单位转换。
- `target`：原列名、目标名称、物理量、单位、分量。例如 `quantity=unverified, unit=native` 可回归；只要单位 / 分量未确认，就不能进入力合成。
- `value_origin`：第一版只接收 measured / simulated 独立标签。旧插值工具仍可用，但插值和预测标签不会混入新的基线审核数据集。

审核检查原始表头、数值、单位、三角形非退化、点位内部性、提供的几何列与顶点的一致性、重复点、目标语义冲突和标签关联。坏行保存在 `rejected_rows.csv`，问题记录在 `issues.csv`。几何矛盾不通过修改坐标或目标值来消除。

如果提供者的坐标系不是 global_xz，应先由明确的坐标变换转换顶点、位置及向量分量。第一版不猜测原点、朝向、旋转或镜像符号。提供的边长 / 角度等几何字段存在时，以 `rtol=1e-4, atol=1e-6` 与重算结果比较；源数据舍入过粗需提供更精确元数据或专门转换记录。

### 聚焦标签

在 manifest 中加入独立标签：

```json
{
  "classes": ["UF", "F", "WF"],
  "label_provenance": "人工标注表的来源与版本",
  "labels": [
    {"experiment_id": "triangle-A-original", "re": 6, "radius_um": 1200, "label": "WF"}
  ]
}
```

二分类使用 `["UF","F"]`。每个 Fzong 截面需要一条标签，模型训练折必须包含声明的全部类别。模型不合并 WF 和 F。预测输出 `probability_UF`、`probability_F`、`probability_WF`（三分类时）与 `predicted_label`。REVIEW 是复核状态，不是第四种聚焦类型。

分类复用现有三点中值平滑和边界特征，样本为整个截面工况。当前特征含边界宽度、位置、符号比例、逐行空间统计与几何 / Re / R；使用固定特征集和默认最大概率判决。需要特征选择、概率校准时仍可使用旧的 `focus-optimize` / `focus-calibrate` 研究流程；新的通用链路暂不自动调参。

## 评估与防泄漏

`geometry` 默认以归一到 um 的三条边长度构造物理几何族，镜像与顶点顺序不同的全等三角形进入同一折。几何边长身份保留至小数点后 6 位；元数据不一致先在审核中解决。不同尺度的相似三角形不是同一个全等组，若要验证整个形状族外推，应增加相似形分组协议。

`spatial` 以已定向几何与坐标为组，同点所有 Re 与 FD/FL/Fzong 数据不可跨折。它只代表点位留出，不代表连续边缘区域或未知几何外推。focus / cascade 禁止使用该协议。

分组少于指定折数时明确报错。外层所有任务共用 `fold_manifest.csv`，不根据文件名各自拆分。输出逐折、逐几何 / Re 的指标、汇总 OOF 和几何等权回归指标。恒定目标的 R² 为 null；不会以 0 或 1 代替不可解释值。

`cascade` 在每个外层训练集中进行一次完整的内层几何交叉拟合：

1. 内层训练 FD/FL，预测被留出的 Fzong 参考位置。
2. 经确认的物理适配器合成预测场，提取折外截面特征。
3. 用独立人工标签拟合分类器。
4. 全部外层训练数据重拟合 FD/FL，对外层测试几何预测并分类。

外层测试几何的所有任务记录都被剔除。Fzong 的原始目标值不作为上游模型特征；其独立真值仅用于合成场误差评估。`metrics.json` 记录内外层组别，测试覆盖每个点 / 截面恰好一次。上游参考位置可以和 FD / FL 的训练位置不同。

`focus` 评估真实 Fzong 输入的分类基线；`cascade` 评估预测场的整条链路。两者均需运行后再比较，不把其中一个当成另一个的精度。固定配置下不存在自动选型；若观察多次外层结果后继续选择配置，仍需新的独立测试集或额外嵌套验证。

## 模型配置与产物

默认使用 ExtraTrees，便于 CPU 上运行可复现基线。TabPFN 使用相同适配接口，在配置中分别指定分类 / 回归 checkpoint：

```json
{
  "regression": {
    "backend": "tabpfn", "seed": 42, "n_estimators": 1,
    "checkpoint": "../../models/models/tabpfn-v2.5-regressor-v2.5_default.ckpt",
    "device": "prefer-cuda"
  },
  "classifier": {
    "backend": "tabpfn", "seed": 42, "n_estimators": 1,
    "checkpoint": "../../models/models/tabpfn-v2-classifier.ckpt",
    "device": "prefer-cuda"
  },
  "seed": 42, "inner_splits": 3, "window_size": 3,
  "physics": null
}
```

Checkpoint 路径相对配置文件解析。缺 checkpoint 或推理失败时不自动换成其他算法。缩放器和缺失值处理器每次仅拟合训练部分，预测分块执行。第一版不自动下采样训练行，资源不足应明确制定训练采样协议后重新评估。

模型包包含训练快照、固定配置、目标与类别契约、实现哈希、依赖版本、checkpoint 哈希、评估关联，以及分类所需训练特征。采用**训练上下文重建**，不加载 pickle：推理时确定性重新拟合估计器。级联分类直接使用已保存的折外特征，不重新运行内层交叉验证。大批量服务可后续在进程内缓存已加载的模型上下文。

代码、依赖、训练表或 checkpoint 发生变化时拒绝旧包，要求重新评估构建。标记 evaluated 表示关联了匹配的完整评估，不代表自动满足某个科学验收阈值。

## Fzong 的接入条件

`examples/system/model.json` 中 `physics=null` 是有意保留的。真实 FD 的 `u` 与单位还未得到确认，系统将输出 FD/FL、空间诊断，并在 `prediction.json` 中明确记录 Fzong / focus 未执行的原因。

物理定义确认后，可以使用以下配置结构；引用与版本必须换成真实依据：

```json
{
  "confirmed": true,
  "formula": "signed_component_sum",
  "version": "your-confirmed-formula-v1",
  "reference": "数据提供者确认的文档编号",
  "component": "x",
  "output_unit": "nN"
}
```

该配置放在模型配置的 `physics` 字段中，要求 FD 和 FL 都是同一个 x 或 z 分量的力。支持 N / mN / uN / nN / pN 之间换算，不使用向量模长代替有符号 Fzong。

若 FD 确实是速度，只支持显式声明的线性转换适配器：`fd_conversion={kind:declared_linear, confirmed:true, gain_N_per_m_s:正数, reference:依据}`。系数及适用域必须由物理研究给出，程序不推断黏度、粒径或阻力公式；如果系数随工况变化，应实现相应新适配器并重新验证，不能套用常数。

## 查询与输出

`query.json` 支持水平规则网格，也支持 `positions:[[x,z],...]` 配合 `position_unit`，后者跳过网格生成。`re_values` 可以多个，查询位置在所有 Re 共用；点 ID 确定性生成，禁止重复坐标。

已拥有真实 Fzong 的情况，使用 observed-field 的 `focus` 模型包，在请求中增加 `provided_fzong={unit,component,values}`，values 必须与查询表逐行对齐。级联模型包使用自身 FD/FL 预测，拒绝混用外部 Fzong 捷径。

运行产物：

```text
audit:     audit.json, issues.csv, rejected_rows.csv, points.csv,
           samples.csv, snapshot.json
evaluate:  point_oof.csv, focus_oof.csv, fold_manifest.csv,
           by_geometry_re.csv, metrics.json, evaluation.json
train:     bundle.json, snapshot/, training_focus_features.csv,
           training_oof_fields.csv, crossfit_audit.json
predict:   request.json, query_points.csv, point_predictions.csv,
           focus_predictions.csv, diagnostics.csv, prediction.json
all:       report.html, source_snapshot/, run_status.json
```

不适用的训练特征 / 折外文件不会生成。使用 CSV 避免增加 Parquet 引擎依赖。预测图保留实际点位，最多展示前 6 个工况，完整数值全部保存。FD / FL 目标保持各自单位；Fzong 单位由 physics 指定，颜色不能在未对齐单位的场之间比较。

诊断覆盖新几何、特征范围、同一已定向几何且相同 R / Re 内的空间凸包与最近距离，以及采样和分类特征变化。无参考真值的推理不会生成 R² 或准确率。

## 新代码的阅读顺序

1. `src/workflows/system.py`：四阶段入口与参数、运行产物。
2. `src/data/contracts.py`、`src/data/audit.py`：来源清单到规范快照。
3. `src/geometry/triangle.py`：唯一的几何、网格与特征实现。
4. `src/regression/models.py`：ExtraTrees / TabPFN 共用训练和预测接口。
5. `src/evaluation/protocols.py`、`src/evaluation/system.py`：划分、指标与五折执行。
6. `src/workflows/engine.py`：字段回归、内层交叉拟合和分类的共享执行路径。
7. `src/physics/force.py`：有条件启用的物理合成。
8. `src/workflows/artifacts.py`：训练上下文和版本完整性。

新增测试在 `tests/test_system_workflows.py`。现有 96 项测试用于保护旧入口与既有行为。

## 本次验证记录（2026-10-03）

- 重构前 96 项测试通过；新增 20 项后，全量 116 项测试通过。
- 新模块的 Ruff 检查、全仓库源码编译和 `doctor` 环境检查通过。
- `system demo` 的完整 CPU ExtraTrees 链路通过：702 条审核 / 折外点记录、18 个 UF/F/WF 截面、新几何 39 个点与 3 个截面，导出 9 张场图。
- 在本机 CUDA 上实际加载 TabPFN v2.5 回归和 v2 分类 checkpoint，验证新适配器的回归输出为有限值，三分类输出形状为 3×3、每行概率和为 1。
- 完整示例索引：`results/system/demo/runs/20261003T054538Z-cd813ae4/demo.json`；TabPFN 接口检查：同目录的 `tabpfn_backend_smoke.json`。
- 已扩展 CI 配置，但未在远程 GitHub 执行。本地完整五折例子使用合成数据与 ExtraTrees；TabPFN 检查是小样本接口验证，不是全量数据评估。

真实 FD-u 的物理定义、真实 Fzong 转换和业务精度尚未验收。现有原始数据没有被改写，也没有替换旧科学模型文件。
