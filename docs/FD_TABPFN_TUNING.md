# TabPFN 专项 FD-u 调优

本入口只比较 TabPFN，不以 ExtraTrees 的得分代替 TabPFN 改进。
默认数据、分组和测试点与前一轮 FD 外推实验一致。

```powershell
.\.venv\Scripts\python.exe -m src fd-tabpfn-tune
```

## 本轮改变

1. 集成次数从 1 增加到 4。它是多个预处理/特征排列等成员的预测集成，
   不是 4 个训练 epoch，也没有更新预训练权重。
2. 对照关闭低取值数数值列的自动类别推断。在当前安装的 TabPFN 中，
   样本超过 100 行且数值列只有 2 或 3 个取值时，默认可能被识别为类别。
   留出几何后，边长、高度、数字标签等恰好可能满足该条件。
   使用 `MIN_UNIQUE_FOR_NUMERICAL_FEATURES=1` 将非恒定数值列保留为连续量。
   仅设置 `categorical_features_indices=[]` 不足以禁止自动推断。
3. 在完整原始特征上追加相对特征，而不是删除原始几何信息。
4. 对照按 Re 一次方、二次方以及 Re 二次方/数字标签缩放目标。
   所有预测还原到原始 u 后计算 R²，不能拿变换空间的分数宣称原始精度提高。

候选菜单在新实验开始前冻结：

| 名称 | 集成数 | 强制数值 | 追加相对特征 | 目标缩放 factor |
| --- | ---: | --- | --- | --- |
| raw_tabpfn | 1 | 否 | 否 | 1 |
| raw_e4 | 4 | 否 | 否 | 1 |
| numeric_raw_e4 | 4 | 是 | 否 | 1 |
| numeric_augmented_e4 | 4 | 是 | 是 | 1 |
| numeric_re1_e4 | 4 | 是 | 否 | Re/30 |
| numeric_re2_e4 | 4 | 是 | 否 | (Re/30)² |
| numeric_re2_radius_e4 | 4 | 是 | 否 | (Re/30)² × 20/radius_label |
| numeric_augmented_re2_e4 | 4 | 是 | 是 | (Re/30)² |

拟合目标为 `u/factor`，推理输出乘回 factor。以上均是待验证的经验假设。
`radius_label` 来自历史文件名，未确认物理单位；不能自动解释为已知半径。

## 验证协议

外层整类三角形留出；同类镜像与所有 Re/数字标签文件一起分组。
内层仅用另外 3 类几何比较上述 8 项，选择几何平均 R² 最高的一项，
再在外层训练集重新拟合并测试。保留原始单成员 TabPFN 作为逐折对照。
不要事后按外层分数修改某折的选择。

各方案使用相同训练上下文，默认 3,000 行；内层每文件约 200 行，
外层每文件约 500 行，按完整位置抽样。本轮没有改变训练抽样策略，
以免同时改变过多因素；4 成员方案的计算预算高于单成员基线。
不能把结果描述成完全相同算力下的比较。

本次使用已有的 4 类几何，不是真正未参与开发的独立盲测。
它能检验内层选型流程的跨几何表现，但不能保证未来所有形状都改善。
更远的几何/工况外推仍需要新增独立仿真或实验标签。

## 代码与输出

- `src/regression/tune_tabpfn_fd.py`：冻结菜单、特征、目标还原、模型配置与专项入口。
- `src/regression/optimize_fd.py`：共享严格研究读取与嵌套分组评估。
- `src/regression/evaluation.py`：共享抽样和指标。
- `tests/test_fd_tabpfn_tuning.py`：保留原特征、目标还原及类别推断的行为测试。

每次运行新建 `results/fd_tabpfn_tuning/runs/<run-id>/`：

- `config.json` / `source_snapshot/`：完整配方、权重哈希和源码。
- `inner_metrics.csv` / `selections.json`：内层比较及逐折选择。
- `fold_metrics.csv` / `summary.csv`：原始 u 空间上的外层分数。
- `predictions.csv`：测试目标、预测、来源文件和原始行号。
- `timings.json`：各次拟合与预测合计耗时。
- `data_inventory.csv`：数据完整性统计，不是 system audit 通过凭证。
- `report.md` / `outer_r2.png`：可复核报告和对比图。

预测核心函数可以复用，但本入口不修改现有生产默认模型、不导出审核通过的模型包。
`nested_selected` 是一个选型流程，不一定是四折都相同的固定配方。
新三角形推理时不能简单按测试几何挑选这里表现最高的折模型。

2026-10-07 实测期间存在其他图形程序共享 GPU 的情况，记录的墙钟时间
受资源竞争影响，不能直接作为严格的方案速度排名。
