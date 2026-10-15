# 代码清理与入口迁移

日期：2026-10-15。范围：当前工作树的源代码、命令入口、测试和使用文档。
这是结构清理，不是重新训练；此前 FD R² 实验记录没有被改写。

## 已删除

- 删除 142 个旧源文件及附属文档/配置，其中包含 121 个 Python 文件及旧扩展的源码、示例和测试。
- 旧预切分 CSV 训练/推理、旧 split 工具、早期独立分析及旧二分类研究脚本。
- 已被 regression 模块替代的实验脚本与兼容包装；不再保留旧命名空间转发。
- 仅旧流程使用的 utils 特征、折文件、目标变换及下载补丁等辅助代码。
- 自动 Re-FL 修复和特定五月批次距离生成脚本。
- 重复的新手指南、旧分类说明、旧服务器方案和旧工作流说明。

逐文件清单见 [cleanup_manifest_20261015.json](cleanup_manifest_20261015.json)。
这 142 个文件不包括下面迁移保留的 5 个插值模块，也不包括未删除的构建缓存。

## 保留并迁移

| 原入口 | 当前实现 | 当前命令（前缀 python -m src） |
| --- | --- | --- |
| predict_fl_positions_from_matching_fd_tabpfn | src/regression/fd_to_fl.py | fd-to-fl |
| predict_fl_at_fd_points_tabpfn | src/regression/fl_at_fd.py | fl-at-fd |
| predict_fd_u_from_interpolated_fl_tabpfn | src/regression/fd_from_fl.py | fd-from-fl |
| interpolate_predicted_fl_files | src/regression/interpolation.py | interpolate |
| evaluate_fd_fl_internal_external | src/regression/evaluation.py | regression-evaluate |
| interpolate_raw_flx_files | src/regression/raw_fl_interpolation.py | interpolate-raw-fl |
| interpolate_raw_flx_tabpfn | src/regression/raw_fl_tabpfn.py | interpolate-raw-fl-tabpfn |
| interpolate_fly_neighbor_midpoints_tabpfn | src/regression/fly_tabpfn.py | interpolate-fly |
| interpolate_fly_neighbor_midpoints | src/regression/fly_midpoints.py | interpolate-fly-linear |
| predict_fl_at_fd_points_from_lines | src/regression/fl_lines.py | fl-lines |

上表前五行原实现已在 regression 中，本次仅删除旧包装；后五行本次实际迁移。
FLy 默认使用 TabPFN 预测新增目标值，线性版本只作为参考。四角单元中心约束保留。
fl-lines 是两条原始指定路径的复现工具，不等同于未知三角形回归。

## 当前结构

| 路径 | 职责 |
| --- | --- |
| src/data | 源数据适配、单位、审核及仍在调用的预处理管线 |
| src/geometry / src/physics | 几何与显式物理转换 |
| src/regression | FD/FL 预测、插值、基线和嵌套验证 |
| src/focus_inspection | 截面特征、F/UF 分类、校准和审核 |
| src/evaluation / src/workflows | 系统评估、模型包和串联推理 |
| utils | 路径、来源追踪、版本化运行目录 |
| web-demo | 当前网页服务；本次不更改其预测算法 |

近期 FD rotation/transfer/zero 实验及较早的受控基线保留，因为它们被当前试验导入，
也用于对照和复现。不能按“文件有日期”或“分数较低”认定代码无用。
重心坐标、数据分组、模型权重和评价公式没有因本次清理而改变。

新增统一命令：fd-transfer-ensemble、fd-transfer-audit、fd-transfer-report、fd-zero-audit。
doctor 从命令注册表检查所有入口；CI 的 Ruff 检查覆盖 src、utils 和全部测试。
移除 config 中仅旧模型使用的参数，以及 publish 在根目录输入时误指向固定旧批次的默认目录。

## 不删除的资产

原始数据、模型文件、历史预测、结果中的 source_snapshot、已有部署包和压缩归档都保留。
旧扩展目录的 src/tabpfn_extensions/hpo/hpo_models 中仍有 checkpoint，不能递归删除该目录。
保留旧扩展 LICENSE 以覆盖残留第三方资产；旧目录已加入忽略规则，防止无扩展名权重被误提交。
已经交付的 server_offline_checkpoint 和 server_retrain_package 是独立交付快照，本次不改写。

## 尚未完成的物理清理

工具拒绝了针对构建副本、缓存和安装元数据的批量删除操作。
因此 build/ 中仍可能含旧源码副本，旧目录中仍可能有 __pycache__；
旧扩展的 egg-info 和示意图也仍在。它们不是维护入口，但不能声称整个磁盘已无旧代码。
不要直接打包现有 build/；发布前需要在干净工作区构建。
完整待清理项及受保护权重的说明记录在 JSON 清单。

## 验证

清理前：153 项单元测试通过。
清理后：162 项单元测试通过；91 个源模块可导入；doctor 全部通过且 CUDA 可用。
Ruff 覆盖 src、utils、tests 和网页预测服务，通过；compileall 及离线 lock 检查通过。
第一次全量测试出现一次 Windows/joblib 线程池 concurrent send_bytes 错误；
不同时运行其他检查后完整重跑，162 项全部通过，没有为此修改模型算法或跳过测试。
合成五折串联 demo 已完成，目录为：

`results/system/cleanup-20261015/runs/20261015T071146Z-b51ba373/`

该 demo 验证审核、五折评估、模型包、新三角形预测能执行，不是科学精度验证。
本次未重新运行耗时的真实数据 TabPFN 五折实验，没有产生新的 R² 声明。
