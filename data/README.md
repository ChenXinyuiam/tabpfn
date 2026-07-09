# Data Directory

这是项目的数据工作区。默认规则：

1. 原始交付文件放 `rawdata/`。
2. 中间处理结果放 `interim/`。
3. 模型训练和推理只读 `processed/`。
4. 每批数据的来源、质量检查、处理记录放 `manifests/`。
5. 历史旧结构逐步放入 `legacy/`。

顶层目录已经收敛为 `rawdata/`、`interim/`、`processed/`、`manifests/`、`legacy/`。

详细规范见 `DATA_LAYOUT.md`。
