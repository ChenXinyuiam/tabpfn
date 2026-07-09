# manifests

这里放每个数据批次的账本。

推荐结构：

```text
manifests/<batch>/
  manifest.json
  validation_report.csv
```

manifest 应记录数据来源、处理步骤、行列数量、缺失值、重复 ID、当前是否可用于训练。
