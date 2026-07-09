# processed

这里放最终建模数据。训练、实验、推理代码应默认只读取这里。

推荐结构：

```text
processed/<batch>/
  samples.csv
  train.csv
  valid.csv
  test.csv
  manifest.json
  figures/
```

`processed/` 里的数据应当已经完成清洗、单位统一、必要特征生成和质量检查。
