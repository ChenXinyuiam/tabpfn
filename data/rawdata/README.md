# rawdata

这里放数据提供方交付的数据，以及项目内部标准化后的 rawdata 入口。

推荐新批次结构：

```text
rawdata/<batch>/
  originals/
  standard/
    samples.csv
    manifest.json
    files.csv
```

`originals/` 保持原样，用于回溯。`standard/samples.csv` 是后续处理脚本读取的入口。
