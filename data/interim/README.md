# interim

这里放中间产物。中间产物应该可以从 `rawdata/` 重新生成。

推荐结构：

```text
interim/<batch>/
  cleaned/
  scaled/
  features/
  augmented/
  reports/
```

当前兼容旧脚本的固定入口：

```text
interim/cleaned/<batch>/
interim/scaled/<batch>/
interim/features/<batch>/
```

不要把唯一版本的数据只放在这里。
