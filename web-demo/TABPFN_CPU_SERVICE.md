# TabPFN CPU 推理服务部署说明

这个服务用于服务器真实模型推理，和宝塔 Node 项目分开运行。

## 端口分工

```text
Node 网站：127.0.0.1:8000
Python TabPFN 服务：127.0.0.1:8100
Nginx 对外域名：80 / 443 / 或你当前的 3001
```

推荐 Nginx 分流：

```text
/       -> 127.0.0.1:8000
/api/   -> 127.0.0.1:8100
```

## 上传文件

Python 服务至少上传：

```text
tabpfn_service.py
requirements-tabpfn-cpu.txt
start_tabpfn_service.sh
TABPFN_CPU_SERVICE.md
```

如果使用训练 CSV 在线 fit，还要上传训练 CSV。

如果使用已保存模型，还要上传四个模型：

```text
TRIFOCUS_FLX_MODEL
TRIFOCUS_FLY_MODEL
TRIFOCUS_FDX_MODEL
TRIFOCUS_FDY_MODEL
```

## 安装依赖

进入 Python 项目目录：

```bash
cd /www/wwwroot/triFocusModel
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tabpfn-cpu.txt
python -m pip cache purge
```

如果硬盘只有 10GB，安装后一定执行：

```bash
python -m pip cache purge
```

## 方式 A：使用训练 CSV 在线 fit

先生成训练 CSV 模板：

```bash
python tabpfn_service.py --write-training-template training_template.csv
```

训练 CSV 必须包含模板里的所有特征列和目标列：

```text
fl_x_pred
fl_y_pred
fd_x_pred
fd_y_pred
```

设置环境变量：

```bash
export TRIFOCUS_MODEL_MODE=training_csv
export TRIFOCUS_TRAINING_CSV=/www/wwwroot/triFocusModel/training.csv
export TRIFOCUS_MAX_TRAIN_ROWS=2000
export TRIFOCUS_TABPFN_ESTIMATORS=1
```

启动：

```bash
bash start_tabpfn_service.sh
```

## 方式 B：使用四个已保存 joblib 模型

推荐比赛部署使用这种方式：本地训练，服务器只加载模型做推理。

当前已经生成了一个可上传模型包：

```text
trifocus-tabpfn-models-500.zip
```

它使用本地 `data/rawdata/2026-06-20` 的 FD/FL 配对数据训练：

```text
训练行数：500
模型：TabPFN CPU
FL-x：真实 FL 单方向投影
Dean-x：匹配 FD u 单方向投影
FL-y / Dean-y：当前本地数据没有对应标签，先用 0 分量占位
```

本地先生成训练 CSV 模板：

```bash
python tabpfn_service.py --write-training-template training_template.csv
```

把真实训练数据整理成模板列名后，执行：

```bash
python train_tabpfn_models.py \
  --training-csv training.csv \
  --output-dir models/tabpfn_server \
  --n-estimators 1 \
  --max-rows 2000
```

会得到：

```text
models/tabpfn_server/fl_x.joblib
models/tabpfn_server/fl_y.joblib
models/tabpfn_server/fd_x.joblib
models/tabpfn_server/fd_y.joblib
models/tabpfn_server/model_manifest.json
```

把这几个文件上传到服务器：

```text
/www/wwwroot/triFocusModel/models/
```

设置环境变量：

```bash
export TRIFOCUS_MODEL_MODE=joblib
export TRIFOCUS_FLX_MODEL=/www/wwwroot/triFocusModel/models/fl_x.joblib
export TRIFOCUS_FLY_MODEL=/www/wwwroot/triFocusModel/models/fl_y.joblib
export TRIFOCUS_FDX_MODEL=/www/wwwroot/triFocusModel/models/fd_x.joblib
export TRIFOCUS_FDY_MODEL=/www/wwwroot/triFocusModel/models/fd_y.joblib
export TRIFOCUS_PREDICT_CHUNK_SIZE=2000
export TRIFOCUS_CACHE=1
```

启动：

```bash
bash start_tabpfn_service.sh
```

## 健康检查

```bash
curl http://127.0.0.1:8100/api/health
```

如果真实模型加载成功，会看到：

```json
{
  "uses_tabpfn": true,
  "model_loaded": true
}
```

如果还没配置模型，服务仍能启动，但会显示：

```json
{
  "uses_tabpfn": false,
  "model_loaded": false
}
```

这时网站可以继续演示，但还不是真实 TabPFN。

## 预测测试

```bash
curl -X POST http://127.0.0.1:8100/api/predict-triangle \
  -H "Content-Type: application/json" \
  -d '{"side_ab_um":690,"side_ac_um":520,"side_bc_um":750}'
```

## Nginx 配置

如果 Node 网站仍跑在 8000，Python 服务跑在 8100：

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8100;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}

location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

这样前端仍然请求 `/api/predict-triangle`，但真实计算会由 Python TabPFN 服务完成。
