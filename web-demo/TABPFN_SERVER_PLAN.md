# 真实 TabPFN 服务器部署方案

当前宝塔 Node 项目只适合运行网站演示后端。真实 TabPFN 应该作为独立 Python 推理服务运行，再由网站或 Nginx 调用。

## 推荐架构

```text
用户浏览器
  -> 域名 / Nginx / 宝塔
  -> Node 网站项目，负责页面
  -> Python TabPFN 推理服务，负责真实模型
  -> models/offline_tabpfn_2_5/
```

不要把 TabPFN 模型直接放进 Node 项目里运行。Node 负责网页，Python 负责模型。

## 服务器需要放哪些内容

最小推理包建议包含：

```text
web-demo/
src/
utils/
pyproject.toml
uv.lock
models/offline_tabpfn_2_5/
```

其中 `models/offline_tabpfn_2_5/` 至少需要：

```text
config.json
tabpfn-v2.5-regressor-v2.5_default.ckpt
```

当前代码的默认模型目录来自：

```text
src/config.py
MODEL_DIR = ROOT_DIR / "models" / "offline_tabpfn_2_5"
```

不要上传：

```text
.venv/
data/rawdata/
outputs/
results/
data.zip
__pycache__/
*.log
```

## 硬盘空间要求

当前服务器只剩 4GB，不建议跑真实 TabPFN。

推荐剩余空间：

```text
最低：12GB - 15GB
推荐：20GB - 30GB
如果安装 GPU 版 PyTorch：30GB+
```

原因是服务器除了模型文件，还需要：

```text
Python 虚拟环境
PyTorch
TabPFN
numpy / pandas / scikit-learn
安装缓存
运行日志
临时文件
```

## 安装环境

在服务器项目目录下创建 Python 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

如果服务器硬盘紧张，安装后清理 pip 缓存：

```bash
python -m pip cache purge
```

## 先验证模型能不能加载

上传模型后，在服务器运行：

```bash
python -m src.inference --mode fd
```

或者：

```bash
python -m src.inference --mode focus
```

如果报错缺少模型文件，优先检查：

```text
models/offline_tabpfn_2_5/config.json
models/offline_tabpfn_2_5/tabpfn-v2.5-regressor-v2.5_default.ckpt
```

## 网站如何接入真实模型

下一步需要新增一个 Python API，例如：

```text
POST /api/predict-triangle
```

这个 API 做：

```text
三条边 + radius_group
  -> 生成 200 个采样点
  -> 构造 TabPFN 特征表
  -> 按 radius_group 路由 FD 模型
  -> 调用 FD / FL 模型
  -> 计算合力和聚焦位置
  -> 返回给前端
```

然后网站有两种接法：

方案 A：Nginx 分流

```text
/               -> Node 网站
/api/           -> Python TabPFN 服务
```

方案 B：Node 转发

```text
浏览器 -> Node server.js -> Python TabPFN 服务
```

生产上更推荐方案 A。

## 宝塔里怎么配置

保留 Node 项目：

```text
用途：展示网页
端口：例如 8000
```

新增 Python 项目或 Supervisor 进程：

```text
用途：TabPFN 推理服务
端口：例如 8100
启动命令：python -m <后续新增的推理 API 模块>
```

Nginx 反向代理：

```text
location / {
    proxy_pass http://127.0.0.1:8000;
}

location /api/ {
    proxy_pass http://127.0.0.1:8100;
}
```

## 当前还缺什么

当前仓库已经有离线 TabPFN 工具和实验推理入口，但网站的真实模型 API 还没有完全接上。

还需要实现：

```text
1. 三条边 + radius_group -> TabPFN 特征表与 FD 模型路由
2. Python API 服务
3. FD / FL 模型调用
4. 返回与当前前端兼容的 JSON
5. 宝塔 Python 服务启动脚本
```

完成后，前端不需要大改，仍然调用：

```text
POST /api/predict-triangle
```
