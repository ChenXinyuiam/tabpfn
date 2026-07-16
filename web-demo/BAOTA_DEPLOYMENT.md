# 宝塔 Node 项目部署说明

本目录已经支持两种运行方式：

- 本地 / Python 方式：`python server.py`
- 宝塔 Node 项目方式：`npm start`，实际运行 `node server.js`

如果宝塔要求“建立 Node 项目”，优先使用 Node 方式。

## 需要上传哪些文件

只上传 `web-demo` 里的这些文件：

```text
index.html
styles.css
app.js
server.js
package.json
API_CONTRACT.md
DEMO_SCRIPT.md
README.md
BAOTA_DEPLOYMENT.md
```

可选上传：

```text
server.py
run_demo.ps1
```

如果只用宝塔 Node 项目，`server.py` 和 `run_demo.ps1` 可以不上传。

不要上传：

```text
.venv/
data/
models/
outputs/
results/
tabpfn-extensions-main/
.git/
__pycache__/
*.log
data.zip
```

这些文件数量多、体积大，并且当前演示网站不需要它们。

## 推荐上传方式

不要直接上传整个项目根目录。推荐上传压缩包：

```text
trifocus-web-demo-node.zip
```

在宝塔文件管理中上传后，解压到类似目录：

```text
/www/wwwroot/trifocus
```

解压后目录里应该能看到：

```text
/www/wwwroot/trifocus/index.html
/www/wwwroot/trifocus/server.js
/www/wwwroot/trifocus/package.json
```

## 宝塔 Node 项目设置

项目目录：

```text
/www/wwwroot/trifocus
```

启动文件：

```text
server.js
```

启动命令：

```bash
npm start
```

Node 版本：

```text
Node.js 16 或更高
```

端口：

```text
8000
```

如果宝塔自动分配端口，`server.js` 会读取环境变量 `PORT`。

## 域名和反向代理

推荐访问链路：

```text
用户访问域名
  -> 宝塔 / Nginx
  -> Node 项目
  -> server.js
```

确认接口可用：

```text
https://你的域名/api/health
```

正常会返回：

```json
{
  "status": "ok",
  "engine": "node-demo-backend",
  "uses_tabpfn": false
}
```

## 项目修改后如何更新

小修改，例如改页面文字、样式、前端逻辑：

```text
重新上传 index.html / styles.css / app.js
然后在宝塔里重启 Node 项目
```

后端接口修改：

```text
重新上传 server.js
然后在宝塔里重启 Node 项目
```

依赖或启动方式修改：

```text
重新上传 package.json
在宝塔里重新安装依赖或重启 Node 项目
```

最稳妥的更新方式：

```text
1. 本地重新生成 trifocus-web-demo-node.zip
2. 宝塔上传压缩包
3. 解压覆盖旧文件
4. 重启 Node 项目
5. 打开 /api/health 检查
```

## 当前模型说明

当前 Node 后端是演示代理模型：

```text
uses_tabpfn = false
```

它可以完整展示网站流程，但不是真实 TabPFN 推理。后续如果要接入真实 TabPFN，建议改为 Python 推理服务，Node/Nginx 只负责转发请求。
