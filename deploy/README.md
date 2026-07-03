# 部署指南（Flight Log Check Demo）

整套部署就是 **Docker Compose + Nginx 反代 + 你的云端 API key**。本目录 `deploy/` 包含所有运行时需要的配置。

## 0. 服务器最低配置

- **OS**：Ubuntu 22.04 / 24.04 LTS（其他发行版也行，命令略改）
- **规格**：1 vCPU + 2 GB RAM 起步，2 vCPU + 4 GB RAM 更舒服
- **网络**：放行 22 / 9080（或你自定义的端口）；其他端口不要对外暴露
- **域名**（建议）：先 DNS A 记录指到服务器 IP，方便上 HTTPS

## 1. 装 Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker          # 或者重新登录，让 docker 命令免 sudo
docker --version       # 期望 >= 24
docker compose version # 期望 >= v2.20
```

## 2. 拉代码

```bash
sudo mkdir -p /opt/flight-log-check
sudo chown -R $USER:$USER /opt/flight-log-check
cd /opt/flight-log-check
git clone <your-repo-url> .
```

> 没建远端仓？先在本机建好 GitHub / Gitee / GitLab 仓，把代码 push 上去，再回来拉。

## 3. 配环境变量

```bash
cp .env.example .env
nano .env              # 或 vim .env
```

必填项（其它保留默认即可）：

| 变量 | 谁用 | 不填会怎样 |
|---|---|---|
| `PADDLEOCR_AISTUDIO_TOKEN` | 主链路 OCR | 上传图会回退到 mock，前端所有字段 `Unresolved` |
| `SILICONFLOW_API_KEY` | DeepSeek 字段清洗 | 同上，cleaner 静默失败 |
| `ALIYUN_API_KEY` | ROI-VLM 复核 | 不致命，只是少一道二次确认 |

> **demo 模式**（点 "示例" 标签）不调用任何云端接口，所以即使 `.env` 全空，主页的缓存示例也能正常展示。

## 4. 起服务

```bash
docker compose build           # 第一次会装依赖、构镜像，约 2-5 分钟
docker compose up -d           # 后台启动 app + nginx
docker compose ps              # 两个容器都是 healthy/Up
```

## 5. 验证

```bash
# 看 demo 缓存（不调用云端）
curl -fsS http://localhost:9080/api/demo | head -c 200

# 跟健康检查同款端点
curl -fsS -o /dev/null -w "%{http_code}\n" http://localhost:9080/api/demo
# 期望：200

# 看日志
docker compose logs -f app
```

打开浏览器访问 `http://<服务器IP>/`，应该看到 demo 首页，点「上传新图」选一张表单一拍就能跑全链路。

## 6. 上 HTTPS（强烈建议）

```bash
sudo apt install -y certbot
sudo certbot certonly --standalone -d your.domain.com
```

把证书复制到 compose 期望的目录：

```bash
sudo mkdir -p /opt/flight-log-check/deploy/certs
sudo cp /etc/letsencrypt/live/your.domain.com/fullchain.pem /opt/flight-log-check/deploy/certs/
sudo cp /etc/letsencrypt/live/your.domain.com/privkey.pem   /opt/flight-log-check/deploy/certs/
```

打开 `deploy/nginx.conf`，把底部 443 server 块的注释去掉，**并把端口映射改成 `"443:443"`**（在 `docker-compose.yml`），保存后：

```bash
docker compose up -d --force-recreate nginx
```

最后加一条自动续期（certbot 自带）：

```bash
sudo certbot renew --dry-run   # 测一次
# 真实续期通过 /etc/cron.d/certbot 自动跑；记得续期后同步证书到 deploy/certs/
```

## 7. 更新 / 回滚

```bash
# 拉新代码 + 重启
cd /opt/flight-log-check
git pull
docker compose build
docker compose up -d

# 出问题回滚到上一个 tag
git tag                      # 看看打过哪些 tag
git checkout v0.0.2
docker compose build
docker compose up -d

# 彻底停掉
docker compose down
```

## 8. 运维速查

| 想看的东西 | 命令 |
|---|---|
| 容器状态 | `docker compose ps` |
| 实时日志（app） | `docker compose logs -f app` |
| 实时日志（nginx） | `docker compose logs -f nginx` |
| 进入容器排错 | `docker compose exec app bash` |
| 占用 | `docker stats` |
| 清理旧镜像 | `docker image prune -f` |
| 备份 `out/` 和 `outputs/` | `tar -czf backup-$(date +%F).tgz out outputs` |

## 9. 常见问题

**Q：上传图片后页面一直"核查中"，但服务没崩？**
A：大概率 PP-OCRv6 / VLM 那一头在排队。把 `proxy_read_timeout` 调到 180s，或者看一下 `docker compose logs app` 里有没有 4xx/5xx。

**Q：访问 `/api/demo` 返回 200，但前端图表是空白？**
A：浏览器开发者工具看 Network 里 `/outputs/demo_sample/...` 是不是 404。是的话说明 demo 缓存没打进镜像——重新 `docker compose build --no-cache` 试试。

**Q：`.env` 改了之后没生效？**
A：`docker compose up -d` 不会重新加载 env。需要 `docker compose up -d --force-recreate app`。

**Q：能不能不装 docker？**
A：可以。`uv sync --python 3.12` 后用 `uv run uvicorn formcheck.app:app --host 0.0.0.0 --port 8003` 直接跑，前面再自己套一个 nginx/systemd。Docker 只是为了让 "新机器上来就能跑"。