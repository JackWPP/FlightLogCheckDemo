# 飞行记录单自动核查 Demo

一个面向固定版式飞行记录单的自动核查演示项目。Demo 打开后会默认展示一份缓存示例；上传新图时才调用云端 OCR/VLM。

## 当前能力

- 默认示例：打开页面即可看到原图、配准图、PP-OCRv6 检测图、电子化字段、问题列表和字段证据。
- OCR 主链路：原始上传图直接交给 PaddleOCR AI Studio `PP-OCRv6`，使用其内置文档矫正能力。
- 字段结构化：PP-OCRv6 输出文字块后，本地按 `fields.yaml` 做字段归属，再用 DeepSeek cleaner 清洗为字段值。
- 规则判断：所有 pass/fail 均由本地 validators 完成，不让 LLM 直接决定合规。
- 审核友好问题列表：字段级失败全部保存在 `all_problems`，右侧展示问题由 LLM/本地兜底压缩到默认 4 条。
- 持久任务队列：上传新图走 SQLite 任务队列，支持 pending/running/done/failed、批量上传和刷新恢复。
- ROI 兜底：默认从 PaddleOCR 输出的 OCR 图与字段候选 blocks 裁切局部证据，再用视觉模型复核；只有复核值通过本地规则才替换。
- 证据保留：每次运行保存 `report.json`、`field_candidates.json`、`ocr_blocks.json`、OCR 检测图、ROI 图。

## 快速启动

需要 Python 3.11 或 3.12，以及 `uv`。当前依赖包含 `numpy==1.26.x`，不要使用 Python 3.13 创建环境。

```powershell
uv sync --python 3.12
uv run --python 3.12 uvicorn formcheck.app:app --host 127.0.0.1 --port 8003
```

打开：

```text
http://127.0.0.1:8003/
```

也可以使用启动脚本：

```powershell
.\scripts\run_demo.ps1
```

## 模型配置

复制 `.env.example` 为 `.env`，填入本地 key。不要提交真实 key。

```powershell
copy .env.example .env
```

主要配置：

- `PADDLEOCR_AISTUDIO_TOKEN`：调用 PaddleOCR AI Studio `PP-OCRv6`。
- `SILICONFLOW_API_KEY`：DeepSeek cleaner，默认模型 `deepseek-ai/DeepSeek-V4-Flash`。
- `ALIYUN_API_KEY`：ROI 级视觉复核，默认建议使用图像 OCR 模型 `qwen3.5-ocr`。
- `ROI_REVIEW_PROVIDER` / `ROI_REVIEW_MODEL`：控制复核模型；若主模型超时或报错，可通过 `ROI_REVIEW_FALLBACK_MODEL` 自动降级。
- `ROI_REVIEW_CONCURRENCY`：ROI 复核并发数，默认 `3`，用于让数字/编号类失败字段并发进入 qwen 复核。
- `ROI_REVIEW_MAX_FIELDS`：单张表最多实时复核的 ROI 数，默认 `12`，`0` 表示不限制；超过预算的低优先级字段会保留人工复核证据，不阻塞整张单。
- `ROI_REVIEW_CACHE_ENABLED`：ROI/VLM 复核缓存，默认 `1`。同一 ROI 图、字段和模型重复复核时复用 `outputs/runtime/roi_review_cache/`。
- `ROI_REVIEW_ACCEPT_MIN_CONFIDENCE`：ROI/VLM 复核自动采纳阈值，默认 `0.65`；低于阈值的通过结果只作为证据保留并标记需复核。
- `CLEANER_SECTION_TIMEOUT_SECONDS` / `CLEANER_TOTAL_BUDGET_SECONDS`：Cleaner 分区请求超时与总预算，默认 `75` / `90`。
- `ISSUE_TRIAGE_TIMEOUT_SECONDS`：问题终裁超时，默认 `45`。
- `ISSUE_DISPLAY_LIMIT`：右侧展示问题数量上限，默认 `4`；`all_problems` 不受影响。
- `REGISTRATION_MODE`：默认 `off`。可选 `optional` / `required`，只在需要旧版 SIFT 配准 ROI 时开启。
- `PPOCR_CACHE_ENABLED`：默认 `1`。同一张图片、同一组 PP-OCR 参数会复用 `outputs/runtime/ocr_cache/`，避免反复提交云端 OCR。
- `PPOCR_POLL_INTERVAL_SECONDS` / `PPOCR_MAX_WAIT_SECONDS`：控制 PaddleOCR 异步 job 轮询间隔和最大等待时间。

说明：默认缓存示例不需要 key；只有上传新图并实时分析时才需要云端 key。

## 项目结构

```text
assets/                 示例原图、扫描底图、canonical 底图
docs/                   技术报告 HTML
outputs/demo_sample/    默认演示缓存，不调用云端即可展示
scripts/                环境、调试、模型对比和启动脚本
src/formcheck/          后端核心逻辑
static/                 本地 Web Demo
tests/                  单元测试
fields.yaml             字段、ROI、规则和候选归属配置
```

## Pipeline

```text
上传图片
  -> PP-OCRv6 原图文档矫正 + 整页 OCR
  -> OCR blocks 字段归属
  -> DeepSeek cleaner 清洗字段值
  -> 本地 validators 规则校验
  -> 从 PaddleOCR OCR 图/blocks 裁切失败关键字段 ROI，并发交给 ROI-VLM double check
  -> LLM issue triage 压缩展示问题，保留 all_problems 全量证据
  -> Web 展示问题列表与证据
```

默认线上链路不跑本地 SIFT 配准，主路径依赖 PaddleOCR 的文档矫正与整页 OCR。旧版配准链路仍保留为可选辅助：设置 `REGISTRATION_MODE=optional` 时，如果配准成功会额外生成 `warped.png` 和旧式 ROI；设置 `required` 时配准失败会中断。

## 当前示例结果

默认示例使用 `outputs/demo_sample/` 中的缓存结果：

- OCR blocks：252
- 字段总数：29
- 通过字段：21
- 失败字段：8
- 需复核字段：13
- 当前规则来自甲方带序号检查单：注册号、滑油栏、故障报告、处理措施、APU 累计、适航放行共 29 个检查点。

上传新图的 `report.json` 会包含 `timings`，用于定位线上慢点，例如 `ppocr_submit_ms`、`ppocr_poll_ms`、`ppocr_download_ms`、`assignment_ms`、`cleaner_ms`、`review_ms`、`issue_triage_ms` 和 `total_ms`。`summary.ocr_cache_hit` 为 `true` 时表示本次整页 OCR 走缓存，没有重新访问 PaddleOCR 云端。

任务队列数据保存在 `outputs/runtime/tasks.sqlite3`，PP-OCR、Cleaner 和 ROI/VLM 复核分别缓存在 `outputs/runtime/ocr_cache/`、`outputs/runtime/cleaner_cache/`、`outputs/runtime/roi_review_cache/`。同一张图重复上传时会尽量复用缓存。

## 线上排错

后端会输出结构化 JSONL 日志，默认位置：

```text
outputs/runtime/logs/app.jsonl
```

单个任务排错时，先从 UI 任务卡片或 `/api/tasks?session_id=...` 拿到 `task_id`，再访问：

```text
/api/tasks/<task_id>/logs
```

日志会按 `request_id / task_id / run_id / session_id` 串起 HTTP 请求、任务队列、PP-OCR、Cleaner、ROI-VLM 和 pipeline 阶段。详细说明见 [`docs/operations_logging.md`](docs/operations_logging.md)。

## 测试

```powershell
uv run --python 3.12 pytest
uv run --python 3.12 python -m compileall src scripts
```

安全检查示例：

```powershell
rg -n "sk-|Authorization:|bearer [A-Za-z0-9]|API_KEY=.*[A-Za-z0-9]" src scripts static tests outputs README.md .env.example fields.yaml
```

## 演示资料

技术报告：

```text
docs/technical_report.html
```

建议演示顺序：

1. 打开 Demo 首页，看默认缓存示例。
2. 切换原图、OCR 图、扫描底图、AI 底图。
3. 点击数字、签名或证照类字段，展示 PP-OCR ROI 与候选证据。
4. 点击问题列表，展示最短问题输出和字段证据。
5. 上传新图，说明实时链路会调用云端模型。

## 注意事项

- `.env`、`.venv`、`out/` 不作为正式交付内容提交。
- `outputs/demo_sample/` 是可演示缓存，可以保留。
- 当前目标是可演示、可验证、可继续调优，不是生产级准确率。
- 拿到真正空白表后，应重新生成 canonical base，并重新精修 `fields.yaml`。

## 部署

整套用 Docker Compose + Nginx 反代，配置已经写好。详细步骤见 [`deploy/README.md`](deploy/README.md)。

服务器上一键起：

```bash
git clone <this-repo> /opt/flight-log-check
cd /opt/flight-log-check
cp .env.example .env && nano .env       # 填三个云端 key
sudo docker compose build && sudo docker compose up -d
```

> 默认 `Dockerfile` 已经把项目依赖指向清华 pip 镜像（`UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`），基础镜像 `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` 自带 uv。**大陆服务器开箱可用**，不用再手动配 pip mirror。

或者用 helper 脚本：

```bash
./deploy/deploy.sh up       # 构建并起
./deploy/deploy.sh logs     # 看日志
./deploy/deploy.sh update   # 拉新代码重启
./deploy/deploy.sh backup   # 打包 out/ 和 outputs/
```

默认监听 `:9080`（通过 nginx），app 容器本身只在内部 `:8003`。HTTPS 模板在 `deploy/nginx.conf` 里，certbot 跑完把证书放到 `deploy/certs/` 再取消注释并把端口改为 `443`。
