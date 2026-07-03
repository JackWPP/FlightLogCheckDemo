# 飞行记录单自动核查 Demo

一个面向固定版式飞行记录单的自动核查演示项目。Demo 打开后会默认展示一份缓存示例；上传新图时才调用云端 OCR/VLM。

## 当前能力

- 默认示例：打开页面即可看到原图、配准图、PP-OCRv6 检测图、电子化字段、问题列表和字段证据。
- OCR 主链路：原始上传图直接交给 PaddleOCR AI Studio `PP-OCRv6`，使用其内置文档矫正能力。
- 字段结构化：PP-OCRv6 输出文字块后，本地按 `fields.yaml` 做字段归属，再用 DeepSeek cleaner 清洗为字段值。
- 规则判断：所有 pass/fail 均由本地 validators 完成，不让 LLM 直接决定合规。
- ROI 兜底：对“已识别但规则未通过”的关键字段裁切扩展 ROI，再用视觉模型复核；只有复核值通过本地规则才替换。
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
- `ALIYUN_API_KEY`：ROI 级视觉复核，默认模型 `qwen3.7-plus`。
- `ROI_REVIEW_PROVIDER` / `ROI_REVIEW_MODEL`：控制复核模型。

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
  -> 失败/需复核关键字段 ROI-VLM double check
  -> Web 展示问题列表与证据
```

配准链路仍保留，用于 ROI 证据和兜底裁切；PP-OCRv6 不再使用我们自己的 warped 图，避免二次图像变形。

## 当前示例结果

默认示例使用 `outputs/demo_sample/` 中的缓存结果：

- OCR blocks：252
- 字段总数：21
- 通过字段：12
- 失败字段：9
- 需复核字段：8
- APU 累计使用循环：主 OCR 读成 `348`，ROI-VLM 复核为 `3481` 后通过。

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
3. 点击 APU 累计使用循环，展示 ROI-VLM 如何把 `348` 复核为 `3481`。
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
docker compose build && docker compose up -d
```

或者用 helper 脚本：

```bash
./deploy/deploy.sh up       # 构建并起
./deploy/deploy.sh logs     # 看日志
./deploy/deploy.sh update   # 拉新代码重启
./deploy/deploy.sh backup   # 打包 out/ 和 outputs/
```

默认监听 `:9080`（通过 nginx），app 容器本身只在内部 `:8003`。HTTPS 模板在 `deploy/nginx.conf` 里，certbot 跑完把证书放到 `deploy/certs/` 再取消注释并把端口改为 `443`。
