from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ASSETS_DIR, OUT_DIR, OUTPUTS_DIR, ROOT
from .demo import demo_payload, ensure_demo_sample
from .llm_cleaner import DEFAULT_CLEANER_MODEL
from .pipeline import analyze_image, registration_mode


app = FastAPI(title="Flight Log Check Demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/runs", StaticFiles(directory=OUT_DIR), name="runs")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


@app.get("/")
def index() -> FileResponse:
    ensure_demo_sample()
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/report")
def report() -> FileResponse:
    ensure_demo_sample()
    return FileResponse(ROOT / "docs" / "technical_report.html")


@app.get("/api/demo")
def demo() -> dict:
    return demo_payload()


def _public_config() -> dict:
    """Surface which models/keys the server is currently using.

    Returned to the frontend so the upload page can show a small
    "powered by" line. Never includes key values.
    """
    keys = {
        "paddleocr": bool(os.getenv("PADDLEOCR_AISTUDIO_TOKEN")),
        "siliconflow": bool(os.getenv("SILICONFLOW_API_KEY")),
        "aliyun": bool(os.getenv("ALIYUN_API_KEY")),
    }
    return {
        "mode": "hybrid",
        "cleaner_provider": "siliconflow",
        "cleaner_model": os.getenv("CLEANER_MODEL") or DEFAULT_CLEANER_MODEL,
        "roi_provider": os.getenv("ROI_REVIEW_PROVIDER", "aliyun"),
        "roi_model": os.getenv("ROI_REVIEW_MODEL", "qwen3.7-plus"),
        "registration_mode": registration_mode(),
        "ppocr_cache_enabled": os.getenv("PPOCR_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"},
        "keys_configured": keys,
        "ready_for_live_upload": all(keys.values()),
    }


@app.get("/api/config")
def config() -> dict:
    return _public_config()


def _save_upload(file: UploadFile) -> tuple[str, Path]:
    """Save the upload to a fresh run dir. Returns (run_id, upload_path)."""
    run_id = uuid.uuid4().hex[:12]
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    upload_path = run_dir / f"upload{suffix}"
    with upload_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return run_id, upload_path


@app.post("/api/analyze")
def analyze(
    file: UploadFile = File(...),
    # All provider/model/mode knobs are env-driven. Kept as optional form
    # fields so direct API callers can still override per-request.
    provider: str | None = Form(None),
    model: str | None = Form(None),
    mode: str | None = Form(None),
    cleaner_provider: str | None = Form(None),
    cleaner_model: str | None = Form(None),
) -> dict:
    cfg = _public_config()
    resolved_provider = provider or cfg["roi_provider"]
    resolved_model = model or cfg["roi_model"] or None
    resolved_mode = mode or cfg["mode"]
    resolved_cleaner_provider = cleaner_provider or cfg["cleaner_provider"]
    resolved_cleaner_model = cleaner_model or cfg["cleaner_model"]

    run_id, upload_path = _save_upload(file)
    report = analyze_image(
        upload_path,
        provider=resolved_provider,
        model=resolved_model,
        run_id=run_id,
        mode=resolved_mode,
        cleaner_provider=resolved_cleaner_provider,
        cleaner_model=resolved_cleaner_model,
    )
    report["run_id"] = run_id
    report["upload_url"] = f"/runs/{run_id}/{upload_path.name}"
    return report


@app.post("/api/analyze/stream")
async def analyze_stream(file: UploadFile = File(...)) -> StreamingResponse:
    """Same as /api/analyze but streams progress via Server-Sent Events.

    Event format (one JSON object per `data:` line, terminated by blank line):
        data: {"stage": "started", "run_id": "..."}\n\n
        data: {"stage": "registration", "status": "running"}\n\n
        data: {"stage": "registration", "status": "done",   "duration_ms": 1234, "inliers": 87}\n\n
        data: {"stage": "ocr",          "status": "running"}\n\n
        data: {"stage": "ocr",          "status": "done",   "duration_ms": 5432, "blocks": 252}\n\n
        data: {"stage": "validate",     "status": "done",   "duration_ms": 45,   "count": 21}\n\n
        data: {"stage": "review",       "status": "running", "total": 3}\n\n
        data: {"stage": "review",       "status": "done",   "duration_ms": 8765, "total": 3}\n\n
        data: {"stage": "final",        "report": { ...full report... }}\n\n
    """
    cfg = _public_config()
    run_id, upload_path = _save_upload(file)
    q: queue.Queue = queue.Queue()
    loop = asyncio.get_running_loop()

    def run_pipeline() -> None:
        def progress_cb(stage: str, status: str, **extra) -> None:
            payload = {"stage": stage, "status": status, **extra}
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

        try:
            report = analyze_image(
                upload_path,
                provider=cfg["roi_provider"],
                model=cfg["roi_model"] or None,
                run_id=run_id,
                mode=cfg["mode"],
                cleaner_provider=cfg["cleaner_provider"],
                cleaner_model=cfg["cleaner_model"],
                progress_cb=progress_cb,
            )
            report["run_id"] = run_id
            report["upload_url"] = f"/runs/{run_id}/{upload_path.name}"
            q.put_nowait({"stage": "final", "report": report})
        except Exception as exc:  # noqa: BLE001
            q.put_nowait({
                "stage": "error",
                "error": str(exc),
                "trace": traceback.format_exc(limit=5),
            })
        finally:
            q.put_nowait(None)  # sentinel -> end of stream

    threading.Thread(target=run_pipeline, daemon=True).start()

    async def event_gen():
        # Initial "started" so the frontend gets feedback immediately.
        yield f"data: {json.dumps({'stage': 'started', 'run_id': run_id}, ensure_ascii=False)}\n\n"
        while True:
            evt = await loop.run_in_executor(None, q.get)
            if evt is None:
                break
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
