from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ASSETS_DIR, OUT_DIR, OUTPUTS_DIR, ROOT
from .demo import demo_payload, ensure_demo_sample
from .pipeline import analyze_image


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


@app.post("/api/analyze")
def analyze(
    file: UploadFile = File(...),
    provider: str = Form("mock"),
    model: str | None = Form(None),
    mode: str = Form("hybrid"),
    cleaner_provider: str = Form("siliconflow"),
    cleaner_model: str | None = Form(None),
) -> dict:
    run_id = uuid.uuid4().hex[:12]
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    upload_path = run_dir / f"upload{suffix}"
    with upload_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    report = analyze_image(
        upload_path,
        provider=provider,
        model=model or None,
        run_id=run_id,
        mode=mode,
        cleaner_provider=cleaner_provider,
        cleaner_model=cleaner_model or None,
    )
    report["run_id"] = run_id
    report["upload_url"] = f"/runs/{run_id}/{upload_path.name}"
    return report
