from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import load_dotenv


JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"


@dataclass
class PaddleJobResult:
    ok: bool
    job_id: str | None = None
    state: str | None = None
    json_url: str | None = None
    data: dict[str, Any] | None = None
    error: str | None = None


def token_from_env() -> str | None:
    load_dotenv()
    import os

    return os.getenv("PADDLEOCR_AISTUDIO_TOKEN")


def submit_job(
    file_path: str,
    token: str,
    model: str,
    optional_payload: dict[str, Any],
    timeout: int = 60,
) -> str:
    headers = {"Authorization": f"bearer {token}"}
    if file_path.startswith("http"):
        headers["Content-Type"] = "application/json"
        payload = {"fileUrl": file_path, "model": model, "optionalPayload": optional_payload}
        response = requests.post(JOB_URL, json=payload, headers=headers, timeout=timeout)
    else:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = {"model": model, "optionalPayload": json.dumps(optional_payload, ensure_ascii=False)}
        with path.open("rb") as fh:
            response = requests.post(JOB_URL, headers=headers, data=data, files={"file": fh}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload["data"]["jobId"]


def poll_job(token: str, job_id: str, interval: int = 5, max_wait: int = 300) -> PaddleJobResult:
    headers = {"Authorization": f"bearer {token}"}
    start = time.time()
    last_payload: dict[str, Any] | None = None
    while time.time() - start < max_wait:
        response = requests.get(f"{JOB_URL}/{job_id}", headers=headers, timeout=60)
        response.raise_for_status()
        payload = response.json()
        last_payload = payload
        data = payload.get("data") or {}
        state = data.get("state")
        if state == "done":
            return PaddleJobResult(
                ok=True,
                job_id=job_id,
                state=state,
                json_url=(data.get("resultUrl") or {}).get("jsonUrl"),
                data=data,
            )
        if state == "failed":
            return PaddleJobResult(False, job_id=job_id, state=state, data=data, error=data.get("errorMsg"))
        time.sleep(interval)
    return PaddleJobResult(False, job_id=job_id, state="timeout", data=last_payload, error="poll_timeout")


def download_jsonl(json_url: str) -> list[dict[str, Any]]:
    response = requests.get(json_url, timeout=120)
    response.raise_for_status()
    records: list[dict[str, Any]] = []
    for line in response.text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def default_optional_payload(model: str, use_doc_unwarping: bool, use_orientation: bool) -> dict[str, Any]:
    if model == "PP-OCRv6":
        return {
            "useDocOrientationClassify": use_orientation,
            "useDocUnwarping": use_doc_unwarping,
            "useTextlineOrientation": False,
        }
    return {
        "useDocOrientationClassify": use_orientation,
        "useDocUnwarping": use_doc_unwarping,
        "useChartRecognition": False,
    }
