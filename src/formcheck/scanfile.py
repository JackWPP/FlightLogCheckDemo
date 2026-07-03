from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import load_dotenv


SCANFILE_ENDPOINT = "https://cloud-iqs.aliyuncs.com/scan/file"


@dataclass
class ScanFileResult:
    ok: bool
    image_bytes: bytes | None
    width: int | None = None
    height: int | None = None
    angle: int | None = None
    request_id: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


def scan_file(
    image_path: Path,
    api_key: str,
    auto_crop: str = "true",
    auto_rotate: str = "true",
    timeout: int = 30,
) -> ScanFileResult:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "imageBase64": image_b64,
        "scanFileInputConfig": {
            "autoCrop": auto_crop,
            "autoRotate": auto_rotate,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(SCANFILE_ENDPOINT, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        items = data.get("scanFileInfoList") or []
        if not items:
            return ScanFileResult(False, None, request_id=data.get("requestId"), raw=data, error="empty_scanFileInfoList")
        first = items[0]
        return ScanFileResult(
            ok=True,
            image_bytes=base64.b64decode(first["imageBase64"]),
            width=first.get("width"),
            height=first.get("height"),
            angle=first.get("angle"),
            request_id=data.get("requestId"),
            raw={"searchInformation": data.get("searchInformation")},
        )
    except Exception as exc:
        raw: dict[str, Any] | None = None
        if "response" in locals():
            try:
                raw = response.json()
            except Exception:
                raw = {"status_code": response.status_code, "text": response.text[:1000]}
        return ScanFileResult(False, None, raw=raw, error=str(exc))


def api_key_from_env() -> str | None:
    load_dotenv()
    import os

    return os.getenv("ALIYUN_IQS_API_KEY")
