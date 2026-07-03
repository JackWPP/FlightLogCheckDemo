from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import requests

from .config import OUTPUTS_DIR, load_dotenv
from .paddleocr_aistudio import default_optional_payload, download_jsonl, poll_job, submit_job, token_from_env
from .schemas import OcrBlock


def run_ppocrv6(image_path: Path, run_dir: Path, use_doc_unwarping: bool = True) -> dict[str, Any]:
    load_dotenv()
    token = token_from_env()
    out_dir = run_dir / "ppocrv6"
    out_dir.mkdir(parents=True, exist_ok=True)
    optional_payload = default_optional_payload("PP-OCRv6", use_doc_unwarping, use_orientation=False)
    cache_dir = ppocr_cache_dir(image_path, optional_payload)
    if ppocr_cache_enabled():
        cached = load_ppocr_cache(cache_dir, out_dir)
        if cached:
            return cached
    if not token:
        return {"ok": False, "error": "Missing PADDLEOCR_AISTUDIO_TOKEN in .env", "blocks": [], "ocr_image_url": None}

    try:
        job_id = submit_job(str(image_path), token, "PP-OCRv6", optional_payload)
        job = poll_job(token, job_id, interval=ppocr_poll_interval(), max_wait=ppocr_max_wait())
        job_meta = {
            "ok": job.ok,
            "job_id": job.job_id,
            "state": job.state,
            "json_url": job.json_url,
            "error": job.error,
            "data": job.data,
            "model": "PP-OCRv6",
            "optional_payload": optional_payload,
        }
        (out_dir / "job.json").write_text(json.dumps(job_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if not job.ok or not job.json_url:
            return {"ok": False, "error": job.error or job.state or "ppocr_job_failed", "blocks": [], "ocr_image_url": None}

        records = download_jsonl(job.json_url)
        (out_dir / "result.jsonl").write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
            encoding="utf-8",
        )
        image_paths = download_ppocr_images(records, out_dir)
        blocks = extract_ppocr_blocks(records)
        blocks_json = [block_to_dict(block) for block in blocks]
        (out_dir / "ocr_blocks.json").write_text(json.dumps(blocks_json, ensure_ascii=False, indent=2), encoding="utf-8")
        if ppocr_cache_enabled():
            save_ppocr_cache(cache_dir, blocks_json, image_paths[0] if image_paths else None, job_meta)
        return {
            "ok": True,
            "error": None,
            "blocks": blocks,
            "blocks_json": blocks_json,
            "ocr_image_path": image_paths[0] if image_paths else None,
            "ocr_image_url": None,
            "job_id": job_id,
            "cache_hit": False,
        }
    except Exception as exc:
        (out_dir / "error.json").write_text(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": False, "error": str(exc), "blocks": [], "ocr_image_url": None}


def ppocr_cache_enabled() -> bool:
    return os.getenv("PPOCR_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def ppocr_poll_interval() -> int:
    return int_env("PPOCR_POLL_INTERVAL_SECONDS", 5, minimum=1, maximum=30)


def ppocr_max_wait() -> int:
    return int_env("PPOCR_MAX_WAIT_SECONDS", 300, minimum=30, maximum=1800)


def int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def ppocr_cache_dir(image_path: Path, optional_payload: dict[str, Any]) -> Path:
    key_payload = {
        "model": "PP-OCRv6",
        "optional_payload": optional_payload,
        "image_sha256": file_sha256(image_path),
    }
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return OUTPUTS_DIR / "runtime" / "ocr_cache" / key


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ppocr_cache(cache_dir: Path, out_dir: Path) -> dict[str, Any] | None:
    blocks_path = cache_dir / "ocr_blocks.json"
    meta_path = cache_dir / "job.json"
    if not blocks_path.exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks_json = json.loads(blocks_path.read_text(encoding="utf-8"))
    blocks = [block_from_dict(item) for item in blocks_json]
    shutil.copy2(blocks_path, out_dir / "ocr_blocks.json")
    if meta_path.exists():
        shutil.copy2(meta_path, out_dir / "job.json")
    cached_image = cache_dir / "ocr_image_0.jpg"
    image_path = None
    if cached_image.exists():
        image_path = out_dir / "ocr_image_0.jpg"
        shutil.copy2(cached_image, image_path)
    return {
        "ok": True,
        "error": None,
        "blocks": blocks,
        "blocks_json": blocks_json,
        "ocr_image_path": image_path,
        "ocr_image_url": None,
        "job_id": "cache",
        "cache_hit": True,
    }


def save_ppocr_cache(cache_dir: Path, blocks_json: list[dict[str, Any]], ocr_image_path: Path | None, job_meta: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "ocr_blocks.json").write_text(json.dumps(blocks_json, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {**job_meta, "cache_saved_at": int(time.time())}
    (cache_dir / "job.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if ocr_image_path and ocr_image_path.exists():
        shutil.copy2(ocr_image_path, cache_dir / "ocr_image_0.jpg")


def extract_ppocr_blocks(records: list[dict[str, Any]]) -> list[OcrBlock]:
    blocks: list[OcrBlock] = []
    index = 0
    for record in records:
        result = record.get("result") or {}
        for ocr in result.get("ocrResults") or []:
            pruned = ocr.get("prunedResult") or {}
            texts = pruned.get("rec_texts") or []
            scores = pruned.get("rec_scores") or []
            boxes = pruned.get("rec_boxes") or []
            for text, score, box in zip(texts, scores, boxes):
                if not text:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                blocks.append(
                    OcrBlock(
                        id=f"b{index:04d}",
                        text=str(text),
                        score=float(score or 0.0),
                        box=(x1, y1, x2, y2),
                        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    )
                )
                index += 1
    return blocks


def download_ppocr_images(records: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    page = 0
    for record in records:
        result = record.get("result") or {}
        for ocr in result.get("ocrResults") or []:
            url = ocr.get("ocrImage")
            if not url:
                continue
            path = out_dir / f"ocr_image_{page}.jpg"
            response = requests.get(url, timeout=120)
            if response.status_code == 200:
                path.write_bytes(response.content)
                paths.append(path)
            page += 1
    return paths


def block_to_dict(block: OcrBlock) -> dict[str, Any]:
    return {
        "id": block.id,
        "text": block.text,
        "score": block.score,
        "box": list(block.box),
        "center": list(block.center),
        "source": block.source,
    }


def block_from_dict(payload: dict[str, Any]) -> OcrBlock:
    box = tuple(float(v) for v in payload["box"])
    center = tuple(float(v) for v in payload.get("center") or ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    return OcrBlock(
        id=str(payload["id"]),
        text=str(payload["text"]),
        score=float(payload.get("score") or 0.0),
        box=box,  # type: ignore[arg-type]
        center=center,  # type: ignore[arg-type]
        source=str(payload.get("source") or "ppocrv6"),
    )
