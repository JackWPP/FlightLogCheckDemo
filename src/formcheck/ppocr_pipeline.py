from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from .paddleocr_aistudio import default_optional_payload, download_jsonl, poll_job, submit_job, token_from_env
from .schemas import OcrBlock


def run_ppocrv6(image_path: Path, run_dir: Path, use_doc_unwarping: bool = True) -> dict[str, Any]:
    token = token_from_env()
    out_dir = run_dir / "ppocrv6"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not token:
        return {"ok": False, "error": "Missing PADDLEOCR_AISTUDIO_TOKEN in .env", "blocks": [], "ocr_image_url": None}

    optional_payload = default_optional_payload("PP-OCRv6", use_doc_unwarping, use_orientation=False)
    try:
        job_id = submit_job(str(image_path), token, "PP-OCRv6", optional_payload)
        job = poll_job(token, job_id, interval=5, max_wait=300)
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
        return {
            "ok": True,
            "error": None,
            "blocks": blocks,
            "blocks_json": blocks_json,
            "ocr_image_path": image_paths[0] if image_paths else None,
            "ocr_image_url": None,
            "job_id": job_id,
        }
    except Exception as exc:
        (out_dir / "error.json").write_text(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": False, "error": str(exc), "blocks": [], "ocr_image_url": None}


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
