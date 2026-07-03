from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from formcheck.paddleocr_aistudio import (
    default_optional_payload,
    download_jsonl,
    poll_job,
    submit_job,
    token_from_env,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--model", default="PaddleOCR-VL-1.6", choices=["PaddleOCR-VL-1.6", "PP-OCRv6"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--use-doc-unwarping", action="store_true")
    parser.add_argument("--use-orientation", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--max-wait", type=int, default=300)
    args = parser.parse_args()

    token = token_from_env()
    if not token:
        raise SystemExit("Missing PADDLEOCR_AISTUDIO_TOKEN in .env")

    run_id = args.run_id or f"{Path(args.file).stem}_{args.model.replace('/', '_')}"
    out_dir = Path("out") / "paddleocr_jobs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    optional_payload = default_optional_payload(args.model, args.use_doc_unwarping, args.use_orientation)

    job_id = submit_job(args.file, token, args.model, optional_payload)
    job = poll_job(token, job_id, interval=args.poll_interval, max_wait=args.max_wait)
    meta = {
        "ok": job.ok,
        "job_id": job.job_id,
        "state": job.state,
        "json_url": job.json_url,
        "error": job.error,
        "data": job.data,
        "model": args.model,
        "optional_payload": optional_payload,
    }
    (out_dir / "job.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if not job.ok or not job.json_url:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    records = download_jsonl(job.json_url)
    (out_dir / "result.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
    extracted = extract_outputs(records, out_dir)
    summary = {
        "ok": True,
        "model": args.model,
        "job_id": job_id,
        "records": len(records),
        "markdown_files": extracted["markdown_files"],
        "downloaded_images": extracted["downloaded_images"],
        "sample_text": extracted["sample_text"][:1600],
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def extract_outputs(records: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    markdown_files: list[str] = []
    downloaded_images: list[str] = []
    sample_chunks: list[str] = []
    page_num = 0
    for record in records:
        result = record.get("result") or {}
        for res in result.get("layoutParsingResults") or []:
            markdown = (res.get("markdown") or {}).get("text") or ""
            if markdown:
                path = out_dir / f"doc_{page_num}.md"
                path.write_text(markdown, encoding="utf-8")
                markdown_files.append(str(path))
                sample_chunks.append(markdown)
            for group_name in ["outputImages"]:
                for img_name, url in (res.get(group_name) or {}).items():
                    path = out_dir / f"{img_name}_{page_num}.jpg"
                    if download_url(url, path):
                        downloaded_images.append(str(path))
            page_num += 1
        for res in result.get("ocrResults") or []:
            text = json.dumps(res, ensure_ascii=False)
            sample_chunks.append(text)
            image_url = res.get("ocrImage")
            if image_url:
                path = out_dir / f"ocr_image_{page_num}.jpg"
                if download_url(image_url, path):
                    downloaded_images.append(str(path))
            page_num += 1
    return {"markdown_files": markdown_files, "downloaded_images": downloaded_images, "sample_text": "\n".join(sample_chunks)}


def download_url(url: str, path: Path) -> bool:
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return True


if __name__ == "__main__":
    main()
