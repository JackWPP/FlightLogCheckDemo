from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import requests

from formcheck.config import OUT_DIR, load_dotenv, provider_config


PROMPT = """
你是航空飞行记录单整页 OCR 与版面解析器。
任务：从整张图片中识别文字块、手写内容、勾选状态，并尽量给出它们在页面中的位置。

请严格输出 JSON，不要解释，不要 Markdown。格式：
{
  "summary": "一句话概括图像质量和是否有遮挡",
  "blocks": [
    {
      "text": "识别出的文字",
      "type": "printed|handwritten|checkbox|signature|redaction|unknown",
      "bbox_norm": [x1,y1,x2,y2],
      "confidence": 0.0
    }
  ],
  "field_candidates": {
    "oil_last_row_added": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "oil_qty": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "fault_flight_no": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "fault_station": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "fault_reported_by": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "fault_report_en": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "fault_ref": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "fault_date": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "fault_authorization": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "apu_cum_hours": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "apu_cum_cycles": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "awr_station": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "awr_date": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "awr_deferred_item": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0},
    "awr_license": {"value": "", "evidence": "", "bbox_norm": [0,0,0,0], "confidence": 0.0}
  }
}

bbox_norm 使用 0-1000 的页面归一化坐标。遮挡看不清的字段 value 输出 "UNREADABLE"，不要猜。
"""


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="assets/raw/sample_01.jpg")
    parser.add_argument("--provider", default="siliconflow")
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.5")
    parser.add_argument("--run-id", default="layout_ocr")
    args = parser.parse_args()

    out_dir = OUT_DIR / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result = call_layout_model(Path(args.image), args.provider, args.model)
    out_path = out_dir / "layout_result.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summarize(result), ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


def call_layout_model(image_path: Path, provider: str, model: str) -> dict[str, Any]:
    cfg = provider_config(provider)
    if not cfg.api_key:
        return {"ok": False, "error": f"missing API key for {provider}"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        ],
        "temperature": 0,
    }
    try:
        resp = requests.post(
            cfg.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return {"ok": True, "provider": provider, "model": model, "parsed": parse_jsonish(content), "raw_content": content, "usage": data.get("usage")}
    except Exception as exc:
        raw: Any = None
        if "resp" in locals():
            try:
                raw = resp.json()
            except Exception:
                raw = resp.text[:2000]
        return {"ok": False, "provider": provider, "model": model, "error": str(exc), "raw": raw}


def image_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def parse_jsonish(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"text": content}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    parsed = result.get("parsed") or {}
    fields = parsed.get("field_candidates") if isinstance(parsed, dict) else None
    if not isinstance(fields, dict):
        return {"ok": result.get("ok"), "error": result.get("error"), "raw_preview": str(result.get("raw_content") or result.get("raw"))[:800]}
    return {
        "ok": result.get("ok"),
        "model": result.get("model"),
        "summary": parsed.get("summary"),
        "block_count": len(parsed.get("blocks") or []),
        "field_candidates": {k: v.get("value") if isinstance(v, dict) else v for k, v in fields.items()},
    }


if __name__ == "__main__":
    main()
