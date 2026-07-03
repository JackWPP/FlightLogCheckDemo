from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests

from .config import provider_config
from .schemas import FieldSpec, RecognitionResult


def _image_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def build_prompt(field: FieldSpec) -> str:
    hints = {
        "numeric_text": "只返回手写或印刷数字，不要返回标签文字；如果有多个数字，请返回与字段名最相关的数字或数字列表。",
        "date_text": "只返回日期值，尽量归一化为 YYYY-MM-DD；不要返回 DATE 标签。",
        "keyed_text": "只返回字段填写值，不要返回字段标签。",
        "free_text": "返回该字段中的完整手写文本；如果有中英文混合，照实返回。",
        "signature_or_text": "返回签名区域读到的姓名；如果无法读出但有签名痕迹，value 写 SIGN_PRESENT。",
    }
    return (
        "你是航空飞行记录单字段识别器。只识别这一个裁切字段。"
        "输出严格 JSON，不要解释。"
        f"字段: {field.label}。"
        f"识别提示: {hints.get(field.recognizer, '只返回字段填写值，不要返回标签文字')}。"
        f"校验规则: {field.validator} {json.dumps(field.params, ensure_ascii=False)}。"
        "如果字段为空，value 和 normalized_value 都输出空字符串。"
        'JSON格式: {"value":"识别值","normalized_value":"归一化值","confidence":0.0}'
    )


def recognize_with_provider(field: FieldSpec, roi_path: Path, provider: str, model: str | None = None) -> RecognitionResult:
    cfg = provider_config(provider)
    if provider == "mock" or not cfg.api_key:
        return RecognitionResult(value="", normalized_value="", confidence=0.0, provider=cfg.name, model=cfg.model)
    selected_model = model or cfg.model
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": selected_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(field)},
                    {"type": "image_url", "image_url": {"url": _image_data_url(roi_path)}},
                ],
            }
        ],
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = _parse_json_content(content)
        return RecognitionResult(
            value=str(parsed.get("value", "")),
            normalized_value=str(parsed.get("normalized_value") or parsed.get("value", "")),
            confidence=float(parsed.get("confidence") or 0.0),
            provider=cfg.name,
            model=selected_model,
            raw={"content": content, "usage": data.get("usage")},
        )
    except Exception as exc:
        return RecognitionResult(
            value="",
            normalized_value="",
            confidence=0.0,
            provider=cfg.name,
            model=selected_model,
            raw={"error": str(exc)},
        )


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
        return json.loads(text)
    return {"value": text, "normalized_value": text, "confidence": 0.5}
