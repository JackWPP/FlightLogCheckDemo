from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image
import requests

from .config import provider_config
from .schemas import FieldSpec, RecognitionResult


def _image_data_url(path: Path) -> str:
    payload = image_payload(path)
    return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"


def image_payload(path: Path) -> bytes:
    max_side = max_image_side()
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


def max_image_side() -> int:
    try:
        value = int(os.getenv("VLM_IMAGE_MAX_SIDE", "900"))
    except ValueError:
        return 900
    return max(320, min(value, 1600))


def build_prompt(field: FieldSpec) -> str:
    hints = {
        "numeric_text": "只返回手写或印刷数字，不要返回标签文字；如果有多个数字，请返回与字段名最相关的数字或数字列表。",
        "date_text": "只返回日期值，尽量归一化为 YYYY-MM-DD；不要返回 DATE 标签。",
        "keyed_text": "只返回字段填写值，不要返回字段标签。",
        "free_text": "忽略印刷表头、格线标签和字段名，只返回手写正文；如果有中英文混合，中文和英文都照实返回。",
        "signature_or_text": "返回签名区域读到的姓名；如果无法读出但有签名痕迹，value 写 SIGN_PRESENT。",
    }
    return (
        "你是航空飞行记录单字段识别器。只识别这一个裁切字段。"
        "输出严格 JSON，不要解释。"
        f"字段: {field.label}。"
        f"识别提示: {hints.get(field.recognizer, '只返回字段填写值，不要返回标签文字')}。"
        f"字段定位提示: {field_specific_hint(field)}。"
        f"校验规则: {field.validator} {json.dumps(field.params, ensure_ascii=False)}。"
        "若规则是 bilingual_text，必须读取正文里的中文和英文，忽略 P/N、S/N、FIN、SIGN 等表格标签。"
        "如果裁切图中有多个相邻格，只读取字段名对应的目标格，不要读取旁边格。"
        "如果字段为空，value 和 normalized_value 都输出空字符串。"
        'JSON格式: {"value":"识别值","normalized_value":"归一化值","confidence":0.0}'
    )


def field_specific_hint(field: FieldSpec) -> str:
    label = field.label
    params = field.params or {}
    if field.validator == "int_range":
        return f"目标是 {label}，合规范围是 {params.get('min')} 到 {params.get('max')}；若图中有多个数字，优先选择该范围和字段列都匹配的数字。"
    if field.validator in {"digit_length", "number_less_than"} and "APU" in label:
        return "目标是 APU 区域底部累计使用时间/循环的小格数字，只读对应格内数字，忽略相邻 APU 小格。"
    if field.validator == "regex" and "授权号" in label:
        return "目标是授权号，通常是 6 位数字；忽略日期、地点、SIGN、AUTHORIZATION NO 标签和签名。"
    if field.validator == "regex" and "执照号" in label:
        return "目标是执照号，通常形如 CAACML 加 8 位数字；忽略地点、放行签署和 LICENSE NO 标签。"
    if field.validator == "name_not_place":
        return "目标是姓名或签名，不要把重庆、渝、日期、数字或放行声明当作姓名。"
    if field.validator == "exact_text":
        return "目标是闭集字段，只返回填写值；重庆和渝等价，NA 和 N/A 等价。"
    if field.validator == "bilingual_text":
        return "目标是正文内容，需要同时保留中文和英文手写正文；忽略所有表头和零件号表格标签。"
    return "只读目标字段值；忽略相邻格、表头、字段标签和格线文字。"


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
        timeout = request_timeout_seconds()
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
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
            raw={"error": str(exc), "timeout_seconds": request_timeout_seconds()},
        )


def request_timeout_seconds() -> int:
    try:
        value = int(os.getenv("VLM_REQUEST_TIMEOUT_SECONDS", "45"))
    except ValueError:
        return 45
    return max(5, min(value, 180))


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
