from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from .config import provider_config


DEFAULT_ISSUE_LIMIT = 4


def issue_display_limit() -> int:
    try:
        value = int(os.getenv("ISSUE_DISPLAY_LIMIT", str(DEFAULT_ISSUE_LIMIT)))
    except ValueError:
        return DEFAULT_ISSUE_LIMIT
    return max(1, min(value, 8))


def triage_issues(
    fields: list[dict[str, Any]],
    provider: str = "siliconflow",
    model: str | None = None,
    cleaner_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    failed = [field for field in fields if not field.get("passed")]
    all_problems = [field.get("message") for field in failed if field.get("message")]
    limit = issue_display_limit()
    fallback = fallback_triage(failed, all_problems, limit)
    if not failed:
        return {
            "problems": ["通过"],
            "all_problems": [],
            "issue_triage": {
                "provider": "local",
                "model": "none",
                "limit": limit,
                "selected": [],
                "suppressed": [],
                "reason": "no_failed_fields",
                "duration_ms": int((time.time() - started) * 1000),
            },
        }

    cfg = provider_config(provider)
    selected_model = model or os.getenv("CLEANER_MODEL") or "deepseek-ai/DeepSeek-V4-Flash"
    if should_skip_triage_for_cleaner_error(cleaner_meta):
        fallback["issue_triage"]["duration_ms"] = int((time.time() - started) * 1000)
        fallback["issue_triage"]["provider"] = "local:fallback"
        fallback["issue_triage"]["model"] = "priority"
        fallback["issue_triage"]["reason"] = "Cleaner异常，跳过LLM问题压缩"
        fallback["issue_triage"]["cleaner_error"] = cleaner_meta.get("cleaner_error") or cleaner_meta.get("error")
        fallback["issue_triage"]["fallback_sections"] = cleaner_meta.get("cleaner_section_meta", {}).get("fallback_sections", [])
        return fallback
    if not cfg.api_key:
        fallback["issue_triage"]["duration_ms"] = int((time.time() - started) * 1000)
        fallback["issue_triage"]["provider"] = f"{cfg.name}:fallback"
        fallback["issue_triage"]["model"] = selected_model
        return fallback

    prompt = build_triage_prompt(failed, limit)
    payload = {
        "model": selected_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(
            cfg.base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=request_timeout_seconds(),
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_content(content)
        selected = [str(item) for item in parsed.get("problems", []) if str(item).strip()]
        selected = selected[:limit] or fallback["problems"]
        suppressed = [item for item in all_problems if item not in selected]
        return {
            "problems": selected,
            "all_problems": all_problems,
            "issue_triage": {
                "provider": cfg.name,
                "model": selected_model,
                "limit": limit,
                "selected": selected,
                "suppressed": suppressed,
                "reason": parsed.get("reason", ""),
                "raw": {"content": content, "usage": data.get("usage")},
                "duration_ms": int((time.time() - started) * 1000),
            },
        }
    except Exception as exc:  # noqa: BLE001
        fallback["issue_triage"]["error"] = str(exc)
        fallback["issue_triage"]["duration_ms"] = int((time.time() - started) * 1000)
        fallback["issue_triage"]["provider"] = f"{cfg.name}:fallback"
        fallback["issue_triage"]["model"] = selected_model
        return fallback


def fallback_triage(failed: list[dict[str, Any]], all_problems: list[str], limit: int) -> dict[str, Any]:
    ranked = sorted(failed, key=lambda field: issue_priority(field))
    selected = [field.get("message") for field in ranked if field.get("message")]
    selected = [str(item) for item in selected[:limit]]
    suppressed = [item for item in all_problems if item not in selected]
    if suppressed and len(selected) < limit:
        selected.append("若干字段需复核")
    return {
        "problems": selected or ["通过"],
        "all_problems": all_problems,
        "issue_triage": {
            "provider": "local:fallback",
            "model": "priority",
            "limit": limit,
            "selected": selected,
            "suppressed": suppressed,
            "reason": "本地优先级兜底",
        },
    }


def should_skip_triage_for_cleaner_error(cleaner_meta: dict[str, Any] | None) -> bool:
    if os.getenv("ISSUE_TRIAGE_SKIP_ON_CLEANER_ERROR", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    if not cleaner_meta:
        return False
    section_meta = cleaner_meta.get("cleaner_section_meta") or {}
    fallback_sections = section_meta.get("fallback_sections") or cleaner_meta.get("fallback_sections") or []
    return bool(cleaner_meta.get("cleaner_error") or cleaner_meta.get("error") or fallback_sections)


def issue_priority(field: dict[str, Any]) -> tuple[int, str]:
    validator = str(field.get("validator") or "")
    recognizer = str(field.get("recognizer") or "")
    label = str(field.get("label") or "")
    if validator in {"regex", "digit_length", "same_day"}:
        return (0, label)
    if "签" in label or recognizer == "signature_or_text":
        return (1, label)
    if validator in {"exact_text", "exact_text_or_ocr_match"}:
        return (2, label)
    return (3, label)


def build_triage_prompt(failed: list[dict[str, Any]], limit: int) -> str:
    payload = [
        {
            "id": field.get("id"),
            "label": field.get("label"),
            "message": field.get("message"),
            "value": field.get("normalized_value") or field.get("value"),
            "validator": field.get("validator"),
            "confidence": field.get("confidence"),
            "needs_review": field.get("needs_review"),
            "source": field.get("source_label"),
            "review_reason": field.get("review_reason"),
        }
        for field in failed
    ]
    return (
        "你是飞行记录单审核助手。目标是减轻人工审核压力。"
        "请从全部规则失败中挑出最值得展示给审核员的少量问题，不要改变规则结果。"
        f"最多输出 {limit} 条问题。优先展示数字、日期、执照号、授权号、签名等高风险问题；"
        "低风险或重复问题可以合并或暂不展示。输出严格 JSON："
        '{"problems":["短问题1"],"reason":"选择理由"}。'
        f"失败字段: {json.dumps(payload, ensure_ascii=False)}"
    )


def request_timeout_seconds() -> int:
    try:
        value = int(os.getenv("ISSUE_TRIAGE_TIMEOUT_SECONDS", os.getenv("CLEANER_REQUEST_TIMEOUT_SECONDS", "45")))
    except ValueError:
        return 45
    return max(5, min(value, 180))


def parse_json_content(content: str) -> dict[str, Any]:
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
