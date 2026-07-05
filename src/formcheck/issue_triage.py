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
    review_pending = [field for field in fields if field.get("passed") and field.get("needs_review")]
    all_problems = [field.get("message") for field in failed if field.get("message")]
    limit = issue_display_limit()
    fallback = fallback_triage(failed, all_problems, limit, review_pending)
    if not failed and not review_pending:
        return {
            "problems": ["通过"],
            "problem_items": [],
            "all_problems": [],
            "review_problems": [],
            "issue_triage": {
                "provider": "local",
                "model": "none",
                "limit": limit,
                "selected": [],
                "suppressed": [],
                "selected_review": [],
                "suppressed_review": [],
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

    prompt = build_triage_prompt(failed + review_pending, limit)
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
        problem_items = problem_items_for_selected(selected, failed, review_pending)
        suppressed = [item for item in all_problems if item not in selected]
        review_problems = [review_message(field) for field in review_pending]
        suppressed_review = [item for item in review_problems if item not in selected]
        return {
            "problems": selected,
            "problem_items": problem_items,
            "all_problems": all_problems,
            "review_problems": review_problems,
            "issue_triage": {
                "provider": cfg.name,
                "model": selected_model,
                "limit": limit,
                "selected": selected,
                "suppressed": suppressed,
                "selected_review": [item for item in selected if item in review_problems],
                "suppressed_review": suppressed_review,
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


def fallback_triage(
    failed: list[dict[str, Any]],
    all_problems: list[str],
    limit: int,
    review_pending: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    review_pending = review_pending or []
    ranked_failed = sorted(failed, key=lambda field: issue_priority(field))
    ranked_review = sorted(review_pending, key=lambda field: issue_priority(field))
    selected = [field.get("message") for field in ranked_failed if field.get("message")]
    selected = [str(item) for item in selected[:limit]]
    review_problems = [review_message(field) for field in ranked_review]
    if len(selected) < limit:
        selected.extend(review_problems[: limit - len(selected)])
    problem_items = problem_items_for_selected(selected, failed, review_pending)
    suppressed = [item for item in all_problems if item not in selected]
    suppressed_review = [item for item in review_problems if item not in selected]
    if (suppressed or suppressed_review) and len(selected) < limit:
        selected.append("若干字段需复核")
        problem_items.append(summary_problem_item("若干字段需复核"))
    return {
        "problems": selected or ["通过"],
        "problem_items": problem_items,
        "all_problems": all_problems,
        "review_problems": review_problems,
        "issue_triage": {
            "provider": "local:fallback",
            "model": "priority",
            "limit": limit,
            "selected": selected,
            "suppressed": suppressed,
            "selected_review": [item for item in selected if item in review_problems],
            "suppressed_review": suppressed_review,
            "reason": "本地优先级兜底",
        },
    }


def review_message(field: dict[str, Any]) -> str:
    label = str(field.get("label") or field.get("id") or "字段").strip()
    reason = str(field.get("review_reason") or "").strip()
    if reason and reason not in {"需人工复核", "needs_review"}:
        return f"{label}需复核：{reason}"
    return f"{label}需复核"


def problem_items_for_selected(
    selected: list[str],
    failed: list[dict[str, Any]],
    review_pending: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, dict[str, Any], str]] = []
    for field in failed:
        message = str(field.get("message") or "")
        if message:
            candidates.append((message, field, "failure"))
    for field in review_pending:
        candidates.append((review_message(field), field, "review"))
    return [problem_item_for_text(text, candidates) for text in selected]


def problem_item_for_text(text: str, candidates: list[tuple[str, dict[str, Any], str]]) -> dict[str, Any]:
    normalized = normalize_problem_text(text)
    for candidate_text, field, kind in candidates:
        candidate_norm = normalize_problem_text(candidate_text)
        if normalized == candidate_norm or (candidate_norm and (normalized in candidate_norm or candidate_norm in normalized)):
            return field_problem_item(text, field, kind)
    for _candidate_text, field, kind in candidates:
        label = normalize_problem_text(str(field.get("label") or ""))
        label_without_number = normalize_problem_text(strip_leading_index(str(field.get("label") or "")))
        if (label and label in normalized) or (label_without_number and label_without_number in normalized):
            return field_problem_item(text, field, kind)
    return summary_problem_item(text)


def field_problem_item(text: str, field: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "text": text,
        "field_id": field.get("id") or "",
        "label": field.get("label") or "",
        "kind": kind,
        "risk_level": issue_risk_level(field),
        "evidence_state": evidence_state(field),
    }


def summary_problem_item(text: str) -> dict[str, Any]:
    return {
        "text": text,
        "field_id": "",
        "label": "",
        "kind": "summary",
        "risk_level": "low",
        "evidence_state": "summary",
    }


def normalize_problem_text(value: str) -> str:
    return "".join(str(value or "").lower().split())


def strip_leading_index(value: str) -> str:
    return str(value or "").lstrip("0123456789. 、-")


def should_skip_triage_for_cleaner_error(cleaner_meta: dict[str, Any] | None) -> bool:
    if os.getenv("ISSUE_TRIAGE_SKIP_ON_CLEANER_ERROR", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    if not cleaner_meta:
        return False
    section_meta = cleaner_meta.get("cleaner_section_meta") or {}
    fallback_sections = section_meta.get("fallback_sections") or cleaner_meta.get("fallback_sections") or []
    return bool(cleaner_meta.get("cleaner_error") or cleaner_meta.get("error") or fallback_sections)


def issue_priority(field: dict[str, Any]) -> tuple[int, str]:
    return (-issue_risk_score(field), str(field.get("label") or ""))


def issue_risk_score(field: dict[str, Any]) -> int:
    validator = str(field.get("validator") or "")
    recognizer = str(field.get("recognizer") or "")
    label = str(field.get("label") or "")
    score = 0
    if validator in {"regex", "digit_length", "same_day", "number_less_than"}:
        score += 80
    elif validator == "int_range":
        score += 70
    elif validator in {"exact_text", "exact_text_or_ocr_match", "name_not_place"}:
        score += 45
    if "签" in label or recognizer == "signature_or_text":
        score += 60
    if any(term in label for term in ("授权号", "执照号", "APU", "滑油", "日期")):
        score += 35
    if field.get("needs_review"):
        score += 12
    raw = field.get("raw") if isinstance(field.get("raw"), dict) else {}
    if raw.get("roi_review"):
        score += 20
        review = raw.get("roi_review")
        if isinstance(review, dict) and review.get("changed_value"):
            score += 12
    if raw.get("roi_review_skipped"):
        score += 18
    return score


def issue_risk_level(field: dict[str, Any]) -> str:
    score = issue_risk_score(field)
    if score >= 130:
        return "high"
    if score >= 80:
        return "medium"
    return "low"


def evidence_state(field: dict[str, Any]) -> str:
    raw = field.get("raw") if isinstance(field.get("raw"), dict) else {}
    if raw.get("roi_review"):
        review = raw.get("roi_review")
        if isinstance(review, dict) and review.get("changed_value"):
            return "roi_reviewed_changed"
        return "roi_reviewed"
    if raw.get("roi_review_skipped"):
        return "roi_review_skipped"
    if field.get("needs_review"):
        return "needs_review"
    if field.get("confidence") is not None:
        return "ocr_cleaned"
    return "unknown"


def review_state(field: dict[str, Any]) -> str:
    if field.get("needs_review"):
        return str(field.get("review_reason") or "needs_review")
    return "not_requested"


def build_triage_prompt(failed: list[dict[str, Any]], limit: int) -> str:
    payload = [
        {
            "id": field.get("id"),
            "label": field.get("label"),
            "kind": "failure" if not field.get("passed") else "review_pending",
            "message": field.get("message") if not field.get("passed") else review_message(field),
            "value": field.get("normalized_value") or field.get("value"),
            "validator": field.get("validator"),
            "confidence": field.get("confidence"),
            "source": field.get("source_label"),
            "risk_level": issue_risk_level(field),
            "risk_score": issue_risk_score(field),
            "evidence_state": evidence_state(field),
            "review_state": review_state(field),
        }
        for field in failed
    ]
    return (
        "你是飞行记录单审核助手。目标是减轻人工审核压力。"
        "请从全部规则失败和高风险复核项中挑出最值得展示给审核员的少量问题，不要改变规则结果。"
        f"最多输出 {limit} 条问题。优先展示 risk_level=high、risk_score 高、"
        "已 ROI 复核仍失败、ROI 因预算跳过、或规则虽通过但证据状态仍需复核的字段；"
        "数字、日期、执照号、授权号、签名优先。低风险或重复问题可以合并或暂不展示。"
        "review_pending 不是规则失败，措辞必须使用“需复核”而不是“不合规”。输出严格 JSON："
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
