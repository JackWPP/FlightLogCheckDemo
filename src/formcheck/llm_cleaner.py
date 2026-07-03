from __future__ import annotations

import json
import os
import hashlib
import time
from pathlib import Path
from typing import Any

import requests

from .config import FIELDS_PATH, OUTPUTS_DIR, provider_config
from .field_assignment import candidates_to_json
from .schemas import FieldCandidate, FieldSpec, RecognitionResult
from .validators import today_str


DEFAULT_CLEANER_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


def clean_field_values(
    fields: list[FieldSpec],
    assignments: dict[str, list[FieldCandidate]],
    provider: str = "siliconflow",
    model: str | None = None,
) -> dict[str, RecognitionResult]:
    results, _meta = clean_field_values_with_meta(fields, assignments, provider, model)
    return results


def clean_field_values_with_meta(
    fields: list[FieldSpec],
    assignments: dict[str, list[FieldCandidate]],
    provider: str = "siliconflow",
    model: str | None = None,
) -> tuple[dict[str, RecognitionResult], dict[str, Any]]:
    started = time.time()
    cfg = provider_config(provider)
    selected_model = model or os.getenv("CLEANER_MODEL") or DEFAULT_CLEANER_MODEL
    fallback = fallback_clean(fields, assignments, cfg.name, selected_model)
    cache_key = cleaner_cache_key(fields, assignments, cfg.name, selected_model)
    cache_dir = OUTPUTS_DIR / "runtime" / "cleaner_cache" / cache_key
    cached = load_cleaner_cache(cache_dir, fields)
    if cached:
        return cached, {
            "provider": cfg.name,
            "model": selected_model,
            "cache_hit": True,
            "duration_ms": int((time.time() - started) * 1000),
        }
    if not cfg.api_key:
        return fallback, {
            "provider": cfg.name,
            "model": selected_model,
            "cache_hit": False,
            "fallback": True,
            "duration_ms": int((time.time() - started) * 1000),
            "error": "missing_api_key",
        }

    prompt = build_cleaner_prompt(fields, assignments)
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
            timeout=cleaner_request_timeout_seconds(),
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_content(content)
        cleaned = parsed.get("fields", parsed)
        if not isinstance(cleaned, dict):
            return fallback
        results = fallback.copy()
        for field in fields:
            item = cleaned.get(field.id)
            if not isinstance(item, dict):
                continue
            raw = {
                "cleaner": {"content": content, "usage": data.get("usage")},
                "candidates": candidates_to_json({field.id: assignments.get(field.id, [])}).get(field.id, []),
                "needs_review": item.get("needs_review"),
                "reason": item.get("reason"),
                "evidence_block_ids": item.get("evidence_block_ids", []),
            }
            results[field.id] = RecognitionResult(
                value=str(item.get("value", "")),
                normalized_value=str(item.get("normalized_value") or item.get("value", "")),
                confidence=float(item.get("confidence") or 0.0),
                provider=cfg.name,
                model=selected_model,
                raw=raw,
                needs_review=bool(item.get("needs_review") or field.assignment.get("needs_review")),
                review_reason=str(item.get("reason") or ""),
            )
        save_cleaner_cache(cache_dir, results)
        return results, {
            "provider": cfg.name,
            "model": selected_model,
            "cache_hit": False,
            "duration_ms": int((time.time() - started) * 1000),
            "usage": data.get("usage"),
        }
    except Exception as exc:
        for result in fallback.values():
            result.raw = {**(result.raw or {}), "cleaner_error": str(exc)}
        return fallback, {
            "provider": cfg.name,
            "model": selected_model,
            "cache_hit": False,
            "fallback": True,
            "duration_ms": int((time.time() - started) * 1000),
            "error": str(exc),
        }


def cleaner_request_timeout_seconds() -> int:
    try:
        value = int(os.getenv("CLEANER_REQUEST_TIMEOUT_SECONDS", "45"))
    except ValueError:
        return 45
    return max(5, min(value, 180))


def cleaner_cache_key(
    fields: list[FieldSpec],
    assignments: dict[str, list[FieldCandidate]],
    provider: str,
    model: str,
) -> str:
    field_hash = ""
    if FIELDS_PATH.exists():
        field_hash = hashlib.sha256(FIELDS_PATH.read_bytes()).hexdigest()
    payload = {
        "provider": provider,
        "model": model,
        "fields_hash": field_hash,
        "candidates": candidates_to_json(assignments),
        "field_ids": [field.id for field in fields],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def load_cleaner_cache(cache_dir: Path, fields: list[FieldSpec]) -> dict[str, RecognitionResult] | None:
    path = cache_dir / "cleaned.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results: dict[str, RecognitionResult] = {}
        field_ids = {field.id for field in fields}
        for field_id, item in payload.items():
            if field_id not in field_ids or not isinstance(item, dict):
                continue
            raw = item.get("raw") or {}
            raw["cleaner_cache_hit"] = True
            results[field_id] = RecognitionResult(
                value=str(item.get("value", "")),
                normalized_value=str(item.get("normalized_value") or item.get("value", "")),
                confidence=float(item.get("confidence") or 0.0),
                provider=str(item.get("provider") or "cleaner_cache"),
                model=str(item.get("model") or ""),
                raw=raw,
                needs_review=bool(item.get("needs_review")),
                review_reason=str(item.get("review_reason") or ""),
            )
        return results if results else None
    except Exception:
        return None


def save_cleaner_cache(cache_dir: Path, results: dict[str, RecognitionResult]) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            field_id: {
                "value": result.value,
                "normalized_value": result.normalized_value,
                "confidence": result.confidence,
                "provider": result.provider,
                "model": result.model,
                "raw": result.raw,
                "needs_review": result.needs_review,
                "review_reason": result.review_reason,
            }
            for field_id, result in results.items()
        }
        (cache_dir / "cleaned.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def build_cleaner_prompt(fields: list[FieldSpec], assignments: dict[str, list[FieldCandidate]]) -> str:
    field_payload = []
    for field in fields:
        if field.recognizer == "checkbox":
            continue
        field_payload.append(
            {
                "id": field.id,
                "label": field.label,
                "section": field.section,
                "recognizer": field.recognizer,
                "validator": field.validator,
                "params": field.params,
                "candidates": [
                    {
                        "block_id": candidate.block.id,
                        "text": candidate.block.text,
                        "ocr_score": round(candidate.block.score, 4),
                        "assignment_score": round(candidate.score, 4),
                        "reason": candidate.reason,
                    }
                    for candidate in assignments.get(field.id, [])[:6]
                ],
            }
        )
    return (
        "你是飞行记录单 OCR 结果清洗器。只根据给定 OCR candidates 归一化字段值，不要判断是否合规。"
        "最终合规由本地规则引擎完成。"
        f"今天日期: {today_str()}。"
        "要求：输出严格 JSON object，顶层 key 为 fields。每个字段格式："
        '{"value":"原始/合并值","normalized_value":"归一化值","confidence":0.0,'
        '"evidence_block_ids":["b0001"],"needs_review":false,"reason":"简短原因"}。'
        "如果没有可靠候选，value 和 normalized_value 输出空字符串，confidence 低于 0.2，needs_review 为 true。"
        "日期尽量归一化为 YYYY-MM-DD；参考手册保留 AMM/TSM 开头编号；执照号去空格并大写；"
        "英文故障描述只保留主要英文手写内容，过滤字段标签。"
        f"OCR字段候选: {json.dumps(field_payload, ensure_ascii=False)}"
    )


def fallback_clean(
    fields: list[FieldSpec],
    assignments: dict[str, list[FieldCandidate]],
    provider: str,
    model: str,
) -> dict[str, RecognitionResult]:
    results: dict[str, RecognitionResult] = {}
    for field in fields:
        candidates = assignments.get(field.id, [])
        text = best_candidate_text(field, candidates)
        confidence = min(candidates[0].score / 3.0, 0.95) if candidates else 0.0
        needs_review = bool(field.assignment.get("needs_review") or not candidates or confidence < 0.45)
        results[field.id] = RecognitionResult(
            value=text,
            normalized_value=normalize_for_field(field, text),
            confidence=confidence,
            provider=f"{provider}:fallback_cleaner",
            model=model,
            raw={"candidates": candidates_to_json({field.id: candidates}).get(field.id, [])},
            needs_review=needs_review,
            review_reason="需人工复核" if needs_review else "",
        )
    return results


def best_candidate_text(field: FieldSpec, candidates: list[FieldCandidate]) -> str:
    if not candidates:
        return ""
    if field.validator == "english_text":
        texts = [candidate.block.text for candidate in candidates[:4] if any(ch.isalpha() for ch in candidate.block.text)]
        return " ".join(texts)
    return candidates[0].block.text


def normalize_for_field(field: FieldSpec, value: str) -> str:
    text = (value or "").strip()
    if field.validator in {"regex", "prefix_or_exact"}:
        return text.replace(" ", "").upper()
    if field.validator == "same_day":
        from .validators import normalize_date

        return normalize_date(text)
    if field.validator == "digit_length":
        import re

        return re.sub(r"\D", "", text)
    return text


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
