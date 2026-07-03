from __future__ import annotations

import json
import os
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
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
    if not cfg.api_key:
        return fallback, {
            "provider": cfg.name,
            "model": selected_model,
            "cache_hit": False,
            "fallback": True,
            "duration_ms": int((time.time() - started) * 1000),
            "error": "missing_api_key",
        }

    section_fields = cleaner_sections(fields, assignments)
    if not section_fields:
        return fallback, {
            "provider": cfg.name,
            "model": selected_model,
            "cache_hit": False,
            "fallback": True,
            "duration_ms": int((time.time() - started) * 1000),
            "skipped": True,
        }

    results = fallback.copy()
    meta: dict[str, Any] = {
        "provider": cfg.name,
        "model": selected_model,
        "cache_hit": False,
        "section_results": {},
        "section_timings": {},
        "section_errors": {},
        "section_cache_hits": {},
        "section_fallback_cached": {},
        "fallback_sections": [],
    }
    max_workers = min(cleaner_section_concurrency(), len(section_fields))
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(clean_section_values, section, section_subset, assignments, cfg, selected_model, fallback): section
        for section, section_subset in section_fields.items()
    }
    try:
        for future in as_completed(futures, timeout=cleaner_total_budget_seconds()):
            section = futures[future]
            try:
                section_results, section_meta = future.result()
            except Exception as exc:  # noqa: BLE001
                section_results, section_meta = section_fallback(
                    section, section_fields[section], fallback, cfg.name, selected_model, f"{type(exc).__name__}: {exc}"
                )
            results.update(section_results)
            merge_section_meta(meta, section, section_meta)
    except TimeoutError:
        meta["error"] = "cleaner_total_budget_exceeded"
    finally:
        for future, section in futures.items():
            if future.done():
                continue
            future.cancel()
            section_results, section_meta = section_fallback(
                section,
                section_fields[section],
                fallback,
                cfg.name,
                selected_model,
                "cleaner_total_budget_exceeded",
            )
            results.update(section_results)
            merge_section_meta(meta, section, section_meta)
        executor.shutdown(wait=False, cancel_futures=True)

    meta["duration_ms"] = int((time.time() - started) * 1000)
    timings = list(meta["section_timings"].values())
    meta["cleaner_total_ms"] = meta["duration_ms"]
    meta["cleaner_section_max_ms"] = max(timings) if timings else 0
    meta["cleaner_section_sum_ms"] = sum(timings)
    meta["cleaner_fallback_count"] = len(meta["fallback_sections"])
    meta["fallback_cached"] = any(meta["section_fallback_cached"].values())
    if meta["fallback_sections"] and not meta.get("error"):
        meta["error"] = "; ".join(
            str(meta["section_errors"].get(section) or section)
            for section in meta["fallback_sections"]
        )
    meta["cache_hit"] = bool(meta["section_cache_hits"]) and all(meta["section_cache_hits"].values())
    return results, meta


def clean_section_values(
    section: str,
    fields: list[FieldSpec],
    assignments: dict[str, list[FieldCandidate]],
    cfg,
    selected_model: str,
    fallback: dict[str, RecognitionResult],
) -> tuple[dict[str, RecognitionResult], dict[str, Any]]:
    started = time.time()
    cache_key = cleaner_cache_key(fields, assignments, cfg.name, selected_model, section)
    cache_dir = OUTPUTS_DIR / "runtime" / "cleaner_cache" / cache_key
    cached = load_cleaner_cache(cache_dir, fields)
    if cached:
        return cached, {
            "provider": cfg.name,
            "model": selected_model,
            "section": section,
            "cache_hit": True,
            "duration_ms": int((time.time() - started) * 1000),
        }

    prompt = build_cleaner_prompt(fields, assignments, section=section)
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
            timeout=cleaner_request_timeout(),
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_content(content)
        cleaned = parsed.get("fields", parsed)
        if not isinstance(cleaned, dict):
            return section_fallback(section, fields, fallback, cfg.name, selected_model, "invalid_cleaner_response")
        results = {field.id: fallback[field.id] for field in fields}
        for field in fields:
            item = cleaned.get(field.id)
            if not isinstance(item, dict):
                continue
            raw = {
                "cleaner": {"content": content, "usage": data.get("usage")},
                "cleaner_section": section,
                "candidates": candidates_to_json({field.id: assignments.get(field.id, [])}).get(field.id, []),
                "needs_review": item.get("needs_review"),
                "reason": item.get("reason"),
                "evidence_block_ids": item.get("evidence_block_ids", []),
            }
            value = str(item.get("value", ""))
            normalized_value = str(item.get("normalized_value") or item.get("value", ""))
            if field.validator == "english_text":
                value = sanitize_english_text(value)
                normalized_value = sanitize_english_text(normalized_value)
            results[field.id] = RecognitionResult(
                value=value,
                normalized_value=normalized_value,
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
            "section": section,
            "cache_hit": False,
            "duration_ms": int((time.time() - started) * 1000),
            "usage": data.get("usage"),
        }
    except Exception as exc:
        results, meta = section_fallback(section, fields, fallback, cfg.name, selected_model, str(exc), started)
        save_cleaner_cache(cache_dir, results, meta={"timeout_cached_at": int(time.time()), "section": section})
        meta["fallback_cached"] = True
        return results, meta


def cleaner_request_timeout_seconds() -> int:
    try:
        value = int(os.getenv("CLEANER_REQUEST_TIMEOUT_SECONDS", "75"))
    except ValueError:
        return 75
    return max(5, min(value, 180))


def cleaner_connect_timeout_seconds() -> int:
    return int_env("CLEANER_CONNECT_TIMEOUT_SECONDS", 10, 1, 60)


def cleaner_section_timeout_seconds() -> int:
    return int_env("CLEANER_SECTION_TIMEOUT_SECONDS", cleaner_request_timeout_seconds(), 5, 300)


def cleaner_total_budget_seconds() -> int:
    return int_env("CLEANER_TOTAL_BUDGET_SECONDS", 90, 5, 600)


def cleaner_section_concurrency() -> int:
    return int_env("CLEANER_SECTION_CONCURRENCY", 3, 1, 8)


def cleaner_timeout_cache_ttl_seconds() -> int:
    return int_env("CLEANER_TIMEOUT_CACHE_TTL_SECONDS", 1800, 0, 86400)


def cleaner_request_timeout() -> tuple[int, int]:
    return cleaner_connect_timeout_seconds(), cleaner_section_timeout_seconds()


def int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


def cleaner_cache_key(
    fields: list[FieldSpec],
    assignments: dict[str, list[FieldCandidate]],
    provider: str,
    model: str,
    section_id: str = "all",
) -> str:
    field_hash = ""
    if FIELDS_PATH.exists():
        field_hash = hashlib.sha256(FIELDS_PATH.read_bytes()).hexdigest()
    payload = {
        "provider": provider,
        "model": model,
        "section_id": section_id,
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
        meta = payload.get("__meta__") if isinstance(payload, dict) else None
        if isinstance(meta, dict) and meta.get("timeout_cached_at"):
            ttl = cleaner_timeout_cache_ttl_seconds()
            if ttl <= 0 or int(time.time()) - int(meta.get("timeout_cached_at") or 0) > ttl:
                return None
        for field_id, item in payload.items():
            if field_id == "__meta__" or field_id not in field_ids or not isinstance(item, dict):
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


def save_cleaner_cache(cache_dir: Path, results: dict[str, RecognitionResult], meta: dict[str, Any] | None = None) -> None:
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
        if meta:
            payload["__meta__"] = meta
        (cache_dir / "cleaned.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def build_cleaner_prompt(fields: list[FieldSpec], assignments: dict[str, list[FieldCandidate]], section: str | None = None) -> str:
    field_payload = []
    for field in fields:
        if field.recognizer == "checkbox":
            continue
        candidates = assignments.get(field.id, [])
        if not candidates:
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
                    for candidate in candidates[:6]
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
        "规则型字段可直接采用最高分候选并做轻量归一化，不要过度改写。"
        "日期尽量归一化为 YYYY-MM-DD；参考手册保留 AMM/TSM 开头编号；执照号去空格并大写；"
        "英文故障描述只保留主要英文手写内容，过滤字段标签。"
        f"当前分区: {section or 'all'}。"
        f"OCR字段候选: {json.dumps(field_payload, ensure_ascii=False)}"
    )


def cleaner_sections(fields: list[FieldSpec], assignments: dict[str, list[FieldCandidate]]) -> dict[str, list[FieldSpec]]:
    sections: dict[str, list[FieldSpec]] = {}
    for field in fields:
        if field.recognizer == "checkbox":
            continue
        if not assignments.get(field.id):
            continue
        sections.setdefault(field.section, []).append(field)
    return sections


def section_fallback(
    section: str,
    fields: list[FieldSpec],
    fallback: dict[str, RecognitionResult],
    provider: str,
    model: str,
    error: str,
    started: float | None = None,
) -> tuple[dict[str, RecognitionResult], dict[str, Any]]:
    results: dict[str, RecognitionResult] = {}
    for field in fields:
        result = clone_result(fallback[field.id])
        result.raw = {
            **(result.raw or {}),
            "cleaner_error": error,
            "cleaner_section": section,
            "cleaner_fallback_reason": error,
        }
        result.needs_review = result.needs_review or result.confidence < 0.8
        if not result.review_reason:
            result.review_reason = "Cleaner超时，使用本地候选兜底"
        results[field.id] = result
    return results, {
        "provider": provider,
        "model": model,
        "section": section,
        "cache_hit": False,
        "fallback": True,
        "duration_ms": int((time.time() - started) * 1000) if started else 0,
        "error": error,
    }


def merge_section_meta(meta: dict[str, Any], section: str, section_meta: dict[str, Any]) -> None:
    meta["section_results"][section] = "fallback" if section_meta.get("fallback") else "ok"
    meta["section_timings"][section] = int(section_meta.get("duration_ms") or 0)
    meta["section_cache_hits"][section] = bool(section_meta.get("cache_hit"))
    meta["section_fallback_cached"][section] = bool(section_meta.get("fallback_cached"))
    if section_meta.get("error"):
        meta["section_errors"][section] = section_meta["error"]
    if section_meta.get("fallback"):
        meta["fallback_sections"].append(section)


def clone_result(result: RecognitionResult) -> RecognitionResult:
    return RecognitionResult(
        value=result.value,
        normalized_value=result.normalized_value,
        confidence=result.confidence,
        provider=result.provider,
        model=result.model,
        raw=dict(result.raw or {}),
        needs_review=result.needs_review,
        review_reason=result.review_reason,
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
        texts = [
            cleaned
            for candidate in candidates[:6]
            if (cleaned := sanitize_english_text(candidate.block.text))
        ]
        return " ".join(texts)
    return candidates[0].block.text


def normalize_for_field(field: FieldSpec, value: str) -> str:
    text = (value or "").strip()
    if field.validator == "english_text":
        return sanitize_english_text(text)
    if field.validator in {"regex", "prefix_or_exact"}:
        return text.replace(" ", "").upper()
    if field.validator == "same_day":
        from .validators import normalize_date

        return normalize_date(text)
    if field.validator == "digit_length":
        import re

        return re.sub(r"\D", "", text)
    return text


def sanitize_english_text(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    parts = re.split(r"\s+", text)
    kept: list[str] = []
    for part in parts:
        cleaned = strip_cjk_noise(part)
        if not cleaned:
            continue
        upper = cleaned.upper()
        if upper in {"FAULT", "FAULTMSG", "FAULTMSGE", "MESSAGE", "MSG"}:
            continue
        if not re.search(r"[A-Za-z]", cleaned):
            continue
        kept.append(cleaned)
    return " ".join(kept)


def strip_cjk_noise(text: str) -> str:
    text = re.sub(r"[\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"[\u3000-\u303f\uff00-\uffef]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
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
