from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from .schemas import FieldSpec, RecognitionResult


def normalize_text(value: str) -> str:
    return (value or "").strip().replace("：", ":").replace("／", "/")


def today_str(tz: str = "Asia/Shanghai") -> str:
    return datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d")


def normalize_date(value: str) -> str:
    text = normalize_text(value)
    text = text.replace(".", "-").replace("/", "-")
    candidates = [
        ("%Y-%m-%d", text),
        ("%Y-%m-%d", "20" + text if re.match(r"^\d{2}-\d{1,2}-\d{1,2}$", text) else text),
        ("%Y%m%d", text),
    ]
    for fmt, candidate in candidates:
        try:
            return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    match = re.search(r"(20\d{2})\D+(\d{1,2})\D+(\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return text


def validate(field: FieldSpec, recognition: RecognitionResult, now: str | None = None) -> tuple[bool, str]:
    value = normalize_text(recognition.normalized_value or recognition.value)
    params = field.params
    validator = field.validator
    expected_today = now or today_str()

    if validator == "int_range":
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return False, field.fail_msg
        number = float(match.group(0))
        return float(params["min"]) <= number <= float(params["max"]), field.fail_msg

    if validator == "exact_text":
        allowed = [normalize_text(str(item)).lower() for item in params.get("allow", [])]
        compact = value.replace(" ", "").lower()
        allowed_compact = [item.replace(" ", "") for item in allowed]
        return compact in allowed_compact, field.fail_msg

    if validator == "checked":
        return value.lower() in {"true", "checked", "yes", "1", "勾选", "已勾选"}, field.fail_msg

    if validator == "english_text":
        has_letter = bool(re.search(r"[A-Za-z]", value))
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", value))
        return has_letter and not has_cjk, field.fail_msg

    if validator == "bilingual_text":
        has_letter = bool(re.search(r"[A-Za-z]", value))
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", value))
        min_letters = int(params.get("min_letters", 1))
        min_cjk = int(params.get("min_cjk", 1))
        return (
            len(re.findall(r"[A-Za-z]", value)) >= min_letters
            and len(re.findall(r"[\u4e00-\u9fff]", value)) >= min_cjk
            and has_letter
            and has_cjk
        ), field.fail_msg

    if validator == "name_not_place":
        compact = value.replace(" ", "").lower()
        blocked = [normalize_text(str(item)).replace(" ", "").lower() for item in params.get("not_allow", ["重庆", "chongqing"])]
        return bool(compact) and compact not in blocked, field.fail_msg

    if validator == "prefix_or_exact":
        upper = value.upper().replace(" ", "")
        if upper in {str(item).upper() for item in params.get("allow_exact", [])}:
            return True, field.fail_msg
        return any(upper.startswith(str(prefix).upper()) for prefix in params.get("prefixes", [])), field.fail_msg

    if validator == "regex":
        compact = value.replace(" ", "").upper()
        return bool(re.fullmatch(params["pattern"], compact)), field.fail_msg

    if validator == "same_day":
        return normalize_date(value) == expected_today, field.fail_msg

    if validator == "digit_length":
        digits = re.sub(r"\D", "", value)
        return len(digits) in {int(n) for n in params.get("allow_lengths", [])}, field.fail_msg

    if validator == "number_less_than":
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return False, field.fail_msg
        number = float(match.group(0))
        return number < float(params["max"]), field.fail_msg

    if validator in {"present", "present_and_match", "exact_text_or_ocr_match"}:
        return bool(value), field.fail_msg

    return False, field.fail_msg
