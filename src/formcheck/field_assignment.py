from __future__ import annotations

import math
import re
from typing import Any

from .schemas import FieldCandidate, FieldSpec, OcrBlock
from .validators import compact_text, normalize_exact_value, normalized_compact


FORM_LABEL_TOKENS = [
    "REPORTED BY",
    "REPORT/WORK",
    "AUTHORIZATION NO",
    "SERVICE COMPLETED",
    "AIRWORTHINESS RELEASE",
    "DEFERRED ITEM",
    "DEFER NO",
    "RELEASE SIGN",
    "LICENSE NO",
    "PIREP",
    "MAREP",
    "ACTIONS",
    "DATE",
    "STATION",
    "REF.",
    "P/N",
    "S/N",
    "FIN",
    "SIGN",
    "报告者",
    "授权号",
    "完成维护类别",
    "保留项目",
    "放行签署",
    "执照号",
    "是否有保",
    "是否有保留",
    "日期",
    "地点",
    "参考手册",
    "处理措施",
    "保留单号",
    "第一联",
    "原始记录",
    "机组",
    "机务",
    "安装件号",
    "拆下件号",
    "拆下序号",
    "安装序号",
    "功能号",
    "工作者签名",
]


def assign_blocks_to_fields(
    fields: list[FieldSpec],
    blocks: list[OcrBlock],
    canonical_size: tuple[int, int],
    max_candidates: int = 8,
) -> dict[str, list[FieldCandidate]]:
    page_size = estimate_ocr_page_size(blocks, canonical_size)
    scale_x = page_size[0] / canonical_size[0]
    scale_y = page_size[1] / canonical_size[1]
    assignments: dict[str, list[FieldCandidate]] = {}
    for field in fields:
        scored: list[FieldCandidate] = []
        target_bbox = tuple(field.assignment.get("search_bbox") or field.bbox)
        scaled_bbox = scale_bbox(target_bbox, scale_x, scale_y)
        expanded = expand_bbox(scaled_bbox, float(field.assignment.get("expand", 0.18)))
        for block in blocks:
            score, reason = score_block_for_field(field, block, scaled_bbox, expanded)
            if score > 0:
                scored.append(FieldCandidate(field.id, block, score, reason))
        scored.sort(key=lambda item: item.score, reverse=True)
        assignments[field.id] = scored[:max_candidates]
    resolve_candidate_conflicts(assignments, {field.id: field for field in fields}, max_candidates)
    return assignments


def resolve_candidate_conflicts(
    assignments: dict[str, list[FieldCandidate]],
    fields_by_id: dict[str, FieldSpec],
    max_candidates: int,
) -> None:
    """Keep one OCR block from becoming the primary evidence for unrelated fields."""
    for _ in range(3):
        primary_by_block: dict[str, list[FieldCandidate]] = {}
        for field_id, candidates in assignments.items():
            field = fields_by_id[field_id]
            if not candidates or allow_shared_primary(field):
                continue
            primary_by_block.setdefault(candidates[0].block.id, []).append(candidates[0])

        changed = False
        for shared in primary_by_block.values():
            if len(shared) <= 1:
                continue
            shared.sort(key=lambda candidate: candidate.score, reverse=True)
            keep = shared[0]
            for candidate in shared[1:]:
                field_candidates = assignments[candidate.field_id]
                assignments[candidate.field_id] = [
                    item for item in field_candidates if item.block.id != keep.block.id
                ][:max_candidates]
                changed = True
        if not changed:
            break


def allow_shared_primary(field: FieldSpec) -> bool:
    return field.validator in {"bilingual_text", "english_text"}


def estimate_ocr_page_size(blocks: list[OcrBlock], canonical_size: tuple[int, int]) -> tuple[float, float]:
    if not blocks:
        return float(canonical_size[0]), float(canonical_size[1])
    max_x = max(block.box[2] for block in blocks)
    max_y = max(block.box[3] for block in blocks)
    # PaddleOCR boxes usually stop slightly inside the page. Inflate by a small
    # margin so template bboxes scale to the same coordinate space.
    return max(max_x * 1.04, 1.0), max(max_y * 1.04, 1.0)


def scale_bbox(bbox: tuple[int, int, int, int], scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y


def expand_bbox(bbox: tuple[float, float, float, float], ratio: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return x1 - w * ratio, y1 - h * ratio, x2 + w * ratio, y2 + h * ratio


def score_block_for_field(
    field: FieldSpec,
    block: OcrBlock,
    bbox: tuple[float, float, float, float],
    expanded: tuple[float, float, float, float],
) -> tuple[float, str]:
    overlap = overlap_ratio(block.box, bbox)
    center_inside = point_inside(block.center, expanded)
    distance_score = center_distance_score(block.center, bbox)
    pattern = pattern_score(field, block.text)
    if incompatible_value(field, block.text):
        return 0.0, ""
    if label_noise(field, block.text):
        return 0.0, ""
    if overlap <= 0 and not center_inside:
        return 0.0, ""

    score = block.score * 0.45 + overlap * 1.5 + distance_score * 0.45 + pattern * 0.7
    value_type = field.assignment.get("value_type")
    if value_type and pattern > 0:
        score += 0.25
    if field.assignment.get("row") == "last":
        y1, _y2 = bbox[1], bbox[3]
        height = max(bbox[3] - bbox[1], 1.0)
        score += max(0.0, min((block.center[1] - y1) / height, 1.0)) * 0.55
    if field.validator == "int_range" and pattern < 0.2:
        score *= 0.45
    reasons = []
    if overlap > 0:
        reasons.append(f"overlap={overlap:.2f}")
    if center_inside:
        reasons.append("center_near")
    if pattern > 0:
        reasons.append(f"pattern={pattern:.2f}")
    return score, ",".join(reasons)


def incompatible_value(field: FieldSpec, text: str) -> bool:
    compact = compact_text(text)
    upper = compact.upper()
    value_type = field.assignment.get("value_type")
    if value_type in {"signature"}:
        return (
            label_noise(field, text)
            or is_station_value(text)
            or is_date_like(text)
            or is_numeric_value(text)
            or is_release_statement(text)
        )
    if value_type in {"authorization"}:
        return not bool(extract_authorization_value(text))
    if value_type in {"license"}:
        return not bool(re.search(r"CA|CAA|CAAC|CAACML|\d{4,8}", upper))
    if value_type in {"text"}:
        return is_station_value(text) or is_date_like(text)
    if field.validator in {"int_range", "digit_length", "number_less_than"}:
        return not bool(re.fullmatch(r"\d+(?:\.\d+)?", compact))
    if field.validator == "same_day":
        return not bool(re.search(r"20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}", text))
    if field.validator == "regex":
        pattern = field.params.get("pattern")
        if pattern and re.fullmatch(pattern, upper):
            return False
        return not bool(re.search(r"CAAC|AUTH|授权|\d{6}", upper))
    if field.validator == "prefix_or_exact":
        exact = normalized_compact(text).upper()
        allow_exact = {normalized_compact(str(item)).upper() for item in field.params.get("allow_exact", [])}
        if exact in allow_exact:
            return False
        return False
    if field.validator == "exact_text":
        allowed = [normalized_compact(str(item)).upper() for item in field.params.get("allow", [])]
        if not allowed:
            return False
        normalized = normalized_compact(text).upper()
        return normalized not in allowed
    return False


def is_station_value(text: str) -> bool:
    normalized = normalize_exact_value(text)
    return normalized == "重庆"


def is_date_like(text: str) -> bool:
    compact = compact_text(text)
    return bool(
        re.fullmatch(r"20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}", compact)
        or re.fullmatch(r"20\d{6}", compact)
    )


def is_numeric_value(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", compact_text(text)))


def is_release_statement(text: str) -> bool:
    upper = text.upper()
    return "CONSIDERED FIT FOR RELEASE" in upper or "适合飞行" in text or "满足适航要求" in text


def extract_authorization_value(text: str) -> str:
    candidates = re.findall(r"(?<!\d)(\d{6})(?!\d)", text)
    return candidates[-1] if candidates else ""


def label_noise(field: FieldSpec, text: str) -> bool:
    if field.recognizer == "checkbox":
        return False
    return looks_like_form_label_text(text)


def looks_like_form_label_text(text: str) -> bool:
    upper = text.upper().strip()
    if not upper:
        return False
    bounded_tokens = {"FIN", "SIGN", "DATE", "STATION"}
    for token in FORM_LABEL_TOKENS:
        if token in bounded_tokens:
            if re.search(rf"(^|[^A-Z]){re.escape(token)}([^A-Z]|$)", upper):
                return True
            continue
        if token in upper:
            return True
    return False


def overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    block_area = max((ax2 - ax1) * (ay2 - ay1), 1.0)
    return inter / block_area


def point_inside(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def center_distance_score(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    diag = math.hypot(x2 - x1, y2 - y1) or 1.0
    distance = math.hypot(x - cx, y - cy)
    return max(0.0, 1.0 - distance / (diag * 1.35))


def pattern_score(field: FieldSpec, text: str) -> float:
    compact = compact_text(text)
    upper = compact.upper()
    if field.recognizer == "checkbox":
        return 0.0
    if field.validator == "regex":
        pattern = field.params.get("pattern")
        if pattern and re.fullmatch(pattern, upper):
            return 1.0
        if "CAAC" in upper or re.search(r"\d{6}", upper):
            return 0.65
    if field.validator == "prefix_or_exact":
        exact = normalized_compact(text).upper()
        if exact in {normalized_compact(str(item)).upper() for item in field.params.get("allow_exact", [])}:
            return 1.0
        if any(upper.startswith(str(prefix).upper()) for prefix in field.params.get("prefixes", [])):
            return 1.0
        if "AMM" in upper or "TSM" in upper:
            return 0.85
        if re.search(r"[A-Z]{2,}\d", upper):
            return 0.35
    if field.validator == "same_day":
        return 1.0 if re.search(r"20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}", text) else 0.0
    if field.validator == "int_range":
        try:
            number = float(compact)
            return 1.0 if float(field.params["min"]) <= number <= float(field.params["max"]) else 0.08
        except (TypeError, ValueError, KeyError):
            return 0.0
    if field.validator in {"digit_length", "number_less_than"}:
        return 0.9 if re.fullmatch(r"\d+(?:\.\d+)?", compact) else 0.0
    if field.validator in {"english_text", "bilingual_text"}:
        if field.validator == "bilingual_text" and label_noise(field, text):
            return 0.0
        has_letter = bool(re.search(r"[A-Za-z]", text))
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
        if field.validator == "bilingual_text":
            return 0.9 if has_letter and has_cjk else (0.45 if has_letter or has_cjk else 0.0)
        return 0.8 if has_letter and not has_cjk else (0.35 if has_letter else 0.0)
    if field.validator == "exact_text":
        normalized = normalized_compact(text).upper()
        allowed = [normalized_compact(str(item)).upper() for item in field.params.get("allow", [])]
        if normalized in allowed:
            return 1.0
    if field.validator == "name_not_place":
        normalized = normalize_exact_value(text)
        return 0.6 if compact and normalized != "重庆" else 0.0
    value_type = field.assignment.get("value_type")
    if value_type == "signature":
        return 0.55 if not incompatible_value(field, text) else 0.0
    if value_type == "authorization":
        return 0.8 if extract_authorization_value(text) else 0.0
    if value_type == "license":
        if upper.startswith("CAACML") or upper.startswith("CAAC"):
            return 0.8
        if upper.startswith("CA") or re.fullmatch(r"\d{4,8}", upper):
            return 0.45
    return 0.0


def candidates_to_json(assignments: dict[str, list[FieldCandidate]]) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for field_id, candidates in assignments.items():
        payload[field_id] = [
            {
                "block_id": candidate.block.id,
                "text": candidate.block.text,
                "block_score": candidate.block.score,
                "assignment_score": candidate.score,
                "box": list(candidate.block.box),
                "center": list(candidate.block.center),
                "reason": candidate.reason,
            }
            for candidate in candidates
        ]
    return payload
