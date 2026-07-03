from __future__ import annotations

import math
import re
from typing import Any

from .schemas import FieldCandidate, FieldSpec, OcrBlock


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
    return assignments


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
    reasons = []
    if overlap > 0:
        reasons.append(f"overlap={overlap:.2f}")
    if center_inside:
        reasons.append("center_near")
    if pattern > 0:
        reasons.append(f"pattern={pattern:.2f}")
    return score, ",".join(reasons)


def incompatible_value(field: FieldSpec, text: str) -> bool:
    compact = text.strip().replace(" ", "")
    upper = compact.upper()
    value_type = field.assignment.get("value_type")
    if value_type in {"signature"}:
        return label_noise(field, text)
    if value_type in {"authorization"}:
        return not bool(re.search(r"\d{4,6}", upper))
    if value_type in {"license"}:
        return not bool(re.search(r"CA|CAA|CAAC|CAACML|\d{4,8}", upper))
    if field.validator in {"int_range", "digit_length"}:
        return not bool(re.fullmatch(r"\d+(?:\.\d+)?", compact))
    if field.validator == "same_day":
        return not bool(re.search(r"20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}", text))
    if field.validator == "regex":
        pattern = field.params.get("pattern")
        if pattern and re.fullmatch(pattern, upper):
            return False
        return not bool(re.search(r"CAAC|AUTH|授权|\d{6}", upper))
    if field.validator == "prefix_or_exact":
        if upper in {str(item).upper() for item in field.params.get("allow_exact", [])}:
            return False
        return not ("AMM" in upper or "TSM" in upper)
    if field.validator == "exact_text":
        allowed = [str(item).replace(" ", "").upper() for item in field.params.get("allow", [])]
        if not allowed:
            return False
        return not any(item and (item in upper or upper in item) for item in allowed)
    return False


def label_noise(field: FieldSpec, text: str) -> bool:
    upper = text.upper().strip()
    label_tokens = [
        "REPORTED BY",
        "REPORT/WORK",
        "AUTHORIZATION NO",
        "SERVICE COMPLETED",
        "DEFERRED ITEM",
        "RELEASE SIGN",
        "LICENSE NO",
        "DATE",
        "STATION",
        "REF.",
        "报告者",
        "授权号",
        "完成维护类别",
        "保留项目",
        "放行签署",
        "执照号",
        "日期",
        "地点",
    ]
    if field.recognizer == "checkbox":
        return False
    return any(token in upper for token in label_tokens)


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
    compact = text.strip().replace(" ", "")
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
        if upper in {str(item).upper() for item in field.params.get("allow_exact", [])}:
            return 1.0
        if any(upper.startswith(str(prefix).upper()) for prefix in field.params.get("prefixes", [])):
            return 1.0
        if "AMM" in upper or "TSM" in upper:
            return 0.85
    if field.validator == "same_day":
        return 1.0 if re.search(r"20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}", text) else 0.0
    if field.validator in {"int_range", "digit_length"}:
        return 0.9 if re.fullmatch(r"\d+(?:\.\d+)?", compact) else 0.0
    if field.validator == "english_text":
        has_letter = bool(re.search(r"[A-Za-z]", text))
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
        return 0.8 if has_letter and not has_cjk else (0.35 if has_letter else 0.0)
    if field.validator == "exact_text":
        allowed = [str(item).replace(" ", "").upper() for item in field.params.get("allow", [])]
        if upper in allowed:
            return 1.0
        if any(item in upper or upper in item for item in allowed if item):
            return 0.65
    value_type = field.assignment.get("value_type")
    if value_type == "signature":
        return 0.55 if not label_noise(field, text) else 0.0
    if value_type == "authorization":
        return 0.8 if re.fullmatch(r"\d{4,6}", compact) else 0.0
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
