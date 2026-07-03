from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from .config import CANONICAL_DIR, FIELDS_PATH, OUT_DIR
from .crop import crop_roi, save_rois
from .field_assignment import assign_blocks_to_fields, candidates_to_json
from .fields import load_fields
from .image_io import imread, imwrite
from .llm_cleaner import DEFAULT_CLEANER_MODEL, clean_field_values
from .model_adapters import recognize_with_provider
from .ppocr_pipeline import block_to_dict, run_ppocrv6
from .recognizers import mock_recognize
from .registration import load_template, register, save_registration_summary
from .schemas import FieldCheck
from .validators import validate


def analyze_image(
    image_path: Path,
    provider: str = "mock",
    model: str | None = None,
    run_id: str = "latest",
    mode: str = "hybrid",
    cleaner_provider: str = "siliconflow",
    cleaner_model: str | None = None,
) -> dict:
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    image = imread(image_path)
    template = load_template(CANONICAL_DIR)
    reg = register(image, template)
    save_registration_summary(run_dir / "registration.json", reg)
    if not reg.ok or reg.warped is None:
        return {"ok": False, "error": reg.reject_reason, "registration": _reg_dict(reg)}
    warped = apply_template_alignment(reg.warped)
    warped_path = run_dir / "warped.png"
    imwrite(warped_path, warped)
    canonical, fields = load_fields(FIELDS_PATH)
    roi_dir = run_dir / "rois"
    roi_paths = save_rois(warped, fields, roi_dir)
    ocr_meta = run_hybrid_ocr(image_path, run_dir, fields, canonical, mode, cleaner_provider, cleaner_model)

    checks: list[FieldCheck] = []
    problems: list[str] = []
    for field in fields:
        roi = crop_roi(warped, field.bbox)
        if field.recognizer == "checkbox" or provider == "mock":
            if field.recognizer == "checkbox":
                recognition = mock_recognize(field, roi)
            else:
                recognition = ocr_meta["cleaned_results"].get(field.id) or mock_recognize(field, roi)
        else:
            recognition = ocr_meta["cleaned_results"].get(field.id)
            if not recognition or not (recognition.value or recognition.normalized_value):
                recognition = recognize_with_provider(field, roi_paths[field.id], provider, model)
        passed, msg = validate(field, recognition)
        if should_roi_review(field, recognition, passed, mode):
            recognition, passed, msg = roi_review_field(
                field,
                recognition,
                passed,
                msg,
                warped,
                run_dir,
            )
        if not passed:
            problems.append(msg)
        checks.append(FieldCheck(field, recognition, passed, msg, roi_url=f"/runs/{run_id}/rois/{field.id}.png"))

    field_dicts = [_check_dict(check) for check in checks]
    report = {
        "ok": not problems,
        "problems": problems or ["通过"],
        "summary": build_summary(field_dicts, ocr_meta["public"]),
        "registration": _reg_dict(reg),
        "warped_url": f"/runs/{run_id}/warped.png",
        "mode": mode,
        "ocr": ocr_meta["public"],
        "fields": field_dicts,
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_hybrid_ocr(
    ocr_input_path: Path,
    run_dir: Path,
    fields,
    canonical: dict[str, int],
    mode: str,
    cleaner_provider: str,
    cleaner_model: str | None,
) -> dict:
    empty = {
        "cleaned_results": {},
        "public": {
            "enabled": False,
            "ok": False,
            "error": None,
            "ocr_image_url": None,
            "blocks": [],
            "candidates": {},
            "cleaner_provider": cleaner_provider,
            "cleaner_model": cleaner_model or DEFAULT_CLEANER_MODEL,
        },
    }
    if mode == "roi_only":
        return empty

    ppocr = run_ppocrv6(ocr_input_path, run_dir)
    blocks = ppocr.get("blocks", [])
    assignments = assign_blocks_to_fields(fields, blocks, (canonical["width"], canonical["height"]))
    candidates_json = candidates_to_json(assignments)
    (run_dir / "field_candidates.json").write_text(json.dumps(candidates_json, ensure_ascii=False, indent=2), encoding="utf-8")
    cleaned = clean_field_values(fields, assignments, provider=cleaner_provider, model=cleaner_model) if blocks else {}
    ocr_image_path = ppocr.get("ocr_image_path")
    ocr_image_url = None
    if ocr_image_path:
        ocr_image_url = f"/runs/{run_dir.name}/ppocrv6/{ocr_image_path.name}"
    public = {
        "enabled": True,
        "ok": bool(ppocr.get("ok")),
        "error": ppocr.get("error"),
        "job_id": ppocr.get("job_id"),
        "input": "original_upload",
        "ocr_image_url": ocr_image_url,
        "blocks": [block_to_dict(block) for block in blocks],
        "candidates": candidates_json,
        "cleaner_provider": cleaner_provider,
        "cleaner_model": cleaner_model or DEFAULT_CLEANER_MODEL,
    }
    return {"cleaned_results": cleaned, "public": public}


def should_roi_review(field, recognition, passed: bool, mode: str) -> bool:
    if mode == "roi_only" or field.recognizer == "checkbox":
        return False
    if field.assignment.get("roi_review_always"):
        return True
    if passed:
        return False
    if field.assignment.get("skip_roi_review"):
        return False
    has_value = bool(recognition.normalized_value or recognition.value)
    if not has_value:
        return False
    if field.assignment.get("roi_review"):
        return True
    if field.validator in {"digit_length", "int_range", "regex"}:
        return True
    return False


def roi_review_field(field, original, original_passed: bool, original_msg: str, warped, run_dir: Path):
    provider = os.getenv("ROI_REVIEW_PROVIDER", "aliyun")
    model = os.getenv("ROI_REVIEW_MODEL", "qwen3.7-plus")
    review_dir = run_dir / "roi_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_bbox = expanded_bbox(field.bbox, warped.shape[1], warped.shape[0], ratio=0.45)
    review_roi = crop_roi(warped, review_bbox)
    review_path = review_dir / f"{field.id}.png"
    imwrite(review_path, review_roi)
    reviewed = recognize_with_provider(field, review_path, provider, model)
    reviewed.provider = f"roi-vlm:{reviewed.provider}"
    reviewed.raw = {
        "roi_review": {
            "roi_url": f"/runs/{run_dir.name}/roi_reviews/{field.id}.png",
            "previous": recognition_dict_for_raw(original),
            "result": reviewed.raw,
        }
    }
    review_passed, review_msg = validate(field, reviewed)
    if review_passed:
        reviewed.needs_review = False
        reviewed.review_reason = "ROI复核通过"
        return reviewed, True, review_msg

    original.raw = {
        **(original.raw or {}),
        "roi_review": {
            "roi_url": f"/runs/{run_dir.name}/roi_reviews/{field.id}.png",
            "value": reviewed.value,
            "normalized_value": reviewed.normalized_value,
            "confidence": reviewed.confidence,
            "provider": reviewed.provider,
            "model": reviewed.model,
            "passed": review_passed,
            "message": review_msg,
            "raw": reviewed.raw,
        },
    }
    original.needs_review = True
    original.review_reason = original.review_reason or "ROI复核未通过"
    return original, original_passed, original_msg


def expanded_bbox(bbox: tuple[int, int, int, int], width: int, height: int, ratio: float = 0.35) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    dx = int(w * ratio)
    dy = int(h * ratio)
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(width, x + w + dx)
    y2 = min(height, y + h + dy)
    return x1, y1, x2 - x1, y2 - y1


def recognition_dict_for_raw(recognition) -> dict:
    return {
        "value": recognition.value,
        "normalized_value": recognition.normalized_value,
        "confidence": recognition.confidence,
        "provider": recognition.provider,
        "model": recognition.model,
        "needs_review": recognition.needs_review,
        "review_reason": recognition.review_reason,
    }


def apply_template_alignment(warped):
    alignment_path = CANONICAL_DIR / "template_alignment.json"
    if not alignment_path.exists():
        return warped
    data = json.loads(alignment_path.read_text(encoding="utf-8"))
    homography = data.get("homography")
    if not homography:
        return warped
    template = imread(CANONICAL_DIR / "template_registration.png")
    matrix = np.array(homography, dtype="float32")
    return cv2.warpPerspective(warped, matrix, (template.shape[1], template.shape[0]), borderValue=(255, 255, 255))


def _reg_dict(reg) -> dict:
    return {
        "ok": reg.ok,
        "inliers": reg.inliers,
        "reproj_rmse": reg.reproj_rmse,
        "reject_reason": reg.reject_reason,
    }


def _check_dict(check: FieldCheck) -> dict:
    rec = check.recognition
    return {
        "id": check.field.id,
        "label": check.field.label,
        "section": check.field.section,
        "bbox": list(check.field.bbox),
        "recognizer": check.field.recognizer,
        "validator": check.field.validator,
        "value": rec.value,
        "normalized_value": rec.normalized_value,
        "confidence": rec.confidence,
        "provider": rec.provider,
        "model": rec.model,
        "source_label": source_label(rec.provider),
        "raw": rec.raw,
        "needs_review": rec.needs_review,
        "review_reason": rec.review_reason,
        "passed": check.passed,
        "message": "" if check.passed else check.message,
        "roi_url": check.roi_url,
    }


def build_summary(fields: list[dict], ocr_public: dict) -> dict:
    failed = [field for field in fields if not field["passed"]]
    review = [field for field in fields if field.get("needs_review")]
    return {
        "field_count": len(fields),
        "passed_count": len(fields) - len(failed),
        "failed_count": len(failed),
        "review_count": len(review),
        "ocr_block_count": len(ocr_public.get("blocks") or []),
        "cleaner_model": ocr_public.get("cleaner_model"),
        "ocr_ok": ocr_public.get("ok"),
    }


def source_label(provider: str) -> str:
    if provider == "vision-rule":
        return "Checkbox"
    if provider.startswith("roi-vlm:"):
        return "ROI-VLM"
    if "fallback_cleaner" in provider:
        return "PP-OCR"
    if provider in {"siliconflow", "aliyun"}:
        return "Cleaner"
    if provider == "mock":
        return "Unresolved"
    return "ROI-VLM"
