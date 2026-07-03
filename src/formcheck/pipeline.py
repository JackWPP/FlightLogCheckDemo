from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import CANONICAL_DIR, FIELDS_PATH, OUT_DIR
from .crop import crop_roi, save_rois
from .field_assignment import assign_blocks_to_fields, candidates_to_json, scale_bbox
from .fields import load_fields
from .image_io import imread, imwrite
from .llm_cleaner import DEFAULT_CLEANER_MODEL, clean_field_values
from .model_adapters import recognize_with_provider
from .ppocr_pipeline import block_to_dict, run_ppocrv6
from .recognizers import mock_recognize
from .registration import load_template, register, save_registration_summary
from .schemas import FieldCheck, FieldSpec, RecognitionResult
from .validators import validate


# Progress callback signature: cb(stage: str, status: str, **extra) -> None
# Stages: registration, ocr, validate, review
# Status: running, done, failed, skipped
ProgressCB = Callable[[str, str], None] | None
REGISTRATION_MODES = {"off", "optional", "required"}


def analyze_image(
    image_path: Path,
    provider: str = "mock",
    model: str | None = None,
    run_id: str = "latest",
    mode: str = "hybrid",
    cleaner_provider: str = "siliconflow",
    cleaner_model: str | None = None,
    progress_cb: ProgressCB = None,
) -> dict:
    def emit(stage: str, status: str, **extra) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, status, **extra)
            except Exception:
                # Never let a buggy progress sink break the pipeline.
                pass

    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    canonical, fields = load_fields(FIELDS_PATH)
    started_at = time.time()
    timings: dict[str, int] = {}

    # --- Stage 1: OCR (PP-OCRv6 page recognition) --------------------------
    t = time.time()
    emit("ocr", "running")
    ocr_meta = run_hybrid_ocr(
        image_path, run_dir, fields, canonical, mode, cleaner_provider, cleaner_model
    )
    blocks_count = len(ocr_meta["public"].get("blocks") or [])
    ocr_status = "done" if ocr_meta["public"].get("ok") else "failed"
    emit("ocr", ocr_status,
         duration_ms=int((time.time() - t) * 1000),
         blocks=blocks_count,
         cache_hit=ocr_meta["public"].get("cache_hit"),
         ocr_image_url=ocr_meta["public"].get("ocr_image_url"),
         ocr_blocks=ocr_meta["public"].get("blocks") or [],
         error=ocr_meta["public"].get("error"))
    timings["ocr_ms"] = int((time.time() - t) * 1000)
    t_roi = time.time()
    ppocr_roi_paths = build_ppocr_roi_evidence(ocr_meta, fields, canonical, run_dir)
    timings["ppocr_roi_ms"] = int((time.time() - t_roi) * 1000)

    # --- Stage 2: optional registration for ROI evidence/fallback ----------
    reg = None
    warped = None
    warped_url = None
    roi_paths: dict[str, Path] = {}
    reg_mode = registration_mode()
    if reg_mode == "off":
        emit("registration", "skipped", reason="REGISTRATION_MODE=off")
        timings["registration_ms"] = 0
    else:
        t = time.time()
        emit("registration", "running", mode=reg_mode)
        image = imread(image_path)
        template = load_template(CANONICAL_DIR)
        reg = register(image, template)
        save_registration_summary(run_dir / "registration.json", reg)
        if not reg.ok or reg.warped is None:
            emit("registration", "failed",
                 duration_ms=int((time.time() - t) * 1000),
                 reason=reg.reject_reason)
            timings["registration_ms"] = int((time.time() - t) * 1000)
            if reg_mode == "required":
                return {"ok": False, "error": reg.reject_reason, "registration": _reg_dict(reg, reg_mode)}
        else:
            emit("registration", "done",
                 duration_ms=int((time.time() - t) * 1000),
                 inliers=reg.inliers)
            warped = apply_template_alignment(reg.warped)
            warped_path = run_dir / "warped.png"
            imwrite(warped_path, warped)
            warped_url = f"/runs/{run_id}/warped.png"
            roi_dir = run_dir / "rois"
            roi_paths = save_rois(warped, fields, roi_dir)
            timings["registration_ms"] = int((time.time() - t) * 1000)

    # --- Stage 3: per-field validation (local rules) ----------------------
    t = time.time()
    emit("validate", "running")
    checks: list[FieldCheck] = []
    for field in fields:
        roi = crop_roi(warped, field.bbox) if warped is not None else None
        roi_path = ppocr_roi_paths.get(field.id) or roi_paths.get(field.id)
        roi_url = evidence_url(run_id, roi_path) if roi_path else None
        if field.recognizer == "checkbox" or provider == "mock":
            if field.recognizer == "checkbox":
                if roi is not None:
                    recognition = mock_recognize(field, roi)
                elif roi_path:
                    recognition = mock_recognize(field, imread(roi_path))
                else:
                    recognition = unresolved_result(field, "未生成可用ROI，勾选框需人工复核")
            else:
                recognition = ocr_meta["cleaned_results"].get(field.id)
                if recognition is None:
                    recognition = mock_recognize(field, roi) if roi is not None else unresolved_result(
                        field, "无OCR候选"
                    )
        else:
            recognition = ocr_meta["cleaned_results"].get(field.id)
            if not recognition or not (recognition.value or recognition.normalized_value):
                if roi_path:
                    recognition = recognize_with_provider(field, roi_path, provider, model)
                else:
                    recognition = unresolved_result(field, "无可靠OCR候选，且未生成可用ROI")
        passed, msg = validate(field, recognition)
        checks.append(FieldCheck(
            field, recognition, passed, msg,
            roi_url=roi_url,
        ))
    emit("validate", "done",
         duration_ms=int((time.time() - t) * 1000),
         count=len(checks))
    timings["validate_ms"] = int((time.time() - t) * 1000)

    # --- Stage 4: ROI-VLM review (prefer PaddleOCR ROI evidence) -----------
    needs_review = [
        c for c in checks
        if should_roi_review(c.field, c.recognition, c.passed, mode) and (ppocr_roi_paths.get(c.field.id) or warped is not None)
    ]
    if needs_review:
        t = time.time()
        emit("review", "running", total=len(needs_review))
        with ThreadPoolExecutor(max_workers=roi_review_concurrency()) as pool:
            futures = {
                pool.submit(review_check, c, run_dir, ppocr_roi_paths.get(c.field.id), warped): checks.index(c)
                for c in needs_review
            }
            for done_count, future in enumerate(as_completed(futures), start=1):
                i = futures[future]
                checks[i] = future.result()
                emit("review", "running", total=len(needs_review), completed=done_count)
        emit("review", "done",
             duration_ms=int((time.time() - t) * 1000),
             total=len(needs_review))
        timings["review_ms"] = int((time.time() - t) * 1000)
    else:
        reason = "no_roi_evidence" if warped is None and not ppocr_roi_paths else "no_fields"
        emit("review", "skipped", total=0, reason=reason)
        timings["review_ms"] = 0

    # --- Build report ------------------------------------------------------
    field_dicts = [_check_dict(check) for check in checks]
    problems = [c.message for c in checks if not c.passed]
    report = {
        "ok": not problems,
        "problems": problems or ["通过"],
        "summary": build_summary(field_dicts, ocr_meta["public"]),
        "registration": _reg_dict(reg, reg_mode),
        "warped_url": warped_url,
        "mode": mode,
        "timings": {**timings, "total_ms": int((time.time() - started_at) * 1000)},
        "ocr": ocr_meta["public"],
        "fields": field_dicts,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def registration_mode() -> str:
    value = os.getenv("REGISTRATION_MODE", "off").strip().lower()
    return value if value in REGISTRATION_MODES else "off"


def roi_review_concurrency() -> int:
    try:
        return max(1, min(6, int(os.getenv("ROI_REVIEW_CONCURRENCY", "3"))))
    except ValueError:
        return 3


def unresolved_result(field: FieldSpec, reason: str) -> RecognitionResult:
    return RecognitionResult(
        value="",
        normalized_value="",
        confidence=0.0,
        provider="unresolved",
        model="none",
        raw={"reason": reason, "field_id": field.id},
        needs_review=True,
        review_reason=reason,
    )


def evidence_url(run_id: str, path: Path) -> str:
    if path.parts[-2] == "ppocr_rois":
        return f"/runs/{run_id}/ppocr_rois/{path.name}"
    if path.parts[-2] == "rois":
        return f"/runs/{run_id}/rois/{path.name}"
    return f"/runs/{run_id}/{path.name}"


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
        "cache_hit": bool(ppocr.get("cache_hit")),
    }
    return {
        "cleaned_results": cleaned,
        "public": public,
        "ocr_image_path": ocr_image_path,
        "assignments": assignments,
    }


def build_ppocr_roi_evidence(ocr_meta: dict, fields: list[FieldSpec], canonical: dict[str, int], run_dir: Path) -> dict[str, Path]:
    ocr_image_path = ocr_meta.get("ocr_image_path")
    if not ocr_image_path:
        return {}
    image = imread(ocr_image_path)
    out_dir = run_dir / "ppocr_rois"
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments = ocr_meta.get("assignments") or {}
    paths: dict[str, Path] = {}
    for field in fields:
        bbox = ppocr_evidence_bbox(field, assignments.get(field.id, []), image.shape[1], image.shape[0], canonical)
        roi = crop_roi(image, bbox, pad=10)
        path = out_dir / f"{field.id}.png"
        imwrite(path, roi)
        paths[field.id] = path
    return paths


def ppocr_evidence_bbox(
    field: FieldSpec,
    candidates,
    width: int,
    height: int,
    canonical: dict[str, int],
) -> tuple[int, int, int, int]:
    candidate_boxes = [candidate.block.box for candidate in candidates[:4]]
    if candidate_boxes:
        x1 = min(box[0] for box in candidate_boxes)
        y1 = min(box[1] for box in candidate_boxes)
        x2 = max(box[2] for box in candidate_boxes)
        y2 = max(box[3] for box in candidate_boxes)
        return clamp_xyxy(expand_xyxy((x1, y1, x2, y2), 0.55), width, height)
    scale_x = width / max(float(canonical["width"]), 1.0)
    scale_y = height / max(float(canonical["height"]), 1.0)
    base = tuple(field.assignment.get("search_bbox") or field.bbox)
    return clamp_xyxy(expand_xyxy(scale_bbox(base, scale_x, scale_y), 0.35), width, height)


def expand_xyxy(bbox: tuple[float, float, float, float], ratio: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    return x1 - w * ratio, y1 - h * ratio, x2 + w * ratio, y2 + h * ratio


def clamp_xyxy(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    ix1 = max(0, min(width - 1, int(round(x1))))
    iy1 = max(0, min(height - 1, int(round(y1))))
    ix2 = max(ix1 + 1, min(width, int(round(x2))))
    iy2 = max(iy1 + 1, min(height, int(round(y2))))
    return ix1, iy1, ix2 - ix1, iy2 - iy1


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
    if field.assignment.get("prefer_roi_vlm") or field.assignment.get("roi_review"):
        return True
    if field.validator in {"digit_length", "int_range", "regex"}:
        return True
    if not has_value:
        return False
    return False


def review_check(check: FieldCheck, run_dir: Path, evidence_path: Path | None, warped) -> FieldCheck:
    try:
        new_rec, new_passed, new_msg = roi_review_field(
            check.field, check.recognition, check.passed, check.message, run_dir,
            evidence_path=evidence_path, warped=warped,
        )
        return FieldCheck(check.field, new_rec, new_passed, new_msg, check.roi_url)
    except Exception as exc:  # noqa: BLE001
        rec = check.recognition
        rec.needs_review = True
        rec.review_reason = rec.review_reason or "ROI复核异常"
        rec.raw = {**(rec.raw or {}), "roi_review_error": str(exc)}
        return FieldCheck(check.field, rec, check.passed, check.message, check.roi_url)


def roi_review_field(
    field,
    original,
    original_passed: bool,
    original_msg: str,
    run_dir: Path,
    evidence_path: Path | None = None,
    warped=None,
):
    provider = os.getenv("ROI_REVIEW_PROVIDER", "aliyun")
    model = os.getenv("ROI_REVIEW_MODEL", "qwen3.7-plus")
    review_dir = run_dir / "roi_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / f"{field.id}.png"
    if evidence_path:
        review_path.write_bytes(evidence_path.read_bytes())
    elif warped is not None:
        review_bbox = expanded_bbox(field.bbox, warped.shape[1], warped.shape[0], ratio=0.45)
        review_roi = crop_roi(warped, review_bbox)
        imwrite(review_path, review_roi)
    else:
        original.needs_review = True
        original.review_reason = original.review_reason or "缺少ROI复核图"
        return original, original_passed, original_msg
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


def _reg_dict(reg, mode: str | None = None) -> dict:
    if reg is None:
        return {
            "mode": mode or registration_mode(),
            "ok": False,
            "enabled": False,
            "inliers": 0,
            "reproj_rmse": None,
            "reject_reason": "REGISTRATION_MODE=off",
        }
    return {
        "mode": mode or registration_mode(),
        "enabled": True,
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
        "ocr_cache_hit": ocr_public.get("cache_hit"),
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
    if provider == "unresolved":
        return "Unresolved"
    return "ROI-VLM"
