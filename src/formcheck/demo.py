from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import ASSETS_DIR, OUT_DIR, OUTPUTS_DIR
from .issue_triage import triage_issues
from .pipeline import build_summary, source_label


DEMO_RUN_ID = "local_29_rule_smoke"
DEMO_DIR = OUTPUTS_DIR / "demo_sample"


def ensure_demo_sample() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    copy_file(ASSETS_DIR / "raw" / "sample_01.jpg", DEMO_DIR / "sample_01.jpg")
    copy_file(ASSETS_DIR / "raw" / "base_scan.jpg", DEMO_DIR / "base_scan.jpg")
    copy_file(ASSETS_DIR / "canonical" / "experimental_gpt_image_base.png", DEMO_DIR / "gpt_base.png")
    copy_file(OUT_DIR / DEMO_RUN_ID / "warped.png", DEMO_DIR / "warped.png")
    copy_file(OUT_DIR / DEMO_RUN_ID / "ppocrv6" / "ocr_image_0.jpg", DEMO_DIR / "ocr_image.jpg")
    copy_file(OUT_DIR / DEMO_RUN_ID / "report.json", DEMO_DIR / "report.json")
    copy_file(OUT_DIR / DEMO_RUN_ID / "field_candidates.json", DEMO_DIR / "field_candidates.json")
    copy_file(OUT_DIR / DEMO_RUN_ID / "ppocrv6" / "ocr_blocks.json", DEMO_DIR / "ocr_blocks.json")
    src_roi_dir = OUT_DIR / DEMO_RUN_ID / "rois"
    dst_roi_dir = DEMO_DIR / "rois"
    if src_roi_dir.exists():
        dst_roi_dir.mkdir(exist_ok=True)
        for path in src_roi_dir.glob("*.png"):
            copy_file(path, dst_roi_dir / path.name)
    src_ppocr_roi_dir = OUT_DIR / DEMO_RUN_ID / "ppocr_rois"
    dst_ppocr_roi_dir = DEMO_DIR / "ppocr_rois"
    if src_ppocr_roi_dir.exists():
        dst_ppocr_roi_dir.mkdir(exist_ok=True)
        for path in src_ppocr_roi_dir.glob("*.png"):
            copy_file(path, dst_ppocr_roi_dir / path.name)
    manifest = build_manifest()
    (DEMO_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def demo_payload() -> dict[str, Any]:
    ensure_demo_sample()
    report_path = DEMO_DIR / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = json.loads((DEMO_DIR / "field_candidates.json").read_text(encoding="utf-8"))
    blocks = json.loads((DEMO_DIR / "ocr_blocks.json").read_text(encoding="utf-8"))
    report["run_id"] = "demo_sample"
    report["upload_url"] = "/outputs/demo_sample/sample_01.jpg"
    report["warped_url"] = "/outputs/demo_sample/warped.png"
    report.setdefault("ocr", {})
    report["ocr"]["ocr_image_url"] = "/outputs/demo_sample/ocr_image.jpg"
    report["ocr"]["candidates"] = candidates
    report["ocr"]["blocks"] = blocks
    for field in report.get("fields", []):
        ppocr_roi = DEMO_DIR / "ppocr_rois" / f"{field['id']}.png"
        field["roi_url"] = (
            f"/outputs/demo_sample/ppocr_rois/{field['id']}.png"
            if ppocr_roi.exists()
            else f"/outputs/demo_sample/rois/{field['id']}.png"
        )
        field.setdefault("source_label", source_label(field.get("provider", "")))
        high_risk = field["id"] in {
            "ac_reg_digits",
            "action_authorization",
            "action_release_sign",
            "awr_release_sign",
            "awr_license",
        }
        unresolved = not (field.get("value") or field.get("normalized_value"))
        field.setdefault("needs_review", bool(high_risk or unresolved))
        field.setdefault("review_reason", "需人工复核" if field.get("needs_review") else "")
    report["summary"] = build_summary(report.get("fields", []), report.get("ocr", {}))
    triage = triage_issues(report.get("fields", []), provider="mock")
    report["all_problems"] = triage["all_problems"]
    report["problems"] = triage["problems"]
    report["problem_items"] = triage.get("problem_items", [])
    report["review_problems"] = triage.get("review_problems", [])
    report["issue_triage"] = triage["issue_triage"]
    return {"manifest": build_manifest(), "report": report}


def build_manifest() -> dict[str, Any]:
    return {
        "id": "demo_sample",
        "title": "飞行记录单示例",
        "description": "缓存示例，不调用云端模型。",
        "images": {
            "upload": "/outputs/demo_sample/sample_01.jpg",
            "warped": "/outputs/demo_sample/warped.png",
            "ocr": "/outputs/demo_sample/ocr_image.jpg",
            "scan_base": "/outputs/demo_sample/base_scan.jpg",
            "gpt_base": "/outputs/demo_sample/gpt_base.png",
        },
        "artifacts": {
            "report": "/outputs/demo_sample/report.json",
            "field_candidates": "/outputs/demo_sample/field_candidates.json",
            "ocr_blocks": "/outputs/demo_sample/ocr_blocks.json",
        },
    }


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)
