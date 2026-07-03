from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from formcheck.config import CANONICAL_DIR, FIELDS_PATH, OUT_DIR, load_dotenv
from formcheck.crop import crop_roi, make_roi_grid, save_rois
from formcheck.fields import load_fields
from formcheck.enhance import enhance_roi_for_ocr
from formcheck.image_io import imread, imwrite
from formcheck.model_adapters import recognize_with_provider
from formcheck.pipeline import apply_template_alignment
from formcheck.recognizers import mock_recognize
from formcheck.registration import load_template, register, save_registration_summary
from formcheck.validators import validate


DEFAULT_FIELDS = [
    "fault_flight_no",
    "fault_station",
    "fault_report_en",
    "fault_ref",
    "fault_date",
    "fault_authorization",
    "apu_cum_hours",
    "apu_cum_cycles",
    "awr_date",
    "awr_license",
]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="assets/raw/sample_01.jpg")
    parser.add_argument("--run-id", default="model_compare")
    parser.add_argument("--fields", nargs="*", default=DEFAULT_FIELDS)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="provider:model, for example siliconflow:deepseek-ai/DeepSeek-OCR",
    )
    args = parser.parse_args()

    models = _parse_model_args(args.model) or _models_from_env()
    if not models:
        raise SystemExit("No models configured. Pass --model provider:model or set env vars.")

    run_dir = OUT_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    image = imread(Path(args.image))
    template = load_template(CANONICAL_DIR)
    reg = register(image, template)
    save_registration_summary(run_dir / "registration.json", reg)
    if not reg.ok or reg.warped is None:
        print(json.dumps({"ok": False, "registration": reg.reject_reason}, ensure_ascii=False, indent=2))
        return

    warped = apply_template_alignment(reg.warped)
    warped_path = run_dir / "warped.png"
    imwrite(warped_path, warped)
    _, all_fields = load_fields(FIELDS_PATH)
    fields = [field for field in all_fields if field.id in set(args.fields)]
    roi_dir = run_dir / "rois"
    roi_paths = save_rois(warped, fields, roi_dir)
    enhanced_roi_dir = run_dir / "rois_enhanced"
    enhanced_roi_dir.mkdir(parents=True, exist_ok=True)
    enhanced_roi_paths = {}
    for field in fields:
        enhanced = enhance_roi_for_ocr(crop_roi(warped, field.bbox))
        path = enhanced_roi_dir / f"{field.id}.png"
        imwrite(path, enhanced)
        enhanced_roi_paths[field.id] = path
    make_roi_grid(warped, fields, run_dir / "roi_grid.png")

    rows = []
    for field in fields:
        roi = crop_roi(warped, field.bbox)
        if field.recognizer == "checkbox":
            recognition = mock_recognize(field, roi)
            passed, msg = validate(field, recognition)
            rows.append(_row(field, "local", "checkbox", recognition, passed, msg))
            continue
        for provider, model in models:
            recognition = recognize_with_provider(field, enhanced_roi_paths[field.id], provider, model)
            passed, msg = validate(field, recognition)
            rows.append(_row(field, provider, model, recognition, passed, msg))

    output = {
        "ok": True,
        "registration": {
            "inliers": reg.inliers,
            "reproj_rmse": reg.reproj_rmse,
            "warped": str(warped_path),
        },
        "models": [{"provider": provider, "model": model} for provider, model in models],
        "fields": rows,
    }
    out_path = run_dir / "comparison.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_summary(output), ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


def _models_from_env() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key, provider in [("SILICONFLOW_COMPARE_MODELS", "siliconflow"), ("ALIYUN_COMPARE_MODELS", "aliyun")]:
        for model in os.getenv(key, "").split(","):
            model = model.strip()
            if model:
                pairs.append((provider, model))
    return pairs


def _parse_model_args(items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        if ":" not in item:
            raise SystemExit(f"Model must be provider:model, got {item}")
        provider, model = item.split(":", 1)
        pairs.append((provider, model))
    return pairs


def _row(field, provider, model, recognition, passed, msg) -> dict:
    return {
        "field_id": field.id,
        "label": field.label,
        "provider": provider,
        "model": model,
        "value": recognition.value,
        "normalized_value": recognition.normalized_value,
        "confidence": recognition.confidence,
        "passed": passed,
        "message": "" if passed else msg,
        "raw": recognition.raw,
    }


def _summary(output: dict) -> dict:
    rows = []
    for row in output["fields"]:
        rows.append(
            {
                "field": row["field_id"],
                "model": f"{row['provider']}:{row['model']}",
                "value": row["normalized_value"] or row["value"],
                "passed": row["passed"],
                "error": (row.get("raw") or {}).get("error"),
            }
        )
    return {"registration": output["registration"], "rows": rows}


if __name__ == "__main__":
    main()
