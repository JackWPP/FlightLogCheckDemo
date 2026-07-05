from __future__ import annotations

import cv2
import numpy as np

from formcheck.pipeline import analyze_image
from formcheck.schemas import FieldSpec, RecognitionResult


def test_no_key_report_keeps_fine_grained_timings(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("formcheck.pipeline.OUT_DIR", tmp_path / "out")
    monkeypatch.setenv("PADDLEOCR_AISTUDIO_TOKEN", "")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "")
    monkeypatch.setenv("ALIYUN_API_KEY", "")
    image = tmp_path / "upload.jpg"
    cv2.imwrite(str(image), np.full((30, 40, 3), 255, dtype=np.uint8))

    report = analyze_image(image, run_id="timing-test")

    timings = report["timings"]
    assert "ppocr_submit_ms" in timings
    assert "ppocr_poll_ms" in timings
    assert "assignment_ms" in timings
    assert "cleaner_ms" in timings
    assert "issue_triage_ms" in timings
    assert report["all_problems"]
    assert len(report["problems"]) <= 4


def test_failed_reviewable_field_is_marked_waiting_for_roi(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("formcheck.pipeline.OUT_DIR", tmp_path / "out")
    field = FieldSpec(
        id="apu_cum_hours",
        label="22 APU累计使用时间",
        section="apu",
        bbox=(0, 0, 100, 50),
        recognizer="numeric_text",
        validator="number_less_than",
        params={"max": 99999},
        fail_msg="APU时间不小于99999",
    )
    monkeypatch.setattr("formcheck.pipeline.load_fields", lambda _path: ({"width": 100, "height": 50}, [field]))

    def fake_hybrid(*_args, **_kwargs):
        return {
            "cleaned_results": {
                field.id: RecognitionResult(value="114121", normalized_value="114121", provider="test", model="test")
            },
            "public": {"ok": True, "blocks": [], "timings": {}, "cleaner_model": "test"},
            "ocr_image_path": None,
            "blocks": [],
            "assignments": {},
        }

    monkeypatch.setattr("formcheck.pipeline.run_hybrid_ocr", fake_hybrid)
    monkeypatch.setattr("formcheck.pipeline.triage_issues", lambda fields, **_kwargs: {
        "problems": ["APU时间不小于99999"],
        "all_problems": ["APU时间不小于99999"],
        "issue_triage": {"provider": "local"},
    })
    image = tmp_path / "upload.jpg"
    cv2.imwrite(str(image), np.full((30, 40, 3), 255, dtype=np.uint8))

    report = analyze_image(image, run_id="review-state-test")
    result = report["fields"][0]

    assert not result["passed"]
    assert result["needs_review"]
    assert result["review_reason"] == "规则失败，等待ROI复核"
