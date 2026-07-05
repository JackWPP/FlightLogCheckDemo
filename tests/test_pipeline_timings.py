from __future__ import annotations

import cv2
import numpy as np

from formcheck.pipeline import (
    analyze_image,
    numeric_candidate_ambiguity_reason,
    roi_review_priority,
    select_roi_reviews,
    should_roi_review,
)
from formcheck.schemas import FieldCandidate, FieldSpec, OcrBlock, RecognitionResult


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


def test_dense_numeric_fields_with_close_candidates_are_marked_for_review() -> None:
    field = FieldSpec(
        id="apu_cum_cycles",
        label="23 APU累计使用循环",
        section="apu",
        bbox=(0, 0, 100, 50),
        recognizer="numeric_text",
        validator="number_less_than",
        params={"max": 99999},
        fail_msg="APU循环不小于99999",
    )
    candidates = [
        FieldCandidate(field.id, OcrBlock("b1", "5789", 0.99, (0, 0, 40, 20), (20, 10)), 2.6, "best"),
        FieldCandidate(field.id, OcrBlock("b2", "114121", 0.99, (45, 0, 100, 20), (72, 10)), 2.1, "near"),
    ]

    reason = numeric_candidate_ambiguity_reason(field, candidates)
    recognition = RecognitionResult("5789", "5789", needs_review=bool(reason), review_reason=reason)

    assert reason == "存在相近数字候选，等待ROI复核"
    assert should_roi_review(field, recognition, passed=True, mode="hybrid")


def test_clear_numeric_winner_is_not_marked_ambiguous() -> None:
    field = FieldSpec(
        id="oil_eng1_added",
        label="02 发动机1加注量",
        section="oil",
        bbox=(0, 0, 100, 50),
        recognizer="numeric_text",
        validator="int_range",
        params={"min": 0, "max": 4},
        fail_msg="发动机1加注量不在0-4",
    )
    candidates = [
        FieldCandidate(field.id, OcrBlock("b1", "0", 0.99, (0, 0, 40, 20), (20, 10)), 2.6, "best"),
        FieldCandidate(field.id, OcrBlock("b2", "19.5", 0.99, (45, 0, 100, 20), (72, 10)), 1.2, "far"),
    ]

    assert numeric_candidate_ambiguity_reason(field, candidates) == ""


def test_roi_review_selection_prioritizes_high_risk_failures(monkeypatch) -> None:
    monkeypatch.setenv("ROI_REVIEW_MAX_FIELDS", "2")

    license_field = FieldSpec(
        id="awr_license",
        label="29 适航放行-执照号",
        section="airworthiness_release",
        bbox=(0, 0, 1, 1),
        recognizer="keyed_text",
        validator="regex",
        params={"pattern": r"^CAACML[0-9]{8}$"},
        fail_msg="执照号不合规",
    )
    date_field = FieldSpec(
        id="awr_date",
        label="26 适航放行-日期",
        section="airworthiness_release",
        bbox=(0, 0, 1, 1),
        recognizer="date_text",
        validator="same_day",
        params={},
        fail_msg="日期不是今天",
    )
    review_passed_field = FieldSpec(
        id="apu_cum_cycles",
        label="23 APU累计使用循环",
        section="apu",
        bbox=(0, 0, 1, 1),
        recognizer="numeric_text",
        validator="number_less_than",
        params={"max": 99999},
        fail_msg="APU循环不小于99999",
    )
    low_field = FieldSpec(
        id="action_station",
        label="15 处理措施-地点",
        section="fault_action",
        bbox=(0, 0, 1, 1),
        recognizer="keyed_text",
        validator="exact_text",
        params={"allow": ["重庆"]},
        fail_msg="地点不是重庆",
    )
    checks = [
        make_check(review_passed_field, passed=True, needs_review=True),
        make_check(low_field, passed=False),
        make_check(license_field, passed=False),
        make_check(date_field, passed=False),
    ]

    selected, skipped = select_roi_reviews(checks)

    assert [check.field.id for check in selected] == ["awr_license", "apu_cum_cycles"]
    assert [check.field.id for check in skipped] == ["awr_date", "action_station"]
    assert roi_review_priority(selected[0]) > roi_review_priority(skipped[-1])


def test_roi_review_selection_can_be_unlimited(monkeypatch) -> None:
    monkeypatch.setenv("ROI_REVIEW_MAX_FIELDS", "0")
    field = FieldSpec("f", "字段", "s", (0, 0, 1, 1), "text", "present", {}, "fail")
    checks = [make_check(field, passed=False), make_check(field, passed=False)]

    selected, skipped = select_roi_reviews(checks)

    assert len(selected) == 2
    assert skipped == []


def make_check(field: FieldSpec, passed: bool, needs_review: bool = False):
    from formcheck.schemas import FieldCheck

    recognition = RecognitionResult(
        value="value",
        normalized_value="value",
        provider="test",
        model="test",
        needs_review=needs_review,
    )
    return FieldCheck(field=field, recognition=recognition, passed=passed, message="" if passed else field.fail_msg)
