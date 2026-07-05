from __future__ import annotations

import cv2
import numpy as np

from formcheck.pipeline import (
    build_ppocr_roi_evidence,
    ocr_bbox_to_image_bbox,
    ppocr_evidence_bbox,
    ppocr_visual_page_region,
    roi_review_field,
    recognize_roi_with_fallback,
    registration_mode,
    should_roi_review,
)
from formcheck.schemas import FieldCandidate, FieldSpec, OcrBlock, RecognitionResult


def test_build_ppocr_roi_evidence_uses_ppocr_image(tmp_path, monkeypatch) -> None:
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (90, 80), (120, 105), (0, 0, 0), -1)
    ocr_image = tmp_path / "ocr_image.jpg"
    cv2.imwrite(str(ocr_image), image)

    field = FieldSpec(
        id="apu_cum_cycles",
        label="APU累计使用循环",
        section="apu",
        bbox=(100, 100, 80, 40),
        recognizer="text",
        validator="digit_length",
        params={"allow_lengths": [4, 5]},
        fail_msg="APU循环不是4或5位",
    )
    block = OcrBlock(
        id="b0001",
        text="348",
        score=0.9,
        box=(90, 80, 120, 105),
        center=(105, 92.5),
    )
    candidate = FieldCandidate(field.id, block, 1.0, "test")

    paths = build_ppocr_roi_evidence(
        {"ocr_image_path": ocr_image, "assignments": {field.id: [candidate]}},
        [field],
        {"width": 2400, "height": 1800},
        tmp_path,
    )

    assert field.id in paths
    assert paths[field.id].parent.name == "ppocr_rois"
    crop = cv2.imread(str(paths[field.id]))
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert crop.min() == 0


def test_registration_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("REGISTRATION_MODE", raising=False)

    assert registration_mode() == "off"


def test_failed_numeric_fields_go_to_roi_review_even_without_value() -> None:
    field = FieldSpec(
        id="apu_cum_cycles",
        label="APU累计使用循环",
        section="apu",
        bbox=(690, 1678, 230, 100),
        recognizer="numeric_text",
        validator="digit_length",
        params={"allow_lengths": [4, 5]},
        fail_msg="APU循环不是4或5位",
    )
    recognition = RecognitionResult(value="", normalized_value="", provider="unresolved")

    assert should_roi_review(field, recognition, passed=False, mode="hybrid")


def test_roi_review_fields_use_precise_field_bbox_not_tight_candidate_box() -> None:
    field = FieldSpec(
        id="apu_cum_cycles",
        label="APU累计使用循环",
        section="apu",
        bbox=(690, 1678, 230, 100),
        recognizer="numeric_text",
        validator="digit_length",
        params={"allow_lengths": [4, 5]},
        fail_msg="APU循环不是4或5位",
        assignment={
            "search_bbox": [690, 1660, 230, 124],
            "value_type": "numeric",
            "roi_review": True,
            "roi_review_always": True,
        },
    )
    block = OcrBlock(
        id="b0243",
        text="348",
        score=0.9,
        box=(410, 900, 455, 930),
        center=(432.5, 915),
    )
    candidate = FieldCandidate(field.id, block, 1.0, "test")

    x, y, w, h = ppocr_evidence_bbox(
        field,
        [candidate],
        width=1200,
        height=892,
        canonical={"width": 2400, "height": 1784},
        page_size=(2400, 1784),
    )

    assert 140 <= w <= 155
    assert 55 <= h <= 70
    assert 338 <= x <= 346
    assert 820 <= y <= 836


def test_ppocr_evidence_prefers_field_bbox_over_wide_search_bbox() -> None:
    field = FieldSpec(
        id="oil_eng1_qty",
        label="发动机1滑油量",
        section="oil",
        bbox=(310, 635, 155, 58),
        recognizer="numeric_text",
        validator="int_range",
        params={"min": 15, "max": 25},
        fail_msg="滑油量不在15-25",
        assignment={
            "search_bbox": [250, 570, 420, 140],
            "value_type": "numeric",
            "roi_review": True,
        },
    )
    block = OcrBlock(
        id="b0094",
        text="20.5",
        score=0.9,
        box=(250, 620, 620, 690),
        center=(435, 655),
    )
    candidate = FieldCandidate(field.id, block, 1.0, "test")

    x, _y, w, _h = ppocr_evidence_bbox(
        field,
        [candidate],
        width=2400,
        height=1784,
        canonical={"width": 2400, "height": 1784},
        page_size=(2400, 1784),
    )

    assert 295 <= x <= 315
    assert 190 <= w <= 210


def test_ppocr_visual_page_region_uses_left_half_for_stitched_debug_image() -> None:
    region = ppocr_visual_page_region(
        image_width=2000,
        image_height=750,
        page_width=1339.52,
        page_height=1003.6,
    )

    assert region == (0.0, 0.0, 1000.0, 750.0)


def test_ocr_bbox_maps_to_left_half_of_ppocr_debug_image() -> None:
    x, y, w, h = ocr_bbox_to_image_bbox(
        (385.0, 933.0, 513.0, 1003.0),
        image_width=2000,
        image_height=750,
        page_width=1339.52,
        page_height=1003.6,
    )

    assert 280 <= x <= 300
    assert 690 <= y <= 700
    assert 90 <= w <= 105
    assert 50 <= h <= 60


def test_aliyun_roi_review_falls_back_to_ocr_model(tmp_path, monkeypatch) -> None:
    field = FieldSpec(
        id="apu_cum_cycles",
        label="APU累计使用循环",
        section="apu",
        bbox=(690, 1678, 230, 100),
        recognizer="numeric_text",
        validator="digit_length",
        params={"allow_lengths": [4, 5]},
        fail_msg="APU循环不是4或5位",
    )
    roi_path = tmp_path / "roi.png"
    roi_path.write_bytes(b"fake")
    calls = []

    def fake_recognize(field_arg, roi_path_arg, provider_arg, model_arg):
        calls.append(model_arg)
        if model_arg == "qwen3.7-plus":
            return RecognitionResult(
                value="",
                normalized_value="",
                confidence=0.0,
                provider="aliyun",
                model=model_arg,
                raw={"error": "timeout"},
            )
        return RecognitionResult(
            value="3481",
            normalized_value="3481",
            confidence=0.9,
            provider="aliyun",
            model=model_arg,
            raw={"content": "{}"},
        )

    monkeypatch.setenv("ROI_REVIEW_FALLBACK_MODEL", "qwen3.5-ocr")
    monkeypatch.setenv("ROI_REVIEW_CACHE_ENABLED", "0")
    monkeypatch.setattr("formcheck.pipeline.recognize_with_provider", fake_recognize)

    result = recognize_roi_with_fallback(field, roi_path, "aliyun", "qwen3.7-plus")

    assert calls == ["qwen3.7-plus", "qwen3.5-ocr"]
    assert result.normalized_value == "3481"
    assert result.raw["fallback_model"] == "qwen3.5-ocr"


def test_low_confidence_roi_review_does_not_override_original_value(tmp_path, monkeypatch) -> None:
    field = FieldSpec(
        id="apu_cum_cycles",
        label="APU累计使用循环",
        section="apu",
        bbox=(690, 1678, 230, 100),
        recognizer="numeric_text",
        validator="number_less_than",
        params={"max": 99999},
        fail_msg="APU循环不小于99999",
    )
    original = RecognitionResult(
        value="348",
        normalized_value="348",
        confidence=0.72,
        provider="siliconflow:fallback_cleaner",
        model="cleaner",
    )
    evidence = tmp_path / "evidence.png"
    evidence.write_bytes(b"roi-evidence")

    def fake_review(field_arg, roi_path_arg, provider_arg, model_arg):
        return RecognitionResult(
            value="3481",
            normalized_value="3481",
            confidence=0.42,
            provider="aliyun",
            model=model_arg,
            raw={"content": "{}"},
        )

    monkeypatch.setenv("ROI_REVIEW_ACCEPT_MIN_CONFIDENCE", "0.65")
    monkeypatch.setattr("formcheck.pipeline.recognize_roi_with_fallback", fake_review)

    result, passed, message = roi_review_field(
        field,
        original,
        original_passed=True,
        original_msg="",
        run_dir=tmp_path,
        evidence_path=evidence,
    )

    assert passed
    assert message == ""
    assert result.normalized_value == "348"
    assert result.needs_review
    assert result.review_reason == "ROI复核置信度低于0.65，需人工确认"
    assert result.raw["roi_review"]["normalized_value"] == "3481"
    assert result.raw["roi_review"]["accepted"] is False
    assert result.raw["roi_review"]["changed_value"] is True
    assert result.raw["roi_review"]["previous_normalized_value"] == "348"
    assert result.raw["roi_review"]["review_normalized_value"] == "3481"


def test_high_confidence_roi_review_can_override_original_value(tmp_path, monkeypatch) -> None:
    field = FieldSpec(
        id="apu_cum_cycles",
        label="APU累计使用循环",
        section="apu",
        bbox=(690, 1678, 230, 100),
        recognizer="numeric_text",
        validator="number_less_than",
        params={"max": 99999},
        fail_msg="APU循环不小于99999",
    )
    original = RecognitionResult(
        value="348",
        normalized_value="348",
        confidence=0.72,
        provider="siliconflow:fallback_cleaner",
        model="cleaner",
        needs_review=True,
    )
    evidence = tmp_path / "evidence.png"
    evidence.write_bytes(b"roi-evidence")

    def fake_review(field_arg, roi_path_arg, provider_arg, model_arg):
        return RecognitionResult(
            value="3481",
            normalized_value="3481",
            confidence=0.91,
            provider="aliyun",
            model=model_arg,
            raw={"content": "{}"},
        )

    monkeypatch.setenv("ROI_REVIEW_ACCEPT_MIN_CONFIDENCE", "0.65")
    monkeypatch.setattr("formcheck.pipeline.recognize_roi_with_fallback", fake_review)

    result, passed, message = roi_review_field(
        field,
        original,
        original_passed=True,
        original_msg="",
        run_dir=tmp_path,
        evidence_path=evidence,
    )

    assert passed
    assert message == field.fail_msg
    assert result.normalized_value == "3481"
    assert not result.needs_review
    assert result.review_reason == "ROI复核通过"
    assert result.raw["roi_review"]["accepted"] is True
    assert result.raw["roi_review"]["changed_value"] is True
    assert result.raw["roi_review"]["previous_normalized_value"] == "348"
    assert result.raw["roi_review"]["review_normalized_value"] == "3481"


def test_roi_review_success_is_cached(tmp_path, monkeypatch) -> None:
    field = FieldSpec(
        id="apu_cum_cycles",
        label="APU累计使用循环",
        section="apu",
        bbox=(690, 1678, 230, 100),
        recognizer="numeric_text",
        validator="digit_length",
        params={"allow_lengths": [4, 5]},
        fail_msg="APU循环不是4或5位",
    )
    roi_path = tmp_path / "roi.png"
    roi_path.write_bytes(b"fake-success")
    calls = []

    def fake_recognize(field_arg, roi_path_arg, provider_arg, model_arg):
        calls.append(model_arg)
        return RecognitionResult(
            value="3481",
            normalized_value="3481",
            confidence=0.91,
            provider="aliyun",
            model=model_arg,
            raw={"content": "{}"},
        )

    monkeypatch.setattr("formcheck.pipeline.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setenv("ROI_REVIEW_CACHE_ENABLED", "1")
    monkeypatch.setattr("formcheck.pipeline.recognize_with_provider", fake_recognize)

    first = recognize_roi_with_fallback(field, roi_path, "aliyun", "qwen3.5-ocr")
    second = recognize_roi_with_fallback(field, roi_path, "aliyun", "qwen3.5-ocr")

    assert calls == ["qwen3.5-ocr"]
    assert first.normalized_value == "3481"
    assert second.normalized_value == "3481"
    assert second.raw["roi_review_cache_hit"] is True


def test_roi_review_cache_is_model_specific(tmp_path, monkeypatch) -> None:
    field = FieldSpec(
        id="awr_license",
        label="适航放行-执照号",
        section="airworthiness_release",
        bbox=(0, 0, 1, 1),
        recognizer="keyed_text",
        validator="regex",
        params={"pattern": r"^CAACML[0-9]{8}$"},
        fail_msg="执照号不合规",
    )
    roi_path = tmp_path / "license.png"
    roi_path.write_bytes(b"license-roi")
    calls = []

    def fake_recognize(field_arg, roi_path_arg, provider_arg, model_arg):
        calls.append(model_arg)
        return RecognitionResult(
            value=model_arg,
            normalized_value=model_arg,
            confidence=0.8,
            provider="aliyun",
            model=model_arg,
            raw={"content": "{}"},
        )

    monkeypatch.setattr("formcheck.pipeline.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr("formcheck.pipeline.recognize_with_provider", fake_recognize)

    one = recognize_roi_with_fallback(field, roi_path, "aliyun", "model-a")
    two = recognize_roi_with_fallback(field, roi_path, "aliyun", "model-b")

    assert calls == ["model-a", "model-b"]
    assert one.normalized_value == "model-a"
    assert two.normalized_value == "model-b"


def test_roi_review_cache_can_be_disabled(tmp_path, monkeypatch) -> None:
    field = FieldSpec(
        id="apu_cum_cycles",
        label="APU累计使用循环",
        section="apu",
        bbox=(690, 1678, 230, 100),
        recognizer="numeric_text",
        validator="digit_length",
        params={"allow_lengths": [4, 5]},
        fail_msg="APU循环不是4或5位",
    )
    roi_path = tmp_path / "roi.png"
    roi_path.write_bytes(b"cache-disabled")
    calls = []

    def fake_recognize(field_arg, roi_path_arg, provider_arg, model_arg):
        calls.append(model_arg)
        return RecognitionResult(
            value="3481",
            normalized_value="3481",
            confidence=0.91,
            provider="aliyun",
            model=model_arg,
            raw={"content": "{}"},
        )

    monkeypatch.setattr("formcheck.pipeline.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setenv("ROI_REVIEW_CACHE_ENABLED", "0")
    monkeypatch.setattr("formcheck.pipeline.recognize_with_provider", fake_recognize)

    first = recognize_roi_with_fallback(field, roi_path, "aliyun", "qwen3.5-ocr")
    second = recognize_roi_with_fallback(field, roi_path, "aliyun", "qwen3.5-ocr")

    assert calls == ["qwen3.5-ocr", "qwen3.5-ocr"]
    assert first.normalized_value == "3481"
    assert second.normalized_value == "3481"
    assert "roi_review_cache_hit" not in (second.raw or {})


def test_roi_review_errors_are_not_cached(tmp_path, monkeypatch) -> None:
    field = FieldSpec(
        id="action_authorization",
        label="授权号",
        section="fault_action",
        bbox=(0, 0, 1, 1),
        recognizer="keyed_text",
        validator="regex",
        params={"pattern": r"^[0-9]{6}$"},
        fail_msg="授权号不是6位数字",
    )
    roi_path = tmp_path / "auth.png"
    roi_path.write_bytes(b"auth-roi")
    calls = []

    def fake_recognize(field_arg, roi_path_arg, provider_arg, model_arg):
        calls.append(model_arg)
        return RecognitionResult(
            value="",
            normalized_value="",
            confidence=0.0,
            provider="aliyun",
            model=model_arg,
            raw={"error": "timeout"},
        )

    monkeypatch.setattr("formcheck.pipeline.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setenv("ROI_REVIEW_FALLBACK_MODEL", "")
    monkeypatch.setattr("formcheck.pipeline.recognize_with_provider", fake_recognize)

    recognize_roi_with_fallback(field, roi_path, "aliyun", "qwen3.5-ocr")
    recognize_roi_with_fallback(field, roi_path, "aliyun", "qwen3.5-ocr")

    assert calls == ["qwen3.5-ocr", "qwen3.5-ocr"]
