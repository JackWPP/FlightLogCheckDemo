from __future__ import annotations

import cv2
import numpy as np

from formcheck.pipeline import build_ppocr_roi_evidence, registration_mode, should_roi_review
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
