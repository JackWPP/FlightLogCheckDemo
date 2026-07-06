from __future__ import annotations

from PIL import Image

from formcheck.model_adapters import build_prompt, image_payload, parsed_confidence
from formcheck.model_adapters import request_timeout_seconds
from formcheck.schemas import FieldSpec


def test_vlm_request_timeout_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("VLM_REQUEST_TIMEOUT_SECONDS", "2")
    assert request_timeout_seconds() == 5

    monkeypatch.setenv("VLM_REQUEST_TIMEOUT_SECONDS", "999")
    assert request_timeout_seconds() == 180

    monkeypatch.setenv("VLM_REQUEST_TIMEOUT_SECONDS", "31")
    assert request_timeout_seconds() == 31


def test_vlm_request_timeout_falls_back_on_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("VLM_REQUEST_TIMEOUT_SECONDS", "slow")
    assert request_timeout_seconds() == 25


def test_ocr_model_value_gets_conservative_default_confidence() -> None:
    assert parsed_confidence({"confidence": "0.0"}, "qwen3.5-ocr", "20.5", "") == 0.72
    assert parsed_confidence({"confidence": "0.0"}, "qwen3.7-plus", "20.5", "") == 0.0
    assert parsed_confidence({"confidence": "0.81"}, "qwen3.5-ocr", "20.5", "") == 0.81


def test_image_payload_resizes_large_roi(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLM_IMAGE_MAX_SIDE", "400")
    path = tmp_path / "roi.png"
    Image.new("RGB", (1200, 250), "white").save(path)

    payload = image_payload(path)
    out = tmp_path / "encoded.png"
    out.write_bytes(payload)

    with Image.open(out) as image:
        assert max(image.size) == 400


def test_numeric_vlm_prompt_includes_range_and_neighbor_guard() -> None:
    field = FieldSpec(
        id="oil_eng1_qty",
        label="03 发动机1滑油量",
        section="oil",
        bbox=(310, 635, 155, 58),
        recognizer="numeric_text",
        validator="int_range",
        params={"min": 15, "max": 25},
        fail_msg="滑油量不在15-25",
    )

    prompt = build_prompt(field)

    assert "15 到 25" in prompt
    assert "多个相邻格" in prompt
    assert "目标格" in prompt


def test_signature_and_license_prompts_warn_about_neighbor_values() -> None:
    signature = FieldSpec(
        id="awr_release_sign",
        label="28 适航放行-放行签署",
        section="airworthiness_release",
        bbox=(0, 0, 1, 1),
        recognizer="signature_or_text",
        validator="name_not_place",
        params={},
        fail_msg="fail",
    )
    license_field = FieldSpec(
        id="awr_license",
        label="29 适航放行-执照号",
        section="airworthiness_release",
        bbox=(0, 0, 1, 1),
        recognizer="keyed_text",
        validator="regex",
        params={"pattern": r"^CAACML[0-9]{8}$"},
        fail_msg="fail",
    )

    assert "不要把重庆" in build_prompt(signature)
    license_prompt = build_prompt(license_field)
    assert "CAACML" in license_prompt
    assert "忽略地点" in license_prompt
