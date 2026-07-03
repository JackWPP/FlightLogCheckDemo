from __future__ import annotations

from PIL import Image

from formcheck.model_adapters import image_payload
from formcheck.model_adapters import request_timeout_seconds


def test_vlm_request_timeout_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("VLM_REQUEST_TIMEOUT_SECONDS", "2")
    assert request_timeout_seconds() == 5

    monkeypatch.setenv("VLM_REQUEST_TIMEOUT_SECONDS", "999")
    assert request_timeout_seconds() == 180

    monkeypatch.setenv("VLM_REQUEST_TIMEOUT_SECONDS", "31")
    assert request_timeout_seconds() == 31


def test_vlm_request_timeout_falls_back_on_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("VLM_REQUEST_TIMEOUT_SECONDS", "slow")
    assert request_timeout_seconds() == 45


def test_image_payload_resizes_large_roi(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VLM_IMAGE_MAX_SIDE", "400")
    path = tmp_path / "roi.png"
    Image.new("RGB", (1200, 250), "white").save(path)

    payload = image_payload(path)
    out = tmp_path / "encoded.png"
    out.write_bytes(payload)

    with Image.open(out) as image:
        assert max(image.size) == 400
