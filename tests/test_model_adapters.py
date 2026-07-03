from __future__ import annotations

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
