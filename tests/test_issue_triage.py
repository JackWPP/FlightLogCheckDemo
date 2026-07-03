from __future__ import annotations

from formcheck.issue_triage import fallback_triage, triage_issues


def test_issue_triage_fallback_limits_display_and_keeps_all(monkeypatch) -> None:
    monkeypatch.setenv("ISSUE_DISPLAY_LIMIT", "2")
    fields = [
        {"id": "station", "label": "地点", "validator": "exact_text", "message": "地点不是重庆", "passed": False},
        {"id": "license", "label": "执照号", "validator": "regex", "message": "执照号不合规", "passed": False},
        {"id": "date", "label": "日期", "validator": "same_day", "message": "日期不是今天", "passed": False},
    ]

    result = triage_issues(fields, provider="mock")

    assert result["all_problems"] == ["地点不是重庆", "执照号不合规", "日期不是今天"]
    assert len(result["problems"]) == 2
    assert set(result["problems"]) == {"执照号不合规", "日期不是今天"}
    assert result["issue_triage"]["suppressed"]


def test_fallback_triage_returns_pass_when_no_failures() -> None:
    result = fallback_triage([], [], 4)

    assert result["problems"] == ["通过"]
    assert result["all_problems"] == []
