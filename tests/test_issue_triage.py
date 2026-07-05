from __future__ import annotations

from formcheck.issue_triage import build_triage_prompt, evidence_state, fallback_triage, issue_risk_level, triage_issues


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


def test_issue_triage_skips_llm_when_cleaner_failed(monkeypatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    fields = [
        {"id": "date", "label": "日期", "validator": "same_day", "message": "日期不是今天", "passed": False},
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("triage LLM should be skipped")

    monkeypatch.setattr("formcheck.issue_triage.requests.post", fail_if_called)
    result = triage_issues(
        fields,
        provider="siliconflow",
        cleaner_meta={
            "cleaner_error": "timeout",
            "cleaner_section_meta": {"fallback_sections": ["apu"]},
        },
    )

    assert result["problems"] == ["日期不是今天"]
    assert result["issue_triage"]["provider"] == "local:fallback"
    assert result["issue_triage"]["reason"] == "Cleaner异常，跳过LLM问题压缩"


def test_issue_triage_prompt_includes_risk_and_evidence_state() -> None:
    fields = [
        {
            "id": "license",
            "label": "29 适航放行-执照号",
            "validator": "regex",
            "message": "执照号不合规",
            "passed": False,
            "needs_review": True,
            "review_reason": "超过ROI复核预算，保留人工复核",
            "raw": {"roi_review_skipped": {"reason": "budget_exceeded"}},
        }
    ]

    prompt = build_triage_prompt(fields, 4)

    assert '"risk_level": "high"' in prompt
    assert '"evidence_state": "roi_review_skipped"' in prompt
    assert "ROI 因预算跳过" in prompt
    assert issue_risk_level(fields[0]) == "high"
    assert evidence_state(fields[0]) == "roi_review_skipped"


def test_fallback_triage_prefers_high_risk_review_skipped_item(monkeypatch) -> None:
    monkeypatch.setenv("ISSUE_DISPLAY_LIMIT", "1")
    fields = [
        {
            "id": "station",
            "label": "15 处理措施-地点",
            "validator": "exact_text",
            "message": "地点不是重庆",
            "passed": False,
        },
        {
            "id": "license",
            "label": "29 适航放行-执照号",
            "validator": "regex",
            "message": "执照号不合规",
            "passed": False,
            "needs_review": True,
            "raw": {"roi_review_skipped": {"reason": "budget_exceeded"}},
        },
    ]

    result = triage_issues(fields, provider="mock")

    assert result["problems"] == ["执照号不合规"]
