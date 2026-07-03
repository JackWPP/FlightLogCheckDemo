from __future__ import annotations

from formcheck.field_assignment import assign_blocks_to_fields
import requests

from formcheck.llm_cleaner import clean_field_values_with_meta, fallback_clean, sanitize_english_text
from formcheck.schemas import FieldCandidate, FieldSpec, OcrBlock


def test_fallback_clean_uses_best_candidate() -> None:
    field = FieldSpec(
        id="fault_ref",
        label="处理措施-参考手册",
        section="fault_action",
        bbox=(400, 400, 300, 80),
        recognizer="keyed_text",
        validator="prefix_or_exact",
        params={"prefixes": ["AMM", "TSM"], "allow_exact": ["N/A"]},
        fail_msg="fail",
    )
    block = OcrBlock("b0002", "TSM21-28-00-810-802A", 0.9, (410, 410, 620, 450), (515, 430))
    edge = OcrBlock("edge", "x", 0.1, (950, 950, 990, 990), (970, 970))
    assignments = assign_blocks_to_fields([field], [block, edge], (1000, 1000))

    result = fallback_clean([field], assignments, "mock", "cleaner")["fault_ref"]

    assert result.normalized_value == "TSM21-28-00-810-802A"
    assert result.confidence and result.confidence > 0


def test_fallback_clean_filters_chinese_noise_from_english_field() -> None:
    field = FieldSpec(
        id="fault_report_en",
        label="故障报告-英文内容",
        section="fault_action",
        bbox=(0, 0, 500, 100),
        recognizer="free_text",
        validator="english_text",
        params={"min_letters": 1},
        fail_msg="fail",
    )
    blocks = [
        OcrBlock("b0001", "库信息EW:", 0.95, (10, 10, 80, 30), (45, 20)),
        OcrBlock("b0002", "COND AFT CRG ISOL VALVE", 0.95, (90, 10, 300, 30), (195, 20)),
    ]
    assignments = {
        field.id: [
            FieldCandidate(field.id, blocks[0], 1.0, "test"),
            FieldCandidate(field.id, blocks[1], 0.9, "test"),
        ]
    }

    result = fallback_clean([field], assignments, "mock", "cleaner")["fault_report_en"]

    assert result.normalized_value == "EW: COND AFT CRG ISOL VALVE"
    assert "库" not in result.normalized_value


def test_sanitize_english_text_removes_cjk_tokens() -> None:
    assert sanitize_english_text("FaultmesSge:E/W. COND AFT CRG ISOL VALVE 库信息EW:") == (
        "FaultmesSge:E/W. COND AFT CRG ISOL VALVE EW:"
    )


def test_cleaner_timeout_fallback_is_cached(tmp_path, monkeypatch) -> None:
    field = FieldSpec(
        id="fault_ref",
        label="处理措施-参考手册",
        section="fault_action",
        bbox=(400, 400, 300, 80),
        recognizer="keyed_text",
        validator="prefix_or_exact",
        params={"prefixes": ["AMM", "TSM"], "allow_exact": ["N/A"]},
        fail_msg="fail",
    )
    block = OcrBlock("b0002", "TSM21-28-00-810-802A", 0.9, (410, 410, 620, 450), (515, 430))
    assignments = {field.id: [FieldCandidate(field.id, block, 1.0, "test")]}

    monkeypatch.setattr("formcheck.llm_cleaner.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")

    def timeout(*args, **kwargs):
        raise requests.Timeout("read timeout")

    monkeypatch.setattr("formcheck.llm_cleaner.requests.post", timeout)

    first, first_meta = clean_field_values_with_meta([field], assignments, provider="siliconflow", model="cleaner")
    second, second_meta = clean_field_values_with_meta([field], assignments, provider="siliconflow", model="cleaner")

    assert first[field.id].normalized_value == "TSM21-28-00-810-802A"
    assert "cleaner_error" in first[field.id].raw
    assert first_meta["fallback_cached"] is True
    assert second[field.id].normalized_value == "TSM21-28-00-810-802A"
    assert second_meta["cache_hit"] is True
