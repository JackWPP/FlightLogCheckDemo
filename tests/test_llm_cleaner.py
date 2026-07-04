from __future__ import annotations

from formcheck.field_assignment import assign_blocks_to_fields
import requests

from formcheck.llm_cleaner import clean_field_values_with_meta, cleaner_sections, fallback_clean, normalize_for_field, sanitize_english_text
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


def test_fallback_clean_bilingual_filters_form_labels_and_merges_body() -> None:
    field = FieldSpec(
        id="action_description_bilingual",
        label="处理措施-中英文内容",
        section="fault_action",
        bbox=(0, 0, 500, 200),
        recognizer="free_text",
        validator="bilingual_text",
        params={"min_letters": 1, "min_cjk": 1},
        fail_msg="fail",
    )
    blocks = [
        OcrBlock("label", "安装件号P/N ON", 0.99, (0, 0, 100, 20), (50, 10)),
        OcrBlock("zh", "完成飞机遭鸟击后的检查，检查正常", 0.9, (0, 40, 250, 70), (125, 55)),
        OcrBlock("en", "Finished the inspection after a Bird strike. check o/k", 0.9, (0, 80, 450, 110), (225, 95)),
    ]
    assignments = {
        field.id: [
            FieldCandidate(field.id, blocks[0], 1.2, "label"),
            FieldCandidate(field.id, blocks[1], 1.0, "body"),
            FieldCandidate(field.id, blocks[2], 0.9, "body"),
        ]
    }

    result = fallback_clean([field], assignments, "mock", "cleaner")[field.id]

    assert "安装件号" not in result.normalized_value
    assert "完成飞机" in result.normalized_value
    assert "Finished" in result.normalized_value


def test_normalize_for_field_station_and_na_variants() -> None:
    station = FieldSpec("station", "地点", "s", (0, 0, 1, 1), "keyed_text", "exact_text", {"allow": ["重庆"]}, "fail")
    ref = FieldSpec("ref", "参考手册", "s", (0, 0, 1, 1), "keyed_text", "prefix_or_exact", {"allow_exact": ["N/A"]}, "fail")

    assert normalize_for_field(station, "渝") == "重庆"
    assert normalize_for_field(ref, "N-A") == "N/A"


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


def test_cleaner_sections_skip_checkbox_and_empty_candidates() -> None:
    text_field = FieldSpec("fault_ref", "参考手册", "fault_action", (0, 0, 1, 1), "keyed_text", "prefix_or_exact", {}, "fail")
    empty_field = FieldSpec("apu_cum_cycles", "APU循环", "apu", (0, 0, 1, 1), "numeric_text", "digit_length", {}, "fail")
    checkbox = FieldSpec("crew_checkbox", "机组", "fault_action", (0, 0, 1, 1), "checkbox", "checked", {}, "fail")
    block = OcrBlock("b0001", "TSM21", 0.9, (0, 0, 10, 10), (5, 5))
    assignments = {text_field.id: [FieldCandidate(text_field.id, block, 1.0, "test")]}

    sections = cleaner_sections([text_field, empty_field, checkbox], assignments)

    assert list(sections) == ["fault_action"]
    assert sections["fault_action"] == [text_field]


def test_section_cleaner_success_and_timeout_are_merged(tmp_path, monkeypatch) -> None:
    fault = FieldSpec("fault_ref", "参考手册", "fault_action", (0, 0, 1, 1), "keyed_text", "prefix_or_exact", {}, "fail")
    apu = FieldSpec("apu_cum_cycles", "APU循环", "apu", (0, 0, 1, 1), "numeric_text", "digit_length", {"allow_lengths": [4, 5]}, "fail")
    fault_block = OcrBlock("b0001", "TSM21-28", 0.9, (0, 0, 10, 10), (5, 5))
    apu_block = OcrBlock("b0002", "348", 0.9, (0, 0, 10, 10), (5, 5))
    assignments = {
        fault.id: [FieldCandidate(fault.id, fault_block, 1.0, "test")],
        apu.id: [FieldCandidate(apu.id, apu_block, 1.0, "test")],
    }

    monkeypatch.setattr("formcheck.llm_cleaner.OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("CLEANER_SECTION_CONCURRENCY", "2")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"fields":{"fault_ref":{"value":"TSM21-28","normalized_value":"TSM21-28","confidence":0.93,"needs_review":false,"reason":""}}}'
                        }
                    }
                ],
                "usage": {"total_tokens": 10},
            }

    def fake_post(url, headers, json, timeout):
        content = json["messages"][0]["content"]
        if "当前分区: apu" in content:
            raise requests.Timeout("apu timeout")
        return Response()

    monkeypatch.setattr("formcheck.llm_cleaner.requests.post", fake_post)

    results, meta = clean_field_values_with_meta([fault, apu], assignments, provider="siliconflow", model="cleaner")

    assert results[fault.id].provider == "siliconflow"
    assert results[fault.id].normalized_value == "TSM21-28"
    assert results[apu.id].provider == "siliconflow:fallback_cleaner"
    assert results[apu.id].normalized_value == "348"
    assert meta["section_results"]["fault_action"] == "ok"
    assert meta["section_results"]["apu"] == "fallback"
    assert meta["fallback_sections"] == ["apu"]
    assert "apu timeout" in meta["section_errors"]["apu"]
