from __future__ import annotations

from formcheck.field_assignment import assign_blocks_to_fields
from formcheck.llm_cleaner import fallback_clean
from formcheck.schemas import FieldSpec, OcrBlock


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
