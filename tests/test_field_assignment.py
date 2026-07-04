from __future__ import annotations

from formcheck.field_assignment import assign_blocks_to_fields
from formcheck.schemas import FieldSpec, OcrBlock


def test_assigns_block_to_scaled_field_bbox() -> None:
    field = FieldSpec(
        id="apu_cum_hours",
        label="APU累计使用时间",
        section="apu",
        bbox=(500, 1600, 200, 100),
        recognizer="numeric_text",
        validator="digit_length",
        params={"allow_lengths": [4, 5]},
        fail_msg="fail",
    )
    block = OcrBlock("b0001", "2238", 0.99, (520, 1610, 600, 1680), (560, 1645))
    edge = OcrBlock("edge", "x", 0.1, (950, 1950, 990, 1990), (970, 1970))

    assignments = assign_blocks_to_fields([field], [block, edge], (1000, 2000))

    assert assignments["apu_cum_hours"]
    assert assignments["apu_cum_hours"][0].block.text == "2238"


def test_row_last_prefers_pf_row_candidate() -> None:
    field = FieldSpec(
        id="oil_eng1_added",
        label="发动机1加注量",
        section="oil",
        bbox=(0, 0, 100, 100),
        recognizer="numeric_text",
        validator="int_range",
        params={"min": 0, "max": 4},
        fail_msg="fail",
        assignment={"row": "last"},
    )
    upper_row = OcrBlock("upper", "4", 0.99, (10, 20, 30, 40), (20, 30))
    pf_row = OcrBlock("pf", "0", 0.99, (20, 72, 42, 92), (31, 82))

    assignments = assign_blocks_to_fields([field], [upper_row, pf_row], (100, 100))

    assert assignments[field.id][0].block.text == "0"


def test_int_range_penalizes_neighbor_column_value() -> None:
    field = FieldSpec(
        id="oil_eng1_qty",
        label="发动机1滑油量",
        section="oil",
        bbox=(0, 0, 100, 100),
        recognizer="numeric_text",
        validator="int_range",
        params={"min": 15, "max": 25},
        fail_msg="fail",
    )
    neighbor_added = OcrBlock("neighbor", "0", 0.99, (10, 10, 95, 90), (52, 50))
    oil_qty = OcrBlock("qty", "19.5", 0.99, (76, 10, 130, 90), (103, 50))

    assignments = assign_blocks_to_fields([field], [neighbor_added, oil_qty], (100, 100))

    assert assignments[field.id][0].block.text == "19.5"
