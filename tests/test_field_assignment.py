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
