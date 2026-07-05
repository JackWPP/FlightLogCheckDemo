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


def test_station_alias_enters_exact_text_candidates() -> None:
    field = FieldSpec(
        id="action_station",
        label="处理地点",
        section="fault_action",
        bbox=(0, 0, 120, 80),
        recognizer="keyed_text",
        validator="exact_text",
        params={"allow": ["重庆", "渝", "Chongqing"]},
        fail_msg="fail",
    )
    block = OcrBlock("station", "渝", 0.98, (10, 10, 60, 55), (35, 32))

    assignments = assign_blocks_to_fields([field], [block], (120, 80))

    assert assignments[field.id][0].block.text == "渝"


def test_na_variant_enters_exact_text_candidates() -> None:
    field = FieldSpec(
        id="fault_flight_no",
        label="航班号",
        section="fault_action",
        bbox=(0, 0, 120, 80),
        recognizer="keyed_text",
        validator="exact_text",
        params={"allow": ["N/A", "NA"]},
        fail_msg="fail",
    )
    block = OcrBlock("na", "N-A", 0.98, (10, 10, 80, 55), (45, 32))

    assignments = assign_blocks_to_fields([field], [block], (120, 80))

    assert assignments[field.id][0].block.text == "N-A"


def test_prefix_or_exact_retains_noncompliant_reference_as_evidence() -> None:
    field = FieldSpec(
        id="action_ref",
        label="参考手册",
        section="fault_action",
        bbox=(0, 0, 300, 80),
        recognizer="keyed_text",
        validator="prefix_or_exact",
        params={"prefixes": ["AMM", "TSM"], "allow_exact": ["N/A", "NA"]},
        fail_msg="fail",
    )
    block = OcrBlock("fla", "FLA320-05-51-14-200-803-A-X", 0.95, (10, 10, 260, 55), (135, 32))

    assignments = assign_blocks_to_fields([field], [block], (300, 80))

    assert assignments[field.id][0].block.text.startswith("FLA320")


def test_bilingual_assignment_filters_form_labels_before_body() -> None:
    field = FieldSpec(
        id="action_description_bilingual",
        label="处理措施-中英文内容",
        section="fault_action",
        bbox=(0, 0, 800, 220),
        recognizer="free_text",
        validator="bilingual_text",
        params={"min_letters": 1, "min_cjk": 1},
        fail_msg="fail",
    )
    blocks = [
        OcrBlock("label1", "安装件号P/N ON", 0.99, (20, 20, 180, 50), (100, 35)),
        OcrBlock("label2", "工作者签名SIGN", 0.99, (190, 20, 350, 50), (270, 35)),
        OcrBlock("zh", "完成飞机遭鸟击后的检查，检查正常", 0.93, (40, 80, 520, 120), (280, 100)),
        OcrBlock("en", "Finished the inspection after a Bird strike. check o/k", 0.93, (40, 130, 760, 170), (400, 150)),
    ]

    assignments = assign_blocks_to_fields([field], blocks, (800, 220))
    texts = [candidate.block.text for candidate in assignments[field.id]]

    assert "安装件号P/N ON" not in texts
    assert "工作者签名SIGN" not in texts
    assert texts[:2] == ["Finished the inspection after a Bird strike. check o/k", "完成飞机遭鸟击后的检查，检查正常"] or texts[:2] == [
        "完成飞机遭鸟击后的检查，检查正常",
        "Finished the inspection after a Bird strike. check o/k",
    ]


def test_signature_rejects_station_and_release_statement() -> None:
    field = FieldSpec(
        id="action_release_sign",
        label="处理措施-放行签署",
        section="fault_action",
        bbox=(0, 0, 240, 100),
        recognizer="signature_or_text",
        validator="name_not_place",
        params={"not_allow": ["重庆", "渝", "Chongqing"]},
        fail_msg="fail",
        assignment={"value_type": "signature"},
    )
    station = OcrBlock("station", "重庆", 0.99, (20, 10, 80, 60), (50, 35))
    statement = OcrBlock(
        "statement",
        "飞机技术状态满足适航要求，适合飞行，The A/C is considered fit for Release to Service",
        0.99,
        (20, 10, 220, 60),
        (120, 35),
    )
    name = OcrBlock("name", "李冬", 0.9, (110, 10, 170, 60), (140, 35))

    assignments = assign_blocks_to_fields([field], [station, statement, name], (240, 100))

    assert [candidate.block.text for candidate in assignments[field.id]] == ["李冬"]


def test_authorization_rejects_date_candidate() -> None:
    field = FieldSpec(
        id="action_authorization",
        label="授权号",
        section="fault_action",
        bbox=(0, 0, 260, 100),
        recognizer="keyed_text",
        validator="regex",
        params={"pattern": "^[0-9]{6}$"},
        fail_msg="fail",
        assignment={"value_type": "authorization"},
    )
    date = OcrBlock("date", "2026.06.29", 0.99, (20, 10, 170, 60), (95, 35))
    auth = OcrBlock("auth", "017867", 0.88, (170, 10, 250, 60), (210, 35))

    assignments = assign_blocks_to_fields([field], [date, auth], (260, 100))

    assert assignments[field.id][0].block.text == "017867"
    assert "2026.06.29" not in [candidate.block.text for candidate in assignments[field.id]]


def test_authorization_accepts_mixed_text_when_six_digit_value_exists() -> None:
    field = FieldSpec(
        id="action_authorization",
        label="授权号",
        section="fault_action",
        bbox=(0, 0, 260, 100),
        recognizer="keyed_text",
        validator="regex",
        params={"pattern": "^[0-9]{6}$"},
        fail_msg="fail",
        assignment={"value_type": "authorization"},
    )
    mixed = OcrBlock("mixed", "2026.06.29 017867", 0.99, (20, 10, 250, 60), (135, 35))

    assignments = assign_blocks_to_fields([field], [mixed], (260, 100))

    assert assignments[field.id][0].block.text == "2026.06.29 017867"


def test_text_field_rejects_station_only_candidate() -> None:
    field = FieldSpec(
        id="action_installed_part_info",
        label="安装件信息",
        section="fault_action",
        bbox=(0, 0, 200, 100),
        recognizer="keyed_text",
        validator="present",
        params={},
        fail_msg="fail",
        assignment={"value_type": "text"},
    )
    station = OcrBlock("station", "重庆", 0.99, (20, 10, 80, 60), (50, 35))

    assignments = assign_blocks_to_fields([field], [station], (200, 100))

    assert assignments[field.id] == []


def test_shared_primary_block_is_kept_for_best_matching_field_only() -> None:
    station_field = FieldSpec(
        id="action_station",
        label="处理地点",
        section="fault_action",
        bbox=(0, 0, 120, 80),
        recognizer="keyed_text",
        validator="exact_text",
        params={"allow": ["重庆", "渝", "Chongqing"]},
        fail_msg="fail",
    )
    sign_field = FieldSpec(
        id="action_release_sign",
        label="放行签署",
        section="fault_action",
        bbox=(0, 0, 120, 80),
        recognizer="signature_or_text",
        validator="name_not_place",
        params={"not_allow": ["重庆", "渝", "Chongqing"]},
        fail_msg="fail",
        assignment={"value_type": "signature"},
    )
    station = OcrBlock("station", "重庆", 0.99, (10, 10, 90, 60), (50, 35))

    assignments = assign_blocks_to_fields([station_field, sign_field], [station], (120, 80))

    assert assignments[station_field.id][0].block.text == "重庆"
    assert assignments[sign_field.id] == []
