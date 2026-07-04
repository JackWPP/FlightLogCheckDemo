from __future__ import annotations

from formcheck.schemas import FieldSpec, RecognitionResult
from formcheck.validators import validate


def field(validator: str, params: dict, fail_msg: str = "fail") -> FieldSpec:
    return FieldSpec("f", "字段", "s", (0, 0, 1, 1), "text", validator, params, fail_msg)


def test_int_range() -> None:
    ok, _ = validate(field("int_range", {"min": 0, "max": 4}), RecognitionResult("3"))
    assert ok
    ok, msg = validate(field("int_range", {"min": 0, "max": 4}, "范围错"), RecognitionResult("5"))
    assert not ok and msg == "范围错"


def test_regex_license() -> None:
    spec = field("regex", {"pattern": r"^CAACML[0-9]{8}$"})
    assert validate(spec, RecognitionResult("CAACML12345678"))[0]
    assert not validate(spec, RecognitionResult("CAAC123"))[0]


def test_same_day_injected() -> None:
    spec = field("same_day", {})
    assert validate(spec, RecognitionResult("2026.07.02"), now="2026-07-02")[0]
    assert not validate(spec, RecognitionResult("2026-07-01"), now="2026-07-02")[0]


def test_english_text() -> None:
    spec = field("english_text", {})
    assert validate(spec, RecognitionResult("LND AFT CAG ISOL VALVE"))[0]
    assert not validate(spec, RecognitionResult("故障 message"))[0]


def test_bilingual_text() -> None:
    spec = field("bilingual_text", {"min_letters": 1, "min_cjk": 1})
    assert validate(spec, RecognitionResult("更换氧气瓶 replace oxygen cylinder"))[0]
    assert not validate(spec, RecognitionResult("replace oxygen cylinder"))[0]
    assert not validate(spec, RecognitionResult("更换氧气瓶"))[0]


def test_name_not_place() -> None:
    spec = field("name_not_place", {"not_allow": ["重庆", "Chongqing"]})
    assert validate(spec, RecognitionResult("李四"))[0]
    assert not validate(spec, RecognitionResult("重庆"))[0]
    assert not validate(spec, RecognitionResult(""))[0]


def test_digit_length() -> None:
    spec = field("digit_length", {"allow_lengths": [4, 5]})
    assert validate(spec, RecognitionResult("2238"))[0]
    assert validate(spec, RecognitionResult("12345"))[0]
    assert not validate(spec, RecognitionResult("123"))[0]


def test_number_less_than() -> None:
    spec = field("number_less_than", {"max": 99999})
    assert validate(spec, RecognitionResult("99888"))[0]
    assert not validate(spec, RecognitionResult("99999"))[0]
    assert not validate(spec, RecognitionResult(""))[0]
