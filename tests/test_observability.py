from __future__ import annotations

import json
import logging

from formcheck.observability import JsonFormatter


def test_json_formatter_keeps_list_and_dict_payload_values() -> None:
    record = logging.LogRecord(
        name="formcheck",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="event",
        args=(),
        exc_info=None,
    )
    record.event = "ocr.done"
    record.payload = {
        "section_results": {"oil": "ok"},
        "candidates": [{"text": "20.5"}],
        "empty": "",
        "none_value": None,
    }

    formatted = JsonFormatter().format(record)
    payload = json.loads(formatted)

    assert payload["section_results"] == {"oil": "ok"}
    assert payload["candidates"] == [{"text": "20.5"}]
    assert "empty" not in payload
    assert "none_value" not in payload
