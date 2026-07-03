from __future__ import annotations

from pathlib import Path

from formcheck.fields import load_fields


def test_fields_complete() -> None:
    canonical, fields = load_fields(Path("fields.yaml"))
    assert canonical["width"] == 2400
    assert canonical["height"] > 1000
    assert len(fields) >= 20
    for item in fields:
        x, y, w, h = item.bbox
        assert item.id
        assert item.label
        assert item.section
        assert item.recognizer
        assert item.validator
        assert item.fail_msg
        assert isinstance(item.assignment, dict)
        assert x >= 0 and y >= 0 and w > 0 and h > 0
        assert x + w <= canonical["width"]
        assert y + h <= canonical["height"]
        search_bbox = item.assignment.get("search_bbox")
        if search_bbox:
            sx, sy, sw, sh = search_bbox
            assert sx >= 0 and sy >= 0 and sw > 0 and sh > 0
            assert sx + sw <= canonical["width"]
            assert sy + sh <= canonical["height"]
