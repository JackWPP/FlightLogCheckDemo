from __future__ import annotations

from formcheck.demo import demo_payload


def test_demo_payload_loads_cached_report() -> None:
    payload = demo_payload()
    report = payload["report"]

    assert payload["manifest"]["images"]["ocr"].endswith("ocr_image.jpg")
    assert report["run_id"] == "demo_sample"
    assert report["upload_url"].startswith("/outputs/demo_sample/")
    assert report["ocr"]["blocks"]
    assert report["summary"]["field_count"] == len(report["fields"])
    assert all(
        field["roi_url"].startswith(("/outputs/demo_sample/ppocr_rois/", "/outputs/demo_sample/rois/"))
        for field in report["fields"]
    )
