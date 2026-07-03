from __future__ import annotations

import cv2
import numpy as np

from formcheck.pipeline import analyze_image


def test_no_key_report_keeps_fine_grained_timings(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("formcheck.pipeline.OUT_DIR", tmp_path / "out")
    monkeypatch.delenv("PADDLEOCR_AISTUDIO_TOKEN", raising=False)
    image = tmp_path / "upload.jpg"
    cv2.imwrite(str(image), np.full((30, 40, 3), 255, dtype=np.uint8))

    report = analyze_image(image, run_id="timing-test")

    timings = report["timings"]
    assert "ppocr_submit_ms" in timings
    assert "ppocr_poll_ms" in timings
    assert "assignment_ms" in timings
    assert "cleaner_ms" in timings
    assert "issue_triage_ms" in timings
    assert report["all_problems"]
    assert len(report["problems"]) <= 4
