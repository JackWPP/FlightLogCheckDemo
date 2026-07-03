from __future__ import annotations

import cv2
import numpy as np

from formcheck.ppocr_pipeline import extract_ppocr_blocks, load_ppocr_cache, ppocr_cache_dir, save_ppocr_cache


def test_extract_ppocr_blocks() -> None:
    records = [
        {
            "result": {
                "ocrResults": [
                    {
                        "prunedResult": {
                            "rec_texts": ["TSM21-28-00-810-802A", "2238"],
                            "rec_scores": [0.87, 0.99],
                            "rec_boxes": [[10, 20, 110, 40], [50, 80, 90, 120]],
                        }
                    }
                ]
            }
        }
    ]

    blocks = extract_ppocr_blocks(records)

    assert [block.text for block in blocks] == ["TSM21-28-00-810-802A", "2238"]
    assert blocks[0].id == "b0000"
    assert blocks[1].center == (70.0, 100.0)


def test_ppocr_cache_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("formcheck.ppocr_pipeline.OUTPUTS_DIR", tmp_path / "outputs")
    image = tmp_path / "upload.jpg"
    image.write_bytes(b"same image")
    ocr_image = tmp_path / "ocr_image_0.jpg"
    cv2.imwrite(str(ocr_image), np.full((10, 12, 3), 255, dtype=np.uint8))
    blocks_json = [
        {
            "id": "b0000",
            "text": "3481",
            "score": 0.99,
            "box": [1, 2, 8, 9],
            "center": [4.5, 5.5],
            "source": "ppocrv6",
        }
    ]
    optional_payload = {"useDocUnwarping": True}
    cache_dir = ppocr_cache_dir(image, optional_payload)

    save_ppocr_cache(cache_dir, blocks_json, ocr_image, {"job_id": "job-1"})
    cached = load_ppocr_cache(cache_dir, tmp_path / "run" / "ppocrv6")

    assert cached is not None
    assert cached["cache_hit"] is True
    assert cached["job_id"] == "cache"
    assert [block.text for block in cached["blocks"]] == ["3481"]
    assert cached["ocr_image_path"].exists()
