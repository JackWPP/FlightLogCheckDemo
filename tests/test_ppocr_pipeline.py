from __future__ import annotations

from formcheck.ppocr_pipeline import extract_ppocr_blocks


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
