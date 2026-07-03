from __future__ import annotations

import json
from pathlib import Path

from formcheck.config import CANONICAL_DIR, FIELDS_PATH
from formcheck.crop import make_roi_grid
from formcheck.fields import load_fields
from formcheck.image_io import imread
from formcheck.pipeline import analyze_image


def main() -> None:
    report = analyze_image(Path("assets/raw/sample_01.jpg"), provider="mock", run_id="sample_01")
    print(json.dumps(report["registration"], ensure_ascii=False, indent=2))
    _, fields = load_fields(FIELDS_PATH)
    template = imread(CANONICAL_DIR / "template.png")
    make_roi_grid(template, fields, Path("out/template_rois.png"))
    print("ROI grid: out/template_rois.png")
    print("Report: out/sample_01/report.json")


if __name__ == "__main__":
    main()
