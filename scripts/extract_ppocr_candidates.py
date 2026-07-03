from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PATTERNS = {
    "dates": re.compile(r"20\d{2}[.\-/]\d{2}[.\-/]\d{2}"),
    "refs": re.compile(r"(?:AMM|TSM)[A-Z0-9\-.]+", re.I),
    "license_fragments": re.compile(r"CAACML|CAAC|CAA|CA", re.I),
    "digit_3_5": re.compile(r"^\d{3,5}$"),
    "english_fault": re.compile(r"[A-Za-z][A-Za-z0-9,./\\\- ]{12,}"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    candidates = extract_candidates(Path(args.jsonl))
    out_path = Path(args.out) if args.out else Path(args.jsonl).with_name("field_candidates.json")
    out_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(candidates, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


def extract_candidates(path: Path) -> dict[str, Any]:
    rows = load_ppocr_rows(path)
    return {
        "dates": match_rows(rows, "dates"),
        "refs": match_rows(rows, "refs"),
        "license_fragments": match_rows(rows, "license_fragments"),
        "digit_3_5": match_rows(rows, "digit_3_5"),
        "english_fault": match_rows(rows, "english_fault"),
        "field_guesses": guess_fields(rows),
    }


def load_ppocr_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        result = record.get("result") or {}
        for ocr in result.get("ocrResults") or []:
            pruned = ocr.get("prunedResult") or {}
            texts = pruned.get("rec_texts") or []
            scores = pruned.get("rec_scores") or []
            boxes = pruned.get("rec_boxes") or []
            for text, score, box in zip(texts, scores, boxes):
                rows.append({"text": text, "score": score, "box": box, "center": center(box)})
    return rows


def center(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def match_rows(rows: list[dict[str, Any]], pattern_name: str) -> list[dict[str, Any]]:
    pattern = PATTERNS[pattern_name]
    return [row for row in rows if pattern.search(row["text"])]


def guess_fields(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    guesses: dict[str, list[dict[str, Any]]] = {
        "fault_ref": [],
        "apu_cum_hours": [],
        "apu_cum_cycles": [],
        "awr_date": [],
        "fault_report_en": [],
        "awr_license": [],
    }
    for row in rows:
        x, y = row["center"]
        text = row["text"]
        if PATTERNS["refs"].search(text):
            guesses["fault_ref"].append(row)
        if PATTERNS["dates"].search(text) and y > 800:
            guesses["awr_date"].append(row)
        if PATTERNS["english_fault"].search(text) and 360 < y < 520:
            guesses["fault_report_en"].append(row)
        if re.fullmatch(r"\d{3,5}", text):
            if 300 < x < 400 and y > 850:
                guesses["apu_cum_hours"].append(row)
            if 390 < x < 480 and y > 850:
                guesses["apu_cum_cycles"].append(row)
        if PATTERNS["license_fragments"].search(text):
            guesses["awr_license"].append(row)
    return guesses


if __name__ == "__main__":
    main()
