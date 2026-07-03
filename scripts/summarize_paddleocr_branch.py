from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "ppocrv6_unwarp": Path("out/paddleocr_jobs/ppocrv6_sample_01_unwarp"),
    "ppocrv6_no_unwarp": Path("out/paddleocr_jobs/ppocrv6_sample_01_no_unwarp"),
    "paddleocr_vl_unwarp": Path("out/paddleocr_jobs/paddleocr_vl_sample_01_unwarp"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(DEFAULT_RUNS)
    write_deliverables(report, out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved report: {out_dir / 'paddleocr_branch_report.md'}")


def build_report(runs: dict[str, Path]) -> dict[str, Any]:
    ppocr_unwarp = load_candidates(runs["ppocrv6_unwarp"] / "field_candidates.json")
    ppocr_no_unwarp = load_candidates(runs["ppocrv6_no_unwarp"] / "field_candidates.json")
    vl_doc = read_optional(runs["paddleocr_vl_unwarp"] / "doc_0.md")

    return {
        "runs": {name: summarize_run(path) for name, path in runs.items()},
        "ppocrv6_unwarp": summarize_candidates(ppocr_unwarp),
        "ppocrv6_no_unwarp": summarize_candidates(ppocr_no_unwarp),
        "paddleocr_vl": summarize_vl_doc(vl_doc),
        "recommendation": {
            "primary": "PP-OCRv6 as full-page text-block candidate generator",
            "secondary": "Template/anchor matching maps detected blocks into business fields",
            "fallback": "Field-level VLM/OCR only for ambiguous handwriting, signatures, and redacted/low-confidence fields",
            "not_recommended": "Letting PaddleOCR-VL markdown directly decide compliance in v1",
        },
    }


def summarize_run(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return {"exists": False, "path": str(path)}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "model": data.get("model"),
        "job_id": data.get("job_id"),
        "records": data.get("records"),
        "markdown_files": data.get("markdown_files", []),
        "downloaded_images": data.get("downloaded_images", []),
    }


def load_candidates(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_candidates(data: dict[str, Any]) -> dict[str, Any]:
    guesses = data.get("field_guesses") or {}
    return {
        "counts": {
            "dates": len(data.get("dates") or []),
            "refs": len(data.get("refs") or []),
            "digit_3_5": len(data.get("digit_3_5") or []),
            "english_fault": len(data.get("english_fault") or []),
            "license_fragments": len(data.get("license_fragments") or []),
        },
        "field_guesses": {
            "fault_ref": compact_rows(guesses.get("fault_ref")),
            "fault_report_en": compact_rows(guesses.get("fault_report_en"), limit=3),
            "apu_cum_hours": compact_rows(guesses.get("apu_cum_hours")),
            "apu_cum_cycles": compact_rows(guesses.get("apu_cum_cycles")),
            "awr_date": compact_rows(guesses.get("awr_date")),
            "awr_license": compact_rows(guesses.get("awr_license")),
        },
    }


def compact_rows(rows: list[dict[str, Any]] | None, limit: int = 5) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in (rows or [])[:limit]:
        compact.append(
            {
                "text": row.get("text"),
                "score": round(float(row.get("score", 0.0)), 4),
                "box": row.get("box"),
                "center": row.get("center"),
            }
        )
    return compact


def read_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def summarize_vl_doc(text: str) -> dict[str, Any]:
    probes = ["TSM", "AMM", "2238", "348", "CAAC", "Fault", "COND AFT"]
    return {
        "markdown_chars": len(text),
        "contains": {probe: (probe in text) for probe in probes},
        "table_like": "<table" in text.lower(),
        "preview": text[:500],
    }


def write_deliverables(report: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "paddleocr_branch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "paddleocr_branch_report.md").write_text(render_markdown(report), encoding="utf-8")
    copy_if_exists(Path("out/paddleocr_jobs/ppocrv6_sample_01_unwarp/ocr_image_0.jpg"), out_dir / "ppocrv6_unwarp_ocr_image.jpg")
    copy_if_exists(
        Path("out/paddleocr_jobs/ppocrv6_sample_01_no_unwarp/ocr_image_0.jpg"),
        out_dir / "ppocrv6_no_unwarp_ocr_image.jpg",
    )
    copy_if_exists(
        Path("out/paddleocr_jobs/paddleocr_vl_sample_01_unwarp/layout_det_res_0.jpg"),
        out_dir / "paddleocr_vl_layout_det.jpg",
    )
    copy_if_exists(
        Path("out/paddleocr_jobs/paddleocr_vl_sample_01_unwarp/doc_0.md"),
        out_dir / "paddleocr_vl_doc_0.md",
    )


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def render_markdown(report: dict[str, Any]) -> str:
    unwarp = report["ppocrv6_unwarp"]
    no_unwarp = report["ppocrv6_no_unwarp"]
    vl = report["paddleocr_vl"]
    lines = [
        "# PaddleOCR branch report",
        "",
        "## Verdict",
        "",
        "- PP-OCRv6 is the stronger first-stage candidate generator for this demo.",
        "- Document unwarping improves the coordinate regularity enough to recover APU field guesses in the current heuristic.",
        "- PaddleOCR-VL produces a useful macro table/Markdown view, but it missed the key handwritten probes in the Markdown, so it should not be the compliance source of truth yet.",
        "",
        "## PP-OCRv6 with document unwarping",
        "",
        f"- Counts: `{json.dumps(unwarp['counts'], ensure_ascii=False)}`",
        f"- Reference manual: `{texts(unwarp, 'fault_ref')}`",
        f"- Fault English/action candidates: `{texts(unwarp, 'fault_report_en')}`",
        f"- APU hours: `{texts(unwarp, 'apu_cum_hours')}`",
        f"- APU cycles: `{texts(unwarp, 'apu_cum_cycles')}`",
        f"- Airworthiness date candidates: `{texts(unwarp, 'awr_date')}`",
        "",
        "## PP-OCRv6 without document unwarping",
        "",
        f"- Counts: `{json.dumps(no_unwarp['counts'], ensure_ascii=False)}`",
        f"- Reference manual: `{texts(no_unwarp, 'fault_ref')}`",
        f"- Fault English/action candidates: `{texts(no_unwarp, 'fault_report_en')}`",
        f"- APU hours: `{texts(no_unwarp, 'apu_cum_hours')}`",
        f"- APU cycles: `{texts(no_unwarp, 'apu_cum_cycles')}`",
        "",
        "## PaddleOCR-VL",
        "",
        f"- Markdown chars: `{vl['markdown_chars']}`",
        f"- Contains probes: `{json.dumps(vl['contains'], ensure_ascii=False)}`",
        f"- Table-like output: `{vl['table_like']}`",
        "",
        "## Recommended v2 pipeline",
        "",
        "1. Run a document preprocessor/scan enhancement branch first.",
        "2. Run PP-OCRv6 full-page OCR with document unwarping enabled.",
        "3. Use template anchors or normalized boxes to assign OCR blocks to business fields.",
        "4. Run local validators for numeric/date/regex/checkbox rules.",
        "5. Escalate only low-confidence fields to field-level VLM.",
    ]
    return "\n".join(lines) + "\n"


def texts(summary: dict[str, Any], field: str) -> str:
    rows = summary["field_guesses"].get(field) or []
    return " | ".join(str(row["text"]) for row in rows)


if __name__ == "__main__":
    main()
