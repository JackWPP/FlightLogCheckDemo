from __future__ import annotations

import argparse
from pathlib import Path

from formcheck.enhance import normalize_illumination, scanner_bw, warp_document
from formcheck.image_io import imread, imwrite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("-o", "--out-dir", default="out/preprocess")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = imread(image_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imwrite(out_dir / f"{image_path.stem}_color_normalized.png", normalize_illumination(image))
    imwrite(out_dir / f"{image_path.stem}_scanner_bw.png", scanner_bw(image))
    warped, stats = warp_document(image)
    imwrite(out_dir / f"{image_path.stem}_document_warped.png", warped)
    imwrite(out_dir / f"{image_path.stem}_document_warped_color_normalized.png", normalize_illumination(warped))
    imwrite(out_dir / f"{image_path.stem}_document_warped_bw.png", scanner_bw(warped))
    (out_dir / f"{image_path.stem}_document_warp_stats.json").write_text(
        __import__("json").dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
