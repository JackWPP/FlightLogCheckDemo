from __future__ import annotations

import argparse
import json
from pathlib import Path

from formcheck.scanfile import api_key_from_env, scan_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("-o", "--out", default="out/scanfile/scanfile_result.jpg")
    parser.add_argument("--auto-crop", default="true", choices=["true", "false", "force"])
    parser.add_argument("--auto-rotate", default="true", choices=["true", "false", "force"])
    args = parser.parse_args()

    api_key = api_key_from_env()
    if not api_key:
        raise SystemExit("Missing ALIYUN_IQS_API_KEY in .env")

    result = scan_file(Path(args.image), api_key, auto_crop=args.auto_crop, auto_rotate=args.auto_rotate)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "ok": result.ok,
        "width": result.width,
        "height": result.height,
        "angle": result.angle,
        "request_id": result.request_id,
        "error": result.error,
        "raw": result.raw,
        "output": str(out_path) if result.ok else None,
    }
    if result.ok and result.image_bytes:
        out_path.write_bytes(result.image_bytes)
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
