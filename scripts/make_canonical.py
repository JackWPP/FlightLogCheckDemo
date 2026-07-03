from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from formcheck.enhance import scanner_bw
from formcheck.image_io import imread, imwrite


DEFAULT_CORNERS = {
    # Manual outer-border points for the provided high-resolution second image.
    "base_source.jpg": [[315, 104], [5068, 66], [5092, 3744], [260, 3678]],
}


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    height = int(round(image.shape[0] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def warp_to_canonical(image: np.ndarray, corners: list[list[float]], width: int) -> tuple[np.ndarray, np.ndarray]:
    src = order_points(np.array(corners, dtype="float32"))
    top = np.linalg.norm(src[1] - src[0])
    bottom = np.linalg.norm(src[2] - src[3])
    left = np.linalg.norm(src[3] - src[0])
    right = np.linalg.norm(src[2] - src[1])
    aspect = ((left + right) / 2) / max(1.0, ((top + bottom) / 2))
    height = int(round(width * aspect))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return warped, matrix


def make_scanner_base(warped: np.ndarray) -> np.ndarray:
    return scanner_bw(warped)


def estimate_alignment(src_image: np.ndarray, dst_image: np.ndarray) -> tuple[np.ndarray, dict]:
    gray_src = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create(nfeatures=10000)
    kp_src, desc_src = detector.detectAndCompute(gray_src, None)
    kp_dst, desc_dst = detector.detectAndCompute(gray_dst, None)
    if desc_src is None or desc_dst is None:
        raise ValueError("Cannot compute template alignment features")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    matches = matcher.knnMatch(desc_src, desc_dst, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    src = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, getattr(cv2, "USAC_MAGSAC", cv2.RANSAC), 4.0)
    if matrix is None or mask is None:
        raise ValueError("Cannot estimate template alignment homography")
    return matrix, {
        "source_features": len(kp_src),
        "target_features": len(kp_dst),
        "good_matches": len(good),
        "inliers": int(mask.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="assets/raw/base_scan.jpg")
    parser.add_argument("-o", "--out", default="assets/canonical")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--corners", help="JSON corners: [[x,y],[x,y],[x,y],[x,y]]")
    parser.add_argument("--scanned", action="store_true", help="Input is already perspective-corrected by a scanner app.")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = imread(image_path)
    corners = json.loads(args.corners) if args.corners else DEFAULT_CORNERS.get(image_path.name)
    scanned = args.scanned or corners is None
    if scanned:
        warped = resize_to_width(image, args.width)
        matrix = np.array(
            [[args.width / image.shape[1], 0, 0], [0, args.width / image.shape[1], 0], [0, 0, 1]],
            dtype="float32",
        )
        corners = [[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]]
    else:
        warped, matrix = warp_to_canonical(image, corners, args.width)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scanner = make_scanner_base(warped)
    photo_registration_written = False
    imwrite(out_dir / "template_registration.png", warped)
    imwrite(out_dir / "template.png", scanner)
    photo_source = Path("assets/raw/base_source.jpg")
    if image_path.name != "base_source.jpg" and photo_source.exists():
        photo_corners = DEFAULT_CORNERS.get(photo_source.name)
        if photo_corners:
            photo_warped, _ = warp_to_canonical(imread(photo_source), photo_corners, args.width)
            photo_warped = cv2.resize(photo_warped, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_AREA)
            imwrite(out_dir / "template_photo_registration.png", photo_warped)
            alignment, alignment_stats = estimate_alignment(photo_warped, warped)
            (out_dir / "template_alignment.json").write_text(
                json.dumps(
                    {
                        "source": "template_photo_registration.png",
                        "target": "template_registration.png",
                        "homography": alignment.tolist(),
                        "stats": alignment_stats,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            photo_registration_written = True
    meta = {
        "source": str(image_path),
        "width": int(warped.shape[1]),
        "height": int(warped.shape[0]),
        "scanned_input": scanned,
        "template_png": "scanner-cleaned black/white base for ROI and demo display",
        "template_registration_png": "color warped base for feature matching",
        "template_photo_registration_png": "phone-photo warped base for feature matching fallback" if photo_registration_written else None,
        "corners": corners,
        "homography": matrix.tolist(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "template_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
