from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .image_io import imwrite
from .schemas import FieldSpec


def crop_roi(image: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 6) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    return image[y0:y1, x0:x1].copy()


def save_rois(image: np.ndarray, fields: list[FieldSpec], out_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for field in fields:
        roi = crop_roi(image, field.bbox)
        path = out_dir / f"{field.id}.png"
        imwrite(path, roi)
        paths[field.id] = path
    return paths


def make_roi_grid(image: np.ndarray, fields: list[FieldSpec], out_path: Path) -> None:
    thumbs = []
    cell_w, cell_h = 300, 120
    for field in fields:
        roi = crop_roi(image, field.bbox, pad=8)
        scale = min(cell_w / max(1, roi.shape[1]), (cell_h - 28) / max(1, roi.shape[0]))
        resized = cv2.resize(roi, (max(1, int(roi.shape[1] * scale)), max(1, int(roi.shape[0] * scale))))
        canvas = np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        cv2.putText(canvas, field.id[:34], (6, cell_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
        thumbs.append(canvas)
    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    grid = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        y = (idx // cols) * cell_h
        x = (idx % cols) * cell_w
        grid[y : y + cell_h, x : x + cell_w] = thumb
    imwrite(out_path, grid)
