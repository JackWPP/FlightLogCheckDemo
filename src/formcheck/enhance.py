from __future__ import annotations

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def detect_page_quad(image: np.ndarray) -> tuple[np.ndarray | None, dict]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Flight-log pages are bright and low-to-medium saturation after excluding the dark background.
    mask = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([179, 125, 255]))
    h, w = mask.shape
    close = max(15, (min(h, w) // 30) | 1)
    open_size = max(7, (min(h, w) // 90) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"reason": "no_contours"}
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    area_ratio = area / float(h * w)
    if area_ratio < 0.35:
        return None, {"reason": "area_too_small", "area_ratio": area_ratio}
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype("float32")
    else:
        rect = cv2.minAreaRect(contour)
        quad = cv2.boxPoints(rect).astype("float32")
    quad = order_points(quad)
    q_area = abs(cv2.contourArea(quad))
    stats = {
        "area_ratio": area_ratio,
        "quad_area_ratio": q_area / float(h * w),
        "points": quad.tolist(),
    }
    if stats["quad_area_ratio"] < 0.45:
        return None, {**stats, "reason": "quad_area_too_small"}
    return quad, stats


def warp_document(image: np.ndarray, target_width: int = 2400) -> tuple[np.ndarray, dict]:
    quad, stats = detect_page_quad(image)
    if quad is None:
        return image.copy(), {"ok": False, **stats}
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    aspect = ((left + right) / 2) / max(1.0, ((top + bottom) / 2))
    target_height = int(round(target_width * aspect))
    dst = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, matrix, (target_width, target_height), borderValue=(255, 255, 255))
    return warped, {"ok": True, **stats, "width": target_width, "height": target_height}


def normalize_illumination(image: np.ndarray, blur_size: int = 81) -> np.ndarray:
    if blur_size % 2 == 0:
        blur_size += 1
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    background = cv2.medianBlur(l, blur_size)
    l_norm = cv2.divide(l, background, scale=255)
    l_norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_norm)
    return cv2.cvtColor(cv2.merge([l_norm, a, b]), cv2.COLOR_LAB2BGR)


def scanner_bw(image: np.ndarray) -> np.ndarray:
    normalized = normalize_illumination(image)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=7, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        11,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def enhance_roi_for_ocr(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    scale = min(4.0, max(1.0, 900 / max(h, w)))
    if scale > 1.05:
        roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return normalize_illumination(roi, blur_size=31)
