from __future__ import annotations

import cv2
import numpy as np

from .schemas import FieldSpec, RecognitionResult


def recognize_checkbox(roi: np.ndarray) -> RecognitionResult:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Ignore the printed border by looking at dark stroke density in the center.
    h, w = gray.shape
    center = gray[max(0, h // 5) : min(h, h * 4 // 5), max(0, w // 5) : min(w, w * 4 // 5)]
    if center.size == 0:
        return RecognitionResult(value="false", normalized_value="false", confidence=0.0, provider="vision-rule")
    dark = center < 150
    density = float(dark.mean())
    checked = density > 0.035
    return RecognitionResult(
        value="checked" if checked else "unchecked",
        normalized_value="checked" if checked else "false",
        confidence=min(1.0, density / 0.12),
        provider="vision-rule",
        model="dark-pixel-density",
        raw={"dark_density": density},
    )


def mock_recognize(field: FieldSpec, roi: np.ndarray) -> RecognitionResult:
    if field.recognizer == "checkbox":
        return recognize_checkbox(roi)
    return RecognitionResult(value="", normalized_value="", confidence=0.0, provider="mock", model="empty")
