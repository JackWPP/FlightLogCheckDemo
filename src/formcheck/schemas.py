from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldSpec:
    id: str
    label: str
    section: str
    bbox: tuple[int, int, int, int]
    recognizer: str
    validator: str
    params: dict[str, Any] = field(default_factory=dict)
    fail_msg: str = ""
    assignment: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecognitionResult:
    value: str
    normalized_value: str | None = None
    confidence: float | None = None
    provider: str = "mock"
    model: str = "mock"
    raw: Any = None
    needs_review: bool = False
    review_reason: str = ""


@dataclass
class FieldCheck:
    field: FieldSpec
    recognition: RecognitionResult
    passed: bool
    message: str
    roi_url: str | None = None


@dataclass(frozen=True)
class OcrBlock:
    id: str
    text: str
    score: float
    box: tuple[float, float, float, float]
    center: tuple[float, float]
    source: str = "ppocrv6"


@dataclass(frozen=True)
class FieldCandidate:
    field_id: str
    block: OcrBlock
    score: float
    reason: str
