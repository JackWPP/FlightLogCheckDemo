from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import FieldSpec


def load_fields(path: Path) -> tuple[dict[str, int], list[FieldSpec]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    canonical = data.get("canonical", {})
    fields: list[FieldSpec] = []
    for item in data.get("fields", []):
        bbox = item["bbox"]
        fields.append(
            FieldSpec(
                id=item["id"],
                label=item["label"],
                section=item["section"],
                bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                recognizer=item["recognizer"],
                validator=item["validator"],
                params=dict(item.get("params") or {}),
                fail_msg=item.get("fail_msg") or f"{item['label']}不符合要求",
                assignment=dict(item.get("assignment") or {}),
            )
        )
    return {"width": int(canonical["width"]), "height": int(canonical["height"])}, fields


def fields_by_id(fields: list[FieldSpec]) -> dict[str, FieldSpec]:
    return {field.id: field for field in fields}
