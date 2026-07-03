from __future__ import annotations

import io
from pathlib import Path

from formcheck import tasks


class DummyUpload:
    filename = "sample.jpg"

    def __init__(self, payload: bytes = b"image") -> None:
        self.file = io.BytesIO(payload)


def test_task_lifecycle_and_claim_is_unique(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "tasks.sqlite3"
    monkeypatch.setattr(tasks, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(tasks, "DB_PATH", db_path)
    monkeypatch.setattr(tasks, "_processor", None)

    created = tasks.create_task(DummyUpload(), "session-1", db_path)
    assert created["status"] == "pending"

    claimed = tasks.claim_next(db_path)
    assert claimed is not None
    assert claimed["task_id"] == created["task_id"]
    assert claimed["status"] == "running"
    assert tasks.claim_next(db_path) is None

    report_path = Path(claimed["upload_path"]).parent / "report.json"
    report_path.write_text('{"ok": true}', encoding="utf-8")
    tasks.finish_task(claimed["task_id"], report_path, db_path)

    done = tasks.get_task(claimed["task_id"], db_path)
    assert done["status"] == "done"
    assert done["report"] == {"ok": True}
    assert tasks.list_tasks("session-1", db_path)[0]["task_id"] == claimed["task_id"]
