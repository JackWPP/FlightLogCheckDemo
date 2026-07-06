from __future__ import annotations

import io
from pathlib import Path
import time

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


def test_recover_stale_running_task_returns_it_to_pending(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "tasks.sqlite3"
    monkeypatch.setattr(tasks, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(tasks, "DB_PATH", db_path)
    monkeypatch.setattr(tasks, "_processor", None)
    monkeypatch.setenv("TASK_LEASE_SECONDS", "60")

    created = tasks.create_task(DummyUpload(), "session-1", db_path)
    claimed = tasks.claim_next(db_path)
    assert claimed is not None

    stale_time = time.time() - 120
    with tasks.connect(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET lease_expires_at = ?, updated_at = ? WHERE task_id = ?",
            (stale_time, stale_time, claimed["task_id"]),
        )

    recovered = tasks.recover_stale_tasks(db_path)
    task = tasks.get_task(created["task_id"], db_path)

    assert recovered == 1
    assert task["status"] == "pending"
    assert task["progress"]["reason"] == "recovered_stale_running_task"
    assert task["error"] is None
