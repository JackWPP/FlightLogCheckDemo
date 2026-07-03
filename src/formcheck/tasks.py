from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from .config import OUT_DIR, OUTPUTS_DIR


DB_PATH = OUTPUTS_DIR / "runtime" / "tasks.sqlite3"
STATUSES = {"pending", "running", "done", "failed", "cancelled"}
_worker_lock = threading.Lock()
_worker_active = False
_processor = None


def configure_processor(processor) -> None:
    global _processor
    _processor = processor


def init_db(path: Path | None = None) -> None:
    db_path = path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                progress TEXT NOT NULL,
                run_id TEXT NOT NULL,
                upload_path TEXT NOT NULL,
                report_path TEXT,
                error TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session_created ON tasks(session_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks(status, created_at)")


def connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def create_task(file: UploadFile, session_id: str, path: Path | None = None) -> dict[str, Any]:
    init_db(path)
    task_id = uuid.uuid4().hex[:12]
    run_id = task_id
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    upload_path = run_dir / f"upload{suffix}"
    with upload_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    created_at = time.time()
    progress = {"stage": "pending", "status": "pending"}
    with connect(path) as conn:
        conn.execute(
            """
            INSERT INTO tasks (
                task_id, session_id, filename, status, created_at, progress,
                run_id, upload_path
            ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)
            """,
            (
                task_id,
                normalize_session_id(session_id),
                file.filename or upload_path.name,
                created_at,
                json.dumps(progress, ensure_ascii=False),
                run_id,
                str(upload_path),
            ),
        )
    task = get_task(task_id, path)
    start_worker()
    return task


def normalize_session_id(session_id: str | None) -> str:
    text = (session_id or "").strip()
    return text[:80] if text else "default"


def list_tasks(session_id: str, path: Path | None = None) -> list[dict[str, Any]]:
    init_db(path)
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE session_id = ? ORDER BY created_at DESC LIMIT 200",
            (normalize_session_id(session_id),),
        ).fetchall()
    return [row_to_task(row) for row in rows]


def get_task(task_id: str, path: Path | None = None) -> dict[str, Any]:
    init_db(path)
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        raise KeyError(task_id)
    task = row_to_task(row)
    if task.get("report_path"):
        report_path = Path(task["report_path"])
        if report_path.exists():
            task["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    return task


def claim_next(path: Path | None = None) -> dict[str, Any] | None:
    init_db(path)
    with connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM tasks WHERE status = 'pending' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        now = time.time()
        progress = {"stage": "started", "status": "running"}
        conn.execute(
            "UPDATE tasks SET status = 'running', started_at = ?, progress = ? WHERE task_id = ? AND status = 'pending'",
            (now, json.dumps(progress, ensure_ascii=False), row["task_id"]),
        )
        conn.execute("COMMIT")
    return get_task(row["task_id"], path)


def update_progress(task_id: str, progress: dict[str, Any], path: Path | None = None) -> None:
    init_db(path)
    with connect(path) as conn:
        conn.execute(
            "UPDATE tasks SET progress = ? WHERE task_id = ?",
            (json.dumps(progress, ensure_ascii=False), task_id),
        )


def finish_task(task_id: str, report_path: Path, path: Path | None = None) -> None:
    progress = {"stage": "final", "status": "done"}
    with connect(path) as conn:
        conn.execute(
            """
            UPDATE tasks
            SET status = 'done', finished_at = ?, progress = ?, report_path = ?, error = NULL
            WHERE task_id = ?
            """,
            (time.time(), json.dumps(progress, ensure_ascii=False), str(report_path), task_id),
        )


def fail_task(task_id: str, error: str, path: Path | None = None) -> None:
    progress = {"stage": "error", "status": "failed", "error": error}
    with connect(path) as conn:
        conn.execute(
            """
            UPDATE tasks
            SET status = 'failed', finished_at = ?, progress = ?, error = ?
            WHERE task_id = ?
            """,
            (time.time(), json.dumps(progress, ensure_ascii=False), error, task_id),
        )


def row_to_task(row: sqlite3.Row) -> dict[str, Any]:
    progress = json.loads(row["progress"] or "{}")
    task = {key: row[key] for key in row.keys()}
    task["progress"] = progress
    task["upload_url"] = f"/runs/{row['run_id']}/{Path(row['upload_path']).name}"
    if row["report_path"]:
        task["report_url"] = f"/runs/{row['run_id']}/report.json"
    return task


def start_worker() -> None:
    global _worker_active
    if _processor is None:
        return
    with _worker_lock:
        if _worker_active:
            return
        _worker_active = True
    threading.Thread(target=worker_loop, daemon=True).start()


def worker_loop() -> None:
    global _worker_active
    try:
        while True:
            task = claim_next()
            if task is None:
                return
            process_claimed_task(task)
    finally:
        with _worker_lock:
            _worker_active = False


def process_claimed_task(task: dict[str, Any]) -> None:
    assert _processor is not None
    task_id = task["task_id"]

    def progress_cb(stage: str, status: str, **extra) -> None:
        update_progress(task_id, {"stage": stage, "status": status, **extra})

    try:
        report = _processor(task, progress_cb)
        report_path = OUT_DIR / task["run_id"] / "report.json"
        finish_task(task_id, report_path)
    except Exception as exc:  # noqa: BLE001
        fail_task(task_id, str(exc))
