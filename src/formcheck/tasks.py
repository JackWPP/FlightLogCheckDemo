from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from .config import OUT_DIR, OUTPUTS_DIR
from .observability import current_request_id, log_event, log_exception, reset_context, set_context


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
                error TEXT,
                updated_at REAL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                lease_expires_at REAL,
                request_id TEXT
            )
            """
        )
        ensure_column(conn, "tasks", "updated_at", "REAL")
        ensure_column(conn, "tasks", "attempt_count", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "tasks", "lease_expires_at", "REAL")
        ensure_column(conn, "tasks", "request_id", "TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session_created ON tasks(session_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks(status, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_lease ON tasks(status, lease_expires_at)")


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


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
    normalized_session = normalize_session_id(session_id)
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
                task_id, session_id, filename, status, created_at, updated_at, progress,
                run_id, upload_path, request_id
            ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                normalized_session,
                file.filename or upload_path.name,
                created_at,
                created_at,
                json.dumps(progress, ensure_ascii=False),
                run_id,
                str(upload_path),
                current_request_id(),
            ),
        )
    task = get_task(task_id, path)
    log_event(
        "task.created",
        task_id=task_id,
        run_id=run_id,
        session_id=normalized_session,
        filename=task.get("filename"),
        upload_path=str(upload_path),
        size_bytes=upload_path.stat().st_size,
    )
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
            """
            UPDATE tasks
            SET status = 'running',
                started_at = ?,
                updated_at = ?,
                lease_expires_at = ?,
                attempt_count = COALESCE(attempt_count, 0) + 1,
                progress = ?
            WHERE task_id = ? AND status = 'pending'
            """,
            (now, now, now + task_lease_seconds(), json.dumps(progress, ensure_ascii=False), row["task_id"]),
        )
        conn.execute("COMMIT")
    task = get_task(row["task_id"], path)
    log_event(
        "task.claimed",
        task_id=task["task_id"],
        run_id=task["run_id"],
        session_id=task["session_id"],
        filename=task.get("filename"),
    )
    return task


def update_progress(task_id: str, progress: dict[str, Any], path: Path | None = None) -> None:
    init_db(path)
    now = time.time()
    with connect(path) as conn:
        conn.execute(
            "UPDATE tasks SET progress = ?, updated_at = ?, lease_expires_at = ? WHERE task_id = ?",
            (json.dumps(progress, ensure_ascii=False), now, now + task_lease_seconds(), task_id),
        )
    log_event("task.progress", task_id=task_id, **progress)


def finish_task(task_id: str, report_path: Path, path: Path | None = None) -> None:
    progress = {"stage": "final", "status": "done"}
    now = time.time()
    with connect(path) as conn:
        conn.execute(
            """
            UPDATE tasks
            SET status = 'done', finished_at = ?, updated_at = ?, lease_expires_at = NULL,
                progress = ?, report_path = ?, error = NULL
            WHERE task_id = ?
            """,
            (now, now, json.dumps(progress, ensure_ascii=False), str(report_path), task_id),
        )
    log_event("task.finished", task_id=task_id, report_path=str(report_path))


def fail_task(task_id: str, error: str, path: Path | None = None) -> None:
    progress = {"stage": "error", "status": "failed", "error": error}
    now = time.time()
    with connect(path) as conn:
        conn.execute(
            """
            UPDATE tasks
            SET status = 'failed', finished_at = ?, updated_at = ?, lease_expires_at = NULL,
                progress = ?, error = ?
            WHERE task_id = ?
            """,
            (now, now, json.dumps(progress, ensure_ascii=False), error, task_id),
        )
    log_event("task.failed", "error", task_id=task_id, error=error)


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
    log_event("task.worker.start")
    threading.Thread(target=worker_loop, daemon=True).start()


def recover_stale_tasks(path: Path | None = None) -> int:
    init_db(path)
    now = time.time()
    progress = {
        "stage": "pending",
        "status": "pending",
        "reason": "recovered_stale_running_task",
    }
    with connect(path) as conn:
        cursor = conn.execute(
            """
            UPDATE tasks
            SET status = 'pending',
                updated_at = ?,
                lease_expires_at = NULL,
                progress = ?,
                error = NULL
            WHERE status = 'running'
              AND (lease_expires_at IS NULL OR lease_expires_at < ?)
            """,
            (now, json.dumps(progress, ensure_ascii=False), now),
        )
    count = int(cursor.rowcount or 0)
    if count:
        log_event("task.recovered_stale", count=count)
    return count


def task_lease_seconds() -> int:
    try:
        value = int(os.getenv("TASK_LEASE_SECONDS", "1800"))
    except ValueError:
        return 1800
    return max(60, min(value, 24 * 60 * 60))


def worker_loop() -> None:
    global _worker_active
    try:
        while True:
            task = claim_next()
            if task is None:
                log_event("task.worker.idle")
                return
            process_claimed_task(task)
    except Exception as exc:  # noqa: BLE001
        log_exception("task.worker.exception", exc)
        raise
    finally:
        with _worker_lock:
            _worker_active = False


def process_claimed_task(task: dict[str, Any]) -> None:
    assert _processor is not None
    task_id = task["task_id"]
    tokens = set_context(task_id=task_id, run_id=task["run_id"], session_id=task["session_id"])

    def progress_cb(stage: str, status: str, **extra) -> None:
        update_progress(task_id, {"stage": stage, "status": status, **extra})

    try:
        report = _processor(task, progress_cb)
        report_path = OUT_DIR / task["run_id"] / "report.json"
        finish_task(task_id, report_path)
    except Exception as exc:  # noqa: BLE001
        log_exception("task.exception", exc)
        fail_task(task_id, str(exc))
    finally:
        reset_context(tokens)
