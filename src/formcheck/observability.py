from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .config import OUTPUTS_DIR


request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
task_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("task_id", default="")
run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")

LOGGER_NAME = "formcheck"
SENSITIVE_KEYS = {"api_key", "authorization", "token", "password", "secret", "bearer"}
_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname.lower(),
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "") or request_id_var.get(),
            "task_id": getattr(record, "task_id", "") or task_id_var.get(),
            "run_id": getattr(record, "run_id", "") or run_id_var.get(),
            "session_id": getattr(record, "session_id", "") or session_id_var.get(),
        }
        extra = getattr(record, "payload", None)
        if isinstance(extra, dict):
            payload.update(sanitize_payload(extra))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps({k: v for k, v in payload.items() if v not in {"", None}}, ensure_ascii=False)


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    log_dir().mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level())
    logger.propagate = False
    logger.handlers.clear()

    handler = RotatingFileHandler(
        log_file(),
        maxBytes=int_env("APP_LOG_MAX_BYTES", 10 * 1024 * 1024),
        backupCount=int_env("APP_LOG_BACKUP_COUNT", 7),
        encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    if os.getenv("APP_LOG_STDOUT", "1").strip().lower() not in {"0", "false", "no", "off"}:
        console = logging.StreamHandler()
        console.setFormatter(JsonFormatter())
        logger.addHandler(console)

    _configured = True


def log_event(event: str, level: str = "info", **payload: Any) -> None:
    configure_logging()
    logger = logging.getLogger(LOGGER_NAME)
    log = getattr(logger, level.lower(), logger.info)
    log(event, extra={"event": event, "payload": payload})


def log_exception(event: str, exc: BaseException, **payload: Any) -> None:
    configure_logging()
    logging.getLogger(LOGGER_NAME).error(
        event,
        exc_info=(type(exc), exc, exc.__traceback__),
        extra={"event": event, "payload": payload},
    )


def set_context(
    request_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
) -> list[contextvars.Token]:
    tokens: list[contextvars.Token] = []
    if request_id is not None:
        tokens.append(request_id_var.set(request_id))
    if task_id is not None:
        tokens.append(task_id_var.set(task_id))
    if run_id is not None:
        tokens.append(run_id_var.set(run_id))
    if session_id is not None:
        tokens.append(session_id_var.set(session_id))
    return tokens


def reset_context(tokens: list[contextvars.Token]) -> None:
    for token in reversed(tokens):
        token.var.reset(token)


def current_request_id() -> str:
    return request_id_var.get()


def read_recent_logs(
    task_id: str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    path = log_file()
    if not path.exists():
        return []
    limit = max(1, min(limit, 1000))
    records: list[dict[str, Any]] = []
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if len(records) >= limit:
            break
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if task_id and record.get("task_id") != task_id:
            continue
        if run_id and record.get("run_id") != run_id:
            continue
        if request_id and record.get("request_id") != request_id:
            continue
        records.append(record)
    return list(reversed(records))


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            lowered = text_key.lower()
            if any(sensitive in lowered for sensitive in SENSITIVE_KEYS):
                sanitized[text_key] = "[redacted]"
            else:
                sanitized[text_key] = sanitize_payload(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return str(value)
    return value


def exception_summary(exc: BaseException, limit: int = 5) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "trace": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=limit)),
    }


def log_dir() -> Path:
    return Path(os.getenv("APP_LOG_DIR") or (OUTPUTS_DIR / "runtime" / "logs"))


def log_file() -> Path:
    return log_dir() / os.getenv("APP_LOG_FILE", "app.jsonl")


def log_level() -> int:
    return getattr(logging, os.getenv("APP_LOG_LEVEL", "INFO").strip().upper(), logging.INFO)


def int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default
