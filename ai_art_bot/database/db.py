import sqlite3
from datetime import datetime, timezone
import json
from typing import Any

from utils.config import DB_PATH, ensure_directories


def _connect() -> sqlite3.Connection:
    ensure_directories()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                image_path TEXT NOT NULL,
                upscaled_path TEXT,
                uploaded INTEGER NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                prompt_mode TEXT NOT NULL,
                pipeline_mode TEXT NOT NULL,
                source_image_path TEXT,
                source_upscaled_path TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                error_message TEXT,
                settings_json TEXT,
                image_path TEXT,
                upscaled_path TEXT,
                upload_json TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )


def save_image_record(
    prompt: str,
    image_path: str,
    upscaled_path: str | None = None,
    uploaded: bool = False,
) -> int:
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO images (prompt, image_path, upscaled_path, uploaded, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (prompt, image_path, upscaled_path, int(uploaded), timestamp),
        )
        return int(cursor.lastrowid)


def mark_uploaded(record_id: int, uploaded: bool = True, upscaled_path: str | None = None) -> None:
    with _connect() as connection:
        connection.execute(
            """
            UPDATE images
            SET uploaded = ?, upscaled_path = COALESCE(?, upscaled_path)
            WHERE id = ?
            """,
            (int(uploaded), upscaled_path, record_id),
        )


def list_images(limit: int = 20) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, prompt, image_path, upscaled_path, uploaded, timestamp
            FROM images
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def enqueue_task(
    *,
    prompt: str,
    prompt_mode: str,
    pipeline_mode: str,
    settings: dict[str, Any] | None = None,
    source_image_path: str | None = None,
    source_upscaled_path: str | None = None,
) -> int:
    timestamp = _now_iso()
    settings_json = json.dumps(settings or {}, ensure_ascii=True)
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO tasks (
                prompt,
                prompt_mode,
                pipeline_mode,
                source_image_path,
                source_upscaled_path,
                status,
                settings_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)
            """,
            (
                prompt,
                prompt_mode,
                pipeline_mode,
                source_image_path,
                source_upscaled_path,
                settings_json,
                timestamp,
                timestamp,
            ),
        )
        return int(cursor.lastrowid)


def get_task(task_id: int) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM tasks
            WHERE id = ?
            """,
            (task_id,),
        ).fetchone()
    return dict(row) if row else None


def list_tasks(limit: int = 100, status: str | None = None) -> list[dict]:
    with _connect() as connection:
        if status:
            rows = connection.execute(
                """
                SELECT *
                FROM tasks
                WHERE status = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM tasks
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return [dict(row) for row in rows]


def update_task_status(task_id: int, status: str, error_message: str | None = None) -> None:
    now = _now_iso()
    started_at = now if status == "running" else None
    finished_at = now if status in {"success", "failure", "no_info"} else None
    with _connect() as connection:
        connection.execute(
            """
            UPDATE tasks
            SET
                status = ?,
                error_message = ?,
                started_at = COALESCE(?, started_at),
                finished_at = COALESCE(?, finished_at),
                updated_at = ?
            WHERE id = ?
            """,
            (status, error_message, started_at, finished_at, now, task_id),
        )


def update_task_result(
    task_id: int,
    *,
    image_path: str | None = None,
    upscaled_path: str | None = None,
    upload_payload: dict[str, Any] | None = None,
) -> None:
    now = _now_iso()
    upload_json = json.dumps(upload_payload, ensure_ascii=True) if upload_payload is not None else None
    with _connect() as connection:
        connection.execute(
            """
            UPDATE tasks
            SET
                image_path = COALESCE(?, image_path),
                upscaled_path = COALESCE(?, upscaled_path),
                upload_json = COALESCE(?, upload_json),
                updated_at = ?
            WHERE id = ?
            """,
            (image_path, upscaled_path, upload_json, now, task_id),
        )


def task_settings(task: dict) -> dict[str, Any]:
    raw = task.get("settings_json")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def delete_task(task_id: int) -> None:
    with _connect() as connection:
        connection.execute(
            """
            DELETE FROM tasks WHERE id = ?
            """,
            (task_id,),
        )