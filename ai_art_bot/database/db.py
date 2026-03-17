import sqlite3
from datetime import datetime, UTC

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


def save_image_record(
    prompt: str,
    image_path: str,
    upscaled_path: str | None = None,
    uploaded: bool = False,
) -> int:
    timestamp = datetime.now(UTC).isoformat()
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