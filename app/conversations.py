from contextlib import contextmanager
import json
import os
import sqlite3
from pathlib import Path


DATABASE_PATH = Path(os.environ.get("CHAT_DB_PATH", "/data/chats.sqlite3"))


def _connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _connection():
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_chat_store():
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                messages_json TEXT NOT NULL
            )
            """
        )


def list_conversations():
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title, updated_at, messages_json
            FROM conversations
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "updatedAt": row["updated_at"],
            "messages": json.loads(row["messages_json"]),
        }
        for row in rows
    ]


def save_conversation(conversation, *, only_if_missing=False):
    statement = (
        """
        INSERT OR IGNORE INTO conversations (id, title, updated_at, messages_json)
        VALUES (?, ?, ?, ?)
        """
        if only_if_missing else
        """
        INSERT INTO conversations (id, title, updated_at, messages_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            updated_at = excluded.updated_at,
            messages_json = excluded.messages_json
        """
    )
    with _connection() as connection:
        cursor = connection.execute(
            statement,
            (
                conversation.id,
                conversation.title,
                conversation.updatedAt,
                json.dumps(
                    [message.model_dump() for message in conversation.messages],
                    ensure_ascii=False,
                ),
            ),
        )
    return cursor.rowcount > 0


def delete_conversation(conversation_id):
    with _connection() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
    return cursor.rowcount > 0
