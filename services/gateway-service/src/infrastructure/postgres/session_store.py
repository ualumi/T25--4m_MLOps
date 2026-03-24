"""Хранение пользовательских сессий в PostgreSQL."""

from __future__ import annotations

from secrets import token_urlsafe

import psycopg

from src.application.ports.session_store import SessionStorePort
from src.domain.entities.user_session import UserSession


class PostgresSessionStore(SessionStorePort):
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._ensure_schema()

    def _connect(self):
        return psycopg.connect(self._database_url)

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        user_id TEXT PRIMARY KEY,
                        token TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            connection.commit()

    def create(self, user_id: str) -> UserSession:
        token = token_urlsafe(24)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO user_sessions (user_id, token)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET token = EXCLUDED.token, created_at = CURRENT_TIMESTAMP
                    """,
                    (user_id, token),
                )
            connection.commit()
        return UserSession(user_id=user_id, token=token)

    def is_valid(self, user_id: str, token: str) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1
                    FROM user_sessions
                    WHERE user_id = %s AND token = %s
                    """,
                    (user_id, token),
                )
                row = cursor.fetchone()
        return row is not None
