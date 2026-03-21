"""Хранение пользовательских сессий в памяти."""

from __future__ import annotations

from secrets import token_urlsafe

from src.application.ports.session_store import SessionStorePort
from src.domain.entities.user_session import UserSession


class InMemorySessionStore(SessionStorePort):
    def __init__(self) -> None:
        self._sessions: dict[str, str] = {}

    def create(self, user_id: str) -> UserSession:
        token = token_urlsafe(24)
        self._sessions[user_id] = token
        return UserSession(user_id=user_id, token=token)

    def is_valid(self, user_id: str, token: str) -> bool:
        return self._sessions.get(user_id) == token
