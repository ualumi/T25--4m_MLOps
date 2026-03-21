"""Сценарий создания пользовательской сессии."""

from src.application.ports.session_store import SessionStorePort
from src.domain.entities.user_session import UserSession


class ConnectUserUseCase:
    def __init__(self, store: SessionStorePort) -> None:
        self._store = store

    def execute(self, user_id: str) -> UserSession:
        return self._store.create(user_id=user_id)
