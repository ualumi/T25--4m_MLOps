"""Абстракция хранилища пользовательских сессий."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.user_session import UserSession


class SessionStorePort(ABC):
    @abstractmethod
    def create(self, user_id: str) -> UserSession:
        """Создаёт и возвращает новую сессию."""

    @abstractmethod
    def is_valid(self, user_id: str, token: str) -> bool:
        """Проверяет, что пара user_id/token корректна."""
