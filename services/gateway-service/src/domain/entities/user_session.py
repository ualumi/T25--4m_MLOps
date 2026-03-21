"""Сущность пользовательской сессии."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UserSession:
    user_id: str
    token: str
