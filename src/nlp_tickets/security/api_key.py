"""Autenticación servicio a servicio por API key con roles (deny-by-default)."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    CLIENT = "client"
    REVIEWER = "reviewer"


@dataclass(frozen=True)
class Principal:
    role: Role
    api_key: str

    @property
    def can_review(self) -> bool:
        return self.role is Role.REVIEWER


class ApiKeyAuthenticator:
    def __init__(self, client_keys: list[str], reviewer_keys: list[str]) -> None:
        self._keys = [(key, Role.REVIEWER) for key in reviewer_keys] + [
            (key, Role.CLIENT) for key in client_keys
        ]

    def authenticate(self, presented: str | None) -> Principal | None:
        """Compara contra todas las claves en tiempo constante (sin cortocircuito)."""
        if not presented:
            return None
        match: Principal | None = None
        for key, role in self._keys:
            if hmac.compare_digest(presented.encode(), key.encode()) and match is None:
                match = Principal(role=role, api_key=key)
        return match
