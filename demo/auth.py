"""Inicio de sesión de la demo: contraseñas con hash Argon2id y bloqueo tras intentos fallidos.

Las credenciales viven solo en el servidor (variables de entorno o st.secrets) como hashes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60
_hasher = PasswordHasher()


class Role(StrEnum):
    CLIENT = "client"
    REVIEWER = "reviewer"


@dataclass(frozen=True)
class Account:
    username: str
    password_hash: str
    role: Role


@dataclass
class LoginGuard:
    """Cuenta intentos fallidos por sesión y bloquea temporalmente (fuerza bruta)."""

    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    locked_until: float = 0.0

    def locked_for(self) -> int:
        remaining = self.locked_until - self.clock()
        return max(0, int(remaining) + (1 if remaining > 0 else 0))

    def register_failure(self) -> None:
        self.failures += 1
        if self.failures >= MAX_ATTEMPTS:
            self.locked_until = self.clock() + LOCKOUT_SECONDS
            self.failures = 0

    def reset(self) -> None:
        self.failures = 0
        self.locked_until = 0.0


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def authenticate(accounts: list[Account], username: str, password: str) -> Account | None:
    """Verifica siempre un hash (aunque el usuario no exista) para no filtrar qué usuarios hay."""
    account = next((a for a in accounts if a.username == username.strip()), None)
    candidate = account.password_hash if account else _DUMMY_HASH
    try:
        _hasher.verify(candidate, password)
    except (VerificationError, InvalidHashError):
        return None
    return account


_DUMMY_HASH = _hasher.hash("usuario-inexistente")


def build_accounts(settings: dict[str, str]) -> list[Account]:
    accounts = []
    for user_key, hash_key, role in (
        ("DEMO_USER", "DEMO_PASSWORD_HASH", Role.CLIENT),
        ("REVIEWER_USER", "REVIEWER_PASSWORD_HASH", Role.REVIEWER),
    ):
        username, password_hash = settings.get(user_key, ""), settings.get(hash_key, "")
        if username and password_hash.startswith("$argon2"):
            accounts.append(Account(username, password_hash, role))
    return accounts
