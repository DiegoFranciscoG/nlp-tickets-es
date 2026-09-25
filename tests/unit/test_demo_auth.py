"""Inicio de sesión de la demo Streamlit (módulo puro, sin Streamlit)."""

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"


@pytest.fixture(scope="module")
def auth() -> Iterator[ModuleType]:
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(DEMO_DIR))
        import auth as module  # noqa: PLC0415

        yield module


def test_cuentas_desde_configuracion(auth: ModuleType) -> None:
    accounts = auth.build_accounts(
        {
            "DEMO_USER": "demo",
            "DEMO_PASSWORD_HASH": auth.hash_password("clave-de-prueba-123"),
            "REVIEWER_USER": "revisor",
            "REVIEWER_PASSWORD_HASH": "texto-plano-no-es-hash",
        }
    )
    assert [(a.username, a.role) for a in accounts] == [("demo", auth.Role.CLIENT)]


def test_autenticacion(auth: ModuleType) -> None:
    account = auth.Account("demo", auth.hash_password("clave-de-prueba-123"), auth.Role.CLIENT)
    assert auth.authenticate([account], " demo ", "clave-de-prueba-123") == account
    assert auth.authenticate([account], "demo", "incorrecta") is None
    assert auth.authenticate([account], "otro", "clave-de-prueba-123") is None


def test_bloqueo_tras_intentos_fallidos(auth: ModuleType) -> None:
    now = [0.0]
    guard = auth.LoginGuard(clock=lambda: now[0])
    for _ in range(auth.MAX_ATTEMPTS - 1):
        guard.register_failure()
    assert guard.locked_for() == 0
    guard.register_failure()
    assert guard.locked_for() > 0
    now[0] = auth.LOCKOUT_SECONDS + 1.0
    assert guard.locked_for() == 0
    guard.register_failure()
    guard.reset()
    assert guard.failures == 0
