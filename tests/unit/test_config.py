import pytest
from pydantic import ValidationError

from nlp_tickets.config import Settings

VALID_KEY = "k" * 32


def make(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql://user:pwd@localhost:5432/db",
        "api_keys_client": VALID_KEY,
        "api_keys_reviewer": "r" * 40,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DATABASE_URL", "API_KEYS_CLIENT", "API_KEYS_REVIEWER", "CORS_ALLOWED_ORIGINS"):
        monkeypatch.delenv(name, raising=False)


def test_no_arranca_sin_secretos() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_rechaza_api_keys_cortas() -> None:
    with pytest.raises(ValidationError, match="32"):
        make(api_keys_client="corta")


def test_rechaza_url_que_no_es_postgres() -> None:
    with pytest.raises(ValidationError, match="PostgreSQL"):
        make(database_url="sqlite:///app.db")


@pytest.mark.parametrize(
    "url",
    [
        "postgres://u:p@host/db",
        "postgresql://u:p@host/db",
        "postgresql+psycopg://u:p@host/db",
    ],
)
def test_normaliza_url_al_driver_psycopg(url: str) -> None:
    assert make(database_url=url).sqlalchemy_url == "postgresql+psycopg://u:p@host/db"


def test_varias_claves_separadas_por_coma() -> None:
    settings = make(api_keys_client=f"{VALID_KEY}, {'z' * 33}")
    assert settings.client_keys == [VALID_KEY, "z" * 33]


def test_cors_con_comodin_no_permitido() -> None:
    with pytest.raises(ValueError, match="comodín"):
        _ = make(cors_allowed_origins="*").cors_origins


def test_umbral_vacio_es_none() -> None:
    assert make(confidence_threshold_override="").confidence_threshold_override is None


def test_secretos_no_aparecen_en_repr() -> None:
    assert VALID_KEY not in repr(make())
