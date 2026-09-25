"""Configuración por variables de entorno. Sin valores por defecto para secretos."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_API_KEY_LENGTH = 32


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    docs_enabled: bool = True

    database_url: SecretStr
    api_keys_client: SecretStr
    api_keys_reviewer: SecretStr

    cors_allowed_origins: str = ""
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10_000)
    max_text_length: int = Field(default=5000, ge=100, le=20_000)

    model_dir: Path = Path("artifacts/model")
    model_url: str = ""
    model_sha256: str = ""
    confidence_threshold_override: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("confidence_threshold_override", mode="before")
    @classmethod
    def _empty_as_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("api_keys_client", "api_keys_reviewer")
    @classmethod
    def _validate_keys(cls, value: SecretStr) -> SecretStr:
        keys = _split_csv(value.get_secret_value())
        if not keys:
            raise ValueError("define al menos una API key")
        if any(len(key) < MIN_API_KEY_LENGTH for key in keys):
            raise ValueError(f"cada API key debe tener al menos {MIN_API_KEY_LENGTH} caracteres")
        return value

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, value: SecretStr) -> SecretStr:
        url = value.get_secret_value()
        if not url.startswith(("postgresql+psycopg://", "postgresql://", "postgres://")):
            raise ValueError("DATABASE_URL debe ser una URL de PostgreSQL")
        return value

    @property
    def sqlalchemy_url(self) -> str:
        """Normaliza la URL de Neon/Render (``postgres://``) al driver psycopg 3."""
        url = self.database_url.get_secret_value()
        for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix) :]
        return url

    @property
    def client_keys(self) -> list[str]:
        return _split_csv(self.api_keys_client.get_secret_value())

    @property
    def reviewer_keys(self) -> list[str]:
        return _split_csv(self.api_keys_reviewer.get_secret_value())

    @property
    def cors_origins(self) -> list[str]:
        origins = _split_csv(self.cors_allowed_origins)
        if "*" in origins:
            raise ValueError("CORS con comodín '*' no está permitido; usa orígenes explícitos")
        return origins


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # los campos obligatorios vienen del entorno; si faltan, no arranca
