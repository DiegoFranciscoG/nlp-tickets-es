"""Infraestructura de integración: PostgreSQL 17 + pgvector real con Testcontainers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

PGVECTOR_IMAGE = "pgvector/pgvector:0.8.6-pg17"
ROOT = Path(__file__).resolve().parents[2]


def _docker_available() -> bool:
    try:
        import docker  # type: ignore[import-untyped]  # noqa: PLC0415

        docker.from_env().ping()
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("Docker no está disponible para Testcontainers")
    from testcontainers.community.postgres import PostgresContainer  # noqa: PLC0415

    with PostgresContainer(PGVECTOR_IMAGE, driver="psycopg") as container:
        url = container.get_connection_url()
        config = Config(str(ROOT / "alembic.ini"))
        config.attributes["database_url"] = url
        command.upgrade(config, "head")
        yield url


@pytest.fixture(scope="session")
def engine(database_url: str) -> Iterator[Engine]:
    engine = create_engine(database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Sesión aislada: todo lo que haga el test se revierte al terminar."""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection, join_transaction_mode="create_savepoint")()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def clean_tables(engine: Engine) -> Iterator[None]:
    yield
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE feedback, predictions, ticket_labels, tickets, model_metrics, "
                "model_versions RESTART IDENTITY CASCADE"
            )
        )
