import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from nlp_tickets.db.base import Base
from nlp_tickets.db.models import Ticket

pytestmark = pytest.mark.integration


def test_migraciones_coinciden_con_los_modelos(engine: Engine) -> None:
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_catalogos_cargados(engine: Engine) -> None:
    with engine.connect() as conn:
        counts = {
            table: conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()  # noqa: S608
            for table in ("categories", "teams", "intents")
        }
    assert counts == {"categories": 9, "teams": 6, "intents": 77}


def test_extension_vector_e_indice_hnsw(engine: Engine) -> None:
    with engine.connect() as conn:
        assert conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname='vector'")
        ).scalar()
        index = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname='ix_tickets_embedding_hnsw'")
        ).scalar_one()
    assert "hnsw" in index
    assert "vector_cosine_ops" in index


def test_check_constraint_de_canal(session: Session) -> None:
    session.add(Ticket(text_anonymized="hola", channel="fax", source="api"))
    with pytest.raises(IntegrityError):
        session.flush()
