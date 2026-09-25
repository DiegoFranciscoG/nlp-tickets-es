"""Flujo completo de la API contra PostgreSQL + pgvector reales (modelo sustituido por un doble)."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import sessionmaker

from nlp_tickets.config import Settings
from nlp_tickets.db.models import Feedback, ModelVersion, Prediction, Ticket
from nlp_tickets.domain.catalog import load_catalog
from nlp_tickets.main import create_app
from nlp_tickets.repositories.models import CatalogRepository
from nlp_tickets.repositories.tickets import FeedbackRepository, TicketRepository
from nlp_tickets.services.feedback_service import TrainingDataExporter
from nlp_tickets.services.seed_service import SeedService
from tests.fakes import FakeModel

pytestmark = pytest.mark.integration

CLIENT_KEY = "client-key-" + "c" * 32
REVIEWER_KEY = "reviewer-key-" + "r" * 32
CLIENT = {"X-API-Key": CLIENT_KEY}
REVIEWER = {"X-API-Key": REVIEWER_KEY}
SEED_ROWS = [
    ("Mi tarjeta todavía no llega", "card_arrival", "chat"),
    ("¿Cuándo llega la tarjeta nueva?", "card_arrival", "web"),
    ("Me robaron la billetera con la tarjeta", "lost_or_stolen_card", "phone"),
    ("¿Cuál es el tipo de cambio del euro?", "exchange_rate", "email"),
]


def _settings(database_url: str, rate_limit: int = 1000) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        database_url=database_url,
        api_keys_client=CLIENT_KEY,
        api_keys_reviewer=REVIEWER_KEY,
        cors_allowed_origins="http://localhost:8501",
        rate_limit_per_minute=rate_limit,
    )


@pytest.fixture(scope="module")
def prepared_db(engine: Engine, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    factory = sessionmaker(bind=engine)
    catalog = load_catalog()
    model = FakeModel(catalog)
    seed_file = tmp_path_factory.mktemp("seed") / "seed.csv"
    with seed_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "intent_code", "channel", "created_at"])
        writer.writerows((t, i, c, "2026-03-01T10:00+00:00") for t, i, c in SEED_ROWS)
    with factory() as session:
        session.add(
            ModelVersion(
                name=model.name,
                version=model.version,
                backend="onnx",
                artifact_sha256="0" * 64,
                confidence_threshold=Decimal("0.500"),
                dataset_revision="test",
                is_active=True,
                trained_at=datetime.now(UTC),
            )
        )
        service = SeedService(CatalogRepository(session).load(), TicketRepository(session), model)
        assert service.seed(seed_file) == len(SEED_ROWS)
        assert service.seed(seed_file) == 0  # idempotente
        session.commit()
    yield
    with engine.begin() as conn:
        for table in (
            "feedback",
            "predictions",
            "ticket_labels",
            "tickets",
            "model_metrics",
            "model_versions",
        ):
            conn.exec_driver_sql(f"DELETE FROM {table}")  # noqa: S608


@pytest.fixture
def client(database_url: str, engine: Engine, prepared_db: None) -> Iterator[TestClient]:
    app = create_app(
        _settings(database_url),
        model=FakeModel(load_catalog()),
        session_factory=sessionmaker(bind=engine, expire_on_commit=False),
    )
    with TestClient(app) as test_client:
        yield test_client


class TestSeguridad:
    def test_health_es_publico(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body == {"status": "ok", "model_loaded": True, "model_version": "test-1"}
        assert client.get("/health/ready").status_code == 200

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("post", "/api/v1/classify"),
            ("get", "/api/v1/catalog"),
            ("get", "/api/v1/model"),
            ("post", "/api/v1/tickets/similar"),
            ("post", "/api/v1/feedback"),
        ],
    )
    def test_deny_by_default(self, client: TestClient, method: str, path: str) -> None:
        response = client.request(method, path, json={})
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"

    def test_clave_invalida(self, client: TestClient) -> None:
        response = client.get("/api/v1/catalog", headers={"X-API-Key": "x" * 40})
        assert response.status_code == 401

    def test_cabeceras_de_seguridad(self, client: TestClient) -> None:
        headers = client.get("/api/v1/catalog", headers=CLIENT).headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["Content-Security-Policy"].startswith("default-src 'none'")
        assert headers["Cache-Control"] == "no-store"
        assert "X-Request-ID" in headers

    def test_cors_solo_origen_permitido(self, client: TestClient) -> None:
        allowed = client.options(
            "/api/v1/catalog",
            headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "GET"},
        )
        denied = client.options(
            "/api/v1/catalog",
            headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "GET"},
        )
        assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8501"
        assert "access-control-allow-origin" not in denied.headers

    def test_error_de_validacion_no_devuelve_la_entrada(self, client: TestClient) -> None:
        secreto = "correo privado persona@example.com"
        response = client.post(
            "/api/v1/classify", headers=CLIENT, json={"text": secreto, "channel": "fax"}
        )
        assert response.status_code == 422
        assert "persona@example.com" not in response.text

    def test_campos_desconocidos_rechazados(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/classify", headers=CLIENT, json={"text": "hola hola", "admin": True}
        )
        assert response.status_code == 422

    def test_cuerpo_demasiado_grande(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/classify",
            headers={**CLIENT, "Content-Type": "application/json"},
            content=b'{"text": "' + b"a" * 40_000 + b'"}',
        )
        assert response.status_code == 413

    def test_rate_limit(self, database_url: str, engine: Engine, prepared_db: None) -> None:
        app = create_app(
            _settings(database_url, rate_limit=2),
            model=FakeModel(load_catalog()),
            session_factory=sessionmaker(bind=engine),
        )
        with TestClient(app) as limited:
            codes = [limited.get("/api/v1/catalog", headers=CLIENT).status_code for _ in range(3)]
        assert codes == [200, 200, 429]


class TestClasificacion:
    def test_clasifica_anonimiza_y_persiste(self, client: TestClient, engine: Engine) -> None:
        response = client.post(
            "/api/v1/classify",
            headers=CLIENT,
            json={
                "text": "Hola, soy Andrea Paredes, mi tarjeta no llega. Correo andrea@example.com",
                "channel": "email",
                "external_ref": "HD-1",
                "explain": True,
                "include_similar": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["intent"]["code"] == "card_arrival"
        assert body["category"]["code"] == "cards"
        assert body["team"]["code"] == "card_operations"
        assert body["decision"] == "auto_route"
        assert "<NOMBRE>" in body["anonymized_text"]
        assert "<CORREO>" in body["anonymized_text"]
        assert body["explanation"]["anonymized_entities"] == {"NOMBRE": 1, "CORREO": 1}
        assert body["explanation"]["top_terms"]
        similar = body["similar_tickets"]
        assert similar
        assert all(item["resolution_text"] for item in similar)
        scores = [item["similarity"] for item in similar]
        assert scores == sorted(scores, reverse=True)
        with sessionmaker(bind=engine)() as session:
            ticket = session.get(Ticket, body["ticket_id"])
            assert ticket is not None
            assert "andrea@example.com" not in ticket.text_anonymized
            assert "Andrea" not in ticket.text_anonymized
            assert session.get(Prediction, body["prediction_id"]) is not None

    def test_mismo_external_ref_reusa_el_ticket(self, client: TestClient) -> None:
        first = client.post(
            "/api/v1/classify",
            headers=CLIENT,
            json={"text": "tarjeta no llega", "external_ref": "HD-7"},
        ).json()
        second = client.post(
            "/api/v1/classify",
            headers=CLIENT,
            json={"text": "la tarjeta sigue sin llegar", "external_ref": "HD-7"},
        ).json()
        assert first["ticket_id"] == second["ticket_id"]
        assert first["prediction_id"] != second["prediction_id"]

    def test_fraude_escala_y_va_a_revision(self, client: TestClient) -> None:
        body = client.post(
            "/api/v1/classify",
            headers=CLIENT,
            json={"text": "Veo una compra que no reconozco, creo que es fraude", "persist": False},
        ).json()
        assert body["ticket_id"] is None
        assert body["urgency"]["code"] == "high"
        assert "fraude" in body["urgency"]["rules"]

    def test_baja_confianza_va_a_revision(
        self, database_url: str, engine: Engine, prepared_db: None
    ) -> None:
        app = create_app(
            _settings(database_url),
            model=FakeModel(load_catalog(), confidence=0.2),
            session_factory=sessionmaker(bind=engine),
        )
        with TestClient(app) as low:
            body = low.post(
                "/api/v1/classify", headers=CLIENT, json={"text": "algo raro", "persist": False}
            ).json()
        assert body["decision"] == "human_review"
        assert "low_confidence" in body["review_reasons"]

    def test_texto_solo_con_datos_personales(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/classify", headers=CLIENT, json={"text": "a@example.com", "persist": False}
        )
        assert response.status_code == 200  # queda "<CORREO>", que es texto válido

    def test_similares_por_texto_y_por_ticket(self, client: TestClient) -> None:
        by_text = client.post(
            "/api/v1/tickets/similar",
            headers=CLIENT,
            json={"text": "¿Cuál es el tipo de cambio del euro?", "limit": 2},
        ).json()["items"]
        assert by_text[0]["similarity"] == pytest.approx(1.0, abs=1e-3)
        ticket_id = client.post(
            "/api/v1/classify", headers=CLIENT, json={"text": "tarjeta no llega aún"}
        ).json()["ticket_id"]
        by_ticket = client.get(f"/api/v1/tickets/{ticket_id}/similar?limit=3", headers=CLIENT)
        assert by_ticket.status_code == 200
        assert all(item["ticket_id"] != ticket_id for item in by_ticket.json()["items"])
        missing = client.get(
            "/api/v1/tickets/00000000-0000-0000-0000-000000000000/similar", headers=CLIENT
        )
        assert missing.status_code == 404

    def test_catalogo_y_modelo(self, client: TestClient) -> None:
        catalog = client.get("/api/v1/catalog", headers=CLIENT).json()
        assert len(catalog["intents"]) == 77
        model = client.get("/api/v1/model", headers=CLIENT).json()
        assert model["version"] == "test-1"


class TestRetroalimentacion:
    def _prediction(self, client: TestClient, text: str = "tarjeta no llega todavía") -> str:
        return str(
            client.post("/api/v1/classify", headers=CLIENT, json={"text": text}).json()[
                "prediction_id"
            ]
        )

    def test_cliente_no_puede_corregir(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/feedback",
            headers=CLIENT,
            json={"prediction_id": self._prediction(client), "corrected_urgency": "high"},
        )
        assert response.status_code == 403

    def test_revisor_corrige_y_se_exporta(self, client: TestClient, engine: Engine) -> None:
        prediction_id = self._prediction(client, "la tarjeta no llega, quiero otra")
        response = client.post(
            "/api/v1/feedback",
            headers=REVIEWER,
            json={
                "prediction_id": prediction_id,
                "corrected_intent": "getting_spare_card",
                "corrected_team": "card_operations",
                "comment": "Llamé a la clienta al 0990000001",
            },
        )
        assert response.status_code == 201
        with sessionmaker(bind=engine)() as session:
            stored = session.scalar(select(Feedback).where(Feedback.id == response.json()["id"]))
            assert stored is not None
            assert stored.reviewer_ref.startswith("rev_")
            assert REVIEWER_KEY not in stored.reviewer_ref
            assert stored.comment_anonymized == "Llamé a la clienta al <TELEFONO>"
            rows = TrainingDataExporter(load_catalog(), FeedbackRepository(session)).export()
        labels = {row.label for row in rows}
        assert load_catalog().intent("getting_spare_card").label_id in labels

    @pytest.mark.parametrize(
        ("payload", "status"),
        [
            ({"corrected_intent": "no_existe"}, 422),
            ({"corrected_team": "no_existe"}, 422),
            ({"corrected_category": "no_existe"}, 422),
            ({}, 422),
        ],
    )
    def test_validaciones(self, client: TestClient, payload: dict[str, str], status: int) -> None:
        body = {"prediction_id": self._prediction(client), **payload}
        assert client.post("/api/v1/feedback", headers=REVIEWER, json=body).status_code == status

    def test_prediccion_inexistente(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/feedback",
            headers=REVIEWER,
            json={
                "prediction_id": "00000000-0000-0000-0000-000000000000",
                "corrected_urgency": "low",
            },
        )
        assert response.status_code == 404


def test_modo_degradado_sin_modelo(database_url: str, engine: Engine, tmp_path: Path) -> None:
    settings = _settings(database_url)
    settings.model_dir = tmp_path / "vacio"
    app = create_app(settings, session_factory=sessionmaker(bind=engine))
    with TestClient(app) as degraded:
        assert degraded.get("/health").json()["status"] == "degraded"
        response = degraded.post("/api/v1/classify", headers=CLIENT, json={"text": "hola hola"})
    assert response.status_code == 503
