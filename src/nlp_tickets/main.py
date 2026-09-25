"""Fábrica de la aplicación: ``uvicorn nlp_tickets.main:create_app --factory``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker

from nlp_tickets import __version__
from nlp_tickets.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from nlp_tickets.api.routes import api_router, health_router
from nlp_tickets.config import Settings, get_settings
from nlp_tickets.db.session import get_session_factory
from nlp_tickets.exceptions.handlers import register_exception_handlers
from nlp_tickets.ml.inference import OnnxTicketModel
from nlp_tickets.ml.model_store import ensure_model
from nlp_tickets.repositories.models import CatalogRepository, ModelRepository
from nlp_tickets.security.api_key import ApiKeyAuthenticator
from nlp_tickets.security.rate_limit import SlidingWindowRateLimiter
from nlp_tickets.services.classification_service import ClassificationService
from nlp_tickets.services.ports import TicketModel
from nlp_tickets.services.registry_service import ModelRegistryService

log = logging.getLogger("nlp_tickets")
REPORT_PATH = Path("reports/metrics.json")
DESCRIPTION = """
Clasifica tickets de soporte en español (intención, categoría, urgencia y sentimiento),
sugiere el equipo destino y decide si el caso requiere **revisión humana**.

- Autenticación: cabecera `X-API-Key` (roles `client` y `reviewer`).
- El texto se anonimiza antes de procesarlo; nunca se guarda el original.
"""


def _load_model(settings: Settings) -> OnnxTicketModel | None:
    try:
        if not ensure_model(settings.model_dir, settings.model_url, settings.model_sha256):
            log.warning("no hay modelo en %s: la API arranca en modo degradado", settings.model_dir)
            return None
        return OnnxTicketModel(settings.model_dir, expected_sha256=settings.model_sha256)
    except Exception as exc:  # el servicio sigue vivo para exponer /health
        log.error("no se pudo cargar el modelo: %s", type(exc).__name__)
        return None


def _bootstrap(app: FastAPI, model: TicketModel | None) -> None:
    """Carga el catálogo desde la BD, valida el modelo y lo registra como versión activa."""
    session: Session = app.state.session_factory()
    try:
        catalog = CatalogRepository(session).load()
        app.state.catalog = catalog
        if isinstance(model, OnnxTicketModel):
            ClassificationService.check_compatibility(model.meta.intent_labels, catalog)
            ModelRegistryService(ModelRepository(session)).register(model.meta, REPORT_PATH)
            session.commit()
        app.state.model = model
    except Exception as exc:
        session.rollback()
        log.error("arranque degradado (catálogo/registro): %s", type(exc).__name__)
        app.state.model = None
    finally:
        session.close()


def create_app(
    settings: Settings | None = None,
    *,
    model: TicketModel | None = None,
    session_factory: sessionmaker[Session] | None = None,
    load_model: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.session_factory = session_factory or get_session_factory()
        loaded = model if model is not None else (_load_model(settings) if load_model else None)
        _bootstrap(app, loaded)
        yield

    app = FastAPI(
        title="nlp-tickets-es",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    app.state.model = None
    app.state.catalog = None
    app.state.authenticator = ApiKeyAuthenticator(settings.client_keys, settings.reviewer_keys)
    app.state.rate_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)

    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.app_env == "production")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
        allow_credentials=False,
        max_age=600,
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(api_router)
    return app
