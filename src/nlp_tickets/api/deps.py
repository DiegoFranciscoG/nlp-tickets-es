"""Dependencias de FastAPI: sesión, autenticación, rate limiting y servicios."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from nlp_tickets.domain.catalog import Catalog
from nlp_tickets.exceptions.errors import (
    AuthenticationError,
    ForbiddenError,
    ModelUnavailableError,
    RateLimitError,
)
from nlp_tickets.repositories.models import CatalogRepository, ModelRepository
from nlp_tickets.repositories.tickets import (
    FeedbackRepository,
    PredictionRepository,
    TicketRepository,
)
from nlp_tickets.security.api_key import ApiKeyAuthenticator, Principal
from nlp_tickets.services.classification_service import ClassificationService
from nlp_tickets.services.feedback_service import FeedbackService
from nlp_tickets.services.ports import TicketModel
from nlp_tickets.services.similarity_service import SimilarityService

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key")


def get_session(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


def get_principal(
    request: Request, api_key: Annotated[str | None, Security(api_key_header)]
) -> Principal:
    authenticator: ApiKeyAuthenticator = request.app.state.authenticator
    principal = authenticator.authenticate(api_key)
    if principal is None:
        raise AuthenticationError()
    identity = hashlib.sha256(principal.api_key.encode()).hexdigest()[:16]
    allowed, _ = request.app.state.rate_limiter.allow(identity)
    if not allowed:
        raise RateLimitError()
    return principal


def require_reviewer(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
    if not principal.can_review:
        raise ForbiddenError()
    return principal


ClientDep = Annotated[Principal, Depends(get_principal)]
ReviewerDep = Annotated[Principal, Depends(require_reviewer)]


def get_model(request: Request) -> TicketModel:
    model: TicketModel | None = request.app.state.model
    if model is None:
        raise ModelUnavailableError()
    return model


def get_catalog(request: Request, session: SessionDep) -> Catalog:
    catalog: Catalog | None = request.app.state.catalog
    if catalog is None:
        catalog = CatalogRepository(session).load()
        request.app.state.catalog = catalog
    return catalog


ModelDep = Annotated[TicketModel, Depends(get_model)]
CatalogDep = Annotated[Catalog, Depends(get_catalog)]


def classification_service(
    request: Request, session: SessionDep, model: ModelDep, catalog: CatalogDep
) -> ClassificationService:
    return ClassificationService(
        model,
        catalog,
        TicketRepository(session),
        PredictionRepository(session),
        ModelRepository(session),
        threshold_override=request.app.state.settings.confidence_threshold_override,
    )


def similarity_service(session: SessionDep, model: ModelDep) -> SimilarityService:
    return SimilarityService(model, TicketRepository(session))


def feedback_service(session: SessionDep, catalog: CatalogDep) -> FeedbackService:
    return FeedbackService(catalog, PredictionRepository(session), FeedbackRepository(session))
