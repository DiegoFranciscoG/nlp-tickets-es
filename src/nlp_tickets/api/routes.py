"""Controladores REST. Toda ruta bajo /api/v1 exige API key (deny-by-default)."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy import text

from nlp_tickets.api.deps import (
    CatalogDep,
    ClientDep,
    ReviewerDep,
    SessionDep,
    classification_service,
    feedback_service,
    similarity_service,
)
from nlp_tickets.exceptions.errors import ModelUnavailableError
from nlp_tickets.mappers.classification import (
    to_catalog_response,
    to_classify_response,
    to_feedback_response,
    to_metric_items,
    to_similar,
)
from nlp_tickets.repositories.models import ModelRepository
from nlp_tickets.schemas.tickets import (
    CatalogResponse,
    ClassifyRequest,
    ClassifyResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    SimilarRequest,
    SimilarResponse,
)
from nlp_tickets.services.classification_service import ClassificationService, ClassifyCommand
from nlp_tickets.services.feedback_service import FeedbackCommand, FeedbackService
from nlp_tickets.services.similarity_service import SimilarityService

ERRORS: dict[int | str, dict[str, object]] = {
    401: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}

health_router = APIRouter(tags=["salud"])
api_router = APIRouter(prefix="/api/v1", responses=ERRORS)


@health_router.get("/health", response_model=HealthResponse, summary="Liveness (público)")
def health(request: Request) -> HealthResponse:
    model = request.app.state.model
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model.version if model is not None else None,
    )


@health_router.get("/health/ready", response_model=HealthResponse, summary="Readiness con BD")
def ready(request: Request, session: SessionDep) -> HealthResponse:
    session.execute(text("SELECT 1"))
    return health(request)


@api_router.post(
    "/classify",
    response_model=ClassifyResponse,
    tags=["clasificación"],
    summary="Clasifica, enruta y decide si requiere revisión humana",
)
def classify(
    body: ClassifyRequest,
    _: ClientDep,
    catalog: CatalogDep,
    service: Annotated[ClassificationService, Depends(classification_service)],
) -> ClassifyResponse:
    result = service.classify(ClassifyCommand(**body.model_dump()))
    return to_classify_response(result, catalog, body.include_similar)


@api_router.post(
    "/tickets/similar",
    response_model=SimilarResponse,
    tags=["búsqueda semántica"],
    summary="Tickets resueltos parecidos a un texto nuevo",
)
def similar_to_text(
    body: SimilarRequest,
    _: ClientDep,
    catalog: CatalogDep,
    service: Annotated[SimilarityService, Depends(similarity_service)],
) -> SimilarResponse:
    return SimilarResponse(
        items=to_similar(service.similar_to_text(body.text, body.limit), catalog)
    )


@api_router.get(
    "/tickets/{ticket_id}/similar",
    response_model=SimilarResponse,
    tags=["búsqueda semántica"],
    summary="Tickets resueltos parecidos a un ticket existente",
)
def similar_to_ticket(
    ticket_id: uuid.UUID,
    _: ClientDep,
    catalog: CatalogDep,
    service: Annotated[SimilarityService, Depends(similarity_service)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> SimilarResponse:
    return SimilarResponse(items=to_similar(service.similar_to_ticket(ticket_id, limit), catalog))


@api_router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["retroalimentación"],
    summary="Registra la corrección de un revisor (rol reviewer)",
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def submit_feedback(
    body: FeedbackRequest,
    principal: ReviewerDep,
    service: Annotated[FeedbackService, Depends(feedback_service)],
) -> FeedbackResponse:
    feedback = service.submit(FeedbackCommand(**body.model_dump()), principal.api_key)
    return to_feedback_response(feedback)


@api_router.get(
    "/model", response_model=ModelInfoResponse, tags=["modelo"], summary="Modelo activo y métricas"
)
def model_info(request: Request, _: ClientDep, session: SessionDep) -> ModelInfoResponse:
    model = request.app.state.model
    if model is None:
        raise ModelUnavailableError()
    repository = ModelRepository(session)
    active = repository.get_active()
    return ModelInfoResponse(
        name=model.name,
        version=model.version,
        confidence_threshold=model.confidence_threshold,
        trained_at=model.trained_at,
        metrics=to_metric_items(repository.metrics(active.id)) if active else [],
    )


@api_router.get(
    "/catalog", response_model=CatalogResponse, tags=["catálogo"], summary="Categorías y equipos"
)
def catalog(_: ClientDep, catalog: CatalogDep) -> CatalogResponse:
    return to_catalog_response(catalog)
