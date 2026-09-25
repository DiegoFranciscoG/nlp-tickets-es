"""Conversión entre resultados de dominio y DTO de la API."""

from __future__ import annotations

from nlp_tickets.db.models import Feedback, ModelMetric
from nlp_tickets.domain.catalog import Catalog
from nlp_tickets.repositories.tickets import SimilarRow
from nlp_tickets.schemas.tickets import (
    CatalogItem,
    CatalogResponse,
    ClassifyResponse,
    Explanation,
    FeedbackResponse,
    IntentItem,
    LabelScore,
    ModelMetricItem,
    ModelRef,
    SentimentResult,
    SimilarTicket,
    TermWeight,
    UrgencyResult,
)
from nlp_tickets.services.classification_service import ClassificationResult, Scored


def _label(scored: Scored) -> LabelScore:
    return LabelScore(code=scored.code, name=scored.name, score=round(scored.score, 4))


def to_similar(rows: list[SimilarRow], catalog: Catalog) -> list[SimilarTicket]:
    return [
        SimilarTicket(
            ticket_id=row.ticket.id,
            similarity=round(row.similarity, 4),
            intent=catalog.intent(row.intent_code).name_es if row.intent_code else None,
            category=catalog.category_name(row.category_code) if row.category_code else None,
            text_anonymized=row.ticket.text_anonymized,
            resolution_text=row.ticket.resolution_text,
        )
        for row in rows
    ]


def to_classify_response(
    result: ClassificationResult, catalog: Catalog, include_similar: bool
) -> ClassifyResponse:
    return ClassifyResponse(
        ticket_id=result.ticket_id,
        prediction_id=result.prediction_id,
        model=ModelRef(name=result.model_name, version=result.model_version),
        decision=result.decision,
        review_reasons=list(result.review_reasons),
        confidence_threshold=result.threshold,
        intent=_label(result.intent),
        category=_label(result.category),
        team=_label(result.team),
        urgency=UrgencyResult(
            code=result.urgency,
            score=round(result.urgency_score, 4),
            model_code=result.model_urgency,
            escalated_by_rule=result.escalated_by_rule,
            rules=list(result.rules),
        ),
        sentiment=SentimentResult(code=result.sentiment, score=round(result.sentiment_score, 4)),
        suggested_resolution=result.suggested_resolution,
        anonymized_text=result.anonymized_text,
        explanation=Explanation(
            top_terms=[TermWeight(term=t, weight=round(w, 4)) for t, w in result.top_terms],
            rules_triggered=list(result.rules),
            anonymized_entities=result.entities,
        ),
        similar_tickets=to_similar(result.similar, catalog) if include_similar else None,
        latency_ms=result.latency_ms,
    )


def to_feedback_response(feedback: Feedback) -> FeedbackResponse:
    return FeedbackResponse(
        id=feedback.id,
        prediction_id=feedback.prediction_id,
        status=feedback.status,
        created_at=feedback.created_at,
    )


def to_catalog_response(catalog: Catalog) -> CatalogResponse:
    return CatalogResponse(
        categories=[CatalogItem(code=c.code, name=c.name) for c in catalog.categories],
        teams=[CatalogItem(code=t.code, name=t.name) for t in catalog.teams],
        intents=[
            IntentItem(
                code=i.code,
                name=i.name_es,
                category=i.category_code,
                team=i.team_code,
                base_urgency=i.base_urgency,
            )
            for i in catalog.intents
        ],
    )


def to_metric_items(metrics: list[ModelMetric]) -> list[ModelMetricItem]:
    def num(value: object) -> float | None:
        return None if value is None else float(value)  # type: ignore[arg-type]

    return [
        ModelMetricItem(
            task=m.task,
            split=m.split,
            macro_f1=num(m.macro_f1),
            accuracy=num(m.accuracy),
            ece=num(m.ece),
            coverage=num(m.coverage),
        )
        for m in metrics
    ]
