"""DTO de la API (contrato con el helpdesk P34). Validación estricta y límites de tamaño."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nlp_tickets.domain.enums import Channel, Decision, ReviewReason, Sentiment, Urgency

TicketText = Annotated[str, Field(min_length=3, max_length=5000)]
ExternalRef = Annotated[str, Field(max_length=64, pattern=r"^[A-Za-z0-9._:\-]+$")]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ClassifyRequest(StrictModel):
    text: TicketText
    channel: Channel = Channel.WEB
    external_ref: ExternalRef | None = None
    persist: bool = True
    explain: bool = False
    include_similar: bool = False
    similar_limit: int = Field(default=3, ge=1, le=10)

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "text": "Me robaron la tarjeta y veo compras que no reconozco, ayuda urgente",
                    "channel": "chat",
                    "external_ref": "HD-2026-000123",
                    "explain": True,
                    "include_similar": True,
                }
            ]
        },
    )


class LabelScore(BaseModel):
    code: str
    name: str
    score: Score


class UrgencyResult(BaseModel):
    code: Urgency
    score: Score
    model_code: Urgency = Field(description="Urgencia predicha por el modelo antes de las reglas")
    escalated_by_rule: bool
    rules: list[str]


class SentimentResult(BaseModel):
    code: Sentiment
    score: Score


class TermWeight(BaseModel):
    term: str
    weight: float


class Explanation(BaseModel):
    top_terms: list[TermWeight]
    rules_triggered: list[str]
    anonymized_entities: dict[str, int]


class ModelRef(BaseModel):
    name: str
    version: str


class SimilarTicket(BaseModel):
    ticket_id: uuid.UUID
    similarity: Score
    intent: str | None
    category: str | None
    text_anonymized: str
    resolution_text: str | None


class ClassifyResponse(BaseModel):
    ticket_id: uuid.UUID | None
    prediction_id: uuid.UUID | None
    model: ModelRef
    decision: Decision
    review_reasons: list[ReviewReason]
    confidence_threshold: Score
    intent: LabelScore
    category: LabelScore
    team: LabelScore
    urgency: UrgencyResult
    sentiment: SentimentResult
    suggested_resolution: str
    anonymized_text: str
    explanation: Explanation
    similar_tickets: list[SimilarTicket] | None = None
    latency_ms: int


class SimilarRequest(StrictModel):
    text: TicketText
    limit: int = Field(default=5, ge=1, le=20)


class SimilarResponse(BaseModel):
    items: list[SimilarTicket]


class FeedbackRequest(StrictModel):
    prediction_id: uuid.UUID
    corrected_intent: str | None = Field(default=None, max_length=60)
    corrected_category: str | None = Field(default=None, max_length=40)
    corrected_team: str | None = Field(default=None, max_length=40)
    corrected_urgency: Urgency | None = None
    corrected_sentiment: Sentiment | None = None
    comment: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _at_least_one_correction(self) -> FeedbackRequest:
        corrections = (
            self.corrected_intent,
            self.corrected_category,
            self.corrected_team,
            self.corrected_urgency,
            self.corrected_sentiment,
        )
        if all(value is None for value in corrections):
            raise ValueError("indica al menos una corrección")
        return self


class FeedbackResponse(BaseModel):
    id: uuid.UUID
    prediction_id: uuid.UUID
    status: str
    created_at: datetime


class CatalogItem(BaseModel):
    code: str
    name: str


class IntentItem(CatalogItem):
    category: str
    team: str
    base_urgency: Urgency


class CatalogResponse(BaseModel):
    categories: list[CatalogItem]
    teams: list[CatalogItem]
    intents: list[IntentItem]


class ModelMetricItem(BaseModel):
    task: str
    split: str
    macro_f1: float | None
    accuracy: float | None
    ece: float | None
    coverage: float | None
    selective_accuracy: float | None


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    confidence_threshold: float
    trained_at: str
    metrics: list[ModelMetricItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str | None = None
