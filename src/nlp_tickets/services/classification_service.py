"""Caso de uso principal: clasificar y enrutar un ticket."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from decimal import Decimal

import numpy as np

from nlp_tickets.anonymization import Anonymizer
from nlp_tickets.db.models import Prediction, Ticket
from nlp_tickets.domain.catalog import Catalog
from nlp_tickets.domain.enums import (
    Channel,
    Decision,
    ReviewReason,
    Sentiment,
    TicketSource,
    Urgency,
)
from nlp_tickets.exceptions.errors import ValidationFailedError
from nlp_tickets.ml.routing import RoutingMatrices
from nlp_tickets.repositories.models import ModelRepository
from nlp_tickets.repositories.tickets import PredictionRepository, SimilarRow, TicketRepository
from nlp_tickets.services.decision_policy import DecisionInput, decide
from nlp_tickets.services.ports import TicketModel
from nlp_tickets.services.urgency_rules import evaluate_escalation

SENTIMENT_ORDER = tuple(Sentiment)


class CatalogModelMismatchError(RuntimeError):
    """El catálogo de la base no coincide con las salidas del modelo."""


@dataclass(frozen=True)
class ClassifyCommand:
    text: str
    channel: Channel = Channel.WEB
    external_ref: str | None = None
    persist: bool = True
    explain: bool = False
    include_similar: bool = False
    similar_limit: int = 3


@dataclass(frozen=True)
class Scored:
    code: str
    name: str
    score: float


@dataclass(frozen=True)
class ClassificationResult:
    ticket_id: uuid.UUID | None
    prediction_id: uuid.UUID | None
    model_name: str
    model_version: str
    decision: Decision
    review_reasons: tuple[ReviewReason, ...]
    threshold: float
    intent: Scored
    category: Scored
    team: Scored
    urgency: Urgency
    urgency_score: float
    model_urgency: Urgency
    escalated_by_rule: bool
    rules: tuple[str, ...]
    sentiment: Sentiment
    sentiment_score: float
    suggested_resolution: str
    anonymized_text: str
    entities: dict[str, int]
    top_terms: list[tuple[str, float]]
    latency_ms: int
    similar: list[SimilarRow] = field(default_factory=list)


def _score(value: float) -> Decimal:
    return Decimal(f"{min(1.0, max(0.0, value)):.4f}")


class ClassificationService:
    def __init__(
        self,
        model: TicketModel,
        catalog: Catalog,
        tickets: TicketRepository,
        predictions: PredictionRepository,
        models: ModelRepository,
        *,
        anonymizer: Anonymizer | None = None,
        threshold_override: float | None = None,
    ) -> None:
        self._model = model
        self._catalog = catalog
        self._routing = RoutingMatrices.from_catalog(catalog)
        self._tickets = tickets
        self._predictions = predictions
        self._models = models
        self._anonymizer = anonymizer or Anonymizer()
        self._threshold = (
            threshold_override if threshold_override is not None else model.confidence_threshold
        )

    @staticmethod
    def check_compatibility(model_intents: list[str], catalog: Catalog) -> None:
        if model_intents != catalog.intent_codes:
            raise CatalogModelMismatchError(
                "las intenciones del modelo no coinciden (en orden) con la tabla intents"
            )

    @property
    def threshold(self) -> float:
        return self._threshold

    def classify(self, command: ClassifyCommand) -> ClassificationResult:
        started = time.perf_counter()
        anonymized = self._anonymizer.anonymize(command.text)
        text = anonymized.text.strip()
        if len(text) < 3:
            raise ValidationFailedError("el texto quedó vacío tras anonimizar")

        output = self._model.predict(text)
        intent_probs = output.intent_probs
        category_probs = self._routing.category_probs(intent_probs)
        team_probs = self._routing.team_probs(intent_probs)
        urgency_probs = self._routing.urgency_probs(intent_probs)

        intent_def = self._catalog.intents[int(np.argmax(intent_probs))]
        category_idx = int(np.argmax(category_probs))
        team_idx = int(np.argmax(team_probs))
        urgency_order = list(Urgency)
        model_urgency = urgency_order[int(np.argmax(urgency_probs))]

        escalation = evaluate_escalation(text)
        escalated = escalation.escalate and model_urgency is not Urgency.HIGH
        urgency = Urgency.HIGH if escalated else model_urgency
        urgency_score = float(urgency_probs[urgency_order.index(urgency)])

        outcome = decide(
            DecisionInput(
                team_score=float(team_probs[team_idx]),
                predicted_urgency=model_urgency,
                high_urgency_probability=float(urgency_probs[urgency_order.index(Urgency.HIGH)]),
                escalated_by_rule=escalated,
                threshold=self._threshold,
            )
        )
        sentiment_idx = int(np.argmax(output.sentiment_probs))
        category_code = self._routing.category_codes[category_idx]
        team_code = self._routing.team_codes[team_idx]
        top_terms = self._model.explain(text, intent_def.label_id) if command.explain else []

        result = ClassificationResult(
            ticket_id=None,
            prediction_id=None,
            model_name=self._model.name,
            model_version=self._model.version,
            decision=outcome.decision,
            review_reasons=outcome.reasons,
            threshold=self._threshold,
            intent=Scored(
                intent_def.code, intent_def.name_es, float(intent_probs[intent_def.label_id])
            ),
            category=Scored(
                category_code,
                self._catalog.category_name(category_code),
                float(category_probs[category_idx]),
            ),
            team=Scored(team_code, self._catalog.team_name(team_code), float(team_probs[team_idx])),
            urgency=urgency,
            urgency_score=urgency_score,
            model_urgency=model_urgency,
            escalated_by_rule=escalated,
            rules=escalation.rules,
            sentiment=SENTIMENT_ORDER[sentiment_idx],
            sentiment_score=float(output.sentiment_probs[sentiment_idx]),
            suggested_resolution=intent_def.standard_resolution,
            anonymized_text=text,
            entities=anonymized.counts,
            top_terms=top_terms,
            latency_ms=0,
        )
        ticket_id, prediction_id = (None, None)
        if command.persist:
            elapsed = int((time.perf_counter() - started) * 1000)
            ticket_id, prediction_id = self._persist(command, result, output.embedding, elapsed)
        similar = (
            self._tickets.search_similar(
                output.embedding,
                embedding_model=self._embedding_model_id,
                limit=command.similar_limit,
                exclude_id=ticket_id,
            )
            if command.include_similar
            else []
        )
        return replace(
            result,
            ticket_id=ticket_id,
            prediction_id=prediction_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            similar=similar,
        )

    @property
    def _embedding_model_id(self) -> str:
        return f"{self._model.name}:{self._model.version}"

    def _persist(
        self,
        command: ClassifyCommand,
        result: ClassificationResult,
        embedding: np.ndarray,
        latency_ms: int,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        ticket = (
            self._tickets.get_by_external_ref(command.external_ref)
            if command.external_ref
            else None
        )
        if ticket is None:
            ticket = Ticket(
                external_ref=command.external_ref,
                source=TicketSource.API.value,
                channel=command.channel.value,
                text_anonymized=result.anonymized_text,
            )
        else:  # el helpdesk reenvía un ticket editado: se reclasifica el mismo registro
            ticket.text_anonymized = result.anonymized_text
            ticket.channel = command.channel.value
        ticket.embedding = np.asarray(embedding, dtype=np.float32).tolist()
        ticket.embedding_model = self._embedding_model_id
        self._tickets.add(ticket)

        active = self._models.get_active()
        if active is None or active.version != self._model.version:
            raise RuntimeError("la versión del modelo cargado no está registrada como activa")
        prediction = Prediction(
            ticket_id=ticket.id,
            model_version_id=active.id,
            intent_code=result.intent.code,
            intent_score=_score(result.intent.score),
            category_code=result.category.code,
            category_score=_score(result.category.score),
            team_code=result.team.code,
            team_score=_score(result.team.score),
            urgency=result.urgency.value,
            urgency_score=_score(result.urgency_score),
            urgency_escalated=result.escalated_by_rule,
            sentiment=result.sentiment.value,
            sentiment_score=_score(result.sentiment_score),
            decision=result.decision.value,
            review_reasons=[r.value for r in result.review_reasons],
            explanation={
                "top_terms": [{"term": t, "weight": round(w, 4)} for t, w in result.top_terms],
                "rules_triggered": list(result.rules),
                "anonymized_entities": result.entities,
            },
            latency_ms=latency_ms,
        )
        self._predictions.add(prediction)
        return ticket.id, prediction.id
