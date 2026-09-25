"""Acceso a datos de tickets, predicciones y retroalimentación (SQL siempre parametrizado)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from nlp_tickets.db.models import Feedback, Prediction, Ticket, TicketLabel
from nlp_tickets.ml.routing import FloatArray


@dataclass(frozen=True)
class SimilarRow:
    ticket: Ticket
    similarity: float
    intent_code: str | None
    category_code: str | None


class TicketRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, ticket: Ticket) -> Ticket:
        self._session.add(ticket)
        self._session.flush()
        return ticket

    def get(self, ticket_id: uuid.UUID) -> Ticket | None:
        return self._session.get(Ticket, ticket_id)

    def get_by_external_ref(self, external_ref: str) -> Ticket | None:
        return self._session.scalar(select(Ticket).where(Ticket.external_ref == external_ref))

    def count_by_source(self, source: str) -> int:
        return int(self._session.scalar(select(func.count()).where(Ticket.source == source)) or 0)

    def search_similar(
        self,
        embedding: FloatArray,
        *,
        embedding_model: str,
        limit: int,
        exclude_id: uuid.UUID | None = None,
    ) -> list[SimilarRow]:
        """k vecinos por distancia coseno (índice HNSW) entre tickets resueltos del mismo modelo."""
        vector = np.asarray(embedding, dtype=np.float32).tolist()
        distance = Ticket.embedding.cosine_distance(vector)
        statement = (
            select(
                Ticket,
                distance.label("distance"),
                TicketLabel.intent_code,
                TicketLabel.category_code,
            )
            .outerjoin(TicketLabel, TicketLabel.ticket_id == Ticket.id)
            .where(Ticket.resolution_text.is_not(None))
            .where(Ticket.embedding_model == embedding_model)
            .order_by(distance)
            .limit(limit)
        )
        if exclude_id is not None:
            statement = statement.where(Ticket.id != exclude_id)
        return [
            SimilarRow(ticket, max(0.0, 1.0 - float(dist)), intent, category)
            for ticket, dist, intent, category in self._session.execute(statement).all()
        ]

    def add_label(self, label: TicketLabel) -> None:
        self._session.add(label)


class PredictionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, prediction: Prediction) -> Prediction:
        self._session.add(prediction)
        self._session.flush()
        return prediction

    def get(self, prediction_id: uuid.UUID) -> Prediction | None:
        return self._session.get(Prediction, prediction_id)


class FeedbackRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, feedback: Feedback) -> Feedback:
        self._session.add(feedback)
        self._session.flush()
        self._session.refresh(feedback)
        return feedback

    def accepted_with_intent(self) -> list[tuple[Feedback, str]]:
        """Correcciones aptas para reentrenar: aceptadas y con intención corregida."""
        statement = (
            select(Feedback, Ticket.text_anonymized)
            .join(Ticket, Ticket.id == Feedback.ticket_id)
            .where(Feedback.status == "accepted")
            .where(Feedback.corrected_intent_code.is_not(None))
            .order_by(Feedback.created_at)
        )
        return [(fb, text) for fb, text in self._session.execute(statement).all()]

    def mark_used(self, feedback_ids: Sequence[uuid.UUID], model_version_id: int) -> int:
        if not feedback_ids:
            return 0
        result = self._session.execute(
            update(Feedback)
            .where(Feedback.id.in_(list(feedback_ids)))
            .values(used_in_model_version_id=model_version_id)
        )
        return int(result.rowcount or 0)  # type: ignore[attr-defined]
