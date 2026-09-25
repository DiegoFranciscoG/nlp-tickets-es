"""Bucle de retroalimentación: correcciones humanas auditadas que alimentan el reentrenamiento."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from nlp_tickets.anonymization import Anonymizer
from nlp_tickets.db.models import Feedback
from nlp_tickets.domain.catalog import Catalog
from nlp_tickets.domain.enums import FeedbackStatus, Sentiment, Urgency
from nlp_tickets.exceptions.errors import NotFoundError, ValidationFailedError
from nlp_tickets.repositories.tickets import FeedbackRepository, PredictionRepository


def reviewer_pseudonym(api_key: str) -> str:
    """Seudónimo estable del revisor (LOPDP art. 4): nunca se guarda la clave ni un nombre."""
    return "rev_" + hashlib.sha256(api_key.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class FeedbackCommand:
    prediction_id: uuid.UUID
    corrected_intent: str | None = None
    corrected_category: str | None = None
    corrected_team: str | None = None
    corrected_urgency: Urgency | None = None
    corrected_sentiment: Sentiment | None = None
    comment: str | None = None


class FeedbackService:
    def __init__(
        self,
        catalog: Catalog,
        predictions: PredictionRepository,
        feedback: FeedbackRepository,
        anonymizer: Anonymizer | None = None,
    ) -> None:
        self._catalog = catalog
        self._predictions = predictions
        self._feedback = feedback
        self._anonymizer = anonymizer or Anonymizer()

    def submit(self, command: FeedbackCommand, reviewer_key: str) -> Feedback:
        prediction = self._predictions.get(command.prediction_id)
        if prediction is None:
            raise NotFoundError("predicción no encontrada")
        self._validate_codes(command)
        comment = self._anonymizer.anonymize(command.comment).text if command.comment else None
        return self._feedback.add(
            Feedback(
                prediction_id=prediction.id,
                ticket_id=prediction.ticket_id,
                corrected_intent_code=command.corrected_intent,
                corrected_category_code=command.corrected_category,
                corrected_team_code=command.corrected_team,
                corrected_urgency=command.corrected_urgency.value
                if command.corrected_urgency
                else None,
                corrected_sentiment=command.corrected_sentiment.value
                if command.corrected_sentiment
                else None,
                reviewer_ref=reviewer_pseudonym(reviewer_key),
                comment_anonymized=comment,
                status=FeedbackStatus.ACCEPTED.value,
            )
        )

    def _validate_codes(self, command: FeedbackCommand) -> None:
        if command.corrected_intent and command.corrected_intent not in self._catalog.intent_codes:
            raise ValidationFailedError("intención corregida desconocida")
        if (
            command.corrected_category
            and command.corrected_category not in self._catalog.category_codes
        ):
            raise ValidationFailedError("categoría corregida desconocida")
        if command.corrected_team and command.corrected_team not in self._catalog.team_codes:
            raise ValidationFailedError("equipo corregido desconocido")


@dataclass(frozen=True)
class TrainingRow:
    feedback_id: uuid.UUID
    text: str
    label: int


class TrainingDataExporter:
    """Convierte la retroalimentación aceptada en filas (texto, etiqueta) para reentrenar.

    Controles contra envenenamiento (OWASP ML02): solo ``accepted``, solo con intención corregida,
    una fila por ticket (gana la corrección más reciente) y un máximo de filas por reentrenamiento.
    """

    def __init__(self, catalog: Catalog, feedback: FeedbackRepository) -> None:
        self._catalog = catalog
        self._feedback = feedback

    def export(self, max_rows: int | None = None) -> list[TrainingRow]:
        latest_by_ticket: dict[uuid.UUID, TrainingRow] = {}
        for item, text in self._feedback.accepted_with_intent():
            if item.corrected_intent_code is None:
                continue
            label = self._catalog.intent(item.corrected_intent_code).label_id
            latest_by_ticket[item.ticket_id] = TrainingRow(item.id, text, label)
        rows = list(latest_by_ticket.values())
        return rows[-max_rows:] if max_rows else rows
