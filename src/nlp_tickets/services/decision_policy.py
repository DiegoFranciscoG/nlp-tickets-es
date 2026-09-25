"""Política de decisión: enrutar automáticamente o enviar a revisión humana (LOPDP art. 20, E5).

- ``low_confidence``: la probabilidad calibrada del equipo es menor que el umbral de validación.
- ``high_urgency_low_margin``: el modelo no predice urgencia alta, pero le da una probabilidad
  relevante. Omitir un caso de fraude cuesta más que revisarlo (costes asimétricos).
- ``escalated_by_rule``: una regla subió la urgencia sobre la del modelo; una persona lo confirma.
"""

from __future__ import annotations

from dataclasses import dataclass

from nlp_tickets.domain.enums import Decision, ReviewReason, Urgency

HIGH_URGENCY_REVIEW_PROBABILITY = 0.25


@dataclass(frozen=True)
class DecisionInput:
    team_score: float
    predicted_urgency: Urgency
    high_urgency_probability: float
    escalated_by_rule: bool
    threshold: float


@dataclass(frozen=True)
class DecisionOutcome:
    decision: Decision
    reasons: tuple[ReviewReason, ...]


def decide(data: DecisionInput) -> DecisionOutcome:
    reasons: list[ReviewReason] = []
    if data.team_score < data.threshold:
        reasons.append(ReviewReason.LOW_CONFIDENCE)
    if (
        data.predicted_urgency is not Urgency.HIGH
        and data.high_urgency_probability >= HIGH_URGENCY_REVIEW_PROBABILITY
    ):
        reasons.append(ReviewReason.HIGH_URGENCY_LOW_MARGIN)
    if data.escalated_by_rule:
        reasons.append(ReviewReason.ESCALATED_BY_RULE)
    decision = Decision.HUMAN_REVIEW if reasons else Decision.AUTO_ROUTE
    return DecisionOutcome(decision=decision, reasons=tuple(reasons))
