"""Escalamiento de urgencia por señales explícitas de fraude o emergencia (SUPUESTO 2).

La regla solo sube la urgencia, nunca la baja, y siempre queda registrada en la explicación.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# nombre de la regla -> patrón sobre texto en minúsculas y sin tildes
ESCALATION_RULES: dict[str, re.Pattern[str]] = {
    "fraude": re.compile(r"\b(?:fraude|fraudulent[oa]s?|estafa(?:ron|do|da)?)\b"),
    "robo": re.compile(r"\b(?:robo|robaron|robad[oa]s?|hurto)\b"),
    "acceso_no_autorizado": re.compile(
        r"\b(?:hackea(?:ron|do|da)|jaquea(?:ron|do)|clonaron|clonad[oa]|suplantaci[o]n)\b"
    ),
    "movimiento_no_reconocido": re.compile(
        r"\bno (?:reconozco|autorice|hice|realice)\b.{0,40}\b(?:cargo|cobro|pago|retiro|compra|"
        r"movimiento|transaccion|consumo)s?\b"
    ),
    "urgente": re.compile(r"\b(?:urgente|urgencia|emergencia|inmediatamente)\b"),
}


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


@dataclass(frozen=True)
class EscalationResult:
    escalate: bool
    rules: tuple[str, ...]


def evaluate_escalation(text: str) -> EscalationResult:
    normalized = _normalize(text)
    matched = tuple(
        name for name, pattern in ESCALATION_RULES.items() if pattern.search(normalized)
    )
    return EscalationResult(escalate=bool(matched), rules=matched)
