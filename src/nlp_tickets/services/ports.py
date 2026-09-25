"""Puertos de la capa de servicios: el modelo se inyecta y en los tests se usa un doble."""

from __future__ import annotations

from typing import Protocol

from nlp_tickets.ml.inference import ModelOutput
from nlp_tickets.ml.routing import FloatArray


class TicketModel(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def confidence_threshold(self) -> float: ...

    @property
    def trained_at(self) -> str: ...

    def predict(self, text: str) -> ModelOutput: ...

    def embed(self, text: str) -> FloatArray: ...

    def explain(self, text: str, intent_index: int, top_k: int = 5) -> list[tuple[str, float]]: ...
