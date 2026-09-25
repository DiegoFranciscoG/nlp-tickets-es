"""Búsqueda semántica de tickets resueltos parecidos (embeddings + pgvector)."""

from __future__ import annotations

import uuid

from nlp_tickets.anonymization import Anonymizer
from nlp_tickets.exceptions.errors import NotFoundError
from nlp_tickets.repositories.tickets import SimilarRow, TicketRepository
from nlp_tickets.services.ports import TicketModel


class SimilarityService:
    def __init__(
        self, model: TicketModel, tickets: TicketRepository, anonymizer: Anonymizer | None = None
    ) -> None:
        self._model = model
        self._tickets = tickets
        self._anonymizer = anonymizer or Anonymizer()

    @property
    def embedding_model_id(self) -> str:
        return f"{self._model.name}:{self._model.version}"

    def similar_to_text(self, text: str, limit: int) -> list[SimilarRow]:
        embedding = self._model.embed(self._anonymizer.anonymize(text).text)
        return self._tickets.search_similar(
            embedding, embedding_model=self.embedding_model_id, limit=limit
        )

    def similar_to_ticket(self, ticket_id: uuid.UUID, limit: int) -> list[SimilarRow]:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise NotFoundError("ticket no encontrado")
        embedding = (
            ticket.embedding
            if ticket.embedding is not None and ticket.embedding_model == self.embedding_model_id
            else self._model.embed(ticket.text_anonymized)
        )
        return self._tickets.search_similar(
            embedding, embedding_model=self.embedding_model_id, limit=limit, exclude_id=ticket.id
        )
