"""Carga los tickets de ejemplo (resueltos) con sus embeddings. Idempotente."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np

from nlp_tickets.db.models import Ticket, TicketLabel
from nlp_tickets.domain.catalog import Catalog
from nlp_tickets.domain.enums import LabelSource, TicketSource
from nlp_tickets.ml.routing import FloatArray
from nlp_tickets.repositories.tickets import TicketRepository


class BatchEmbedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def raw_outputs(
        self, texts: list[str], batch_size: int = 256
    ) -> tuple[FloatArray, FloatArray, FloatArray]: ...


class SeedService:
    def __init__(self, catalog: Catalog, tickets: TicketRepository, model: BatchEmbedder) -> None:
        self._catalog = catalog
        self._tickets = tickets
        self._model = model

    def seed(self, path: Path) -> int:
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if self._tickets.count_by_source(TicketSource.SEED.value) >= len(rows):
            return 0
        _, _, embeddings = self._model.raw_outputs([row["text"] for row in rows])
        embedding_model = f"{self._model.name}:{self._model.version}"
        for row, embedding in zip(rows, embeddings, strict=True):
            intent = self._catalog.intent(row["intent_code"])
            created = datetime.fromisoformat(row["created_at"])
            ticket = self._tickets.add(
                Ticket(
                    text_anonymized=row["text"],
                    channel=row["channel"],
                    source=TicketSource.SEED.value,
                    created_at=created,
                    resolution_text=intent.standard_resolution,
                    resolved_at=created + timedelta(hours=4),
                    embedding=np.asarray(embedding, dtype=np.float32).tolist(),
                    embedding_model=embedding_model,
                )
            )
            self._tickets.add_label(
                TicketLabel(
                    ticket_id=ticket.id,
                    intent_code=intent.code,
                    category_code=intent.category_code,
                    team_code=intent.team_code,
                    urgency=intent.base_urgency.value,
                    label_source=LabelSource.DATASET.value,
                )
            )
        return len(rows)
