"""Entidades ORM. Reflejan docs/modelo-datos.md y la migración inicial."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from nlp_tickets.db.base import Base

EMBEDDING_DIM = 384
_SCORE = Numeric(5, 4)


def _in(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


URGENCIES = ("low", "medium", "high")
SENTIMENTS = ("negative", "neutral", "positive")
CHANNELS = ("email", "web", "chat", "phone", "social", "branch")


class Category(Base):
    __tablename__ = "categories"

    code: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str | None] = mapped_column(Text)


class Team(Base):
    __tablename__ = "teams"

    code: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str | None] = mapped_column(Text)


class Intent(Base):
    __tablename__ = "intents"
    __table_args__ = (
        CheckConstraint(_in("base_urgency", URGENCIES), name="base_urgency"),
        CheckConstraint("label_id BETWEEN 0 AND 76", name="label_id_range"),
    )

    code: Mapped[str] = mapped_column(String(60), primary_key=True)
    label_id: Mapped[int] = mapped_column(SmallInteger, unique=True)
    source_label: Mapped[str] = mapped_column(String(60))
    name_es: Mapped[str] = mapped_column(String(80))
    category_code: Mapped[str] = mapped_column(ForeignKey("categories.code"))
    team_code: Mapped[str] = mapped_column(ForeignKey("teams.code"))
    base_urgency: Mapped[str] = mapped_column(String(10))
    standard_resolution: Mapped[str] = mapped_column(Text)


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        CheckConstraint(_in("channel", CHANNELS), name="channel"),
        CheckConstraint(_in("source", ("api", "seed", "dataset")), name="source"),
        CheckConstraint("char_length(text_anonymized) BETWEEN 1 AND 5000", name="text_length"),
        Index(
            "ix_tickets_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_ref: Mapped[str | None] = mapped_column(String(64), unique=True)
    text_anonymized: Mapped[str] = mapped_column(Text)
    channel: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(20))
    language: Mapped[str] = mapped_column(String(2), server_default=text("'es'"))
    resolution_text: Mapped[str | None] = mapped_column(Text)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[Any | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    labels: Mapped[list[TicketLabel]] = relationship(
        back_populates="ticket", cascade="all, delete-orphan"
    )


class TicketLabel(Base):
    __tablename__ = "ticket_labels"
    __table_args__ = (
        UniqueConstraint("ticket_id", "label_source", name="uq_ticket_labels_ticket_source"),
        CheckConstraint(_in("urgency", URGENCIES), name="urgency"),
        CheckConstraint(f"sentiment IS NULL OR {_in('sentiment', SENTIMENTS)}", name="sentiment"),
        CheckConstraint(_in("label_source", ("dataset", "human", "catalog")), name="label_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticket_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"))
    intent_code: Mapped[str | None] = mapped_column(ForeignKey("intents.code"))
    category_code: Mapped[str] = mapped_column(ForeignKey("categories.code"))
    team_code: Mapped[str] = mapped_column(ForeignKey("teams.code"))
    urgency: Mapped[str] = mapped_column(String(10))
    sentiment: Mapped[str | None] = mapped_column(String(10))
    label_source: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    ticket: Mapped[Ticket] = relationship(back_populates="labels")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
        CheckConstraint(_in("backend", ("onnx", "sklearn")), name="backend"),
        CheckConstraint("confidence_threshold BETWEEN 0 AND 1", name="threshold_range"),
        Index(
            "uq_model_versions_single_active",
            "is_active",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80))
    version: Mapped[str] = mapped_column(String(40))
    backend: Mapped[str] = mapped_column(String(20))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    confidence_threshold: Mapped[Decimal] = mapped_column(Numeric(4, 3))
    dataset_revision: Mapped[str] = mapped_column(String(80))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(_in("urgency", URGENCIES), name="urgency"),
        CheckConstraint(_in("sentiment", SENTIMENTS), name="sentiment"),
        CheckConstraint(_in("decision", ("auto_route", "human_review")), name="decision"),
        CheckConstraint("latency_ms >= 0", name="latency"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tickets.id", ondelete="CASCADE"), index=True
    )
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    intent_code: Mapped[str] = mapped_column(ForeignKey("intents.code"))
    intent_score: Mapped[Decimal] = mapped_column(_SCORE)
    category_code: Mapped[str] = mapped_column(ForeignKey("categories.code"))
    category_score: Mapped[Decimal] = mapped_column(_SCORE)
    team_code: Mapped[str] = mapped_column(ForeignKey("teams.code"))
    team_score: Mapped[Decimal] = mapped_column(_SCORE)
    urgency: Mapped[str] = mapped_column(String(10))
    urgency_score: Mapped[Decimal] = mapped_column(_SCORE)
    urgency_escalated: Mapped[bool] = mapped_column(Boolean)
    sentiment: Mapped[str] = mapped_column(String(10))
    sentiment_score: Mapped[Decimal] = mapped_column(_SCORE)
    decision: Mapped[str] = mapped_column(String(20))
    review_reasons: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    explanation: Mapped[dict[str, Any]] = mapped_column(JSONB)
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    ticket: Mapped[Ticket] = relationship()
    model_version: Mapped[ModelVersion] = relationship()


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint(_in("status", ("accepted", "rejected")), name="status"),
        CheckConstraint(
            f"corrected_urgency IS NULL OR {_in('corrected_urgency', URGENCIES)}",
            name="corrected_urgency",
        ),
        CheckConstraint(
            f"corrected_sentiment IS NULL OR {_in('corrected_sentiment', SENTIMENTS)}",
            name="corrected_sentiment",
        ),
        CheckConstraint(
            "num_nonnulls(corrected_intent_code, corrected_category_code, corrected_team_code,"
            " corrected_urgency, corrected_sentiment) >= 1",
            name="at_least_one_correction",
        ),
        CheckConstraint("char_length(comment_anonymized) <= 1000", name="comment_length"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), index=True
    )
    ticket_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"))
    corrected_intent_code: Mapped[str | None] = mapped_column(ForeignKey("intents.code"))
    corrected_category_code: Mapped[str | None] = mapped_column(ForeignKey("categories.code"))
    corrected_team_code: Mapped[str | None] = mapped_column(ForeignKey("teams.code"))
    corrected_urgency: Mapped[str | None] = mapped_column(String(10))
    corrected_sentiment: Mapped[str | None] = mapped_column(String(10))
    reviewer_ref: Mapped[str] = mapped_column(String(64))
    comment_anonymized: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(10), server_default=text("'accepted'"))
    used_in_model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    __table_args__ = (
        UniqueConstraint("model_version_id", "task", "split", name="uq_model_metrics_version_task"),
        CheckConstraint(
            _in("task", ("intent", "category", "team", "urgency", "sentiment", "ood")),
            name="task",
        ),
        CheckConstraint(_in("split", ("validation", "test", "ood")), name="split"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE")
    )
    task: Mapped[str] = mapped_column(String(20))
    split: Mapped[str] = mapped_column(String(20))
    macro_f1: Mapped[Decimal | None] = mapped_column(_SCORE)
    accuracy: Mapped[Decimal | None] = mapped_column(_SCORE)
    ece: Mapped[Decimal | None] = mapped_column(_SCORE)
    coverage: Mapped[Decimal | None] = mapped_column(_SCORE)
    selective_accuracy: Mapped[Decimal | None] = mapped_column(_SCORE)
    n_samples: Mapped[int] = mapped_column(Integer)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
