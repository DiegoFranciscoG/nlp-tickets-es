"""Esquema inicial: catálogos, tickets con pgvector, predicciones, retroalimentación y métricas.

Revision ID: 0001
Revises:
Create Date: 2026-09-25
"""

import csv
from collections.abc import Sequence
from pathlib import Path

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CATALOG_DIR = Path(__file__).resolve().parents[2] / "data" / "catalog"
SCORE = sa.Numeric(5, 4)
URGENCY_IN = "IN ('low', 'medium', 'high')"
SENTIMENT_IN = "IN ('negative', 'neutral', 'positive')"


def _read(name: str) -> list[dict[str, str]]:
    with (CATALOG_DIR / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _now() -> sa.TextClause:
    return sa.text("now()")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    categories = op.create_table(
        "categories",
        sa.Column("code", sa.String(40), nullable=False),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("code", name="pk_categories"),
    )
    teams = op.create_table(
        "teams",
        sa.Column("code", sa.String(40), nullable=False),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("code", name="pk_teams"),
    )
    intents = op.create_table(
        "intents",
        sa.Column("code", sa.String(60), nullable=False),
        sa.Column("label_id", sa.SmallInteger(), nullable=False),
        sa.Column("source_label", sa.String(60), nullable=False),
        sa.Column("name_es", sa.String(80), nullable=False),
        sa.Column("category_code", sa.String(40), nullable=False),
        sa.Column("team_code", sa.String(40), nullable=False),
        sa.Column("base_urgency", sa.String(10), nullable=False),
        sa.Column("standard_resolution", sa.Text(), nullable=False),
        sa.CheckConstraint(f"base_urgency {URGENCY_IN}", name="ck_intents_base_urgency"),
        sa.CheckConstraint("label_id BETWEEN 0 AND 76", name="ck_intents_label_id_range"),
        sa.ForeignKeyConstraint(
            ["category_code"], ["categories.code"], name="fk_intents_category_code_categories"
        ),
        sa.ForeignKeyConstraint(["team_code"], ["teams.code"], name="fk_intents_team_code_teams"),
        sa.PrimaryKeyConstraint("code", name="pk_intents"),
        sa.UniqueConstraint("label_id", name="uq_intents_label_id"),
    )

    op.create_table(
        "tickets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("external_ref", sa.String(64), nullable=True),
        sa.Column("text_anonymized", sa.Text(), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("language", sa.String(2), server_default=sa.text("'es'"), nullable=False),
        sa.Column("resolution_text", sa.Text(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("embedding_model", sa.String(80), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.CheckConstraint(
            "channel IN ('email', 'web', 'chat', 'phone', 'social', 'branch')",
            name="ck_tickets_channel",
        ),
        sa.CheckConstraint("source IN ('api', 'seed', 'dataset')", name="ck_tickets_source"),
        sa.CheckConstraint(
            "char_length(text_anonymized) BETWEEN 1 AND 5000", name="ck_tickets_text_length"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_tickets"),
        sa.UniqueConstraint("external_ref", name="uq_tickets_external_ref"),
    )
    op.create_index(
        "ix_tickets_embedding_hnsw",
        "tickets",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "ticket_labels",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ticket_id", sa.Uuid(), nullable=False),
        sa.Column("intent_code", sa.String(60), nullable=True),
        sa.Column("category_code", sa.String(40), nullable=False),
        sa.Column("team_code", sa.String(40), nullable=False),
        sa.Column("urgency", sa.String(10), nullable=False),
        sa.Column("sentiment", sa.String(10), nullable=True),
        sa.Column("label_source", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.CheckConstraint(f"urgency {URGENCY_IN}", name="ck_ticket_labels_urgency"),
        sa.CheckConstraint(
            f"sentiment IS NULL OR sentiment {SENTIMENT_IN}", name="ck_ticket_labels_sentiment"
        ),
        sa.CheckConstraint(
            "label_source IN ('dataset', 'human', 'catalog')", name="ck_ticket_labels_label_source"
        ),
        sa.ForeignKeyConstraint(
            ["ticket_id"],
            ["tickets.id"],
            name="fk_ticket_labels_ticket_id_tickets",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["intent_code"], ["intents.code"], name="fk_ticket_labels_intent_code_intents"
        ),
        sa.ForeignKeyConstraint(
            ["category_code"], ["categories.code"], name="fk_ticket_labels_category_code_categories"
        ),
        sa.ForeignKeyConstraint(
            ["team_code"], ["teams.code"], name="fk_ticket_labels_team_code_teams"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_ticket_labels"),
        sa.UniqueConstraint("ticket_id", "label_source", name="uq_ticket_labels_ticket_source"),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("version", sa.String(40), nullable=False),
        sa.Column("backend", sa.String(20), nullable=False),
        sa.Column("artifact_sha256", sa.String(64), nullable=False),
        sa.Column("confidence_threshold", sa.Numeric(4, 3), nullable=False),
        sa.Column("dataset_revision", sa.String(80), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.CheckConstraint("backend IN ('onnx', 'sklearn')", name="ck_model_versions_backend"),
        sa.CheckConstraint(
            "confidence_threshold BETWEEN 0 AND 1", name="ck_model_versions_threshold_range"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_versions"),
        sa.UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
    )
    op.create_index(
        "uq_model_versions_single_active",
        "model_versions",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("ticket_id", sa.Uuid(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("intent_code", sa.String(60), nullable=False),
        sa.Column("intent_score", SCORE, nullable=False),
        sa.Column("category_code", sa.String(40), nullable=False),
        sa.Column("category_score", SCORE, nullable=False),
        sa.Column("team_code", sa.String(40), nullable=False),
        sa.Column("team_score", SCORE, nullable=False),
        sa.Column("urgency", sa.String(10), nullable=False),
        sa.Column("urgency_score", SCORE, nullable=False),
        sa.Column("urgency_escalated", sa.Boolean(), nullable=False),
        sa.Column("sentiment", sa.String(10), nullable=False),
        sa.Column("sentiment_score", SCORE, nullable=False),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column(
            "review_reasons",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("explanation", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.CheckConstraint(f"urgency {URGENCY_IN}", name="ck_predictions_urgency"),
        sa.CheckConstraint(f"sentiment {SENTIMENT_IN}", name="ck_predictions_sentiment"),
        sa.CheckConstraint(
            "decision IN ('auto_route', 'human_review')", name="ck_predictions_decision"
        ),
        sa.CheckConstraint("latency_ms >= 0", name="ck_predictions_latency"),
        sa.ForeignKeyConstraint(
            ["ticket_id"],
            ["tickets.id"],
            name="fk_predictions_ticket_id_tickets",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_predictions_model_version_id_model_versions",
        ),
        sa.ForeignKeyConstraint(
            ["intent_code"], ["intents.code"], name="fk_predictions_intent_code_intents"
        ),
        sa.ForeignKeyConstraint(
            ["category_code"], ["categories.code"], name="fk_predictions_category_code_categories"
        ),
        sa.ForeignKeyConstraint(
            ["team_code"], ["teams.code"], name="fk_predictions_team_code_teams"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_predictions"),
    )
    op.create_index("ix_predictions_ticket_id", "predictions", ["ticket_id"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("prediction_id", sa.Uuid(), nullable=False),
        sa.Column("ticket_id", sa.Uuid(), nullable=False),
        sa.Column("corrected_intent_code", sa.String(60), nullable=True),
        sa.Column("corrected_category_code", sa.String(40), nullable=True),
        sa.Column("corrected_team_code", sa.String(40), nullable=True),
        sa.Column("corrected_urgency", sa.String(10), nullable=True),
        sa.Column("corrected_sentiment", sa.String(10), nullable=True),
        sa.Column("reviewer_ref", sa.String(64), nullable=False),
        sa.Column("comment_anonymized", sa.Text(), nullable=True),
        sa.Column("status", sa.String(10), server_default=sa.text("'accepted'"), nullable=False),
        sa.Column("used_in_model_version_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.CheckConstraint("status IN ('accepted', 'rejected')", name="ck_feedback_status"),
        sa.CheckConstraint(
            f"corrected_urgency IS NULL OR corrected_urgency {URGENCY_IN}",
            name="ck_feedback_corrected_urgency",
        ),
        sa.CheckConstraint(
            f"corrected_sentiment IS NULL OR corrected_sentiment {SENTIMENT_IN}",
            name="ck_feedback_corrected_sentiment",
        ),
        sa.CheckConstraint(
            "num_nonnulls(corrected_intent_code, corrected_category_code, corrected_team_code,"
            " corrected_urgency, corrected_sentiment) >= 1",
            name="ck_feedback_at_least_one_correction",
        ),
        sa.CheckConstraint(
            "char_length(comment_anonymized) <= 1000", name="ck_feedback_comment_length"
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name="fk_feedback_prediction_id_predictions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["ticket_id"], ["tickets.id"], name="fk_feedback_ticket_id_tickets", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["corrected_intent_code"],
            ["intents.code"],
            name="fk_feedback_corrected_intent_code_intents",
        ),
        sa.ForeignKeyConstraint(
            ["corrected_category_code"],
            ["categories.code"],
            name="fk_feedback_corrected_category_code_categories",
        ),
        sa.ForeignKeyConstraint(
            ["corrected_team_code"], ["teams.code"], name="fk_feedback_corrected_team_code_teams"
        ),
        sa.ForeignKeyConstraint(
            ["used_in_model_version_id"],
            ["model_versions.id"],
            name="fk_feedback_used_in_model_version_id_model_versions",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feedback"),
    )
    op.create_index("ix_feedback_prediction_id", "feedback", ["prediction_id"])

    op.create_table(
        "model_metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("task", sa.String(20), nullable=False),
        sa.Column("split", sa.String(20), nullable=False),
        sa.Column("macro_f1", SCORE, nullable=True),
        sa.Column("accuracy", SCORE, nullable=True),
        sa.Column("ece", SCORE, nullable=True),
        sa.Column("coverage", SCORE, nullable=True),
        sa.Column("selective_accuracy", SCORE, nullable=True),
        sa.Column("n_samples", sa.Integer(), nullable=False),
        sa.Column("mlflow_run_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.CheckConstraint(
            "task IN ('intent', 'category', 'team', 'urgency', 'sentiment', 'ood')",
            name="ck_model_metrics_task",
        ),
        sa.CheckConstraint("split IN ('validation', 'test', 'ood')", name="ck_model_metrics_split"),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_model_metrics_model_version_id_model_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_metrics"),
        sa.UniqueConstraint(
            "model_version_id", "task", "split", name="uq_model_metrics_version_task"
        ),
    )

    op.bulk_insert(categories, _read("categories.csv"))
    op.bulk_insert(teams, _read("teams.csv"))
    op.bulk_insert(
        intents, [{**row, "label_id": int(row["label_id"])} for row in _read("intents.csv")]
    )


def downgrade() -> None:
    for table in (
        "model_metrics",
        "feedback",
        "predictions",
        "model_versions",
        "ticket_labels",
        "tickets",
        "intents",
        "teams",
        "categories",
    ):
        op.drop_table(table)
