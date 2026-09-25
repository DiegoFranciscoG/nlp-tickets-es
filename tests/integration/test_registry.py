"""Registro de versiones y métricas del modelo en la base."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import sessionmaker

from nlp_tickets.db.models import ModelVersion
from nlp_tickets.ml.inference import ModelMeta
from nlp_tickets.repositories.models import ModelRepository
from nlp_tickets.services.registry_service import FINAL_MODEL_KEY, ModelRegistryService

pytestmark = pytest.mark.integration


def _meta(version: str) -> ModelMeta:
    return ModelMeta(
        name="minilm-multitask",
        version=version,
        intent_labels=[],
        sentiment_labels=[],
        temperature_intent=1.2,
        temperature_sentiment=1.1,
        confidence_threshold=0.8123,
        max_length=64,
        pad_id=1,
        pad_token="<pad>",
        embedding_dim=384,
        onnx_sha256="a" * 64,
        dataset_revision="rev",
        trained_at="2026-09-25T12:00:00+00:00",
        extra={},
    )


def _report(version: str) -> dict[str, object]:
    task = {"macro_f1": 0.91, "accuracy": 0.92, "ece": 0.02, "n_samples": 3080}
    return {
        "version": version,
        "data": {"out_of_domain_samples": 2974},
        "models": {
            FINAL_MODEL_KEY: {
                "intent": task,
                "category": task,
                "team": task,
                "urgency": task,
                "sentiment": {**task, "n_samples": 5349},
                "routing": {
                    "coverage_test": 0.87,
                    "auto_routed_accuracy_test": 0.99,
                    "human_review_rate_out_of_domain": 0.9,
                },
            }
        },
    }


def test_registro_idempotente_con_metricas_y_una_sola_version_activa(
    engine: Engine, tmp_path: Path, clean_tables: None
) -> None:
    report_path = tmp_path / "metrics.json"
    factory = sessionmaker(bind=engine)
    with factory() as session:
        registry = ModelRegistryService(ModelRepository(session))
        report_path.write_text(json.dumps(_report("v1")))
        first = registry.register(_meta("v1"), report_path)
        again = registry.register(_meta("v1"), report_path)
        assert first.id == again.id
        metrics = ModelRepository(session).metrics(first.id)
        assert len(metrics) == 6
        ood = next(m for m in metrics if m.task == "ood")
        assert float(ood.coverage or 0) == pytest.approx(0.1)
        team = next(m for m in metrics if m.task == "team")
        assert float(team.coverage or 0) == pytest.approx(0.87)

        report_path.write_text(json.dumps(_report("v2")))
        second = registry.register(_meta("v2"), report_path)
        session.commit()
        active = session.scalars(select(ModelVersion).where(ModelVersion.is_active)).all()
        assert [v.version for v in active] == ["v2"]
        assert ModelRepository(session).get_active() == second
        assert float(second.confidence_threshold) == pytest.approx(0.812)


def test_reporte_de_otra_version_no_se_carga(
    engine: Engine, tmp_path: Path, clean_tables: None
) -> None:
    report_path = tmp_path / "metrics.json"
    report_path.write_text(json.dumps(_report("otra")))
    with sessionmaker(bind=engine)() as session:
        version = ModelRegistryService(ModelRepository(session)).register(_meta("v3"), report_path)
        assert ModelRepository(session).metrics(version.id) == []
        session.rollback()
