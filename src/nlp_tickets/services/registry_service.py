"""Registra la versión del modelo cargado y sus métricas en ``model_versions``/``model_metrics``."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from nlp_tickets.db.models import ModelMetric, ModelVersion
from nlp_tickets.domain.enums import MetricSplit, MetricTask, ModelBackend
from nlp_tickets.ml.inference import ModelMeta
from nlp_tickets.repositories.models import ModelRepository

FINAL_MODEL_KEY = "minilm_multitask_onnx_int8"


def _dec(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(f"{value:.4f}")


def metrics_from_report(report: dict[str, Any], model_version_id: int) -> list[ModelMetric]:
    """Traduce reports/metrics.json a filas (solo el modelo final, partición de prueba y OOD)."""
    final = report["models"][FINAL_MODEL_KEY]
    routing = final["routing"]
    rows = []
    for task in (
        MetricTask.INTENT,
        MetricTask.CATEGORY,
        MetricTask.TEAM,
        MetricTask.URGENCY,
        MetricTask.SENTIMENT,
    ):
        values = final[task.value]
        rows.append(
            ModelMetric(
                model_version_id=model_version_id,
                task=task.value,
                split=MetricSplit.TEST.value,
                macro_f1=_dec(values["macro_f1"]),
                accuracy=_dec(values["accuracy"]),
                ece=_dec(values["ece"]),
                coverage=_dec(routing["coverage_test"]) if task is MetricTask.TEAM else None,
                selective_accuracy=_dec(routing["auto_routed_accuracy_test"])
                if task is MetricTask.TEAM
                else None,
                n_samples=int(values["n_samples"]),
                mlflow_run_id=report.get("mlflow_run_id"),
            )
        )
    rows.append(
        ModelMetric(
            model_version_id=model_version_id,
            task=MetricTask.OOD.value,
            split=MetricSplit.OOD.value,
            coverage=_dec(1.0 - routing["human_review_rate_out_of_domain"]),
            n_samples=int(report["data"]["out_of_domain_samples"]),
        )
    )
    return rows


class ModelRegistryService:
    def __init__(self, models: ModelRepository) -> None:
        self._models = models

    def register(self, meta: ModelMeta, report_path: Path | None = None) -> ModelVersion:
        """Idempotente: registra (si no existe) y activa la versión; carga métricas del reporte."""
        version = self._models.get(meta.name, meta.version)
        if version is None:
            version = self._models.add(
                ModelVersion(
                    name=meta.name,
                    version=meta.version,
                    backend=ModelBackend.ONNX.value,
                    artifact_sha256=meta.onnx_sha256,
                    confidence_threshold=Decimal(f"{meta.confidence_threshold:.3f}"),
                    dataset_revision=meta.dataset_revision,
                    trained_at=datetime.fromisoformat(meta.trained_at),
                )
            )
        if not version.is_active:
            self._models.activate(version)
        if report_path is not None and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("version") == meta.version:
                self._models.replace_metrics(version.id, metrics_from_report(report, version.id))
        return version
