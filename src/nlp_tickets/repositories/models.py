"""Registro de versiones de modelo, métricas y catálogo persistido."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from nlp_tickets.db.models import Category, Intent, ModelMetric, ModelVersion, Team
from nlp_tickets.domain.catalog import Catalog, CategoryDef, IntentDef, TeamDef
from nlp_tickets.domain.enums import Urgency


class ModelRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_active(self) -> ModelVersion | None:
        return self._session.scalar(select(ModelVersion).where(ModelVersion.is_active))

    def get(self, name: str, version: str) -> ModelVersion | None:
        return self._session.scalar(
            select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == version)
        )

    def add(self, model_version: ModelVersion) -> ModelVersion:
        self._session.add(model_version)
        self._session.flush()
        return model_version

    def activate(self, model_version: ModelVersion) -> None:
        self._session.execute(
            update(ModelVersion).where(ModelVersion.is_active).values(is_active=False)
        )
        self._session.flush()
        model_version.is_active = True
        self._session.flush()

    def replace_metrics(self, model_version_id: int, metrics: list[ModelMetric]) -> None:
        for existing in self.metrics(model_version_id):
            self._session.delete(existing)
        self._session.flush()
        self._session.add_all(metrics)
        self._session.flush()

    def metrics(self, model_version_id: int) -> list[ModelMetric]:
        return list(
            self._session.scalars(
                select(ModelMetric)
                .where(ModelMetric.model_version_id == model_version_id)
                .order_by(ModelMetric.split, ModelMetric.task)
            )
        )


class CatalogRepository:
    """La política de enrutamiento vigente vive en la base (sembrada por la migración)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def load(self) -> Catalog:
        categories = [
            CategoryDef(c.code, c.name, c.description or "")
            for c in self._session.scalars(select(Category).order_by(Category.code))
        ]
        teams = [
            TeamDef(t.code, t.name, t.description or "")
            for t in self._session.scalars(select(Team).order_by(Team.code))
        ]
        intents = [
            IntentDef(
                label_id=i.label_id,
                code=i.code,
                source_label=i.source_label,
                name_es=i.name_es,
                category_code=i.category_code,
                team_code=i.team_code,
                base_urgency=Urgency(i.base_urgency),
                standard_resolution=i.standard_resolution,
            )
            for i in self._session.scalars(select(Intent).order_by(Intent.label_id))
        ]
        return Catalog(categories=categories, teams=teams, intents=intents)
