"""Catálogo de enrutamiento: intención -> categoría, equipo y urgencia base.

Fuente de verdad versionada en ``data/catalog/*.csv``; la migración inicial la carga en la base
y el modelo la usa para marginalizar las probabilidades de intención.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nlp_tickets.domain.enums import Urgency

DEFAULT_CATALOG_DIR = Path(__file__).resolve().parents[3] / "data" / "catalog"


@dataclass(frozen=True, slots=True)
class CategoryDef:
    code: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class TeamDef:
    code: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class IntentDef:
    label_id: int
    code: str
    source_label: str
    name_es: str
    category_code: str
    team_code: str
    base_urgency: Urgency
    standard_resolution: str


class CatalogError(ValueError):
    """El catálogo en disco es incoherente."""


@dataclass(frozen=True)
class Catalog:
    categories: Sequence[CategoryDef]
    teams: Sequence[TeamDef]
    intents: Sequence[IntentDef]

    def __post_init__(self) -> None:
        self._validate()

    @property
    def intent_codes(self) -> list[str]:
        return [i.code for i in self.intents]

    @property
    def category_codes(self) -> list[str]:
        return [c.code for c in self.categories]

    @property
    def team_codes(self) -> list[str]:
        return [t.code for t in self.teams]

    def intent(self, code: str) -> IntentDef:
        for intent in self.intents:
            if intent.code == code:
                return intent
        raise KeyError(code)

    def intent_by_source_label(self) -> dict[str, IntentDef]:
        return {i.source_label: i for i in self.intents}

    def category_name(self, code: str) -> str:
        return next(c.name for c in self.categories if c.code == code)

    def team_name(self, code: str) -> str:
        return next(t.name for t in self.teams if t.code == code)

    def _validate(self) -> None:
        ids = [i.label_id for i in self.intents]
        if ids != list(range(len(ids))):
            raise CatalogError("label_id debe ser 0..N-1 en orden")
        codes = self.intent_codes
        if len(set(codes)) != len(codes):
            raise CatalogError("códigos de intención duplicados")
        cats, teams = set(self.category_codes), set(self.team_codes)
        for intent in self.intents:
            if intent.category_code not in cats:
                raise CatalogError(f"categoría desconocida en {intent.code}")
            if intent.team_code not in teams:
                raise CatalogError(f"equipo desconocido en {intent.code}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_catalog(directory: Path = DEFAULT_CATALOG_DIR) -> Catalog:
    categories = [CategoryDef(**row) for row in _read_csv(directory / "categories.csv")]
    teams = [TeamDef(**row) for row in _read_csv(directory / "teams.csv")]
    intents = [
        IntentDef(
            label_id=int(row["label_id"]),
            code=row["code"],
            source_label=row["source_label"],
            name_es=row["name_es"],
            category_code=row["category_code"],
            team_code=row["team_code"],
            base_urgency=Urgency(row["base_urgency"]),
            standard_resolution=row["standard_resolution"],
        )
        for row in _read_csv(directory / "intents.csv")
    ]
    return Catalog(categories=categories, teams=teams, intents=intents)
