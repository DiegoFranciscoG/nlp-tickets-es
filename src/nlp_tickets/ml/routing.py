"""Marginalización de probabilidades: p(categoría) = suma de p(intención) de sus intenciones.

Así categoría, equipo y urgencia siempre son coherentes con la intención y el catálogo puede
cambiar (reasignar un equipo, por ejemplo) sin reentrenar el modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from nlp_tickets.domain.catalog import Catalog
from nlp_tickets.domain.enums import Urgency

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
URGENCY_ORDER: tuple[str, ...] = tuple(u.value for u in Urgency)  # low, medium, high


def _one_hot(indices: list[int], size: int) -> FloatArray:
    matrix = np.zeros((len(indices), size), dtype=np.float64)
    matrix[np.arange(len(indices)), indices] = 1.0
    return matrix


@dataclass(frozen=True)
class RoutingMatrices:
    category_codes: tuple[str, ...]
    team_codes: tuple[str, ...]
    urgency_codes: tuple[str, ...]
    to_category: FloatArray  # (n_intents, n_categories)
    to_team: FloatArray
    to_urgency: FloatArray

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> RoutingMatrices:
        categories = tuple(catalog.category_codes)
        teams = tuple(catalog.team_codes)
        return cls(
            category_codes=categories,
            team_codes=teams,
            urgency_codes=URGENCY_ORDER,
            to_category=_one_hot(
                [categories.index(i.category_code) for i in catalog.intents], len(categories)
            ),
            to_team=_one_hot([teams.index(i.team_code) for i in catalog.intents], len(teams)),
            to_urgency=_one_hot(
                [URGENCY_ORDER.index(i.base_urgency.value) for i in catalog.intents],
                len(URGENCY_ORDER),
            ),
        )

    def category_probs(self, intent_probs: FloatArray) -> FloatArray:
        return intent_probs @ self.to_category

    def team_probs(self, intent_probs: FloatArray) -> FloatArray:
        return intent_probs @ self.to_team

    def urgency_probs(self, intent_probs: FloatArray) -> FloatArray:
        return intent_probs @ self.to_urgency

    def map_labels(self, intent_labels: IntArray, matrix: FloatArray) -> IntArray:
        return np.asarray(matrix[intent_labels].argmax(axis=1), dtype=np.int64)
