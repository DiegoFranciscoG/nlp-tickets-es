"""Dobles de prueba deterministas (no requieren el modelo entrenado)."""

from __future__ import annotations

import hashlib

import numpy as np

from nlp_tickets.domain.catalog import Catalog
from nlp_tickets.ml.inference import ModelOutput
from nlp_tickets.ml.routing import FloatArray

KEYWORD_INTENTS = (
    ("robaron", "lost_or_stolen_card"),
    ("no reconozco", "card_payment_not_recognised"),
    ("transferencia", "pending_transfer"),
    ("tarjeta", "card_arrival"),
)
DEFAULT_INTENT = "exchange_rate"


def fake_embedding(text: str, dim: int = 384) -> FloatArray:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    vector = np.random.default_rng(seed).normal(size=dim)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float64)


class FakeModel:
    """Elige la intención por palabras clave y asigna ``confidence`` a esa intención."""

    def __init__(
        self,
        catalog: Catalog,
        *,
        confidence: float = 0.9,
        sentiment: int = 0,
        version: str = "test-1",
    ) -> None:
        self._catalog = catalog
        self._confidence = confidence
        self._sentiment = sentiment
        self._version = version
        self.explain_calls = 0

    @property
    def name(self) -> str:
        return "fake-model"

    @property
    def version(self) -> str:
        return self._version

    @property
    def confidence_threshold(self) -> float:
        return 0.5

    @property
    def trained_at(self) -> str:
        return "2026-09-25T00:00:00+00:00"

    def intent_for(self, text: str) -> str:
        lowered = text.lower()
        return next((code for key, code in KEYWORD_INTENTS if key in lowered), DEFAULT_INTENT)

    def predict(self, text: str) -> ModelOutput:
        n = len(self._catalog.intents)
        target = self._catalog.intent(self.intent_for(text)).label_id
        probs = np.full(n, (1.0 - self._confidence) / (n - 1))
        probs[target] = self._confidence
        sentiment = np.full(3, 0.1)
        sentiment[self._sentiment] = 0.8
        return ModelOutput(probs, sentiment, self.embed(text))

    def embed(self, text: str) -> FloatArray:
        return fake_embedding(text)

    def explain(self, text: str, intent_index: int, top_k: int = 5) -> list[tuple[str, float]]:
        self.explain_calls += 1
        return [(word, 0.1) for word in text.split()[:top_k]]

    def raw_outputs(
        self, texts: list[str], batch_size: int = 256
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        outputs = [self.predict(t) for t in texts]
        return (
            np.log(np.stack([o.intent_probs for o in outputs])),
            np.log(np.stack([o.sentiment_probs for o in outputs])),
            np.stack([o.embedding for o in outputs]),
        )
