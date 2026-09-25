"""Líneas base: TF-IDF + regresión logística y embeddings congelados + regresión logística."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from nlp_tickets.ml.data import SEED, Splits
from nlp_tickets.ml.metrics import macro_f1
from nlp_tickets.ml.routing import FloatArray

DEFAULT_C_GRID = (2.0, 8.0, 32.0)


def expand_proba(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> FloatArray:
    """Coloca cada columna en su clase: las clases no vistas en train quedan con probabilidad 0."""
    full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    full[:, np.asarray(classes, dtype=np.int64)] = proba
    return full


def build_tfidf_logreg(c: float, seed: int = SEED) -> Pipeline:
    """Palabras (1-2 gramas) + caracteres (2-5 gramas): robusto a faltas y a la morfología."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )
    classifier = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=seed)
    return Pipeline([("features", features), ("clf", classifier)])


@dataclass(frozen=True)
class FittedBaseline:
    pipeline: Pipeline
    c: float
    validation_macro_f1: dict[float, float]
    n_classes: int

    def predict_proba(self, texts: Sequence[str]) -> FloatArray:
        proba = self.pipeline.predict_proba(list(texts))
        return expand_proba(proba, self.pipeline.classes_, self.n_classes)


def fit_tfidf_baseline(
    splits: Splits,
    n_classes: int,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    seed: int = SEED,
) -> FittedBaseline:
    """Elige C en validación (nunca en prueba) y reentrena con train."""
    scores: dict[float, float] = {}
    best: tuple[float, Pipeline] | None = None
    for c in c_grid:
        pipeline = build_tfidf_logreg(c, seed).fit(splits.train["text"], splits.train["label"])
        predicted = pipeline.predict(splits.validation["text"])
        scores[c] = macro_f1(splits.validation["label"].to_numpy(), np.asarray(predicted))
        if best is None or scores[c] > scores[best[0]]:
            best = (c, pipeline)
    if best is None:
        raise ValueError("c_grid no puede estar vacío")
    return FittedBaseline(
        pipeline=best[1], c=best[0], validation_macro_f1=scores, n_classes=n_classes
    )


@dataclass(frozen=True)
class EmbeddingBaseline:
    classifier: LogisticRegression
    n_classes: int

    def predict_proba(self, embeddings: FloatArray) -> FloatArray:
        proba = self.classifier.predict_proba(embeddings)
        return expand_proba(proba, self.classifier.classes_, self.n_classes)


def fit_embedding_baseline(
    train_embeddings: FloatArray, train_labels: np.ndarray, n_classes: int, seed: int = SEED
) -> EmbeddingBaseline:
    classifier = LogisticRegression(
        C=10.0, max_iter=3000, class_weight="balanced", random_state=seed
    )
    return EmbeddingBaseline(classifier.fit(train_embeddings, train_labels), n_classes)
