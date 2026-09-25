"""Preparación de datos: alineación, limpieza, anonimización, deduplicación y particiones.

Reglas (docs/investigacion.md):
- La etiqueta de intención viene del CSV original de PolyAI (D2): el corpus en español es paralelo
  fila a fila y su columna ``rotulo`` fue traducida (629 variantes para 77 clases).
- Sentimiento según el agrupamiento oficial de GoEmotions (D7).
- La validación se estratifica por clase con semilla fija (E2); la prueba oficial no se toca.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nlp_tickets.anonymization import anonymize
from nlp_tickets.domain.catalog import Catalog
from nlp_tickets.domain.enums import Sentiment

SEED = 42
SENTIMENT_LABELS: tuple[str, ...] = tuple(s.value for s in Sentiment)  # negative, neutral, positive
MIN_ALIGNMENT_AGREEMENT = 0.98

GOEMOTIONS_NAMES = (
    "admiration amusement anger annoyance approval caring confusion curiosity desire "
    "disappointment disapproval disgust embarrassment excitement fear gratitude grief joy love "
    "nervousness optimism pride realization relief remorse sadness surprise neutral"
).split()
# Agrupamiento oficial (google-research/goemotions/data/sentiment_dict.json)
GOEMOTIONS_POSITIVE = frozenset(
    "amusement excitement joy love desire optimism caring pride admiration gratitude relief "
    "approval".split()
)
GOEMOTIONS_NEGATIVE = frozenset(
    "fear nervousness remorse embarrassment disappointment sadness grief disgust anger annoyance "
    "disapproval".split()
)

_WHITESPACE = re.compile(r"\s+")


class AlignmentError(ValueError):
    """El corpus en español no está alineado con el original."""


@dataclass(frozen=True)
class AlignmentReport:
    rows: int
    agreement: float
    ambiguous_variants: int


@dataclass(frozen=True)
class Splits:
    """Particiones con columnas ``text`` y ``label`` (entero)."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def fingerprint(self) -> str:
        """Hash de las particiones: prueba que baseline y transformer usan exactamente lo mismo."""
        digest = hashlib.sha256()
        for frame in (self.train, self.validation, self.test):
            for text, label in zip(frame["text"], frame["label"], strict=True):
                digest.update(f"{label}\t{text}\n".encode())
        return digest.hexdigest()

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    return anonymize(_WHITESPACE.sub(" ", normalized).strip())


def dedup_key(text: str) -> str:
    return _WHITESPACE.sub(" ", text.lower()).strip(" .?!¿¡")


def validate_alignment(
    spanish_labels: Sequence[str], original_labels: Sequence[str]
) -> AlignmentReport:
    """Cada variante traducida debe apuntar mayoritariamente a una sola etiqueta original."""
    if len(spanish_labels) != len(original_labels):
        raise AlignmentError(
            f"tamaños distintos: {len(spanish_labels)} frente a {len(original_labels)}"
        )
    frame = pd.DataFrame({"es": list(spanish_labels), "en": list(original_labels)})
    majority = frame.groupby("es")["en"].agg(lambda s: s.value_counts().index[0])
    agreement = float((frame["en"] == frame["es"].map(majority)).mean())
    ambiguous = int((frame.groupby("es")["en"].nunique() > 1).sum())
    if agreement < MIN_ALIGNMENT_AGREEMENT:
        raise AlignmentError(f"concordancia {agreement:.3f} < {MIN_ALIGNMENT_AGREEMENT}")
    return AlignmentReport(rows=len(frame), agreement=agreement, ambiguous_variants=ambiguous)


def _stratified_validation(
    frame: pd.DataFrame, size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, validation = train_test_split(
        frame, test_size=size, stratify=frame["label"], random_state=seed
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def remove_leakage(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Quita duplicados dentro de train y frases de train que también están en test."""
    keys = train["text"].map(dedup_key)
    duplicated = keys.duplicated()
    in_test = keys.isin(set(test["text"].map(dedup_key)))
    cleaned = train.loc[~duplicated & ~in_test].reset_index(drop=True)
    return cleaned, int(duplicated.sum()), int((in_test & ~duplicated).sum())


def build_intent_splits(
    spanish: dict[str, pd.DataFrame],
    original: dict[str, pd.DataFrame],
    catalog: Catalog,
    *,
    extra_train: pd.DataFrame | None = None,
    validation_size: float = 0.1,
    seed: int = SEED,
) -> tuple[Splits, dict[str, float | int]]:
    by_source = catalog.intent_by_source_label()
    frames: dict[str, pd.DataFrame] = {}
    report: dict[str, float | int] = {}
    for split in ("train", "test"):
        alignment = validate_alignment(spanish[split]["rotulo"], original[split]["category"])
        report[f"{split}_alignment_agreement"] = round(alignment.agreement, 4)
        report[f"{split}_ambiguous_variants"] = alignment.ambiguous_variants
        frames[split] = pd.DataFrame(
            {
                "text": spanish[split]["frase"].map(clean_text),
                "label": original[split]["category"].map(lambda c: by_source[c].label_id),
            }
        )
    train, duplicates, leaked = remove_leakage(frames["train"], frames["test"])
    report.update(train_duplicates_removed=duplicates, train_test_leakage_removed=leaked)
    if extra_train is not None and len(extra_train):
        extra = extra_train[["text", "label"]].assign(text=lambda f: f["text"].map(clean_text))
        extra, _, extra_leaked = remove_leakage(extra, frames["test"])
        report.update(feedback_rows_added=len(extra), feedback_leakage_removed=extra_leaked)
        train = pd.concat([train, extra], ignore_index=True)
    train, validation = _stratified_validation(train, validation_size, seed)
    return Splits(train=train, validation=validation, test=frames["test"]), report


def map_goemotions_sentiment(label_ids: Iterable[int]) -> Sentiment | None:
    names = {GOEMOTIONS_NAMES[i] for i in label_ids}
    negative = bool(names & GOEMOTIONS_NEGATIVE)
    positive = bool(names & GOEMOTIONS_POSITIVE)
    if negative and positive:
        return None  # señal contradictoria: se descarta
    if negative:
        return Sentiment.NEGATIVE
    if positive:
        return Sentiment.POSITIVE
    return Sentiment.NEUTRAL  # "neutral" o solo emociones ambiguas


def build_sentiment_splits(
    frames: dict[str, pd.DataFrame], *, train_size: int | None = 12_000, seed: int = SEED
) -> tuple[Splits, dict[str, int]]:
    out: dict[str, pd.DataFrame] = {}
    report: dict[str, int] = {}
    for split, frame in frames.items():
        sentiment = frame["labels"].map(map_goemotions_sentiment)
        keep = sentiment.notna()
        report[f"{split}_contradictory_dropped"] = int((~keep).sum())
        out[split] = pd.DataFrame(
            {
                "text": frame.loc[keep, "text_es"].map(clean_text).to_numpy(),
                "label": sentiment[keep].map(lambda s: SENTIMENT_LABELS.index(str(s))).to_numpy(),
            }
        )
    train = out["train"]
    if train_size is not None and len(train) > train_size:
        train, _ = train_test_split(
            train, train_size=train_size, stratify=train["label"], random_state=seed
        )
    return Splits(
        train=train.reset_index(drop=True),
        validation=out["validation"],
        test=out["test"],
    ), report


def load_massive_ood(path: Path, limit: int | None = None) -> list[str]:
    """Frases del asistente de voz (partición de prueba) usadas como fuera de dominio."""
    texts = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("partition") == "test":
                texts.append(clean_text(record["utt"]))
    return texts[:limit] if limit else texts


def load_raw_banking(
    paths_es: dict[str, Path], paths_en: dict[str, Path]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    spanish = {split: pd.read_csv(path) for split, path in paths_es.items()}
    original = {split: pd.read_csv(path) for split, path in paths_en.items()}
    return spanish, original


def load_raw_goemotions(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {split: pd.read_parquet(path) for split, path in paths.items()}


def label_distribution(labels: Iterable[int], size: int) -> np.ndarray:
    counts = np.bincount(np.fromiter(labels, dtype=np.int64), minlength=size)
    return np.asarray(counts / max(1, int(counts.sum())), dtype=np.float64)
