"""Inferencia en producción: ONNX Runtime + tokenizers (sin PyTorch) y verificación por SHA-256."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from nlp_tickets.ml.routing import FloatArray

META_FILE = "model_meta.json"
_PLACEHOLDER = re.compile(r"^<[A-Z]+>[.,;:!?]*$")
MAX_EXPLAINED_WORDS = 40


class ModelIntegrityError(RuntimeError):
    """El artefacto no coincide con el hash esperado (OWASP ML06)."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(logits: FloatArray, temperature: float = 1.0) -> FloatArray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return np.asarray(exp / exp.sum(axis=-1, keepdims=True), dtype=np.float64)


@dataclass(frozen=True)
class ModelMeta:
    name: str
    version: str
    intent_labels: list[str]
    sentiment_labels: list[str]
    temperature_intent: float
    temperature_sentiment: float
    confidence_threshold: float
    max_length: int
    pad_id: int
    pad_token: str
    embedding_dim: int
    onnx_sha256: str
    dataset_revision: str
    trained_at: str
    extra: dict[str, Any]

    @classmethod
    def load(cls, model_dir: Path) -> ModelMeta:
        raw = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
        fields = {k: raw.pop(k) for k in list(cls.__dataclass_fields__) if k in raw}
        return cls(**{**fields, "extra": raw})


@dataclass(frozen=True)
class ModelOutput:
    intent_probs: FloatArray
    sentiment_probs: FloatArray
    embedding: FloatArray


class OnnxTicketModel:
    """Carga ``model.onnx`` + ``tokenizer.json`` + ``model_meta.json`` desde ``model_dir``."""

    def __init__(self, model_dir: Path, expected_sha256: str = "") -> None:
        self.meta = ModelMeta.load(model_dir)
        onnx_path = model_dir / "model.onnx"
        actual = sha256_file(onnx_path)
        if actual != self.meta.onnx_sha256 or (expected_sha256 and actual != expected_sha256):
            raise ModelIntegrityError("el hash SHA-256 de model.onnx no coincide")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self.meta.max_length)
        self._tokenizer.enable_padding(pad_id=self.meta.pad_id, pad_token=self.meta.pad_token)

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def version(self) -> str:
        return self.meta.version

    @property
    def confidence_threshold(self) -> float:
        return self.meta.confidence_threshold

    @property
    def trained_at(self) -> str:
        return self.meta.trained_at

    def _run(self, texts: list[str]) -> tuple[FloatArray, FloatArray, FloatArray]:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
        attention = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
        intent, sentiment, embedding = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention}
        )
        return (
            np.asarray(intent, dtype=np.float64),
            np.asarray(sentiment, dtype=np.float64),
            np.asarray(embedding, dtype=np.float64),
        )

    def raw_outputs(
        self, texts: list[str], batch_size: int = 256
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Logits sin calibrar y embeddings, por lotes (evaluación y calibración)."""
        chunks = [self._run(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
        return (
            np.concatenate([c[0] for c in chunks]),
            np.concatenate([c[1] for c in chunks]),
            np.concatenate([c[2] for c in chunks]),
        )

    def predict_batch(self, texts: list[str]) -> list[ModelOutput]:
        intent, sentiment, embedding = self._run(texts)
        intent_probs = softmax(intent, self.meta.temperature_intent)
        sentiment_probs = softmax(sentiment, self.meta.temperature_sentiment)
        return [
            ModelOutput(intent_probs[i], sentiment_probs[i], embedding[i])
            for i in range(len(texts))
        ]

    def predict(self, text: str) -> ModelOutput:
        return self.predict_batch([text])[0]

    def embed(self, text: str) -> FloatArray:
        return np.asarray(self._run([text])[2][0], dtype=np.float64)

    def explain(self, text: str, intent_index: int, top_k: int = 5) -> list[tuple[str, float]]:
        """Importancia por oclusión: cuánto baja p(intención) al quitar cada palabra."""
        words = text.split()[:MAX_EXPLAINED_WORDS]
        candidates = [i for i, w in enumerate(words) if not _PLACEHOLDER.match(w)]
        if len(candidates) < 2:  # con una sola palabra la oclusión no informa nada
            return []
        base = softmax(self._run([text])[0], self.meta.temperature_intent)[0, intent_index]
        variants = [" ".join(w for j, w in enumerate(words) if j != i) for i in candidates]
        occluded = softmax(self._run(variants)[0], self.meta.temperature_intent)[:, intent_index]
        scored = [
            (words[i].strip("¿?¡!.,;:"), float(base - p))
            for i, p in zip(candidates, occluded, strict=True)
        ]
        scored = [(w, s) for w, s in scored if w and s > 0]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
