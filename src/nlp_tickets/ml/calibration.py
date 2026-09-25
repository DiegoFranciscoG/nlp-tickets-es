"""Calibración por *temperature scaling* (Guo et al., 2017) y selección del umbral (E5)."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar

from nlp_tickets.ml.routing import FloatArray, IntArray

_EPS = 1e-12


def softmax(logits: FloatArray, temperature: float = 1.0) -> FloatArray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return np.asarray(exp / exp.sum(axis=1, keepdims=True), dtype=np.float64)


def probs_to_logits(probs: FloatArray) -> FloatArray:
    """Para modelos que devuelven probabilidades (p. ej. regresión logística)."""
    return np.asarray(np.log(np.clip(probs, _EPS, 1.0)), dtype=np.float64)


def negative_log_likelihood(logits: FloatArray, y_true: IntArray, temperature: float) -> float:
    probs = softmax(logits, temperature)
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], _EPS, 1.0))))


def fit_temperature(logits: FloatArray, y_true: IntArray) -> float:
    """Busca T > 0 que minimiza la NLL en validación. T > 1 suaviza; T < 1 agudiza."""
    result = minimize_scalar(
        lambda log_t: negative_log_likelihood(logits, y_true, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def choose_threshold(probs: FloatArray, y_true: IntArray, target_accuracy: float = 0.95) -> float:
    """Menor umbral de confianza tal que la exactitud de lo enrutado solo sea >= objetivo.

    Si ningún umbral alcanza el objetivo, devuelve 1.0 (todo a revisión humana).
    """
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    order = np.argsort(-confidences, kind="stable")
    cumulative_accuracy = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    valid = np.nonzero(cumulative_accuracy >= target_accuracy)[0]
    if valid.size == 0:
        return 1.0
    # Último índice que cumple: el umbral es la confianza de ese elemento, redondeada hacia arriba
    # a 3 decimales (conservador: ante la duda, más tickets van a revisión humana).
    return float(min(1.0, math.ceil(confidences[order][valid[-1]] * 1000) / 1000))
