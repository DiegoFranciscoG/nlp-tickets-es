"""Métricas de evaluación (docs/investigacion.md E1-E5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from nlp_tickets.ml.routing import FloatArray, IntArray


def macro_f1(y_true: IntArray, y_pred: IntArray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true: IntArray, y_pred: IntArray) -> float:
    return float(accuracy_score(y_true, y_pred))


def expected_calibration_error(probs: FloatArray, y_true: IntArray, n_bins: int = 15) -> float:
    """ECE con bins de igual ancho sobre la confianza máxima (Guo et al., 2017)."""
    confidences = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in pairwise(edges):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidences[in_bin].mean())
            ece += in_bin.mean() * gap
    return float(ece)


def brier_score(probs: FloatArray, y_true: IntArray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def reliability_curve(
    probs: FloatArray, y_true: IntArray, n_bins: int = 15
) -> tuple[list[float], list[float], list[int]]:
    confidences = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf, mean_acc, counts = [], [], []
    for low, high in pairwise(edges):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            mean_conf.append(float(confidences[in_bin].mean()))
            mean_acc.append(float(correct[in_bin].mean()))
            counts.append(int(in_bin.sum()))
    return mean_conf, mean_acc, counts


@dataclass(frozen=True)
class SelectiveResult:
    threshold: float
    coverage: float
    selective_accuracy: float
    human_review_rate: float


def selective_metrics(probs: FloatArray, y_true: IntArray, threshold: float) -> SelectiveResult:
    confidences = probs.max(axis=1)
    accepted = confidences >= threshold
    coverage = float(accepted.mean())
    if accepted.any():
        sel_acc = float((probs[accepted].argmax(axis=1) == y_true[accepted]).mean())
    else:
        sel_acc = 1.0
    return SelectiveResult(threshold, coverage, sel_acc, 1.0 - coverage)


@dataclass(frozen=True)
class TaskReport:
    macro_f1: float
    accuracy: float
    ece: float
    brier: float
    n_samples: int

    def as_dict(self) -> dict[str, float | int]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def task_report(probs: FloatArray, y_true: IntArray) -> TaskReport:
    y_pred = np.asarray(probs.argmax(axis=1), dtype=np.int64)
    return TaskReport(
        macro_f1=macro_f1(y_true, y_pred),
        accuracy=accuracy(y_true, y_pred),
        ece=expected_calibration_error(probs, y_true),
        brier=brier_score(probs, y_true),
        n_samples=len(y_true),
    )


@dataclass(frozen=True)
class BootstrapDiff:
    delta: float
    ci_low: float
    ci_high: float
    p_value: float  # proporción de réplicas en las que B no supera a A


def paired_bootstrap_macro_f1(
    y_true: IntArray,
    pred_a: IntArray,
    pred_b: IntArray,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> BootstrapDiff:
    """Diferencia macro-F1(B) - macro-F1(A) con IC 95 % por bootstrap pareado sobre la prueba."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        deltas[i] = macro_f1(y_true[idx], pred_b[idx]) - macro_f1(y_true[idx], pred_a[idx])
    low, high = np.percentile(deltas, [2.5, 97.5])
    return BootstrapDiff(
        delta=macro_f1(y_true, pred_b) - macro_f1(y_true, pred_a),
        ci_low=float(low),
        ci_high=float(high),
        p_value=float((deltas <= 0).mean()),
    )


def confusion(y_true: IntArray, y_pred: IntArray, n_classes: int) -> IntArray:
    return np.asarray(confusion_matrix(y_true, y_pred, labels=list(range(n_classes))))
