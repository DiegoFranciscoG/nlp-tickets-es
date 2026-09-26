"""Figuras estáticas para el README y la ficha del modelo (PNG, fondo claro).

Paleta categórica validada para daltonismo (3 series) y rampa azul secuencial para matrices.
Cada figura tiene su tabla equivalente en reports/metrics.json y en el README.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
REFERENCE = "#8a8984"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def _style(ax: Axes, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.xaxis.label.set_color(TEXT_SECONDARY)


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_model_comparison(
    tasks: Sequence[str], scores: dict[str, Sequence[float]], path: Path, title: str
) -> None:
    """Gráfico de puntos por tarea y modelo (máximo 3 modelos).

    Se usan puntos y no barras porque las diferencias son de centésimas: un eje recortado es
    legítimo para posiciones, pero exageraría las diferencias si las marcas fueran barras.
    """
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(tasks))[::-1]
    offsets = np.linspace(0.18, -0.18, len(scores))
    values_all = [v for values in scores.values() for v in values if not np.isnan(v)]
    for i, (model, values) in enumerate(scores.items()):
        ax.scatter(
            values,
            y + offsets[i],
            s=70,
            color=SERIES[i],
            label=model,
            zorder=3,
            edgecolors=SURFACE,
            linewidths=1.5,
        )
        if i == len(scores) - 1:  # solo el modelo final lleva etiqueta directa
            for value, pos in zip(values, y + offsets[i], strict=True):
                if not np.isnan(value):
                    ax.annotate(
                        f"{value:.3f}",
                        (value, pos),
                        textcoords="offset points",
                        xytext=(8, 0),
                        va="center",
                        fontsize=8,
                        color=TEXT,
                    )
    ax.set_yticks(y, tasks)
    ax.set_xlim(max(0.0, min(values_all) - 0.03), 1.0)
    ax.set_xlabel("macro-F1 (prueba)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper left")
    _style(ax, title)
    _save(fig, path)


def plot_confusion(matrix: np.ndarray, labels: Sequence[str], path: Path, title: str) -> None:
    """Matriz normalizada por fila (recall por clase)."""
    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    size = 0.55 * len(labels) + 2.5
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    image = ax.imshow(normalized, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = normalized[i, j]
            if value >= 0.01:
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#ffffff" if value > 0.55 else TEXT,
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    colorbar.outline.set_visible(False)
    _style(ax, title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, path)


def plot_reliability(
    curves: dict[str, tuple[Sequence[float], Sequence[float]]], path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color=REFERENCE, linewidth=1, label="calibración perfecta"
    )
    for i, (name, (confidence, accuracy)) in enumerate(curves.items()):
        ax.plot(
            confidence,
            accuracy,
            marker="o",
            markersize=5,
            linewidth=2,
            color=SERIES[i],
            label=name,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confianza media del bin")
    ax.set_ylabel("exactitud del bin")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")
    _style(ax, title)
    _save(fig, path)


def plot_training_curve(history: Sequence[dict[str, float]], path: Path, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (key, label) in enumerate(
        (("val_intent_macro_f1", "intención"), ("val_sentiment_macro_f1", "sentimiento"))
    ):
        values = [h[key] for h in history]
        ax.plot(epochs, values, marker="o", markersize=5, linewidth=2, color=SERIES[i], label=label)
        ax.annotate(
            f"{values[-1]:.3f}",
            (epochs[-1], values[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            fontsize=8,
            color=TEXT,
        )
    ax.set_xticks(epochs)
    ax.set_xlabel("época")
    ax.set_ylabel("macro-F1 (validación)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    _style(ax, title)
    _save(fig, path)
