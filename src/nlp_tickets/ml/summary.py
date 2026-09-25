"""Genera la sección de resultados del README a partir de reports/metrics.json."""

from __future__ import annotations

import re
from typing import Any

MODELS = (
    ("tfidf_logreg", "TF-IDF + reg. logística"),
    ("embeddings_logreg", "Embeddings congelados + RL"),
    ("minilm_multitask_onnx_int8", "**MiniLM fine-tuning (ONNX INT8)**"),
)
TASKS = (
    ("intent", "Intención (77 clases)"),
    ("category", "Categoría (9)"),
    ("team", "Equipo destino (6)"),
    ("urgency", "Urgencia (3)"),
    ("sentiment", "Sentimiento (3)"),
)
START, END = "<!-- METRICS:START -->", "<!-- METRICS:END -->"


def _int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _pct(value: float) -> str:
    return f"{value * 100:.1f} %"


def _cell(report: dict[str, Any], task: str) -> str:
    return f"{report[task]['macro_f1']:.3f}" if task in report else "—"


def render_metrics_markdown(report: dict[str, Any]) -> str:
    models = report["models"]
    final = models["minilm_multitask_onnx_int8"]
    base = models["tfidf_logreg"]
    sizes = report["data"]["intent"]
    lines = [
        "## Resultados",
        f"Partición oficial de prueba de BANKING77-ES-LA: **{_int(sizes['test'])} tickets**, "
        "nunca usados para elegir hiperparámetros. Todos los modelos usan la misma partición y "
        f"semilla (hash `{report['split_fingerprint'][:12]}`). Entrenamiento "
        f"{_int(sizes['train'])} · validación {_int(sizes['validation'])}. "
        f"Versión `{report['version']}`.",
        "",
        "| Macro-F1 en prueba | " + " | ".join(label for _, label in MODELS) + " |",
        "|---|" + "---:|" * len(MODELS),
    ]
    for task, label in TASKS:
        lines.append(f"| {label} | " + " | ".join(_cell(models[k], task) for k, _ in MODELS) + " |")
    lines += [
        "",
        "**Diferencia frente a TF-IDF** (bootstrap pareado, 1 000 réplicas, IC 95 %):",
        "",
    ]
    for task, label in (("intent", "Intención"), ("team", "Equipo"), ("sentiment", "Sentimiento")):
        diff = report["comparisons_vs_tfidf"][task]
        lines.append(
            f"- {label}: **{diff['delta']:+.3f}** [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]"
        )
    routing, base_routing = final["routing"], base["routing"]
    parity = report["parity_fp32_vs_int8"]
    latency = report["latency_ms_single_text_local_cpu"]
    lines += [
        "",
        "| Enrutamiento y revisión humana | TF-IDF + RL | MiniLM (ONNX INT8) |",
        "|---|---:|---:|",
        f"| Umbral de confianza del equipo (elegido en validación para 99 % de exactitud) | "
        f"{base_routing['threshold']:.3f} | {routing['threshold']:.3f} |",
        f"| Tickets enrutados automáticamente (cobertura) | {_pct(base_routing['coverage_test'])} "
        f"| {_pct(routing['coverage_test'])} |",
        f"| Exactitud de lo enrutado automáticamente | "
        f"{_pct(base_routing['auto_routed_accuracy_test'])} | "
        f"{_pct(routing['auto_routed_accuracy_test'])} |",
        f"| Frases fuera de dominio (MASSIVE) enviadas a revisión humana | "
        f"{_pct(base_routing['human_review_rate_out_of_domain'])} | "
        f"{_pct(routing['human_review_rate_out_of_domain'])} |",
        f"| ECE de intención (sin calibrar → calibrado) | {base['intent_ece_uncalibrated']:.3f} → "
        f"{base['intent']['ece']:.3f} | {final['intent_ece_uncalibrated']:.3f} → "
        f"{final['intent']['ece']:.3f} |",
        "",
        f"Artefacto desplegado: ONNX INT8 de **{parity['onnx_int8_mb']:.0f} MB** "
        f"(FP32: {parity['onnx_fp32_mb']:.0f} MB). Coincidencia de predicciones FP32/INT8: "
        f"{_pct(parity['argmax_agreement'])}. Latencia por ticket en CPU local: "
        f"p50 {latency['p50']} ms · p95 {latency['p95']} ms.",
        "",
        "![Comparación de macro-F1](reports/figures/comparison_macro_f1.png)",
        "",
        "| Equipo destino | Calibración |",
        "|---|---|",
        "| ![Matriz de confusión por equipo](reports/figures/confusion_team.png) "
        "| ![Diagrama de fiabilidad](reports/figures/reliability_intent.png) |",
        "",
        "Métricas completas: [reports/metrics.json](reports/metrics.json) · Ficha del modelo: "
        "[docs/model-card.md](docs/model-card.md).",
    ]
    return "\n".join(lines)


def update_readme(readme: str, report: dict[str, Any]) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(readme):
        raise ValueError("el README no tiene los marcadores de métricas")
    block = f"{START}\n{render_metrics_markdown(report)}\n{END}"
    return pattern.sub(lambda _: block, readme)
