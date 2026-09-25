import pytest

from nlp_tickets.ml.summary import END, START, render_metrics_markdown, update_readme


def _task(f1: float) -> dict[str, float]:
    return {"macro_f1": f1, "accuracy": f1, "ece": 0.02, "n_samples": 3080}


def _model(f1: float, with_sentiment: bool = True) -> dict[str, object]:
    report: dict[str, object] = {
        task: _task(f1) for task in ("intent", "category", "team", "urgency")
    }
    if with_sentiment:
        report["sentiment"] = _task(f1 - 0.2)
    report["intent_ece_uncalibrated"] = 0.1
    report["routing"] = {
        "threshold": 0.8,
        "coverage_test": 0.9,
        "auto_routed_accuracy_test": 0.99,
        "human_review_rate_out_of_domain": 0.95,
    }
    return report


REPORT = {
    "version": "2026.09.25-abc",
    "split_fingerprint": "7a9c6964c0c8dddc",
    "data": {"intent": {"train": 8577, "validation": 953, "test": 3080}},
    "models": {
        "tfidf_logreg": _model(0.88),
        "embeddings_logreg": _model(0.85, with_sentiment=False),
        "minilm_multitask_onnx_int8": _model(0.91),
    },
    "comparisons_vs_tfidf": {
        t: {"delta": 0.03, "ci_low": 0.01, "ci_high": 0.05} for t in ("intent", "team", "sentiment")
    },
    "parity_fp32_vs_int8": {"onnx_int8_mb": 118.1, "onnx_fp32_mb": 470.4, "argmax_agreement": 0.99},
    "latency_ms_single_text_local_cpu": {"p50": 5.1, "p95": 8.8},
}


def test_tabla_con_los_tres_modelos() -> None:
    markdown = render_metrics_markdown(REPORT)
    assert "| Intención (77 clases) | 0.880 | 0.850 | 0.910 |" in markdown
    assert "| Sentimiento (3) | 0.680 | — | 0.710 |" in markdown
    assert "**3 080 tickets**, nunca usados" in markdown
    assert "**+0.030** [+0.010, +0.050]" in markdown


def test_reemplaza_solo_el_bloque_marcado() -> None:
    readme = f"# Título\n{START}\nviejo\n{END}\n## Después"
    updated = update_readme(readme, REPORT)
    assert "viejo" not in updated
    assert updated.startswith("# Título")
    assert updated.endswith("## Después")


def test_sin_marcadores_falla() -> None:
    with pytest.raises(ValueError, match="marcadores"):
        update_readme("# sin marcadores", REPORT)
