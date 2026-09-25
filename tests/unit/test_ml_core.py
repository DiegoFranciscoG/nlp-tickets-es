"""Métricas, calibración, umbral y marginalización (sin PyTorch)."""

import numpy as np
import pytest

from nlp_tickets.domain.catalog import load_catalog
from nlp_tickets.ml.calibration import (
    choose_threshold,
    fit_temperature,
    negative_log_likelihood,
    probs_to_logits,
    softmax,
)
from nlp_tickets.ml.metrics import (
    brier_score,
    confusion,
    expected_calibration_error,
    macro_f1,
    paired_bootstrap_macro_f1,
    reliability_curve,
    selective_metrics,
    task_report,
)
from nlp_tickets.ml.routing import RoutingMatrices


@pytest.fixture(scope="module")
def routing() -> RoutingMatrices:
    return RoutingMatrices.from_catalog(load_catalog())


class TestRouting:
    def test_marginales_suman_uno(self, routing: RoutingMatrices) -> None:
        probs = np.random.default_rng(0).dirichlet(np.ones(77), size=5)
        for matrix in (routing.to_category, routing.to_team, routing.to_urgency):
            np.testing.assert_allclose((probs @ matrix).sum(axis=1), 1.0)

    def test_intencion_segura_se_propaga(self, routing: RoutingMatrices) -> None:
        catalog = load_catalog()
        probs = np.zeros((1, 77))
        probs[0, catalog.intent("compromised_card").label_id] = 1.0
        team = routing.team_codes[int(routing.team_probs(probs).argmax())]
        urgency = routing.urgency_codes[int(routing.urgency_probs(probs).argmax())]
        assert (team, urgency) == ("fraud_security", "high")

    def test_macro_f1_de_categoria_puede_superar_al_de_intencion(
        self, routing: RoutingMatrices
    ) -> None:
        catalog = load_catalog()
        # confundir dos intenciones del mismo equipo no afecta al enrutamiento
        a, b = (
            catalog.intent("pending_transfer").label_id,
            catalog.intent("transfer_timing").label_id,
        )
        labels = np.array([a, a])
        mapped = routing.map_labels(labels, routing.to_team)
        predicted = routing.map_labels(np.array([a, b]), routing.to_team)
        assert macro_f1(mapped, predicted) == 1.0


class TestMetricas:
    def test_macro_f1_pesa_igual_a_clases_raras(self) -> None:
        y_true = np.array([0] * 9 + [1])
        y_pred = np.array([0] * 10)
        assert macro_f1(y_true, y_pred) < 0.5

    def test_ece_cero_si_confianza_igual_a_exactitud(self) -> None:
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert expected_calibration_error(probs, np.array([0, 1])) == pytest.approx(0.0)

    def test_ece_alto_si_sobreconfiado_y_equivocado(self) -> None:
        probs = np.array([[0.99, 0.01]] * 10)
        assert expected_calibration_error(probs, np.array([1] * 10)) > 0.9

    def test_brier_y_reporte(self) -> None:
        probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        y = np.array([0, 1])
        assert brier_score(probs, y) == pytest.approx(((0.2**2) * 2 + (0.3**2) * 2) / 2)
        report = task_report(probs, y).as_dict()
        assert report["accuracy"] == 1.0
        assert report["n_samples"] == 2

    def test_selectiva(self) -> None:
        probs = np.array([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]])
        result = selective_metrics(probs, np.array([0, 1, 1]), threshold=0.6)
        assert result.coverage == pytest.approx(2 / 3)
        assert result.selective_accuracy == 1.0

    def test_selectiva_sin_aceptados(self) -> None:
        result = selective_metrics(np.array([[0.5, 0.5]]), np.array([0]), threshold=0.9)
        assert result.coverage == 0.0
        assert result.human_review_rate == 1.0

    def test_bootstrap_detecta_mejora_clara(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, 300)
        worse = np.where(rng.random(300) < 0.5, y, (y + 1) % 3)
        better = np.where(rng.random(300) < 0.95, y, (y + 1) % 3)
        diff = paired_bootstrap_macro_f1(y, worse, better, n_resamples=200)
        assert diff.delta > 0
        assert diff.ci_low > 0
        assert diff.p_value < 0.05

    def test_confusion_y_curva(self) -> None:
        matrix = confusion(np.array([0, 1, 1]), np.array([0, 0, 1]), 3)
        assert matrix.shape == (3, 3)
        assert matrix.sum() == 3
        conf, acc, counts = reliability_curve(np.array([[0.9, 0.1], [0.6, 0.4]]), np.array([0, 1]))
        assert sum(counts) == 2
        assert len(conf) == len(acc)


class TestCalibracion:
    def test_softmax_estable(self) -> None:
        probs = softmax(np.array([[1000.0, 1000.0]]))
        np.testing.assert_allclose(probs, [[0.5, 0.5]])

    def test_temperatura_recupera_la_verdadera(self) -> None:
        rng = np.random.default_rng(0)
        true_logits = rng.normal(size=(4000, 5)) * 2
        y = np.array([rng.choice(5, p=p) for p in softmax(true_logits)])
        overconfident = true_logits * 3  # equivale a T = 1/3
        temperature = fit_temperature(overconfident, y)
        assert temperature == pytest.approx(3.0, rel=0.15)
        assert negative_log_likelihood(overconfident, y, temperature) < negative_log_likelihood(
            overconfident, y, 1.0
        )

    def test_probs_to_logits_reversible(self) -> None:
        probs = np.array([[0.7, 0.2, 0.1]])
        np.testing.assert_allclose(softmax(probs_to_logits(probs)), probs)

    def test_umbral_alcanza_objetivo(self) -> None:
        probs = np.array([[0.95, 0.05], [0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
        y = np.array([0, 0, 0, 1, 1])
        threshold = choose_threshold(probs, y, target_accuracy=1.0)
        assert threshold == pytest.approx(0.8)
        assert selective_metrics(probs, y, threshold).selective_accuracy == 1.0

    def test_umbral_imposible_manda_todo_a_revision(self) -> None:
        probs = np.array([[0.9, 0.1]])
        assert choose_threshold(probs, np.array([1]), target_accuracy=0.99) == 1.0
