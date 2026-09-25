import pytest

from nlp_tickets.domain.enums import Decision, ReviewReason, Urgency
from nlp_tickets.services.decision_policy import DecisionInput, decide
from nlp_tickets.services.urgency_rules import evaluate_escalation


class TestEscalamiento:
    @pytest.mark.parametrize(
        ("text", "rule"),
        [
            ("Creo que es un FRAUDE", "fraude"),
            ("me robaron el celular", "robo"),
            ("me hackearon la cuenta", "acceso_no_autorizado"),
            ("No reconozco este cargo de ayer", "movimiento_no_reconocido"),
            ("no realicé esa compra", "movimiento_no_reconocido"),
            ("es URGENTE por favor", "urgente"),
        ],
    )
    def test_detecta_senales(self, text: str, rule: str) -> None:
        result = evaluate_escalation(text)
        assert result.escalate
        assert rule in result.rules

    @pytest.mark.parametrize(
        "text",
        ["¿Cuál es el tipo de cambio hoy?", "quiero cambiar mi PIN", "no reconozco la app nueva"],
    )
    def test_no_escala_consultas_normales(self, text: str) -> None:
        assert not evaluate_escalation(text).escalate


def _input(**overrides: object) -> DecisionInput:
    values: dict[str, object] = {
        "team_score": 0.95,
        "predicted_urgency": Urgency.LOW,
        "high_urgency_probability": 0.01,
        "escalated_by_rule": False,
        "threshold": 0.8,
    }
    values.update(overrides)
    return DecisionInput(**values)  # type: ignore[arg-type]


class TestPoliticaDeDecision:
    def test_confianza_alta_se_enruta_solo(self) -> None:
        outcome = decide(_input())
        assert outcome.decision is Decision.AUTO_ROUTE
        assert outcome.reasons == ()

    def test_confianza_baja_va_a_revision(self) -> None:
        outcome = decide(_input(team_score=0.79))
        assert outcome.decision is Decision.HUMAN_REVIEW
        assert outcome.reasons == (ReviewReason.LOW_CONFIDENCE,)

    def test_umbral_es_inclusivo(self) -> None:
        assert decide(_input(team_score=0.8)).decision is Decision.AUTO_ROUTE

    def test_riesgo_de_urgencia_alta_omitida(self) -> None:
        outcome = decide(_input(high_urgency_probability=0.3, predicted_urgency=Urgency.MEDIUM))
        assert ReviewReason.HIGH_URGENCY_LOW_MARGIN in outcome.reasons

    def test_si_ya_predice_alta_no_aplica_margen(self) -> None:
        outcome = decide(_input(high_urgency_probability=0.6, predicted_urgency=Urgency.HIGH))
        assert outcome.decision is Decision.AUTO_ROUTE

    def test_escalamiento_por_regla_se_confirma_con_humano(self) -> None:
        outcome = decide(_input(escalated_by_rule=True))
        assert outcome.reasons == (ReviewReason.ESCALATED_BY_RULE,)
