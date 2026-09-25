"""Pruebas del artefacto real (solo si existe artifacts/model; en CI se omiten)."""

from pathlib import Path

import numpy as np
import pytest

from nlp_tickets.domain.catalog import load_catalog
from nlp_tickets.ml.inference import OnnxTicketModel
from nlp_tickets.ml.model_store import model_present
from nlp_tickets.services.classification_service import ClassificationService

MODEL_DIR = Path("artifacts/model")
pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(not model_present(MODEL_DIR), reason="modelo no entrenado en este entorno"),
]


@pytest.fixture(scope="module")
def model() -> OnnxTicketModel:
    return OnnxTicketModel(MODEL_DIR)


def test_compatible_con_el_catalogo(model: OnnxTicketModel) -> None:
    ClassificationService.check_compatibility(model.meta.intent_labels, load_catalog())


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Me robaron la tarjeta anoche", "lost_or_stolen_card"),
        ("¿Cuándo llega mi tarjeta nueva?", "card_arrival"),
        ("Olvidé mi código de acceso de la aplicación", "passcode_forgotten"),
    ],
)
def test_predicciones_razonables(model: OnnxTicketModel, text: str, intent: str) -> None:
    output = model.predict(text)
    assert model.meta.intent_labels[int(output.intent_probs.argmax())] == intent
    assert output.intent_probs.sum() == pytest.approx(1.0)
    assert np.linalg.norm(output.embedding) == pytest.approx(1.0, abs=1e-3)


def test_explicacion_y_similitud_semantica(model: OnnxTicketModel) -> None:
    terms = model.explain("me robaron la tarjeta en el bus", intent_index=41)
    assert terms
    assert all(weight > 0 for _, weight in terms)
    a = model.embed("mi tarjeta no ha llegado")
    b = model.embed("todavía espero la tarjeta que pedí")
    c = model.embed("¿cuál es el tipo de cambio del dólar?")
    assert float(a @ b) > float(a @ c)
