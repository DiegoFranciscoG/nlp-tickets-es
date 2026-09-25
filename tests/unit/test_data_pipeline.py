"""Preparación de datos y líneas base con datos sintéticos pequeños."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nlp_tickets.domain.catalog import Catalog, load_catalog
from nlp_tickets.domain.enums import Sentiment
from nlp_tickets.ml import data as data_mod
from nlp_tickets.ml.baseline import expand_proba, fit_embedding_baseline, fit_tfidf_baseline
from nlp_tickets.ml.seed_data import build_seed_frame


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


class TestSentimientoGoEmotions:
    @pytest.mark.parametrize(
        ("labels", "expected"),
        [
            ([2], Sentiment.NEGATIVE),  # anger
            ([15], Sentiment.POSITIVE),  # gratitude
            ([27], Sentiment.NEUTRAL),  # neutral
            ([6], Sentiment.NEUTRAL),  # confusion (ambigua)
            ([2, 6], Sentiment.NEGATIVE),  # anger + confusion
        ],
    )
    def test_mapeo_oficial(self, labels: list[int], expected: Sentiment) -> None:
        assert data_mod.map_goemotions_sentiment(labels) is expected

    def test_contradictorio_se_descarta(self) -> None:
        assert data_mod.map_goemotions_sentiment([2, 15]) is None

    def test_particiones_de_sentimiento(self) -> None:
        frame = pd.DataFrame(
            {
                "text_es": ["gracias", "qué rabia", "hola", "gracias y rabia"] * 3,
                "labels": [[15], [2], [27], [15, 2]] * 3,
            }
        )
        splits, report = data_mod.build_sentiment_splits(
            {"train": frame, "validation": frame, "test": frame}, train_size=None
        )
        assert report["train_contradictory_dropped"] == 3
        assert set(splits.train["label"]) == {0, 1, 2}


class TestAlineacion:
    def test_concordancia_alta(self) -> None:
        report = data_mod.validate_alignment(["a", "a", "b", "b"], ["x", "x", "y", "y"])
        assert report.agreement == 1.0
        assert report.ambiguous_variants == 0

    def test_tamanos_distintos(self) -> None:
        with pytest.raises(data_mod.AlignmentError, match="tamaños"):
            data_mod.validate_alignment(["a"], ["x", "y"])

    def test_corpus_desalineado(self) -> None:
        with pytest.raises(data_mod.AlignmentError, match="concordancia"):
            data_mod.validate_alignment(["a", "a", "a", "a"], ["x", "y", "z", "w"])


def _raw_banking(catalog: Catalog, per_intent: int = 12) -> tuple[dict, dict]:
    labels = [i.source_label for i in catalog.intents for _ in range(per_intent)]
    texts = [f"consulta {n} sobre {label}" for n, label in enumerate(labels)]
    spanish = pd.DataFrame({"frase": texts, "rotulo": labels, "cambiado": 0})
    original = pd.DataFrame({"text": texts, "category": labels})
    test_s = spanish.iloc[::3].reset_index(drop=True)
    test_o = original.iloc[::3].reset_index(drop=True)
    return {"train": spanish, "test": test_s}, {"train": original, "test": test_o}


class TestParticiones:
    def test_fuga_de_prueba_se_elimina_y_hash_es_estable(self, catalog: Catalog) -> None:
        spanish, original = _raw_banking(catalog)
        splits, report = data_mod.build_intent_splits(
            spanish, original, catalog, validation_size=0.25
        )
        assert report["train_test_leakage_removed"] == len(spanish["test"])
        train_keys = set(splits.train["text"].map(data_mod.dedup_key))
        assert train_keys.isdisjoint(set(splits.test["text"].map(data_mod.dedup_key)))
        again, _ = data_mod.build_intent_splits(spanish, original, catalog, validation_size=0.25)
        assert splits.fingerprint() == again.fingerprint()

    def test_retroalimentacion_se_agrega_al_entrenamiento(self, catalog: Catalog) -> None:
        spanish, original = _raw_banking(catalog)
        extra = pd.DataFrame(
            {
                "text": ["me cobraron doble en el súper", "consulta 0 sobre activate_my_card"],
                "label": [63, 0],
            }
        )
        base, _ = data_mod.build_intent_splits(spanish, original, catalog, validation_size=0.25)
        splits, report = data_mod.build_intent_splits(
            spanish, original, catalog, extra_train=extra, validation_size=0.25
        )
        assert report["feedback_rows_added"] == 1  # la otra fila estaba en prueba
        assert report["feedback_leakage_removed"] == 1
        assert (
            len(splits.train) + len(splits.validation) == len(base.train) + len(base.validation) + 1
        )

    def test_limpieza_anonimiza(self) -> None:
        assert data_mod.clean_text("  mi   correo es a@example.com ") == "mi correo es <CORREO>"

    def test_ood_solo_particion_de_prueba(self, tmp_path: Path) -> None:
        path = tmp_path / "es-ES.jsonl"
        rows = [{"partition": "test", "utt": "pon música"}, {"partition": "train", "utt": "hola"}]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert data_mod.load_massive_ood(path) == ["pon música"]

    def test_distribucion(self) -> None:
        np.testing.assert_allclose(data_mod.label_distribution([0, 0, 1], 3), [2 / 3, 1 / 3, 0])

    def test_seed_solo_de_entrenamiento(self, catalog: Catalog) -> None:
        spanish, original = _raw_banking(catalog)
        splits, _ = data_mod.build_intent_splits(spanish, original, catalog, validation_size=0.25)
        seed = build_seed_frame(splits, catalog, per_intent=2)
        assert set(seed["text"]).isdisjoint(set(splits.test["text"]))
        assert set(seed["channel"]) <= {"email", "web", "chat", "phone", "social", "branch"}
        assert seed["created_at"].is_monotonic_increasing


class TestLineasBase:
    def test_expand_proba_con_clases_faltantes(self) -> None:
        full = expand_proba(np.array([[0.4, 0.6]]), np.array([1, 3]), 4)
        np.testing.assert_allclose(full, [[0.0, 0.4, 0.0, 0.6]])

    def test_tfidf_aprende_y_elige_c_en_validacion(self) -> None:
        texts = [
            "tarjeta no llega",
            "mi tarjeta aún no llega",
            "cambio de divisa hoy",
            "tasa de cambio de divisa",
        ] * 6
        frame = pd.DataFrame({"text": texts, "label": [0, 0, 1, 1] * 6})
        splits = data_mod.Splits(frame, frame, frame)
        baseline = fit_tfidf_baseline(splits, n_classes=3, c_grid=(1.0, 4.0))
        probs = baseline.predict_proba(["la tarjeta no llega"])
        assert probs.shape == (1, 3)
        assert probs[0].argmax() == 0
        assert set(baseline.validation_macro_f1) == {1.0, 4.0}

    def test_embeddings_con_regresion_logistica(self) -> None:
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, 0.1, (10, 4)), rng.normal(3, 0.1, (10, 4))])
        y = np.array([0] * 10 + [2] * 10)
        model = fit_embedding_baseline(x, y, n_classes=3)
        assert model.predict_proba(np.array([[3.0, 3.0, 3.0, 3.0]])).argmax() == 2
