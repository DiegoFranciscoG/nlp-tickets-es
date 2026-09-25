from dataclasses import replace

import pytest

from nlp_tickets.domain.catalog import Catalog, CatalogError, load_catalog
from nlp_tickets.domain.enums import Urgency


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return load_catalog()


def test_catalogo_completo(catalog: Catalog) -> None:
    assert len(catalog.intents) == 77
    assert len(catalog.categories) == 9
    assert len(catalog.teams) == 6


def test_toda_intencion_tiene_resolucion_y_nombre(catalog: Catalog) -> None:
    for intent in catalog.intents:
        assert intent.standard_resolution.strip()
        assert intent.name_es.strip()


def test_urgencias_criticas_son_de_fraude_o_dinero_en_riesgo(catalog: Catalog) -> None:
    high = {i.code for i in catalog.intents if i.base_urgency is Urgency.HIGH}
    assert {"compromised_card", "lost_or_stolen_card", "card_payment_not_recognised"} <= high
    assert "exchange_rate" not in high


def test_busquedas(catalog: Catalog) -> None:
    assert catalog.intent("card_arrival").category_code == "cards"
    assert catalog.intent_by_source_label()["Refund_not_showing_up"].code == "refund_not_showing_up"
    assert catalog.category_name("cards") == "Tarjetas"
    assert catalog.team_name("fraud_security") == "Fraude y Seguridad"
    with pytest.raises(KeyError):
        catalog.intent("no_existe")


def test_rechaza_categoria_desconocida(catalog: Catalog) -> None:
    broken = [replace(catalog.intents[0], category_code="inexistente"), *catalog.intents[1:]]
    with pytest.raises(CatalogError, match="categoría"):
        Catalog(categories=catalog.categories, teams=catalog.teams, intents=broken)


def test_rechaza_equipo_desconocido(catalog: Catalog) -> None:
    broken = [replace(catalog.intents[0], team_code="inexistente"), *catalog.intents[1:]]
    with pytest.raises(CatalogError, match="equipo"):
        Catalog(categories=catalog.categories, teams=catalog.teams, intents=broken)


def test_rechaza_ids_fuera_de_orden(catalog: Catalog) -> None:
    broken = list(reversed(catalog.intents))
    with pytest.raises(CatalogError, match="label_id"):
        Catalog(categories=catalog.categories, teams=catalog.teams, intents=broken)


def test_rechaza_codigos_duplicados(catalog: Catalog) -> None:
    duplicated = replace(catalog.intents[1], code=catalog.intents[0].code)
    broken = [catalog.intents[0], duplicated, *catalog.intents[2:]]
    with pytest.raises(CatalogError, match="duplicados"):
        Catalog(categories=catalog.categories, teams=catalog.teams, intents=broken)
