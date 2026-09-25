"""Tests de anonimización. Todos los datos son ficticios y se generan en el propio test."""

import re

import pytest

from nlp_tickets.anonymization import Anonymizer, anonymize
from nlp_tickets.anonymization.validators import cedula_check_digit


def fake_cedula(first_nine: str = "170000001") -> str:
    return first_nine + str(cedula_check_digit(first_nine))


@pytest.fixture
def anonymizer() -> Anonymizer:
    return Anonymizer()


class TestCorreosYUrls:
    def test_enmascara_correo(self, anonymizer: Anonymizer) -> None:
        result = anonymizer.anonymize("Escríbanme a cliente.prueba+soporte@example.com por favor")
        assert result.text == "Escríbanme a <CORREO> por favor"
        assert result.counts == {"CORREO": 1}

    def test_enmascara_url_con_token(self, anonymizer: Anonymizer) -> None:
        result = anonymizer.anonymize("El enlace https://example.org/reset?token=abc123 no abre")
        assert result.text == "El enlace <URL> no abre"


class TestIdentificadores:
    def test_cedula_valida_sin_contexto(self, anonymizer: Anonymizer) -> None:
        text = f"el titular {fake_cedula()} no puede ingresar"
        assert anonymizer.anonymize(text).text == "el titular <CEDULA> no puede ingresar"

    def test_cedula_con_guion(self, anonymizer: Anonymizer) -> None:
        number = fake_cedula()
        text = f"C.I. {number[:9]}-{number[9]}"
        assert anonymizer.anonymize(text).text == "C.I. <CEDULA>"

    def test_diez_digitos_invalidos_con_contexto_de_cedula(self, anonymizer: Anonymizer) -> None:
        assert anonymize("mi cédula es 1799999990") == "mi cédula es <CEDULA>"

    def test_diez_digitos_sin_contexto_ni_verificador_se_enmascaran_igual(
        self, anonymizer: Anonymizer
    ) -> None:
        result = anonymizer.anonymize("referencia 1799999990 del pago")
        assert result.text == "referencia <NUMERO> del pago"

    def test_ruc(self, anonymizer: Anonymizer) -> None:
        text = f"Facturar al RUC {fake_cedula()}001"
        assert anonymizer.anonymize(text).text == "Facturar al RUC <RUC>"

    def test_numero_de_cuenta_por_contexto(self, anonymizer: Anonymizer) -> None:
        text = "transferí a la cuenta 2200112233 y no llegó"
        assert anonymizer.anonymize(text).text == "transferí a la cuenta <CUENTA> y no llegó"

    def test_secuencia_larga_con_guiones(self, anonymizer: Anonymizer) -> None:
        assert anonymize("cta. 22-0011-2233-44") == "cta. <CUENTA>"


class TestTelefonos:
    @pytest.mark.parametrize(
        "phone", ["0990000001", "099 000 0001", "099-000-0001", "+593 99 000 0001", "+593990000001"]
    )
    def test_celular(self, anonymizer: Anonymizer, phone: str) -> None:
        assert anonymizer.anonymize(f"llámenme al {phone} hoy").text == "llámenme al <TELEFONO> hoy"

    @pytest.mark.parametrize("phone", ["022000001", "(02) 200-0001", "04 200 0001"])
    def test_convencional(self, anonymizer: Anonymizer, phone: str) -> None:
        assert anonymize(f"teléfono {phone}") == "teléfono <TELEFONO>"

    def test_09_con_contexto_de_cedula_es_cedula(self, anonymizer: Anonymizer) -> None:
        cedula_guayas = fake_cedula("090000002")
        text = f"número de cédula {cedula_guayas}"
        assert anonymizer.anonymize(text).text == "número de cédula <CEDULA>"

    def test_contexto_no_se_confunde_con_subcadenas(self, anonymizer: Anonymizer) -> None:
        # "cancelar" contiene "cel", pero no es contexto de teléfono.
        text = f"quiero cancelar {fake_cedula()}"
        assert anonymizer.anonymize(text).text == "quiero cancelar <CEDULA>"


class TestTarjetas:
    @pytest.mark.parametrize("pan", ["4111111111111111", "4111 1111 1111 1111"])
    def test_pan_valido_por_luhn(self, anonymizer: Anonymizer, pan: str) -> None:
        result = anonymizer.anonymize(f"me cobraron en la tarjeta {pan} dos veces")
        assert result.text == "me cobraron en la tarjeta <TARJETA> dos veces"
        assert result.counts == {"TARJETA": 1}

    def test_ultimos_cuatro_digitos_no_se_tocan(self, anonymizer: Anonymizer) -> None:
        text = "la tarjeta terminada en 1111 fue rechazada"
        assert anonymizer.anonymize(text).text == text


class TestNombres:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Hola, me llamo Andrea Paredes y necesito ayuda", "Hola, me llamo <NOMBRE> y"),
            ("Mi nombre es José Luis Andrade de la Torre.", "Mi nombre es <NOMBRE>."),
            ("Atentamente, Carlos Vera", "Atentamente, <NOMBRE>"),
            ("Sra. Tamayo solicita reposición", "Sra. <NOMBRE> solicita"),
            ("Ing. Mendoza no recibió la transferencia", "Ing. <NOMBRE> no recibió"),
        ],
    )
    def test_nombres_por_disparador(self, anonymizer: Anonymizer, text: str, expected: str) -> None:
        assert anonymizer.anonymize(text).text.startswith(expected)

    def test_nombre_por_listado_con_apellidos(self, anonymizer: Anonymizer) -> None:
        result = anonymizer.anonymize("Hola María Fernanda Salazar, su tarjeta llegó")
        assert result.text == "Hola <NOMBRE>, su tarjeta llegó"

    def test_nombre_sin_tilde_en_listado(self, anonymizer: Anonymizer) -> None:
        assert anonymize("Maria Pérez no puede ingresar") == "<NOMBRE> no puede ingresar"

    def test_palabras_comunes_no_se_confunden_con_nombres(self, anonymizer: Anonymizer) -> None:
        text = "No hay Luz en la agencia y Soy cliente desde 2019"
        assert anonymizer.anonymize(text).text == text


class TestPropiedades:
    def test_texto_sin_datos_personales_no_cambia(self, anonymizer: Anonymizer) -> None:
        text = "¿Por qué mi transferencia de $250 sigue pendiente desde el 15/09?"
        result = anonymizer.anonymize(text)
        assert result.text == text
        assert result.total == 0

    def test_ticket_completo(self, anonymizer: Anonymizer) -> None:
        text = (
            f"Buenas tardes, soy Diana Rojas con cédula {fake_cedula()}. Me robaron la tarjeta "
            "4111 1111 1111 1111 y veo consumos que no reconozco. Mi celular es 099 000 0001 "
            "y mi correo diana.prueba@example.com"
        )
        result = anonymizer.anonymize(text)
        assert result.counts == {"NOMBRE": 1, "CEDULA": 1, "TARJETA": 1, "TELEFONO": 1, "CORREO": 1}
        assert not re.search(r"\d{6,}", result.text)
        assert "example.com" not in result.text
        assert "Diana" not in result.text

    def test_es_idempotente(self, anonymizer: Anonymizer) -> None:
        text = f"soy Pablo Ortiz, CI {fake_cedula()}, correo pablo@example.com"
        once = anonymizer.anonymize(text).text
        assert anonymizer.anonymize(once).text == once

    def test_normaliza_unicode_nfc(self, anonymizer: Anonymizer) -> None:
        decomposed = "José Pérez"  # José Pérez con tildes combinantes
        assert anonymizer.anonymize(f"Sr. {decomposed}").text == "Sr. <NOMBRE>"
