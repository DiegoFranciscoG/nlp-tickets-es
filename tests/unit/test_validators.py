import pytest

from nlp_tickets.anonymization.validators import (
    cedula_check_digit,
    has_valid_province,
    is_luhn_valid,
    is_valid_cedula,
    looks_like_ruc,
    only_digits,
)


def make_cedula(first_nine: str) -> str:
    """Construye un número estructuralmente válido (ficticio) a partir de 9 dígitos."""
    return first_nine + str(cedula_check_digit(first_nine))


class TestCedula:
    @pytest.mark.parametrize("prefix", ["170000001", "090000002", "300000003", "240000004"])
    def test_numero_construido_con_digito_verificador_es_valido(self, prefix: str) -> None:
        assert is_valid_cedula(make_cedula(prefix))

    def test_digito_verificador_incorrecto_es_invalido(self) -> None:
        valid = make_cedula("170000001")
        wrong = valid[:9] + str((int(valid[9]) + 1) % 10)
        assert not is_valid_cedula(wrong)

    @pytest.mark.parametrize("prefix", ["250000001", "000000001", "990000001"])
    def test_provincia_inexistente_es_invalida(self, prefix: str) -> None:
        assert not is_valid_cedula(make_cedula(prefix))

    def test_tercer_digito_de_sociedad_no_es_persona_natural(self) -> None:
        assert not is_valid_cedula(make_cedula("179000001"))

    def test_acepta_guion_antes_del_verificador(self) -> None:
        number = make_cedula("170000001")
        assert is_valid_cedula(f"{number[:9]}-{number[9]}")

    @pytest.mark.parametrize("value", ["", "12345", "17000000011", "abcdefghij"])
    def test_longitud_o_formato_incorrecto(self, value: str) -> None:
        assert not is_valid_cedula(value)

    def test_check_digit_rechaza_entradas_mal_formadas(self) -> None:
        with pytest.raises(ValueError, match="9 dígitos"):
            cedula_check_digit("123")


class TestRuc:
    def test_ruc_de_persona_natural_es_cedula_mas_001(self) -> None:
        assert looks_like_ruc(make_cedula("170000001") + "001")

    def test_ruc_sin_terminacion_001(self) -> None:
        assert not looks_like_ruc(make_cedula("170000001") + "002")

    def test_ruc_con_provincia_invalida(self) -> None:
        assert not looks_like_ruc("9912345678001")


class TestLuhn:
    @pytest.mark.parametrize(
        "pan",
        ["4111111111111111", "4111 1111 1111 1111", "5555-5555-5555-4444", "378282246310005"],
    )
    def test_numeros_de_prueba_publicos_son_validos(self, pan: str) -> None:
        assert is_luhn_valid(pan)

    @pytest.mark.parametrize("pan", ["4111111111111112", "1234", "12345678901234567890"])
    def test_numeros_invalidos(self, pan: str) -> None:
        assert not is_luhn_valid(pan)


def test_helpers() -> None:
    assert only_digits("09-9 12a") == "09912"
    assert has_valid_province("17")
    assert not has_valid_province("7")
