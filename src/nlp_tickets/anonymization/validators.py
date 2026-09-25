"""Validadores de identificadores ecuatorianos y de tarjetas de pago.

- Cédula: 10 dígitos; los dos primeros son la provincia (01-24, o 30 para ecuatorianos
  registrados en el exterior). El dígito verificador se calcula con módulo 10 y coeficientes
  2-1-2-1... (SUPUESTO 3 en docs/investigacion.md: algoritmo de uso extendido, sin
  especificación oficial publicada).
- RUC: 13 dígitos; los dos primeros son la provincia y los tres últimos, ``001`` (SRI, Reglamento
  a la Ley del RUC, art. 3).
- Tarjeta (PAN): algoritmo de Luhn (ISO/IEC 7812-1).
"""

from __future__ import annotations

_PROVINCES = frozenset([*range(1, 25), 30])
_CEDULA_COEFFICIENTS = (2, 1, 2, 1, 2, 1, 2, 1, 2)


def only_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def has_valid_province(digits: str) -> bool:
    return len(digits) >= 2 and int(digits[:2]) in _PROVINCES


def cedula_check_digit(first_nine: str) -> int:
    """Dígito verificador (módulo 10) para los 9 primeros dígitos de una cédula."""
    if len(first_nine) != 9 or not first_nine.isdigit():
        raise ValueError("se esperaban 9 dígitos")
    total = 0
    for digit, coefficient in zip(first_nine, _CEDULA_COEFFICIENTS, strict=True):
        product = int(digit) * coefficient
        total += product - 9 if product >= 10 else product
    return (10 - total % 10) % 10


def is_valid_cedula(value: str) -> bool:
    digits = only_digits(value)
    if len(digits) != 10 or not has_valid_province(digits):
        return False
    if int(digits[2]) >= 6:  # tercer dígito < 6 para personas naturales
        return False
    return cedula_check_digit(digits[:9]) == int(digits[9])


def looks_like_ruc(value: str) -> bool:
    """Estructura del RUC (art. 3): 13 dígitos, provincia válida y terminación 001."""
    digits = only_digits(value)
    return len(digits) == 13 and has_valid_province(digits) and digits.endswith("001")


def is_luhn_valid(value: str) -> bool:
    digits = only_digits(value)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0
