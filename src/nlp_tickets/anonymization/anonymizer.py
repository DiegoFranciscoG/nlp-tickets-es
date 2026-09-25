"""Anonimización de tickets con expresiones regulares y reglas (LOPDP arts. 4, 10 y 37).

Principios de diseño:
- **Exhaustividad antes que precisión**: ante la duda se enmascara. Toda secuencia de 6 o más
  dígitos desaparece aunque no se sepa qué es.
- Detectores por prioridad: un tramo ya enmascarado no puede volver a enmascararse.
- El texto original nunca sale de esta función: ni logs, ni base de datos.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from nlp_tickets.anonymization.names import FIRST_NAMES
from nlp_tickets.anonymization.validators import (
    is_luhn_valid,
    is_valid_cedula,
    looks_like_ruc,
    only_digits,
)


class EntityKind(StrEnum):
    URL = "URL"
    EMAIL = "CORREO"
    CARD = "TARJETA"
    RUC = "RUC"
    PHONE = "TELEFONO"
    CEDULA = "CEDULA"
    ACCOUNT = "CUENTA"
    NUMBER = "NUMERO"
    NAME = "NOMBRE"

    @property
    def placeholder(self) -> str:
        return f"<{self.value}>"


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    kind: EntityKind

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class AnonymizationResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


_UPPER = "A-ZÁÉÍÓÚÑÜ"
_LOWER = "a-záéíóúñü"
_WORD = rf"[{_UPPER}][{_LOWER}]+"
_NAME_SEQUENCE = rf"{_WORD}(?:\s+(?:de\s+(?:la\s+|los\s+)?)?{_WORD}){{0,3}}"

_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b", re.IGNORECASE)
_CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[ -]){3}\d{1,7}(?!\d)|(?<!\d)\d{13,19}(?!\d)")
_RUC_RE = re.compile(r"(?<!\d)\d{13}(?!\d)")
_PHONE_INTL_RE = re.compile(r"(?:\+|\b00)593[\s.-]?\(?0?\d\)?(?:[\s.-]?\d){7,8}(?!\d)")
_PHONE_MOBILE_RE = re.compile(r"(?<!\d)09\d[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_PHONE_LANDLINE_RE = re.compile(r"(?<![\d(])\(?0[2-7]\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_CEDULA_RE = re.compile(r"(?<!\d)\d{9}-?\d(?!\d)")
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{2,}(?:[-.]\d{2,})+(?!\d)|(?<!\d)\d{6,}(?!\d)")
_NAME_TRIGGER_RE = re.compile(
    r"(?i:\b(?:me\s+llamo|mi\s+nombre\s+es|soy|atentamente|att|cordialmente|saludos|firma|"
    r"sr|sra|srta|señor|señora|señorita|don|doña|ing|lic|dr|dra|abg)\b\.?,?:?)\s+"
    rf"(?P<name>{_NAME_SEQUENCE})"
)
_CAPITALIZED_WORD_RE = re.compile(rf"\b{_WORD}\b")
_NAME_SEQUENCE_RE = re.compile(_NAME_SEQUENCE)

# Palabras de contexto (ya en minúsculas y sin tildes), ancladas al inicio de palabra.
_ID_CONTEXT = re.compile(r"\b(?:cedula|c\.\s?i\b|ci\b|identificacion|documento|dni)")
_PHONE_CONTEXT = re.compile(r"\b(?:cel|tel|whats|movil|llam|contacto)")
_ACCOUNT_CONTEXT = re.compile(r"\b(?:cuenta|cta|nro|no\.|iban)")
_CONTEXT_WINDOW = 30


def _strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _has_context(text: str, start: int, keywords: re.Pattern[str]) -> bool:
    window = _strip_accents(text[max(0, start - _CONTEXT_WINDOW) : start].lower())
    return keywords.search(window) is not None


class Anonymizer:
    """Sustituye datos personales por marcadores como ``<CORREO>`` o ``<CEDULA>``."""

    def anonymize(self, text: str) -> AnonymizationResult:
        normalized = unicodedata.normalize("NFC", text)
        spans: list[Span] = []
        for detector in (
            self._urls,
            self._emails,
            self._cards,
            self._rucs,
            self._phones,
            self._cedulas,
            self._long_numbers,
            self._names_by_trigger,
            self._names_by_gazetteer,
        ):
            for span in detector(normalized):
                if not any(span.overlaps(taken) for taken in spans):
                    spans.append(span)
        return self._apply(normalized, spans)

    @staticmethod
    def _apply(text: str, spans: list[Span]) -> AnonymizationResult:
        counts: Counter[str] = Counter()
        result = text
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            result = result[: span.start] + span.kind.placeholder + result[span.end :]
            counts[span.kind.value] += 1
        return AnonymizationResult(text=result, counts=dict(counts))

    @staticmethod
    def _urls(text: str) -> list[Span]:
        return [Span(m.start(), m.end(), EntityKind.URL) for m in _URL_RE.finditer(text)]

    @staticmethod
    def _emails(text: str) -> list[Span]:
        return [Span(m.start(), m.end(), EntityKind.EMAIL) for m in _EMAIL_RE.finditer(text)]

    @staticmethod
    def _cards(text: str) -> list[Span]:
        return [
            Span(m.start(), m.end(), EntityKind.CARD)
            for m in _CARD_RE.finditer(text)
            if is_luhn_valid(m.group())
        ]

    @staticmethod
    def _rucs(text: str) -> list[Span]:
        return [
            Span(m.start(), m.end(), EntityKind.RUC)
            for m in _RUC_RE.finditer(text)
            if looks_like_ruc(m.group())
        ]

    @staticmethod
    def _phones(text: str) -> list[Span]:
        spans = [Span(m.start(), m.end(), EntityKind.PHONE) for m in _PHONE_INTL_RE.finditer(text)]
        for match in _PHONE_MOBILE_RE.finditer(text):
            # 09XXXXXXXX también puede ser una cédula de Guayas (provincia 09): decide el contexto.
            kind = (
                EntityKind.CEDULA
                if _has_context(text, match.start(), _ID_CONTEXT)
                else EntityKind.PHONE
            )
            spans.append(Span(match.start(), match.end(), kind))
        spans.extend(
            Span(m.start(), m.end(), EntityKind.PHONE) for m in _PHONE_LANDLINE_RE.finditer(text)
        )
        return spans

    @staticmethod
    def _cedulas(text: str) -> list[Span]:
        spans = []
        for match in _CEDULA_RE.finditer(text):
            if _has_context(text, match.start(), _PHONE_CONTEXT):
                kind = EntityKind.PHONE
            elif _has_context(text, match.start(), _ACCOUNT_CONTEXT):
                kind = EntityKind.ACCOUNT
            elif is_valid_cedula(match.group()) or _has_context(text, match.start(), _ID_CONTEXT):
                kind = EntityKind.CEDULA
            else:
                kind = EntityKind.NUMBER
            spans.append(Span(match.start(), match.end(), kind))
        return spans

    @staticmethod
    def _long_numbers(text: str) -> list[Span]:
        spans = []
        for match in _LONG_NUMBER_RE.finditer(text):
            if len(only_digits(match.group())) < 6:
                continue
            kind = (
                EntityKind.ACCOUNT
                if _has_context(text, match.start(), _ACCOUNT_CONTEXT)
                else EntityKind.NUMBER
            )
            spans.append(Span(match.start(), match.end(), kind))
        return spans

    @staticmethod
    def _names_by_trigger(text: str) -> list[Span]:
        return [
            Span(m.start("name"), m.end("name"), EntityKind.NAME)
            for m in _NAME_TRIGGER_RE.finditer(text)
        ]

    @staticmethod
    def _names_by_gazetteer(text: str) -> list[Span]:
        spans: list[Span] = []
        for word in _CAPITALIZED_WORD_RE.finditer(text):
            if spans and word.start() < spans[-1].end:
                continue  # ya forma parte de un nombre compuesto detectado
            if _strip_accents(word.group().lower()) not in _FIRST_NAMES_PLAIN:
                continue
            full = _NAME_SEQUENCE_RE.match(text, word.start())
            end = full.end() if full else word.end()
            spans.append(Span(word.start(), end, EntityKind.NAME))
        return spans


_FIRST_NAMES_PLAIN = frozenset(_strip_accents(name) for name in FIRST_NAMES)


def anonymize(text: str) -> str:
    """Atajo funcional usado en los pipelines de datos."""
    return Anonymizer().anonymize(text).text
