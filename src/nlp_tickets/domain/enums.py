"""Valores cerrados del dominio. Se reflejan como CHECK constraints en la base de datos."""

from enum import StrEnum


class Channel(StrEnum):
    EMAIL = "email"
    WEB = "web"
    CHAT = "chat"
    PHONE = "phone"
    SOCIAL = "social"
    BRANCH = "branch"


class TicketSource(StrEnum):
    API = "api"
    SEED = "seed"
    DATASET = "dataset"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return _URGENCY_RANK[self]


_URGENCY_RANK = {Urgency.LOW: 0, Urgency.MEDIUM: 1, Urgency.HIGH: 2}


class Sentiment(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class Decision(StrEnum):
    AUTO_ROUTE = "auto_route"
    HUMAN_REVIEW = "human_review"


class ReviewReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    HIGH_URGENCY_LOW_MARGIN = "high_urgency_low_margin"
    ESCALATED_BY_RULE = "escalated_by_rule"


class LabelSource(StrEnum):
    DATASET = "dataset"
    HUMAN = "human"
    CATALOG = "catalog"


class FeedbackStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ModelBackend(StrEnum):
    ONNX = "onnx"
    SKLEARN = "sklearn"


class MetricTask(StrEnum):
    INTENT = "intent"
    CATEGORY = "category"
    TEAM = "team"
    URGENCY = "urgency"
    SENTIMENT = "sentiment"
    OOD = "ood"


class MetricSplit(StrEnum):
    VALIDATION = "validation"
    TEST = "test"
    OOD = "ood"
