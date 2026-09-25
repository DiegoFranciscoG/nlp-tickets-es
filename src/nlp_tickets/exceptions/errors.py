"""Errores de aplicación con código estable y mensaje seguro (sin detalles internos)."""

from __future__ import annotations


class AppError(Exception):
    status_code = 500
    code = "internal_error"
    message = "Error interno"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        self.message = message or self.message


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "Recurso no encontrado"


class ValidationFailedError(AppError):
    status_code = 422
    code = "validation_failed"
    message = "Datos inválidos"


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthorized"
    message = "API key ausente o inválida"


class ForbiddenError(AppError):
    status_code = 403
    code = "forbidden"
    message = "La API key no tiene permiso para esta operación"


class RateLimitError(AppError):
    status_code = 429
    code = "rate_limited"
    message = "Demasiadas solicitudes; intenta de nuevo en un minuto"


class ModelUnavailableError(AppError):
    status_code = 503
    code = "model_unavailable"
    message = "El modelo no está cargado; revisa MODEL_DIR o MODEL_URL"
