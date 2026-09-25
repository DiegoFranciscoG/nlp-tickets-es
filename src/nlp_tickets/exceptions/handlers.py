"""Manejadores globales: respuestas uniformes, sin stack traces y sin eco de la entrada."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from nlp_tickets.exceptions.errors import AppError, RateLimitError

log = logging.getLogger("nlp_tickets.errors")


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _body(code: str, message: str, request: Request, **extra: object) -> dict[str, object]:
    return {"code": code, "message": message, "request_id": _request_id(request), **extra}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError) -> JSONResponse:
        headers = {"Retry-After": "60"} if isinstance(exc, RateLimitError) else None
        return JSONResponse(
            _body(exc.code, exc.message, request), status_code=exc.status_code, headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # No se devuelve "input": el cuerpo puede contener datos personales sin anonimizar.
        errors = [
            {"loc": list(err.get("loc", [])), "msg": err.get("msg", ""), "type": err.get("type")}
            for err in exc.errors()
        ]
        return JSONResponse(
            _body("validation_failed", "Datos inválidos", request, errors=errors), status_code=422
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}.get(
            exc.status_code, "http_error"
        )
        return JSONResponse(_body(code, str(exc.detail), request), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error(
            "error no controlado request_id=%s tipo=%s", _request_id(request), type(exc).__name__
        )
        return JSONResponse(_body("internal_error", "Error interno", request), status_code=500)
