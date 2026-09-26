# syntax=docker/dockerfile:1.7
# Imagen de la API: sin PyTorch (inferencia con ONNX Runtime), usuario no root.
FROM ghcr.io/astral-sh/uv:0.12.19 AS uv

FROM python:3.12.14-slim-bookworm AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups --no-editable

FROM python:3.12.14-slim-bookworm AS runtime
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/artifacts/model \
    APP_ENV=production
COPY --from=builder /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations ./migrations
COPY data/catalog ./data/catalog
COPY data/seed ./data/seed
COPY reports ./reports
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN mkdir -p /app/artifacts/model && chown -R app:app /app/artifacts
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]
ENTRYPOINT ["entrypoint.sh"]
