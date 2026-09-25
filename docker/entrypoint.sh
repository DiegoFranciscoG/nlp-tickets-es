#!/bin/sh
# Arranque de la API: migraciones -> datos de ejemplo (opcional) -> uvicorn con un solo worker.
set -eu

nlp-tickets migrate

if [ "${SEED_ON_START:-true}" = "true" ]; then
  nlp-tickets seed || echo "seed omitido (sin modelo o sin base disponible)"
fi

exec uvicorn nlp_tickets.main:create_app --factory \
  --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 \
  --proxy-headers --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-127.0.0.1}" \
  --no-server-header
