# Despliegue 100 % gratuito

| Pieza | Servicio | Plan y límites verificados (2026-09-25) | Fuente |
|---|---|---|---|
| Base de datos + pgvector | **Neon** | Free: 0,5 GB y 100 CU-h por proyecto; pgvector incluido | neon.com/docs/extensions/pgvector |
| API FastAPI (ONNX INT8) | **Render** (Docker) | Free: 0,1 CPU, 512 MB, se duerme tras 15 min (~1 min en despertar), 750 h/mes | render.com/docs/compute-plans · render.com/docs/free |
| Demo | **Streamlit Community Cloud** | Hasta 2,7 GB de RAM; se duerme tras 12 h sin tráfico | docs.streamlit.io |
| Pesos del modelo | **GitHub Releases** | Asset `.zip` (~95 MB) con hash SHA-256 en `models/manifest.json` | docs.github.com |
| Entrenamiento | CPU local (o Colab/Kaggle) | ~2 h en CPU de 4 núcleos físicos | este repo |

Ninguno de estos pasos pide tarjeta de crédito. Render Postgres **no** se usa: expira a los 30 días.

## 1. Publicar el modelo (después de «VOY A SUBIR»)
1. `nlp-tickets train` genera `artifacts/nlp-tickets-model-<versión>.zip` y `models/manifest.json`.
2. Crea un Release en GitHub (p. ej. `model-<versión>`) y sube el `.zip` como asset.
3. Copia la URL del asset (`MODEL_URL`) y `bundle_sha256` del manifiesto (`MODEL_SHA256`).

## 2. Base de datos en Neon
1. Crea un proyecto en Neon (región cercana a Render, p. ej. `us-west-2`).
2. Copia la *connection string* (`postgresql://...?sslmode=require`) → `DATABASE_URL`.
3. Desde tu PC, con esa URL en `.env`, carga el esquema, registra el modelo y los datos de ejemplo:
   ```bash
   uv run nlp-tickets migrate
   uv run nlp-tickets register-model
   uv run nlp-tickets seed
   ```
   El seed se hace desde local porque calcular 770 embeddings con 0,1 CPU tarda varios minutos.

## 3. API en Render
1. *New → Blueprint* y elige este repositorio (lee `render.yaml`).
2. En el panel, define las variables marcadas `sync: false`: `DATABASE_URL`, `API_KEYS_CLIENT`, `API_KEYS_REVIEWER` (claves de ≥ 32 caracteres), `CORS_ALLOWED_ORIGINS` (URL de la demo), `MODEL_URL` y `MODEL_SHA256`.
3. Al arrancar, la API descarga el modelo, **verifica su SHA-256**, aplica migraciones y se registra como versión activa. `GET /health` debe responder `{"status": "ok"}`.

## 4. Demo en Streamlit Community Cloud
1. *New app* → repositorio, rama `main`, archivo `demo/app.py`.
2. En *Settings → Secrets*:
   ```toml
   API_BASE_URL = "https://<tu-servicio>.onrender.com"
   DEMO_API_KEY = "<una de API_KEYS_CLIENT>"
   # DEMO_REVIEWER_KEY = "<opcional: habilita la pestaña de revisión>"
   ```
3. Las claves quedan en el servidor de Streamlit; el navegador nunca las ve.

## 5. Reentrenamiento con la retroalimentación
```bash
uv run nlp-tickets export-feedback --out data/processed/feedback.parquet
uv run nlp-tickets train --feedback data/processed/feedback.parquet
uv run nlp-tickets register-model --feedback data/processed/feedback.parquet
```
Publica el nuevo `.zip` como otro Release y actualiza `MODEL_URL`/`MODEL_SHA256` en Render.
Las correcciones quedan marcadas con la versión que las usó (`feedback.used_in_model_version_id`).
