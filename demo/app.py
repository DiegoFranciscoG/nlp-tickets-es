"""Demo Streamlit de nlp-tickets-es.

Las API keys se leen en el servidor (st.secrets o variables de entorno) y nunca llegan al navegador.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import requests
import streamlit as st
from examples import EXAMPLES

TIMEOUT_SECONDS = 60  # Render free despierta en ~1 minuto
CHANNELS = {
    "chat": "Chat",
    "web": "Formulario web",
    "email": "Correo",
    "phone": "Teléfono",
    "social": "Redes sociales",
    "branch": "Agencia",
}
URGENCY = {"high": "🔴 Alta", "medium": "🟠 Media", "low": "🟢 Baja"}
SENTIMENT = {"negative": "Negativo", "neutral": "Neutral", "positive": "Positivo"}
REASONS = {
    "low_confidence": "Confianza del enrutamiento por debajo del umbral",
    "high_urgency_low_margin": "Posible urgencia alta no predicha por el modelo",
    "escalated_by_rule": "Una regla de negocio subió la urgencia",
}


def setting(name: str, default: str = "") -> str:
    with contextlib.suppress(FileNotFoundError):  # sin secrets.toml en local
        if name in st.secrets:
            return str(st.secrets[name])
    return os.environ.get(name, default)


API = setting("API_BASE_URL", "http://localhost:8000").rstrip("/")
CLIENT_KEY = setting("DEMO_API_KEY")
REVIEWER_KEY = setting("DEMO_REVIEWER_KEY")


def call(method: str, path: str, key: str, payload: dict[str, Any] | None = None) -> Any:
    response = requests.request(
        method, f"{API}{path}", json=payload, headers={"X-API-Key": key}, timeout=TIMEOUT_SECONDS
    )
    if response.status_code >= 400:
        detail = response.json().get("message", response.text) if response.content else ""
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


@st.cache_data(ttl=600, show_spinner=False)
def catalog() -> dict[str, Any]:
    return call("GET", "/api/v1/catalog", CLIENT_KEY)


@st.cache_data(ttl=600, show_spinner=False)
def model_info() -> dict[str, Any]:
    return call("GET", "/api/v1/model", CLIENT_KEY)


def show_result(result: dict[str, Any]) -> None:
    if result["decision"] == "auto_route":
        st.success(f"✅ Enrutado automáticamente a **{result['team']['name']}**")
    else:
        reasons = "; ".join(REASONS.get(r, r) for r in result["review_reasons"])
        st.warning(f"👤 Requiere revisión humana: {reasons}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Categoría",
        result["category"]["name"],
        f"{result['category']['score']:.0%}",
        delta_color="off",
    )
    col2.metric(
        "Equipo", result["team"]["name"], f"{result['team']['score']:.0%}", delta_color="off"
    )
    urgency = result["urgency"]
    col3.metric(
        "Urgencia",
        URGENCY[urgency["code"]],
        "escalada por regla" if urgency["escalated_by_rule"] else None,
        delta_color="off",
    )
    col4.metric(
        "Sentimiento",
        SENTIMENT[result["sentiment"]["code"]],
        f"{result['sentiment']['score']:.0%}",
        delta_color="off",
    )

    st.caption(
        f"Intención: **{result['intent']['name']}** ({result['intent']['score']:.0%}) · "
        f"umbral de confianza {result['confidence_threshold']:.3f} · modelo "
        f"{result['model']['version']} · {result['latency_ms']} ms"
    )
    st.markdown("**Texto almacenado (anonimizado)**")
    st.code(result["anonymized_text"], language=None, wrap_lines=True)
    entities = result["explanation"]["anonymized_entities"]
    if entities:
        st.caption(
            "Datos personales enmascarados: " + ", ".join(f"{k}: {v}" for k, v in entities.items())
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**¿Por qué?** Palabras que más sostienen la intención")
        terms = result["explanation"]["top_terms"]
        if terms:
            st.dataframe(
                [{"palabra": t["term"], "peso": round(t["weight"], 3)} for t in terms],
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("Sin términos destacados.")
        if result["explanation"]["rules_triggered"]:
            st.caption("Reglas activadas: " + ", ".join(result["explanation"]["rules_triggered"]))
    with right:
        st.markdown("**Solución sugerida**")
        st.info(result["suggested_resolution"])

    similar = result.get("similar_tickets") or []
    if similar:
        st.markdown("**Tickets resueltos parecidos** (búsqueda semántica con pgvector)")
        st.dataframe(
            [
                {
                    "similitud": f"{s['similarity']:.0%}",
                    "ticket": s["text_anonymized"],
                    "intención": s["intent"],
                    "solución aplicada": s["resolution_text"],
                }
                for s in similar
            ],
            hide_index=True,
            width="stretch",
        )


def classify_tab() -> None:
    choice = st.selectbox("Ejemplo ficticio", ["(escribir mi propio ticket)", *EXAMPLES])
    example = EXAMPLES.get(choice, {"text": "", "channel": "chat"})
    with st.form("classify"):
        text = st.text_area("Texto del ticket", value=example["text"], height=120, max_chars=5000)
        channel = st.selectbox(
            "Canal",
            list(CHANNELS),
            index=list(CHANNELS).index(example["channel"]),
            format_func=CHANNELS.get,
        )
        submitted = st.form_submit_button("Clasificar", type="primary")
    if submitted:
        if len(text.strip()) < 3:
            st.error("Escribe al menos 3 caracteres.")
            return
        with st.spinner("Clasificando… (si la API estaba dormida tarda ~1 minuto)"):
            try:
                st.session_state["result"] = call(
                    "POST",
                    "/api/v1/classify",
                    CLIENT_KEY,
                    {"text": text, "channel": channel, "explain": True, "include_similar": True},
                )
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"No se pudo clasificar: {exc}")
                return
    if "result" in st.session_state:
        show_result(st.session_state["result"])


def review_tab() -> None:
    result = st.session_state.get("result")
    if not REVIEWER_KEY:
        st.info("La revisión humana está deshabilitada en esta demo (falta DEMO_REVIEWER_KEY).")
        return
    if not result or not result.get("prediction_id"):
        st.info("Primero clasifica un ticket en la pestaña «Clasificar».")
        return
    data = catalog()
    intents = {i["code"]: i["name"] for i in data["intents"]}
    teams = {t["code"]: t["name"] for t in data["teams"]}
    st.markdown(f"Corrigiendo la predicción del ticket: _{result['anonymized_text']}_")
    with st.form("feedback"):
        intent = st.selectbox(
            "Intención correcta",
            list(intents),
            index=list(intents).index(result["intent"]["code"]),
            format_func=intents.get,
        )
        team = st.selectbox(
            "Equipo correcto",
            list(teams),
            index=list(teams).index(result["team"]["code"]),
            format_func=teams.get,
        )
        urgency = st.radio(
            "Urgencia correcta",
            list(URGENCY),
            index=list(URGENCY).index(result["urgency"]["code"]),
            format_func=URGENCY.get,
            horizontal=True,
        )
        comment = st.text_input("Comentario (se anonimiza)", max_chars=1000)
        sent = st.form_submit_button("Guardar corrección")
    if sent:
        payload = {
            "prediction_id": result["prediction_id"],
            "corrected_intent": intent,
            "corrected_team": team,
            "corrected_urgency": urgency,
            "comment": comment or None,
        }
        try:
            saved = call("POST", "/api/v1/feedback", REVIEWER_KEY, payload)
            st.success(
                f"Corrección guardada ({saved['id'][:8]}…). Se usará en el próximo reentrenamiento."
            )
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"No se pudo guardar: {exc}")


def metrics_tab() -> None:
    try:
        info = model_info()
    except (requests.RequestException, RuntimeError) as exc:
        st.error(f"No se pudo consultar el modelo: {exc}")
        return
    st.markdown(
        f"Modelo **{info['name']}** versión `{info['version']}` · umbral "
        f"{info['confidence_threshold']:.3f} · entrenado {info['trained_at'][:10]}"
    )
    names = {
        "intent": "Intención (77)",
        "category": "Categoría (9)",
        "team": "Equipo (6)",
        "urgency": "Urgencia (3)",
        "sentiment": "Sentimiento (3)",
        "ood": "Fuera de dominio",
    }
    rows = [
        {
            "tarea": names.get(m["task"], m["task"]),
            "partición": m["split"],
            "macro-F1": m["macro_f1"],
            "exactitud": m["accuracy"],
            "ECE": m["ece"],
            "cobertura automática": m["coverage"],
        }
        for m in info["metrics"]
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Métricas del artefacto ONNX INT8 desplegado en la partición oficial de prueba. La "
        "comparación completa con la línea base está en el README."
    )


def main() -> None:
    st.set_page_config(page_title="Tickets en español", page_icon="🎫", layout="wide")
    st.title("Clasificación y enrutamiento de tickets en español")
    st.caption(
        "Banco ficticio · los datos que escribas se anonimizan antes de guardarse. No ingreses "
        "datos personales reales."
    )
    if not CLIENT_KEY:
        st.error("Configura DEMO_API_KEY (st.secrets o variable de entorno).")
        return
    tab_classify, tab_review, tab_metrics = st.tabs(["Clasificar", "Revisión humana", "Métricas"])
    with tab_classify:
        classify_tab()
    with tab_review:
        review_tab()
    with tab_metrics:
        metrics_tab()


main()
