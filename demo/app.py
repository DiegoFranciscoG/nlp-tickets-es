"""Demo Streamlit de nlp-tickets-es.

Las API keys se leen en el servidor (st.secrets o variables de entorno) y nunca llegan al navegador.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import requests
import streamlit as st
from auth import Account, LoginGuard, Role, authenticate, build_accounts
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
ACCOUNTS = build_accounts(
    {
        name: setting(name)
        for name in ("DEMO_USER", "DEMO_PASSWORD_HASH", "REVIEWER_USER", "REVIEWER_PASSWORD_HASH")
    }
)


@st.cache_resource
def login_guards() -> dict[str, LoginGuard]:
    """Un contador de intentos por usuario, compartido entre sesiones del proceso."""
    return {}


def login_view() -> None:
    st.subheader("Iniciar sesión")
    with st.form("login"):
        username = st.text_input("Usuario", max_chars=64)
        password = st.text_input("Contraseña", type="password", max_chars=128)
        submitted = st.form_submit_button("Entrar", type="primary")
    if not submitted:
        return
    guard = login_guards().setdefault(username.strip().lower(), LoginGuard())
    if wait := guard.locked_for():
        st.error(f"Demasiados intentos fallidos. Espera {wait} s.")
        return
    account = authenticate(ACCOUNTS, username, password)
    if account is None:
        guard.register_failure()
        st.error("Usuario o contraseña incorrectos.")
        return
    guard.reset()
    st.session_state["account"] = account
    st.rerun()


def current_account() -> Account | None:
    account = st.session_state.get("account")
    return account if isinstance(account, Account) else None


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

    urgency = result["urgency"]
    cards = (
        ("Categoría", result["category"]["name"], f"{result['category']['score']:.0%}"),
        ("Equipo destino", result["team"]["name"], f"{result['team']['score']:.0%}"),
        (
            "Urgencia",
            URGENCY[urgency["code"]],
            "escalada por regla" if urgency["escalated_by_rule"] else f"{urgency['score']:.0%}",
        ),
        (
            "Sentimiento",
            SENTIMENT[result["sentiment"]["code"]],
            f"{result['sentiment']['score']:.0%}",
        ),
    )
    for column, (label, value, detail) in zip(st.columns(4), cards, strict=True):
        with column:
            st.caption(label)
            st.markdown(f"#### {value}")
            st.caption(detail)

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
    account = current_account()
    if account is None or account.role is not Role.REVIEWER:
        st.info("Solo las cuentas con rol revisor pueden corregir predicciones.")
        return
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
    order = {
        "intent": "Intención (77 clases)",
        "category": "Categoría (9)",
        "team": "Equipo destino (6)",
        "urgency": "Urgencia (3)",
        "sentiment": "Sentimiento (3)",
    }
    by_task = {m["task"]: m for m in info["metrics"]}
    rows = [
        {
            "tarea": label,
            "macro-F1": f"{by_task[task]['macro_f1']:.3f}",
            "exactitud": f"{by_task[task]['accuracy']:.1%}",
            "ECE (calibración)": f"{by_task[task]['ece']:.3f}",
        }
        for task, label in order.items()
        if task in by_task
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    team, ood = by_task.get("team"), by_task.get("ood")
    if team and team.get("coverage") is not None:
        st.markdown(
            f"**Enrutamiento automático:** {team['coverage']:.1%} de los tickets de prueba, con "
            f"{team['selective_accuracy']:.1%} de exactitud; el resto va a revisión humana."
        )
    if ood and ood.get("coverage") is not None:
        st.markdown(
            f"**Fuera de dominio (MASSIVE):** {1 - ood['coverage']:.1%} de las frases se envía a "
            "revisión humana."
        )
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
    if not CLIENT_KEY or not ACCOUNTS:
        st.error("Configura DEMO_API_KEY y las cuentas de la demo (st.secrets o entorno).")
        return
    account = current_account()
    if account is None:
        login_view()
        return
    with st.sidebar:
        role = "revisor" if account.role is Role.REVIEWER else "cliente"
        st.markdown(f"Sesión: **{account.username}** ({role})")
        if st.button("Cerrar sesión"):
            st.session_state.clear()
            st.rerun()
    tab_classify, tab_review, tab_metrics = st.tabs(["Clasificar", "Revisión humana", "Métricas"])
    with tab_classify:
        classify_tab()
    with tab_review:
        review_tab()
    with tab_metrics:
        metrics_tab()


main()
