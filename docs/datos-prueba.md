# Datos de prueba

Todo es **ficticio**: correos en `example.com` (RFC 2606), números sin titular real y nombres genéricos. No escribas datos personales reales en la demo.

## Cuentas de la demo (Streamlit)
La demo (`http://localhost:8501`, o el `DEMO_PORT` de tu `.env`) pide usuario y contraseña. Las contraseñas **no** están en el repositorio: en `.env` solo se guarda su hash Argon2id.

| Usuario | Rol | Qué puede hacer |
|---|---|---|
| `demo` | cliente | Clasificar tickets, ver la explicación, los tickets similares y las métricas |
| `revisor` | revisor | Todo lo anterior y corregir predicciones (pestaña «Revisión humana») |

Para crear o cambiar una contraseña:
```bash
uv run --with argon2-cffi python demo/hash_password.py   # pega el hash en .env entre comillas simples
```
Tras 5 intentos fallidos la cuenta se bloquea 60 segundos.

## API keys (servicio a servicio)
La API no usa usuario y contraseña: el helpdesk P34 y la demo se autentican con la cabecera `X-API-Key`.
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # genera una clave para .env
```
En Swagger (`http://localhost:8000/docs`, o el `API_PORT` de tu `.env`) pulsa **Authorize** y pega una clave de `API_KEYS_CLIENT` (o de `API_KEYS_REVIEWER` para `/feedback`).

## Tickets de ejemplo
Los mismos textos aparecen en el selector de la demo ([demo/examples.py](../demo/examples.py)) y en [api.http](api.http).

| # | Ticket | Qué se espera ver |
|---|---|---|
| 1 | Buenas tardes, me robaron la tarjeta en el centro de Quito y veo dos compras que no reconozco. Mi nombre es Andrea Paredes, celular 099 000 0001. | Fraude y Seguridad, urgencia alta; `<NOMBRE>` y `<TELEFONO>` enmascarados |
| 2 | Hice una transferencia a la cuenta 2200112233 hace dos días y el destinatario dice que todavía no le llega el dinero. | Pagos y Transferencias; `<CUENTA>` enmascarada |
| 3 | El cajero se tragó mi tarjeta en la agencia del norte, necesito ayuda urgente. | Tarjeta retenida, urgencia alta |
| 4 | Me cobraron dos veces el mismo consumo en el supermercado. Pueden escribirme a cliente.prueba@example.com | Cobro duplicado; `<CORREO>` enmascarado |
| 5 | ¿Cuánto tiempo tarda en llegar la tarjeta física a Cuenca? | Operaciones de Tarjetas, urgencia baja |
| 6 | Quiero saber cuál es el tipo de cambio para pagar en euros con mi tarjeta. | Atención al Cliente, urgencia baja |
| 7 | Intento verificar mi identidad en la app pero siempre rechaza la foto del documento. | Cumplimiento (KYC) |
| 8 | Quiero cerrar mi cuenta, soy Carlos Vera con C.I. 1799999990. | Cierre de cuenta; `<NOMBRE>` y `<CEDULA>` enmascarados |
| 9 | Gracias, el reembolso ya apareció en mi estado de cuenta. ¡Excelente servicio! | Sentimiento positivo |
| 10 | ¿Me recomiendan una receta de encebollado para el domingo? | Fuera de dominio: debería ir a revisión humana (ver la limitación abajo) |

## Resultados reales con el modelo `2026.09.25-7a9c6964`
Obtenidos con `docker compose up` y `POST /api/v1/classify` (`persist: false`). Los porcentajes son probabilidades calibradas.

| Ejemplo | Intención (confianza) | Equipo (confianza) | Urgencia | Sentimiento | Decisión | Motivos | Texto guardado (inicio) |
|---|---|---|---|---|---|---|---|
| Robo de tarjeta con consumos no reconocidos | Tarjeta comprometida (46%) | Fraude y Seguridad (96%) | Alta | neutral | enrutado automático | — | `Buenas tardes, me robaron la tarjeta en el centro de Quito y veo dos c` |
| Transferencia que no llega | Transferencia no recibida por el destinatario (98%) | Pagos y Transferencias (100%) | Media | neutral | enrutado automático | — | `Hice una transferencia a la cuenta <CUENTA> hace dos días y el destina` |
| Cajero retuvo la tarjeta | Llegada de la tarjeta (38%) | Operaciones de Tarjetas (62%) | Alta (regla) | neutral | revisión humana | low_confidence, high_urgency_low_margin, escalated_by_rule | `El cajero se tragó mi tarjeta en la agencia del norte, necesito ayuda ` |
| Cobro duplicado (con correo) | Transacción cobrada dos veces (99%) | Pagos y Transferencias (100%) | Alta | neutral | enrutado automático | — | `Me cobraron dos veces el mismo consumo en el supermercado. Pueden escr` |
| Tiempo de entrega de tarjeta | Tiempo estimado de entrega (80%) | Operaciones de Tarjetas (97%) | Baja | neutral | enrutado automático | — | `¿Cuánto tiempo tarda en llegar la tarjeta física a Cuenca?` |
| Tipo de cambio | Tipo de cambio (49%) | Atención al Cliente (97%) | Baja | positivo | enrutado automático | — | `Quiero saber cuál es el tipo de cambio para pagar en euros con mi tarj` |
| Verificación de identidad fallida | No puedo verificar mi identidad (87%) | Cumplimiento (KYC) (97%) | Media | negativo | enrutado automático | — | `Intento verificar mi identidad en la app pero siempre rechaza la foto ` |
| Cierre de cuenta (con cédula ficticia) | Cerrar cuenta (95%) | Atención al Cliente (96%) | Baja | neutral | enrutado automático | — | `Quiero cerrar mi cuenta, soy <NOMBRE> con C.I. <CEDULA>.` |
| Agradecimiento (sentimiento positivo) | Solicitar reembolso (51%) | Pagos y Transferencias (96%) | Baja | positivo | enrutado automático | — | `Gracias, el reembolso ya apareció en mi estado de cuenta. ¡Excelente s` |
| Fuera de dominio (limitación conocida) | Tiempo estimado de entrega (71%) | Operaciones de Tarjetas (88%) | Baja | positivo | enrutado automático | — | `¿Me recomiendan una receta de encebollado para el domingo?` |

**Lectura honesta de los resultados:**
- De los 9 tickets bancarios, 8 se enrutan solos al equipo correcto con una confianza ≥ 96 %, y los datos personales quedan enmascarados (`<CUENTA>`, `<NOMBRE>`, `<CEDULA>`; en el ejemplo 1 también `<TELEFONO>`).
- *Cajero retuvo la tarjeta*: el modelo confunde la intención (38 %), pero la **política de decisión lo manda a revisión humana** por baja confianza, posible urgencia alta y la regla de escalamiento. Es el comportamiento buscado.
- *Fuera de dominio*: la pregunta sobre una receta se enruta sola a Tarjetas con un 88 % de confianza. El umbral detiene el 86,4 % de las frases fuera de dominio de MASSIVE, **no todas**. Es una limitación conocida; el roadmap incluye un detector por similitud con los tickets conocidos (pgvector).
- *Sentimiento*: el robo sale "neutral" y el tipo de cambio "positivo". El sentimiento se entrenó con comentarios de Reddit traducidos y **no debe usarse para priorizar** (ver la [ficha del modelo](model-card.md)).

## Pruebas de seguridad rápidas
```bash
curl -i http://localhost:8000/api/v1/catalog                                  # 401 sin API key
curl -i -X POST http://localhost:8000/api/v1/feedback -H "X-API-Key: <clave client>" \
     -H "Content-Type: application/json" -d '{"prediction_id":"00000000-0000-0000-0000-000000000000","corrected_urgency":"low"}'   # 403
curl -i -X POST http://localhost:8000/api/v1/classify -H "X-API-Key: <clave client>" \
     -H "Content-Type: application/json" -d '{"text":"hola","channel":"fax"}'      # 422 sin eco del texto
```
