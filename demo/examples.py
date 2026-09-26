"""Tickets de ejemplo 100 % ficticios (correos en example.com, números sin titular real)."""

EXAMPLES: dict[str, dict[str, str]] = {
    "Robo de tarjeta con consumos no reconocidos": {
        "channel": "chat",
        "text": (
            "Buenas tardes, me robaron la tarjeta en el centro de Quito y veo dos compras que no "
            "reconozco. Mi nombre es Andrea Paredes, celular 099 000 0001."
        ),
    },
    "Transferencia que no llega": {
        "channel": "email",
        "text": (
            "Hice una transferencia a la cuenta 2200112233 hace dos días y el destinatario dice "
            "que todavía no le llega el dinero."
        ),
    },
    "Cajero retuvo la tarjeta": {
        "channel": "phone",
        "text": "El cajero se tragó mi tarjeta en la agencia del norte, necesito ayuda urgente.",
    },
    "Cobro duplicado (con correo)": {
        "channel": "web",
        "text": (
            "Me cobraron dos veces el mismo consumo en el supermercado. Pueden escribirme a "
            "cliente.prueba@example.com"
        ),
    },
    "Tiempo de entrega de tarjeta": {
        "channel": "web",
        "text": "¿Cuánto tiempo tarda en llegar la tarjeta física a Cuenca?",
    },
    "Tipo de cambio": {
        "channel": "chat",
        "text": "Quiero saber cuál es el tipo de cambio para pagar en euros con mi tarjeta.",
    },
    "Verificación de identidad fallida": {
        "channel": "social",
        "text": (
            "Intento verificar mi identidad en la app pero siempre rechaza la foto del documento."
        ),
    },
    "Cierre de cuenta (con cédula ficticia)": {
        "channel": "branch",
        "text": "Quiero cerrar mi cuenta, soy Carlos Vera con C.I. 1799999990.",
    },
    "Agradecimiento (sentimiento positivo)": {
        "channel": "email",
        "text": "Gracias, el reembolso ya apareció en mi estado de cuenta. ¡Excelente servicio!",
    },
    "Fuera de dominio (limitación conocida)": {
        "channel": "chat",
        "text": "¿Me recomiendan una receta de encebollado para el domingo?",
    },
}
