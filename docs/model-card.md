# Ficha del modelo: `minilm-multitask`

Estructura según Mitchell et al. (2019), *Model Cards for Model Reporting*, y la guía de fichas de Hugging Face.

## Detalles
- **Tipo:** encoder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache 2.0, 118 M parámetros, destilado) con fine-tuning **multitarea**: una cabeza de intención (77 clases) y otra de sentimiento (3 clases), más un embedding normalizado de 384 dimensiones para la búsqueda semántica.
- **Entrenamiento:** 5 épocas en CPU, AdamW (encoder 5e-5, cabezas 1e-3), lotes alternados de intención (32) y sentimiento (16), vocabulario congelado y semilla 42. Se elige la época con mejor macro-F1 de intención en **validación**.
- **Despliegue:** ONNX con cuantización dinámica INT8, servido con ONNX Runtime sin PyTorch y verificado por SHA-256 al cargarse.
- **Calibración:** *temperature scaling* por tarea, ajustado en validación.
- **Versión y trazabilidad:** `model_meta.json` (revisiones de datos, hash de la partición, temperaturas y umbral), `models/manifest.json` y la corrida de MLflow.

## Uso previsto
- Triaje de tickets de **soporte bancario en español** (dominio BANKING77): sugerir categoría, equipo y urgencia, y decidir si el ticket se enruta solo o pasa a **revisión humana**.
- **Fuera de alcance:** decisiones con efectos jurídicos sobre el cliente (crédito, bloqueo de cuentas, sanciones) sin intervención humana (LOPDP, art. 20), dominios distintos del bancario sin reentrenar, y textos largos de más de 64 tokens (se truncan).

## Datos
| Uso | Conjunto | Tamaño | Licencia |
|---|---|---|---|
| Intención | BANKING77-ES-LA (traducción revisada) con etiquetas del BANKING77 original | 8 577 entrenamiento · 953 validación · 3 080 prueba (oficial) | CC BY 4.0 |
| Sentimiento | GoEmotions traducido al español (EasyNMT), agrupación oficial positivo/negativo/ambiguo | 12 000 · 5 303 · 5 349 | Apache 2.0 |
| Fuera de dominio | MASSIVE 1.1 `es-ES` (partición de prueba) | 2 974 | CC BY 4.0 |

Antes de entrenar se eliminaron 289 duplicados y **184 frases de entrenamiento que también estaban en prueba**. Todo el texto pasa por el mismo anonimizador que usa la API.

## Métricas
Todas se calculan sobre el **artefacto ONNX INT8 desplegado**, en la partición oficial de prueba, y se comparan con la línea base TF-IDF + regresión logística entrenada con la misma partición y semilla. Las cifras están en la sección **Resultados** del [README](../README.md) y en `reports/metrics.json`:
- macro-F1 y exactitud por tarea (intención, categoría, equipo, urgencia, sentimiento);
- ECE antes y después de calibrar, y puntuación de Brier;
- cobertura y exactitud del enrutamiento automático con el umbral elegido en validación;
- porcentaje de frases fuera de dominio que se envían a revisión humana;
- diferencia con la línea base e intervalo de confianza del 95 % (bootstrap pareado).

## Limitaciones conocidas
- **Etiquetas ruidosas:** BANKING77 tiene errores y solapamientos de etiquetas documentados (Ying & Thomas, 2022), por ejemplo entre `pending_top_up` y `top_up_failed`.
- **Artefactos de traducción:** la versión en español conserva errores del original y añade algunos propios ("top-up" traducido como "blusa"). El modelo puede depender de esas formas.
- **Variante regional:** español latinoamericano genérico, sin jerga ecuatoriana específica.
- **Sentimiento fuera de dominio:** GoEmotions son comentarios de Reddit traducidos automáticamente y sin revisión; su rendimiento en tickets bancarios **no está medido**.
- **Urgencia derivada de la intención:** la urgencia base es una política del catálogo (SUPUESTO 1), no una etiqueta humana por ticket. Las reglas de escalamiento cubren solo señales explícitas.
- **Anonimización por reglas:** detecta identificadores con formato (cédula, RUC, teléfono, tarjeta, correo), pero no todos los nombres propios ni direcciones.
- **Longitud:** con 64 tokens se trunca el final de los tickets largos.

## Sesgos y riesgos
- **Sesgos del preentrenamiento:** el modelo base hereda los sesgos de sus corpus web multilingües; no se ha hecho una auditoría de equidad por grupos demográficos (los datos no contienen esos atributos).
- **Sentimiento:** los anotadores de GoEmotions eran angloparlantes de India. Expresiones culturales de queja o cortesía en Ecuador pueden clasificarse mal, así que el sentimiento **no debe usarse para priorizar** clientes.
- **Automatización:** confiar ciegamente en `auto_route` concentraría los errores en las clases raras. Por eso la política envía a revisión humana los casos de baja confianza, los de posible urgencia alta y los escalados por regla.
- **Envenenamiento del bucle de retroalimentación** (OWASP ML02): solo se usan correcciones aceptadas, del rol revisor, una por ticket, con tope por reentrenamiento y trazables a la versión que las consumió.

## Recomendaciones
- Revisar periódicamente las correcciones humanas por intención: si los revisores corrigen el equipo de una intención de forma sistemática, cambia el catálogo, no el modelo.
- Etiquetar una muestra del dominio real (≥ 300 tickets) antes de usar el sentimiento en producción.
- Recalibrar el umbral si cambia la mezcla de intenciones o el coste de enviar un caso a revisión humana.
