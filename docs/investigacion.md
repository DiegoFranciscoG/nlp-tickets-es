# Investigación y fuentes

Toda regla de negocio, dato o decisión técnica del proyecto sale de una fuente de esta tabla.
Fecha de consulta de todas las fuentes: **2026-09-25**.

## 1. Datos (datasets) y licencias

| # | Fuente (oficial / confiable) | URL | Consultada | Qué se tomó de aquí |
|---|---|---|---|---|
| D1 | BANKING77-ES-LA (Centro de Ciencia de Datos POLI/USP), ficha oficial en Hugging Face | https://huggingface.co/datasets/c2d-usp/banking77-es-la | 2026-09-25 | **Corpus principal.** 13 083 consultas de soporte bancario en español latinoamericano (10 003 de entrenamiento y 3 080 de prueba), 77 intenciones. Licencia **CC BY 4.0**. Traducción automática revisada por seis hablantes nativos (columna `cambiado`). Revisión fijada: `11f2c4d028e7df4d5e8e6ad36977cf9bf10cb666`. |
| D2 | BANKING77 original (PolyAI), ficha y repositorio oficial | https://huggingface.co/datasets/PolyAI/banking77 · https://github.com/PolyAI-LDN/task-specific-datasets | 2026-09-25 | Licencia de origen **CC BY 4.0** (verificada en GitHub). Partición oficial entrenamiento/prueba y **etiquetas enteras originales**: se toman de `banking_data/{train,test}.csv` en el commit `57ec275d8078af65b7731c2a98be812d844a6d6b`, porque el corpus en español es paralelo fila a fila. Los nombres en inglés de las 77 intenciones se usan como códigos. |
| D3 | Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI @ ACL | https://aclanthology.org/2020.nlp4convai-1.5/ | 2026-09-25 | Paper revisado por pares que publica BANKING77; cita obligatoria. |
| D4 | Ying & Thomas (2022), *Label Errors in BANKING77*, Insights @ ACL | https://aclanthology.org/2022.insights-1.19/ | 2026-09-25 | BANKING77 contiene etiquetas erróneas o solapadas; se declara en la ficha del modelo como limitación y justifica el umbral de revisión humana. |
| D5 | GoEmotions (Google Research), ficha oficial | https://huggingface.co/datasets/google-research-datasets/go_emotions | 2026-09-25 | Licencia de origen **Apache 2.0**; 27 emociones más "neutral". |
| D6 | Demszky et al. (2020), *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL | https://aclanthology.org/2020.acl-main.372/ | 2026-09-25 | Paper revisado por pares. Sesgos declarados: comentarios de Reddit y anotadores angloparlantes. |
| D7 | Agrupación oficial emoción → sentimiento (`sentiment_dict.json`) | https://github.com/google-research/google-research/blob/master/goemotions/data/sentiment_dict.json | 2026-09-25 | **Regla de etiquetado de sentimiento:** 12 emociones positivas, 11 negativas y 4 ambiguas. Se mapea a `positive` / `negative` / `neutral` (sección 5). |
| D8 | GoEmotions traducido al español (`go_emotions-es-mt`) | https://huggingface.co/datasets/mrm8488/go_emotions-es-mt | 2026-09-25 | **Corpus de sentimiento.** Licencia Apache 2.0, traducción automática con EasyNMT **sin revisión humana**. Revisión fijada: `f881ecdb455e1ef7b7e70164df594a98ddf3424e`. |
| D9 | MASSIVE 1.1 (Amazon Science), ficha oficial | https://huggingface.co/datasets/AmazonScience/massive | 2026-09-25 | Licencia **CC BY 4.0**, locale `es-ES`: 11 514 / 2 033 / 2 974 frases de asistente de voz (60 intenciones, fuera del dominio bancario). **Uso:** conjunto *fuera de dominio* para medir si el umbral envía a revisión humana lo que el modelo no conoce. |
| D10 | FitzGerald et al. (2023), *MASSIVE*, ACL | https://aclanthology.org/2023.acl-long.235/ | 2026-09-25 | Paper revisado por pares; cita obligatoria de MASSIVE. |
| D11 | MultiEURLEX (Chalkidis et al., 2021, EMNLP) | https://huggingface.co/datasets/coastalcph/multi_eurlex · https://aclanthology.org/2021.emnlp-main.559/ | 2026-09-25 | Licencia CC BY-SA 4.0; leyes de la UE con etiquetas EUROVOC. **Descartado:** dominio jurídico ajeno a tickets y la cláusula *ShareAlike* se heredaría al modelo. |
| D12 | TASS (SEPLN), acuerdo de licencia | http://tass.sepln.org/tass_data/download.php | 2026-09-25 | Licencia "Research/Non Commercial": uso interno y solo investigación, sin distribuir derivados. **Descartado:** un modelo público en una demo sería un derivado distribuido. |
| D13 | Customer Support Tickets (Tobi-Bueck / Softoft) | https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets | 2026-09-25 | CC BY-NC 4.0, sintético. La copia de Hugging Face solo tiene inglés y alemán (61 765 tickets). **Descartado** para el MVP: sin español en esa copia y con licencia no comercial. Queda en el roadmap. |
| D14 | Classification of IT Support Tickets (Benítez Pereira, IFSC), Zenodo | https://doi.org/10.5281/zenodo.7648117 · https://huggingface.co/datasets/tasksource/it-support-tickets | 2026-09-25 | CC BY 4.0; 2 229 tickets reales anonimizados, mayoritariamente en inglés y portugués. **Referencia de buenas prácticas:** anonimizó con marcadores `[NAME]` más revisión manual. Se adopta el mismo estilo de marcadores. |

## 2. Modelos base y bibliotecas

| # | Fuente | URL | Consultada | Qué se tomó de aquí |
|---|---|---|---|---|
| M1 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (ficha) | https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 2026-09-25 | **Modelo base elegido.** Licencia Apache 2.0, 12 capas, dimensión 384, multilingüe (incluye español). Es un modelo **destilado**. |
| M2 | Reimers & Gurevych (2020), *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*, EMNLP | https://aclanthology.org/2020.emnlp-main.365/ | 2026-09-25 | Explica la destilación maestro→alumno de M1. Justifica que el modelo sirva a la vez para clasificar y para búsqueda semántica. |
| M3 | BETO (`dccuchile/bert-base-spanish-wwm-cased`), ficha | https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased | 2026-09-25 | Pretende CC BY 4.0, pero los autores advierten que no saben si todos los datos de preentrenamiento lo permiten, "especialmente para uso comercial". **Descartado** por la incertidumbre de licencia y por su tamaño (110 M parámetros, no cabe con holgura en 512 MB). |
| M4 | RoBERTa-BNE (`PlanTL-GOB-ES/roberta-base-bne`), ficha | https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne | 2026-09-25 | Apache 2.0, pero la ficha lo marca como **deprecated** y remite a BSC-LT. **Descartado** por obsoleto y por tamaño (125 M parámetros). |
| M5 | `microsoft/Multilingual-MiniLM-L12-H384` y `intfloat/multilingual-e5-small` | https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384 · https://huggingface.co/intfloat/multilingual-e5-small | 2026-09-25 | Licencia MIT; alternativas del mismo tamaño. Se prefiere M1 porque ya viene entrenado para similitud de oraciones. |
| M6 | Documentación de Hugging Face Transformers | https://huggingface.co/docs/transformers/index | 2026-09-25 | `AutoModel`/`AutoTokenizer` para el fine-tuning. Versión estable verificada en PyPI: 5.17.0. |
| M7 | Documentación de Sentence Transformers (eficiencia, ONNX) | https://sbert.net/docs/sentence_transformer/usage/efficiency.html | 2026-09-25 | Mean pooling, embeddings normalizados y exportación ONNX para inferencia sin PyTorch. Versión 6.1.0. |
| M8 | ONNX Runtime, cuantización | https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html | 2026-09-25 | Cuantización dinámica INT8 de pesos para bajar tamaño y latencia en CPU. |
| M9 | Tunstall et al. (2022), *Efficient Few-Shot Learning Without Prompts* (SetFit) | https://arxiv.org/abs/2209.11055 | 2026-09-25 | Muestra que un clasificador lineal sobre embeddings de oraciones es un punto de comparación fuerte. Se usa como línea base intermedia (embeddings congelados + regresión logística). |
| M10 | Liu et al. (2019), *Multi-Task Deep Neural Networks for NLU*, ACL | https://aclanthology.org/P19-1441/ | 2026-09-25 | Justifica el entrenamiento **multitarea** (encoder compartido; cabezas de intención y sentimiento con lotes alternados). |
| M11 | MLflow, *tracking* y *backend store* | https://mlflow.org/docs/latest/ml/tracking/ | 2026-09-25 | El backend por defecto es ahora SQLite (`sqlite:///mlflow.db`); el de archivos está en mantenimiento. Versión 3.16.1. |

## 3. Métricas y evaluación

| # | Fuente | URL | Consultada | Qué se tomó de aquí |
|---|---|---|---|---|
| E1 | scikit-learn, métricas de clasificación | https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics | 2026-09-25 | **Macro-F1** (`average="macro"`) como métrica principal: pesa igual a todas las clases, sea cual sea su frecuencia. Matriz de confusión. |
| E2 | scikit-learn, validación estratificada | https://scikit-learn.org/stable/modules/cross_validation.html#stratification | 2026-09-25 | La validación se separa del entrenamiento **estratificando por intención** y con semilla fija (42). La prueba oficial se usa una sola vez. |
| E3 | scikit-learn, calibración de probabilidades | https://scikit-learn.org/stable/modules/calibration.html | 2026-09-25 | Curvas de fiabilidad y puntuación de Brier. |
| E4 | Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML (PMLR 70) | https://proceedings.mlr.press/v70/guo17a.html | 2026-09-25 | **ECE** (15 bins) y **temperature scaling** ajustado en validación. Sin calibrar, el umbral de confianza no significa nada. |
| E5 | Geifman & El-Yaniv (2017), *Selective Classification for Deep Neural Networks*, NeurIPS | https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html | 2026-09-25 | **Umbral de revisión humana:** se elige en validación la confianza mínima que garantiza una precisión objetivo en lo que se enruta solo, y se reportan *cobertura* y *riesgo* en prueba. |
| E6 | Mitchell et al. (2019), *Model Cards for Model Reporting*, FAT* (ACM) | https://doi.org/10.1145/3287560.3287596 | 2026-09-25 | Estructura de la ficha del modelo: uso previsto, métricas por grupo, limitaciones y sesgos. |
| E7 | Hugging Face, guía de *model cards* | https://huggingface.co/docs/hub/model-cards | 2026-09-25 | Secciones y metadatos de `docs/model-card.md`. |

## 4. Datos personales, anonimización y seguridad

| # | Fuente | URL | Consultada | Qué se tomó de aquí |
|---|---|---|---|---|
| L1 | **LOPDP Ecuador**, Registro Oficial, Quinto Suplemento N.º 459 (26-may-2021), copia oficial de Cancillería | https://www.cancilleria.gob.ec/wp-content/uploads/2023/03/2021.05.10_ley_organica_de_proteccion_de_datos_personales.pdf | 2026-09-25 | **Art. 4:** "Anonimización: la aplicación de medidas dirigidas a impedir la identificación o reidentificación de una persona natural…" y definición de seudonimización. **Art. 2 c):** los datos anonimizados quedan fuera del ámbito de la ley mientras no sea posible identificar al titular. **Art. 10 e):** "pertinencia y minimización". **Art. 10:** conservación limitada. **Art. 37:** la anonimización, la seudonimización y el cifrado son medidas de seguridad. **Art. 39:** protección desde el diseño y por defecto. |
| L2 | LOPDP, **art. 20**: derecho a no ser objeto de decisiones basadas únicamente en valoraciones automatizadas | (mismo documento L1) | 2026-09-25 | Justifica la **revisión humana** cuando la confianza es baja, la explicación de cada predicción y el registro de la versión del modelo que decidió. |
| L3 | Reglamento General a la LOPDP (Decreto Ejecutivo 904, 2023) | https://www.cosede.gob.ec/wp-content/uploads/2023/12/REGLAMENTO-GENERAL-A-LA-LEY-ORG%C3%81NICA-DE-PROTECCION-DE-DATOS-PERSONALES_compressed-1.pdf | 2026-09-25 | Desarrolla la aplicación de la ley. Se cita como marco; no cambia el diseño técnico. |
| L4 | SRI, Reglamento a la Ley del RUC, **art. 3** | https://www.sri.gob.ec/o/sri-portlet-biblioteca-alfresco-internet/descargar/9822eb12-84d9-4722-88b4-5c95cc3bdda7/TRIBUTAR-REGLAMENTO_A_LA_LEY_DE_REGISTRO_UNICO_DE_CONTRIBUYENTES_RUC-1.pdf | 2026-09-25 | **RUC = 13 dígitos**; los dos primeros son la provincia; para personas naturales son los **10 dígitos de la cédula + `001`**. Regla del detector `<RUC>`. |
| L5 | ARCOTEL, Plan Técnico Fundamental de Numeración (Res. TEL-068-04-CONATEL-2013) | https://www.arcotel.gob.ec/wp-content/uploads/2015/10/plan-tecnico-fundamental-de-numeracion-ptfn.pdf | 2026-09-25 | Número nacional significativo de **9 dígitos**; códigos de área geográficos **22–28**; numeración móvil no geográfica. Regla del detector `<TELEFONO>`: `0[2-7]` + 7 dígitos, `09` + 8 dígitos, `+593` + 9 dígitos. |
| L6 | PCI SSC, PCI DSS v4.0.1 (req. 3.4.1 y 3.5.1) y FAQ de truncado | https://www.pcisecuritystandards.org/faq/articles/Frequently_Asked_Question/how-can-an-entity-meet-pci-dss-requirements-for-pan-masking-and-truncation-if-it-has-migrated-to-8-digit-bins/ | 2026-09-25 | Un número de tarjeta (PAN) no se muestra completo ni se almacena legible. Como este sistema no necesita el PAN, se **elimina por completo** (`<TARJETA>`). La validación de candidatos usa el algoritmo de Luhn (ISO/IEC 7812-1). |
| L7 | IETF RFC 2606, *Reserved Top Level DNS Names* | https://www.rfc-editor.org/rfc/rfc2606 | 2026-09-25 | Los datos ficticios de pruebas y demo usan solo `example.com` / `example.org`: nunca correos reales. |
| S1 | OWASP API Security Top 10 (2023) | https://owasp.org/API-Security/editions/2023/en/0x11-t10/ | 2026-09-25 | API1 (BOLA), API2 (autenticación), API4 (consumo de recursos: rate limiting y tamaño máximo del texto), API8 (configuración). |
| S2 | OWASP Machine Learning Security Top 10 (2023): ML02 *Data Poisoning*, ML06 *Supply Chain* | https://owasp.org/www-project-machine-learning-security-top-10/docs/ML02_2023-Data_Poisoning_Attack | 2026-09-25 | La retroalimentación solo la envía el rol `reviewer`, queda auditada, puede rechazarse y se filtra antes de reentrenar. El modelo se verifica por SHA-256 antes de cargarse. |

## 5. Despliegue gratuito (límites verificados)

| # | Fuente | URL | Consultada | Qué se tomó de aquí |
|---|---|---|---|---|
| P1 | Render, *Compute plans* | https://render.com/docs/compute-plans | 2026-09-25 | Plan web gratuito: **0,1 CPU y 512 MB de RAM**. Obliga a servir el modelo en ONNX INT8, sin PyTorch en la imagen de producción. |
| P2 | Render, *Deploy for Free* | https://render.com/docs/free | 2026-09-25 | Se duerme tras 15 min sin tráfico (≈1 min para despertar), 750 h/mes, sin disco persistente, y Postgres gratis expira a los 30 días. Por eso la base va en Neon. |
| P3 | Neon, extensión pgvector | https://neon.com/docs/extensions/pgvector | 2026-09-25 | pgvector disponible en el plan gratuito (0,5 GB por proyecto). |
| P4 | pgvector (README oficial, v0.8.6) | https://github.com/pgvector/pgvector | 2026-09-25 | Tipo `vector(384)`, operador de distancia coseno `<=>` e índice **HNSW** con `vector_cosine_ops`. Imagen Docker `pgvector/pgvector:0.8.6-pg17`. |
| P5 | Streamlit Community Cloud, *Manage your app* | https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app | 2026-09-25 | Hasta 2,7 GB de RAM; las apps se duermen tras 12 h sin tráfico. La demo llama a la API; la clave vive en `st.secrets`, nunca en el navegador. |

## 6. Reglas de negocio derivadas

| Regla | Fuente | Implementación |
|---|---|---|
| Nunca persistir ni registrar en logs el texto original: solo el anonimizado. | L1 (art. 10, 37, 39) | `anonymization/` corre antes de cualquier repositorio o log. |
| Marcadores: `<CORREO>`, `<CEDULA>`, `<RUC>`, `<TELEFONO>`, `<TARJETA>`, `<CUENTA>`, `<NOMBRE>`, `<URL>`. | L1, L4, L5, L6, D14 | `anonymization/patterns.py` |
| Toda secuencia de 10 dígitos se enmascara aunque su dígito verificador no sea válido (prioridad a la **exhaustividad**). | L1 (art. 4: impedir la reidentificación) | La etiqueta `<CEDULA>` solo se asigna si hay contexto ("cédula", "C.I.") o dígito verificador válido; si no, queda `<NUMERO>`. |
| Si la confianza calibrada es menor que el umbral, la decisión es `human_review`. | L2, E4, E5 | `services/classification_service.py` |
| Categoría, equipo y urgencia base salen de la intención mediante un **catálogo versionado** (tabla `intents`). | Diseño propio (ver SUPUESTO 1) | Cambiar el enrutamiento no exige reentrenar. |
| La urgencia se escala a `high` si el texto contiene señales de fraude o emergencia. | SUPUESTO 2 | `services/urgency_rules.py`; la regla aplicada se devuelve en la explicación. |
| Sentimiento: negativo si hay alguna emoción negativa y ninguna positiva; positivo en el caso inverso; neutral si solo hay "neutral" o ambiguas. Los ejemplos con emociones positivas y negativas a la vez se descartan. | D7 | `ml/data.py::map_goemotions_sentiment` |
| La retroalimentación entra al reentrenamiento solo si está `accepted` y trae la intención corregida. | S2 | `services/feedback_service.py`, `ml/data.py` |
| La comparación entre la línea base y el transformer usa la misma partición, la misma semilla (42) y el mismo conjunto de prueba. | E1, E2 | `ml/splits.py` guarda los índices y su hash. |

## Supuestos (no verificados en una fuente oficial)

1. **SUPUESTO 1: catálogo de enrutamiento.** Las 9 categorías, los 6 equipos y la urgencia base por intención son una política del proyecto para un banco **ficticio** ("Banco Demo"), no una norma oficial. Criterio: urgencia `high` si hay dinero en riesgo o un posible fraude; `medium` si el servicio falla o hay dinero retenido; `low` si es una consulta informativa. Se inspira en la práctica de priorizar por impacto × urgencia de la gestión de servicios (ISO/IEC 20000-1), cuyo texto es de pago y no se pudo verificar literalmente.
2. **SUPUESTO 2: palabras de escalamiento** ("fraude", "robo", "me robaron", "hackearon", "no reconozco", "urgente", "emergencia"). Lista propia; se evalúa aparte y no se mezcla con la métrica del modelo.
3. **SUPUESTO 3: dígito verificador de la cédula (módulo 10, coeficientes 2-1-2-1…).** El Registro Civil no publica una especificación oficial. Se usa el algoritmo de uso extendido **solo para rotular** (`<CEDULA>` frente a `<NUMERO>`); el enmascarado no depende de él.
4. **SUPUESTO 4: detección de nombres.** Con reglas (disparadores como "me llamo", "soy", "Sr./Sra.", firmas tras "Atentamente" y un listado de nombres de pila frecuentes) no se detectan todos los nombres. Es una limitación declarada en la ficha del modelo.
5. **SUPUESTO 5: desplazamiento de dominio del sentimiento.** GoEmotions son comentarios de Reddit traducidos automáticamente. Su rendimiento en tickets bancarios no está medido con un conjunto etiquetado del dominio; queda en el roadmap: etiquetar 300 tickets con la propia herramienta de retroalimentación.

## Hallazgos de calidad de datos

- **Las etiquetas también se tradujeron.** La columna `rotulo` de BANKING77-ES-LA no es un entero (pese a lo que dice la ficha), sino el nombre de la intención traducido frase por frase: hay **629 variantes** para 77 intenciones (por ejemplo `cargo_extra_en_el_extracto` y `carga_adicional_en_la_factura`), y `recarga_por_tarjeta` se usa para dos intenciones distintas. **Decisión:** el corpus es paralelo fila a fila con el original (mismo tamaño y mismo orden, comprobado), así que la etiqueta se toma del CSV original de PolyAI (D2). `ml/data.py::validate_alignment` comprueba en cada ejecución que el tamaño coincide y que cada variante de `rotulo` apunta mayoritariamente a una sola intención.
- **Errores de traducción** conservados a propósito por los autores. Por ejemplo, *"My top-up is still pending"* aparece como *"Mi blusa todavía está pendiente"*. Añaden ruido de entrada y se declaran en la ficha del modelo.
- No hay problemas de codificación: el texto es UTF-8 válido (sin `Ã`, sin U+FFFD). Se verificó con pandas directamente sobre el parquet.
- La ficha de BANKING77-ES-LA tiene incoherencias (tabla numerada 1–77 frente a "rango [0-76]" y un párrafo que menciona "portugués", copiado de otra versión). No afectan a los datos, pero se documentan.
