# Reportes de evaluación

`nlp-tickets train` genera aquí:

- `metrics.json`: métricas de prueba de la línea base TF-IDF, de la línea base con embeddings congelados y del modelo final ONNX INT8 (misma partición y semilla), bootstrap pareado, paridad FP32/INT8, latencia e historial de entrenamiento.
- `figures/`: comparación de macro-F1, matrices de confusión (equipo y categoría), diagrama de fiabilidad y curva de entrenamiento.

La API lee `metrics.json` al arrancar para registrar las métricas del modelo activo en `model_metrics`.
