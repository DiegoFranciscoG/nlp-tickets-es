# Datos de ejemplo (seeds)

`tickets_seed.csv` contiene 770 tickets históricos (10 por intención) para la demo de búsqueda semántica.

- **Origen del texto:** partición de *entrenamiento* de BANKING77-ES-LA (Centro de Ciencia de Datos POLI/USP, 2025), traducción revisada al español latinoamericano de BANKING77 (Casanueva et al., 2020, PolyAI). Licencia **CC BY 4.0**: https://huggingface.co/datasets/c2d-usp/banking77-es-la
- **Cambios realizados:** normalización de espacios, anonimización (sin cambios: el corpus no contiene datos personales), exclusión de frases que también aparecen en la partición de prueba, y asignación de canal y fecha **ficticios** con semilla fija (`nlp-tickets build-seed`).
- **Resolución:** al cargarse, cada ticket recibe la resolución estándar de su intención (`data/catalog/intents.csv`), que describe un banco ficticio.
- No contiene datos personales reales.
