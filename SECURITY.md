# Política de seguridad

## Reportar una vulnerabilidad
No abras un issue público. Escríbeme por GitHub (perfil DiegoFranciscoG) con los pasos para reproducir el problema. Respondo en un máximo de 7 días.

## Prácticas aplicadas en este proyecto
- Secretos solo por variables de entorno (`.env` no se versiona; ver `.env.example`). Si falta una variable crítica, la API no arranca.
- Escaneo de secretos con gitleaks en cada push y pull request; hooks locales de pre-commit.
- Autenticación por API key con roles (`client`, `reviewer`) y comparación en tiempo constante; deny-by-default en todo `/api/**`.
- Rate limiting por clave, CORS con orígenes explícitos, cabeceras de seguridad y errores sin detalles internos.
- Anonimización del texto **antes** de persistirlo o registrarlo: nunca se guarda el texto original del ticket (LOPDP Ecuador, art. 10 y 37).
- Retroalimentación solo con rol `reviewer`, auditada y filtrada antes de reentrenar (mitigación de envenenamiento de datos, OWASP ML02).
- Modelo verificado por hash SHA-256 antes de cargarse (OWASP ML06, cadena de suministro).
- Dependencias fijadas en `uv.lock` y revisadas con Dependabot; imágenes Docker con versión fija y usuario no root.
