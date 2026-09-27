# Terminología y redacción

Lo que se toma como referencia se llama predeterminado.

| No usar | Usar |
|---|---|
| Reproducir el libro | Modo predeterminado (frente a personalizado) |
| Coincide con el libro | Resultado predeterminado. Si se cambia un parámetro o un dato, Configuración personalizada |
| Valores del libro | Valores predeterminados |
| Panel de verificación | Panel de resultados predeterminados |
| Pruebas de paridad | Pruebas de resultados predeterminados |
| CSV precalculados | Artefactos predeterminados |

## Qué es un resultado predeterminado

Es lo que produce la configuración predeterminada, con los datos de `data_modules/` y las versiones de librerías fijadas en `requirements.txt`. Se guarda como instantánea de referencia. En los modelos con componente aleatoria o sensibles a la versión se acepta una tolerancia.

## Excepción: atribución

Las licencias exigen citar el material de origen. Esa cita vive solo en `mltrading/config/atribucion.py`, en `NOTICE` y en la página de referencias de la fase 5. Ahí el material de origen se nombra como fuente, no como criterio de validación.

## Estilo

- Español, voz activa, mayúscula solo al inicio de la frase.
- Un botón conserva el mismo nombre a lo largo de todo el flujo.
- Los errores explican qué pasó y cómo corregirlo, sin disculpas.
- Sin emojis. Los iconos son de la familia Material.

## Comprobación automática

`tests/test_terminology.py` y `tests/test_no_emojis.py` fallan si el código de la interfaz usa una expresión retirada o un emoji. Las reglas están en `tests/reglas_texto.py`.
