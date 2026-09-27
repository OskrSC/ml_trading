# Guía de mantenimiento

Notas prácticas para quien dé continuidad a este proyecto después del lanzamiento.

## Actualizar una dependencia

1. Cambie la versión en `requirements.txt`.
2. Reinstale en un entorno limpio: `pip install -r requirements-dev.txt`.
3. Regenere los artefactos predeterminados si tocó `numpy`, `scipy` o `statsmodels`:
   `python scripts/generar_artefactos.py`.
4. Ejecute `python -m pytest` completo. Los resultados predeterminados están fijados con números exactos en
   las pruebas; si una actualización de una librería los cambia de verdad (no solo en el quinto decimal),
   revise el motivo antes de aceptar el nuevo valor como resultado predeterminado y actualice tanto el código
   como la prueba a la vez, nunca solo la prueba.
5. Ejecute `pip-audit -r requirements.txt` para revisar vulnerabilidades conocidas. El flujo de CI ya lo hace
   en cada cambio, en un job separado que informa sin bloquear el resto.

## Añadir un modelo al registro de clasificación

Los modelos que comparten las variables y la división de JPMorgan (los que aparecen en Modelos supervisados,
Evaluación, Backtesting y el Comparador) siguen un patrón fijo en `mltrading/core/models/`:

1. Cree `mi_modelo.py` con: `PARAMETROS` (tupla de objetos de `params.py`), una función `entrenar(...)` que
   devuelva un `ResultadoEntrenamiento`, y una `ESPECIFICACION` de tipo `EspecificacionModelo`.
2. Si el modelo participará en el simulador de operación en vivo y no necesita escalado, exponga también
   `construir_estimador(parametros, ...)` por separado de `entrenar`, y añada su id a
   `mltrading/core/live/simulator.py::MODELOS_SIMULABLES`.
3. Añádalo a `MODELOS` en `mltrading/core/models/registry.py`.
4. Escriba pruebas de resultado predeterminado en `tests/test_models.py` o un archivo nuevo, siguiendo el
   patrón de los modelos existentes: entrenar con los valores predeterminados y fijar el número exacto de
   exactitud que produce en este entorno.

No hace falta tocar `mltrading/views/modelos.py`, `evaluacion.py`, `backtesting.py` ni `comparador.py`: leen el
registro y construyen sus controles automáticamente a partir de `PARAMETROS`.

## Regenerar el tema tras cambiar los tokens de diseño

```
python scripts/generate_theme.py
```

Nunca edite `.streamlit/config.toml` a mano: se genera desde `mltrading/config/tokens.py` y
`mltrading/config/theme_config.py`, y una prueba (`test_tokens.py`) falla si quedan desincronizados.

## Regenerar las capturas de referencia

```
python scripts/capture_themes.py
```

Requiere Chromium de Playwright. Escribe en `docs/capturas/`. El script tarda varios minutos con las 15
páginas y 4 combinaciones cada una; si se ejecuta en un entorno con límite de tiempo por comando, edite
`PAGINAS` en el propio script para generar un subconjunto por partes.

## Diagnosticar en producción

La página Diagnóstico del entorno (oculta en producción por `MLT_MOSTRAR_DIAGNOSTICO=0`; ver
`docs/despliegue_streamlit_cloud.md`) muestra recursos, paquetes instalados y las últimas entradas del
registro de este proceso. Para verla en producción de forma puntual, cambie ese secreto a `1`, revise, y
vuelva a ponerlo en `0`.

Los registros también se escriben a la salida estándar (`mltrading/core/observabilidad.py`), que la mayoría de
servicios de despliegue capturan en su propio visor de registros; no dependa solo de la página de diagnóstico
para depurar un incidente pasado, porque su buffer en memoria se pierde al reiniciar el proceso.

## Ediciones de terminología

Antes de añadir texto a `mltrading/` o a `docs/*.md`, revise `docs/terminologia.md`. Dos pruebas
(`tests/test_terminology.py` y `tests/test_no_emojis.py`) fallan si aparece una expresión retirada o un
emoji; es la forma más rápida de detectar el problema antes de abrir una revisión de código.

## Licencia del código propio

`NOTICE` y `LICENSE` proponen MIT como licencia por defecto para el código propio (no para `data_modules/` ni
para el contenido conceptual derivado del material de origen, que tienen sus propias licencias). Si el
proyecto va a distribuirse de otra forma, actualice ambos archivos a la vez.
