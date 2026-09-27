# Aprendizaje automático en trading

Laboratorio interactivo, construido con Streamlit, para entrenar, evaluar y probar modelos de aprendizaje automático sobre precios reales de acciones. Cada modelo parte de una configuración predeterminada que la persona puede modificar y comparar.

Estado: **las 7 fases están completadas** (cimientos, datos y variables, primer flujo, resto de modelos, no supervisado, operación en vivo y guía, endurecimiento). Perfil de despliegue: **ligero** (sin TensorFlow; la red neuronal usa una variante con scikit-learn). Antes de un lanzamiento público, revise `docs/checklist_lanzamiento.md`. Ver la hoja de ruta en la página de inicio y el detalle de cada fase en `docs/faseN.md`.

## Ejecutar en local

Requiere Python 3.12.

```
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

TA-Lib se instala desde PyPI. Si en su sistema no hay una rueda binaria disponible, consulte `docs/despliegue_streamlit_cloud.md`. Si ve un error de importación de `scipy` o `statsmodels`, o un módulo que falta, consulte `docs/solucion_problemas.md` antes de nada: casi siempre es un entorno virtual que no coincide con `requirements.txt`.

## Pruebas

```
python -m pytest
```

426 pruebas. Comprueban el catálogo de datos, la preparación de variables y la división contra los artefactos predeterminados, cada modelo (resultado predeterminado y reproducibilidad), el backtesting, el clustering, el simulador de operación en vivo, las recetas (sin deserialización insegura), los tokens de diseño y su sincronía con `.streamlit/config.toml`, el diagnóstico, la observabilidad, la ausencia de emojis y de terminología retirada, y que cada página se ejecuta sin errores y degrada con elegancia ante un fallo de entrenamiento.

Para regenerar el tema después de editar `mltrading/config/tokens.py`:

```
python scripts/generate_theme.py
```

Para regenerar los artefactos predeterminados (la prueba ADF precalculada) tras cambiar los datos o las versiones de `numpy`/`scipy`/`statsmodels`:

```
python scripts/generar_artefactos.py
```

Para capturar las páginas en ambos temas y en móvil (requiere Chromium de Playwright):

```
python scripts/capture_themes.py
```

Ver `docs/mantenimiento.md` para estas y otras tareas recurrentes (añadir un modelo, actualizar una dependencia, diagnosticar en producción).

## Estructura

```
streamlit_app.py        punto de entrada y navegación (15 páginas)
.streamlit/config.toml  temas claro y oscuro (generado desde los tokens)
mltrading/
  config/               tokens de diseño, textos, ajustes, hoja de ruta, glosario, atribución
  ui/                   diseño de página, componentes, gráficos, formato, estado compartido
  views/                una página por archivo
  core/data/            catálogo, lectura y validación de los 16 conjuntos de datos
  core/features/        indicadores, variables, selección y división
  core/models/          registro de modelos de clasificación, regresión y XGBoost multiactivo
  core/evaluation/      métricas de clasificación
  core/backtest/        retornos, curva de capital, registro de operaciones
  core/clustering/      K-Means, jerárquico, PCA y t-SNE
  core/live/            simulador de operación en vivo
  core/recipes/         recetas de modelos (JSON, sin objetos serializados)
  core/env/             diagnóstico del entorno
  core/observabilidad.py  registro estructurado, con buffer en memoria
artefactos/             resultados predeterminados precalculados
data_modules/           datos de origen, sin modificar
tests/  scripts/  docs/
```

`mltrading/core` no importa Streamlit en la lógica de datos, variables y modelos, de modo que se puede probar y reutilizar de forma aislada.

## Apariencia

Los temas claro y oscuro los define Streamlit desde `config.toml`. La persona elige entre Sistema, Claro y Oscuro en el menú de la esquina superior derecha, y el cambio es instantáneo. Los colores de las series de los gráficos son idénticos en ambos temas, y los gráficos que distinguen categorías (como acierto y fallo en Evaluación) no dependen solo del color: también varían en símbolo o posición.

## Seguridad

- Las recetas de modelos son JSON validado campo a campo; cargarlas nunca deserializa un objeto de Python (nada de `pickle` ni `joblib`), y el archivo subido tiene un límite de tamaño.
- `showErrorDetails = "type"` en producción: se ve el tipo de error, no la traza completa ni rutas del servidor.
- `pip-audit -r requirements.txt` se ejecuta en cada cambio (job `seguridad` de la integración continua).

## Observabilidad

`mltrading/core/observabilidad.py` registra a la salida estándar y guarda además las últimas entradas en memoria, visibles en la página Diagnóstico del entorno (oculta en producción salvo que se active a propósito). Ver `docs/mantenimiento.md`.

## Despliegue

Ver `docs/despliegue_streamlit_cloud.md` y, antes de publicar, `docs/checklist_lanzamiento.md`.

## Atribución y licencias

Ver `NOTICE` y `LICENSE`. Los datos y el código de origen provienen de quantra-go-algo/ml-trading-ebook (Apache 2.0). El contenido conceptual se basa en "Machine Learning in Trading", de Ishan Shah y Rekhit Pachanekar (CC BY-SA 4.0). El código propio de este repositorio propone MIT como licencia por defecto, a confirmar antes de un lanzamiento público.

Esta aplicación tiene fines educativos y no constituye una recomendación de inversión.
