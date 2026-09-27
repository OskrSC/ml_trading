# Fase 3: resto de modelos supervisados y comparador

## Alcance

- Tres modelos añadidos al registro genérico (reutilizan Modelos supervisados, Evaluación y Backtesting sin
  cambios en esas páginas): Naive Bayes (capítulo 11), árbol de decisión de clasificación (capítulo 12) y una
  variante de la red neuronal del capítulo 15.
- Página Regresión: regresión lineal simple por mínimos cuadrados ordinarios (capítulo 9) y árbol de
  decisión de regresión (capítulo 12).
- Página XGBoost multiactivo (capítulo 14), con su propio conjunto de datos y su propio flujo.
- Página Comparador de modelos: entrena varios modelos del registro con las mismas variables y la misma
  división, y los compara.

## Decisión: red neuronal sin Keras

El perfil de despliegue elegido es ligero (ver `docs/decisiones.md`, punto 11), sin TensorFlow. El capítulo 15
construye una red con Keras (dos capas ocultas de 128 neuronas ReLU, salida sigmoide, optimizador Adam, error
cuadrático medio como función de pérdida, 7 épocas, lotes de 20). Esta aplicación usa `MLPClassifier` de
scikit-learn con la misma arquitectura de capas, pero con entropía cruzada en lugar de error cuadrático medio,
así que no es una réplica exacta. Además, el código de origen solo fija la semilla del módulo `random` de
Python, no la de numpy ni la de Keras, así que sus resultados no son exactamente reproducibles entre
ejecuciones; los de esta variante sí lo son, con semilla 42 por defecto.

## Decisión: universo de XGBoost sin descargas en vivo

El capítulo 14 descarga cinco acciones estadounidenses (AAPL, AMZN, NFLX, WMT, MSFT) con `yfinance` en el
momento de ejecutarse. Esta aplicación no descarga datos por defecto (fase 0). El universo predeterminado usa
cinco activos ya presentes en `data_modules/`: Reliance Industries, Coca-Cola y tres columnas de `pca.csv`
(Alphabet, Amazon y 3M). La lógica de variables (`pct_change_r` y `std_r` para r de 10 a 55 en pasos de 5),
objetivo (-1 o 1 según el signo del retorno del día siguiente) y división (80/20 por activo, concatenada)
reproduce la del código de origen. La etiqueta -1/1 del código de origen se convierte a 0/1 solo para entrenar,
porque la versión de XGBoost usada aquí no admite -1/1 como en versiones anteriores de la librería (ver
`docs/decisiones.md`, punto 17); se vuelve a -1/1 al mostrar los resultados.

Como los activos no comparten calendario (Reliance cotiza en India, Coca-Cola y los de `pca.csv` en Estados
Unidos, con rangos de fechas distintos), la pestaña Portafolio alinea las curvas de retorno por posición
dentro de cada tramo de prueba, no por fecha. Se documenta así en la propia página.

## Resultados predeterminados

| Elemento | Resultado |
|---|---|
| Naive Bayes, exactitud | 50,93 % |
| Árbol de decisión (clasificación), exactitud | 51,04 % |
| Red neuronal (variante scikit-learn), exactitud | 52,10 % |
| Regresión lineal, JPM sobre BAC (2019), R² | 0,82 |
| Regresión lineal, JPM sobre Nestlé (2019), R² | 0,35 |
| Árbol de regresión, R² en prueba | -0,0172 |
| XGBoost multiactivo, exactitud global | 49,97 % |
| XGBoost, filas de entrenamiento y prueba | 5.844 y 1.465 |

Todos estos valores, y la reproducibilidad de los modelos con semilla, se comprueban en
`tests/test_models_fase3.py`, `tests/test_regression_models.py` y `tests/test_multiasset_xgboost.py`.

## Módulos nuevos

```
mltrading/core/models/
  naive_bayes.py               BernoulliNB (capítulo 11), sin hiperparámetros expuestos
  decision_tree.py             árbol de clasificación (capítulo 12)
  neural_network.py            variante de la red del capítulo 15 con MLPClassifier
  linear_regression.py         OLS con statsmodels (capítulo 9)
  decision_tree_regressor.py   árbol de regresión (capítulo 12)
  multiasset.py                variables, objetivo y división del conjunto multiactivo (capítulo 14)
  xgboost_multiactivo.py       entrenamiento de XGBoost sobre ese conjunto
mltrading/views/
  regresion.py       regresión lineal y árbol de regresión, en pestañas
  xgboost_view.py    XGBoost multiactivo
  comparador.py      comparador de modelos del registro
```

## Por qué el árbol de regresión y XGBoost no usan el registro genérico

El registro de `mltrading/core/models/registry.py` supone una clasificación binaria sobre las variables y la
división de JPMorgan. El árbol de regresión predice un número (el cambio porcentual siguiente) con un objetivo
propio, y XGBoost entrena sobre un conjunto de datos distinto por completo (multiactivo). Forzarlos al mismo
contrato habría complicado el registro para dos casos que no lo necesitan; en su lugar, cada uno tiene su
página y sus propias funciones de entrenamiento.
