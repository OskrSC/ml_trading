# Fase 2: primer flujo completo

## Alcance

- Página Modelos supervisados: registro de modelos, controles generados desde su definición, entrenamiento y tres pestañas de resultado (Resumen, Interpretación, Predicciones).
- Dos modelos: Random Forest (capítulos 5 y 13) y regresión logística (capítulo 10), con escalado ajustado solo en entrenamiento para la segunda.
- Página Evaluación: exactitud, matriz de confusión, informe de clasificación y precisión a lo largo del tiempo.
- Página Backtesting: retornos de la estrategia, curva de capital, drawdown, Sharpe, registro de operaciones y su analítica, con un costo de transacción opcional (0 por defecto).
- El modelo entrenado se guarda en el estado de la sesión y alimenta las dos páginas siguientes sin volver a entrenar.

## Resultados predeterminados

| Elemento | Resultado |
|---|---|
| Random Forest (3 árboles), predicciones | Idénticas a `JPM_predicted_2017_2019.csv` |
| Random Forest, exactitud en prueba | 51,55 % |
| Regresión logística, exactitud en prueba | 50,96 % |
| Regresión logística, Sharpe | 2,75 |
| Matriz de confusión (Random Forest) | [[985, 951], [921, 1007]] |
| Backtest (Random Forest): acumulado, anualizado, volatilidad, Sharpe, drawdown | 28,10 %, 52,20 %, 14,90 %, 2,89, -7,94 % |
| Operaciones del backtest | 169 (todas largas), 102 ganadoras, 67 perdedoras, resultado neto 29,72 |

Todos estos valores se comprueban en `tests/test_models.py`, `tests/test_evaluation.py` y `tests/test_backtest.py`.

## Módulos

```
mltrading/core/models/
  base.py, params.py       contrato y esquema de hiperparámetros
  random_forest.py         Random Forest (capítulos 5 y 13)
  logistic_regression.py   regresión logística (capítulo 10)
  registry.py               registro de modelos disponibles
  escalado.py                StandardScaler ajustado solo en entrenamiento
  entrenamiento.py           orquesta escalado + entrenamiento
mltrading/core/evaluation/metrics.py   exactitud, matriz de confusión, informe
mltrading/core/backtest/
  config.py    configuración del backtest (costo, factor de anualización)
  returns.py   retornos, curva de capital, drawdown, Sharpe
  trades.py    registro de operaciones (vectorizado, sin DataFrame.append)
  analytics.py analítica agregada del registro
```

## Decisiones

- **Registro de modelos.** Cada modelo se describe con una ficha (`EspecificacionModelo`) que incluye su esquema de hiperparámetros. La página Modelos supervisados construye los controles a partir de esa ficha: añadir un modelo no exige tocar la página.
- **Estado compartido.** El modelo entrenado, sus parámetros, la división usada y si la configuración es predeterminada se guardan en `st.session_state["_entrenamiento"]`. Evaluación y Backtesting leen de ahí y piden entrenar primero si no existe.
- **Registro de operaciones vectorizado.** El código de origen recorre las filas con un bucle y usa `DataFrame.append`, retirado de pandas. Se reescribió con una función vectorizada que agrupa las barras donde cambia la posición y empareja cada apertura con su cierre siguiente. Se comprobó que el resultado, con la señal predeterminada, coincide con el del libro: 169 operaciones y el mismo resultado neto.
- **La señal del libro es 0/1, nunca -1.** Por eso el registro de operaciones predeterminado no tiene posiciones cortas. El campo `Posición` admite -1 para cuando, en fases futuras, algún modelo produzca señales de venta en corto.
- **Costo de transacción.** Es una extensión sobre la configuración predeterminada, apagada por defecto (0 puntos básicos). Con cualquier valor mayor que 0, la marca de la página pasa a "Configuración personalizada" y aparece un aviso.
- **Factor de anualización.** Fijo en 252 x 6,5 x 4 (barras de 15 minutos), como el capítulo 7. No es un control de la interfaz todavía; queda como parámetro interno por si en una fase futura se necesitan otras frecuencias.
- **Caché de entrenamiento.** `st.cache_data` evita reentrenar si no cambian el modelo, sus parámetros o la división. La caché se limpia con el mismo mecanismo que la de variables.

## Cómo probar el flujo manualmente

1. Abra Modelos supervisados, deje el modo Predeterminado y pulse «Entrenar Random Forest».
2. Vaya a Evaluación: la exactitud debe marcar 51,55 %.
3. Vaya a Backtesting: el retorno acumulado debe marcar 28,10 % y el Sharpe 2,89.
4. Vuelva a Modelos supervisados, elija Regresión logística y entrénela. Backtesting debe actualizarse solo, con Sharpe 2,75.
