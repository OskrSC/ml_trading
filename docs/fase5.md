# Fase 5: operación en vivo, recetas, glosario y referencias

## Alcance

- Página Operación en vivo (capítulo 8): simula día a día, revelando barras del tramo de prueba, con tres
  criterios de reentrenamiento y uno de referencia sin reentrenar.
- Recetas de modelos: guardar y cargar la configuración completa de un modelo entrenado como JSON, sin
  serializar ningún objeto de Python.
- Página Guía y referencias: fundamentos, glosario de 19 términos, y un resumen conceptual, con palabras
  propias, de las tres familias de modelos que el material de origen trata pero que esta aplicación no
  implementa (no supervisado como introducción general, NLP y aprendizaje por refuerzo).

## Decisión: el simulador se limita a tres modelos

El capítulo 8 no tiene código: describe, en prosa, cómo guardar y cargar un modelo, cómo ampliar los datos con
la jornada más reciente, y tres criterios para decidir cuándo reentrenar (por caída de exactitud, por pérdida
de capital, o de forma periódica, sin importar el desempeño). Esta aplicación implementa esos tres criterios.

El simulador solo admite Random Forest, Naive Bayes y árbol de decisión: los tres modelos del registro que no
necesitan escalado. Los otros dos (regresión logística y la red neuronal) ajustan un `StandardScaler` sobre el
tramo de entrenamiento en cada llamada a `entrenar_modelo`, sin persistirlo; incorporarlos al simulador exigiría
mantener ese escalador actualizado de forma incremental, fuera del flujo estándar de entrenamiento. Se dejó
fuera de alcance por simplicidad, documentado aquí como una limitación conocida.

## Decisión: recetas en JSON, nunca modelos serializados

La fase 0 descartó aceptar archivos `pickle` o `joblib` subidos por la persona, porque deserializarlos ejecuta
código arbitrario. Una receta no es un modelo guardado: es un JSON con los parámetros necesarios (variables,
división, modelo e hiperparámetros) para que la aplicación reconstruya exactamente el mismo modelo a partir de
los datos de `data_modules/`. Cargar una receta nunca ejecuta código del archivo más allá de `json.loads`; una
prueba (`test_ninguna_receta_ejecuta_codigo_arbitrario`) confirma que el módulo no usa `eval`, `exec`,
`pickle` ni `compile`.

## Resultados predeterminados

| Elemento | Resultado |
|---|---|
| Simulador, calendario «Nunca», exactitud sobre las 3.864 barras de prueba | 51,55 % (idéntico al Random Forest predeterminado) |
| Simulador, calendario predeterminado (periódico cada 26 barras) | Reproducible con la misma semilla del Random Forest |
| Receta predeterminada aplicada de nuevo | Reproduce exactamente el resultado del Random Forest predeterminado |

## Módulos nuevos

```
mltrading/core/live/simulator.py     tres criterios de reentrenamiento, estado de la simulación
mltrading/core/recipes/
  schema.py    RecetaModelo, validación campo a campo
  aplicar.py   reconstruye variables, división y modelo desde una receta
mltrading/config/glosario.py         19 términos agrupados por sección
mltrading/views/
  en_vivo.py   página del simulador
  guia.py      fundamentos, glosario, no supervisado, NLP, refuerzo, licencias
```

## Refactor menor en los modelos simulables

`random_forest.py`, `naive_bayes.py` y `decision_tree.py` ahora exponen `construir_estimador(parametros, ...)`
por separado de `entrenar(...)`, para que el simulador pueda reentrenar sin pasar por la interfaz de
ajuste-y-predicción de una sola vez que usan las demás páginas. `entrenar()` sigue funcionando igual;
`tests/test_models.py` y `tests/test_models_fase3.py` no cambiaron y siguen en verde.
