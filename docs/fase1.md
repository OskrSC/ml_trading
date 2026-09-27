# Fase 1: datos, variables y división

## Alcance

- Páginas Variables y objetivo, y División de datos, con modo Predeterminado y Personalizado.
- Construcción de la señal y de las 13 variables candidatas para JPMorgan (barras de 15 minutos).
- Prueba de estacionariedad (ADF) y descarte guiado por correlación.
- División cronológica y comparación con una división aleatoria.
- Cálculo de RSI y ADX con TA-Lib y con un respaldo en numpy.

## Resultados predeterminados

Con los valores predeterminados, la aplicación produce:

| Elemento | Resultado |
|---|---|
| Filas utilizables | 19.317 de 19.370 (52 iniciales por las ventanas de cálculo y 1 final sin retorno futuro) |
| Variables candidatas | 13 |
| No estacionarias (descartadas) | open, high, low, close, sma |
| Par correlacionado (0,832) | volatility y volatility2; se descarta volatility2 |
| Variables finales | pct_change, pct_change2, pct_change5, rsi, adx, corr, volatility |
| División cronológica 80 % | 15.453 filas de entrenamiento y 3.864 de prueba |
| Señal 1 en el total y en prueba | 49,2 % y 49,9 % |

Estos valores se comprueban en `tests/test_variables.py` y `tests/test_division.py` contra los artefactos predeterminados de `data_modules/`: variables y señal idénticas (diferencia menor que 1e-12) y división idéntica en índices y valores.

## Módulos

```
mltrading/core/features/
  config.py          ConfigVariables y su valor predeterminado
  indicadores.py     RSI y ADX con TA-Lib y respaldo en numpy
  construccion.py    señal y variables candidatas
  seleccion.py       prueba ADF, pares correlacionados y regla de descarte
  pipeline.py        flujo completo y ResultadoVariables
  division.py        división cronológica y aleatoria
  predeterminados.py lectura y escritura del artefacto de la prueba ADF
mltrading/ui/estado.py   modo, valores compartidos entre páginas y caché
artefactos/adf_predeterminado.json
scripts/generar_artefactos.py
```

## Reglas de la selección de variables

1. Se descartan las variables con p-valor ADF mayor o igual que el umbral (0,05 por defecto).
2. Entre las estacionarias se buscan pares con correlación absoluta mayor que el umbral (0,70 por defecto).
3. De cada par, empezando por el más correlacionado, se descarta la variable que aparece después en el orden de columnas. La persona puede cambiar esa elección en el modo Personalizado.

## Comportamiento de la interfaz

- El modo elegido se conserva al cambiar de página.
- La marca "Resultado predeterminado" aparece solo si los parámetros, los descartes y la proporción coinciden con los predeterminados. Cualquier cambio la convierte en "Configuración personalizada". Cambiar el motor de indicadores no altera la marca, porque el resultado es el mismo.
- Con una ventana distinta de 26 barras, la prueba ADF se calcula en el momento (unos 15 segundos) y queda en caché durante una hora.
- Las descargas de variables y señal son CSV generados en memoria, sin escribir en el disco del servidor.

## Cómo regenerar el artefacto

Si cambian los datos, las variables candidatas o las versiones de statsmodels o numpy:

```
python scripts/generar_artefactos.py
python -m pytest tests/test_variables.py
```
