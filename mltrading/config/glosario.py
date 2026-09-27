"""Glosario de términos, en el orden en que aparecen en la aplicación."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Termino:
    nombre: str
    nombre_original: str
    definicion: str
    seccion: str


GLOSARIO: tuple[Termino, ...] = (
    Termino("Aprendizaje supervisado", "Supervised learning",
           "Familia de modelos que aprenden a partir de ejemplos con la respuesta correcta ya conocida "
           "(la señal de compra, en esta aplicación).", "Fundamentos"),
    Termino("Aprendizaje no supervisado", "Unsupervised learning",
           "Familia de modelos que buscan patrones o grupos en los datos sin que nadie les diga cuál es la "
           "respuesta correcta.", "Fundamentos"),
    Termino("Variable candidata", "Feature",
           "Cada una de las señales calculadas a partir del precio (el RSI, la volatilidad...) que un modelo "
           "puede usar para predecir.", "Preparación de datos"),
    Termino("Objetivo", "Target",
           "Lo que el modelo intenta predecir: en los modelos de clasificación de esta aplicación, si el precio "
           "sube o baja en la siguiente barra.", "Preparación de datos"),
    Termino("Estacionariedad", "Stationarity",
           "Una variable es estacionaria si su media y su varianza no cambian con el tiempo. Muchos modelos "
           "funcionan mejor con variables estacionarias.", "Preparación de datos"),
    Termino("Prueba ADF", "Augmented Dickey-Fuller test",
           "Prueba estadística que comprueba si una variable es estacionaria. Un p-valor bajo indica que sí lo "
           "es.", "Preparación de datos"),
    Termino("División cronológica", "Chronological split",
           "Separar los datos en un tramo de entrenamiento y otro de prueba respetando el orden temporal, para "
           "no entrenar con información del futuro.", "Preparación de datos"),
    Termino("Sobreajuste", "Overfitting",
           "Cuando un modelo aprende el ruido de los datos de entrenamiento en vez del patrón general, y por "
           "eso funciona peor con datos nuevos.", "Modelado"),
    Termino("Hiperparámetro", "Hyperparameter",
           "Un ajuste del modelo que se decide antes de entrenar (por ejemplo, cuántos árboles tiene un Random "
           "Forest), a diferencia de lo que el modelo aprende por sí solo.", "Modelado"),
    Termino("Exactitud", "Accuracy",
           "Proporción de predicciones correctas sobre el total. Puede ser engañosa si las clases no están "
           "equilibradas.", "Evaluación"),
    Termino("Matriz de confusión", "Confusion matrix",
           "Tabla que cruza lo que el modelo predijo con lo que realmente ocurrió, y muestra en qué se "
           "equivoca.", "Evaluación"),
    Termino("Curva de capital", "Equity curve",
           "Gráfico de cómo habría crecido el capital siguiendo las señales del modelo, con 1 como punto de "
           "partida.", "Backtesting"),
    Termino("Drawdown", "Drawdown",
           "Caída del capital desde su máximo histórico hasta un momento dado, en porcentaje.", "Backtesting"),
    Termino("Sharpe", "Sharpe ratio",
           "Retorno medio dividido entre su volatilidad, ajustado a un periodo anual. Compara el retorno "
           "obtenido con el riesgo asumido.", "Backtesting"),
    Termino("Clúster", "Cluster",
           "Grupo de elementos que un algoritmo de aprendizaje no supervisado considera similares entre sí.",
           "No supervisado"),
    Termino("Componente principal", "Principal component",
           "Una combinación de las variables originales que resume la mayor cantidad posible de su variación "
           "conjunta.", "No supervisado"),
    Termino("Varianza explicada", "Explained variance",
           "Proporción de la variación total de los datos que capturan los componentes principales elegidos.",
           "No supervisado"),
    Termino("Reentrenar", "Retrain",
           "Ajustar de nuevo un modelo con datos más recientes, en lugar de seguir usando el modelo original "
           "indefinidamente.", "Operación en vivo"),
    Termino("Receta", "Recipe",
           "Archivo con la configuración de un modelo (no el modelo en sí) que esta aplicación puede recalcular "
           "de forma determinista.", "Operación en vivo"),
)


def por_seccion() -> dict[str, list[Termino]]:
    salida: dict[str, list[Termino]] = {}
    for t in GLOSARIO:
        salida.setdefault(t.seccion, []).append(t)
    return salida
