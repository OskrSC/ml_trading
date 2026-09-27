"""Árbol de decisión de regresión (capítulo 12, segunda parte)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .params import ParametroEntero

PARAMETROS = (
    ParametroEntero("min_samples_leaf", "Mínimo de casos por hoja", 200, 5, 2000,
                    "El capítulo 12 usa 200: hojas grandes, para suavizar la predicción del retorno."),
)


@dataclass
class ResultadoArbolRegresion:
    estimador: object
    y_pred_prueba: pd.Series
    r_cuadrado: float
    error_cuadratico_medio: float
    interpretacion: pd.DataFrame


def construir_objetivo(X: pd.DataFrame, columna: str = "pct_change") -> pd.Series:
    """El objetivo es el valor de `columna` en la barra siguiente (regresión, no clasificación)."""
    return X[columna].shift(-1).rename("objetivo_regresion")


def entrenar(X_entrenamiento: pd.DataFrame, y_entrenamiento: pd.Series, X_prueba: pd.DataFrame,
             y_prueba: pd.Series, parametros: dict) -> ResultadoArbolRegresion:
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.tree import DecisionTreeRegressor  # importación diferida

    modelo = DecisionTreeRegressor(min_samples_leaf=int(parametros["min_samples_leaf"]))
    modelo.fit(X_entrenamiento, y_entrenamiento)
    pred = pd.Series(modelo.predict(X_prueba), index=X_prueba.index, name="prediccion")
    importancias = pd.DataFrame(
        {"variable": X_entrenamiento.columns, "importancia": modelo.feature_importances_}
    ).sort_values("importancia", ascending=False, ignore_index=True)
    return ResultadoArbolRegresion(
        estimador=modelo, y_pred_prueba=pred,
        r_cuadrado=float(r2_score(y_prueba, pred)),
        error_cuadratico_medio=float(mean_squared_error(y_prueba, pred)),
        interpretacion=importancias,
    )
