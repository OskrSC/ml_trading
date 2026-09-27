"""Regresión lineal simple con mínimos cuadrados ordinarios (capítulo 9)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ResultadoOLS:
    modelo: object  # RegressionResultsWrapper de statsmodels
    datos: pd.DataFrame  # columnas: x, y, predicho
    interseccion: float
    pendiente: float
    r_cuadrado: float
    nombre_x: str
    nombre_y: str


def ajustar_ols(datos: pd.DataFrame, columna_x: str, columna_y: str) -> ResultadoOLS:
    import statsmodels.api as sm  # importación diferida
    from sklearn.metrics import r2_score

    Y = datos[columna_y]
    X = sm.add_constant(datos[columna_x])
    modelo = sm.OLS(Y, X).fit()
    predicho = modelo.predict(X)
    salida = pd.DataFrame({columna_x: datos[columna_x], columna_y: Y, "predicho": predicho})
    return ResultadoOLS(
        modelo=modelo,
        datos=salida,
        interseccion=float(modelo.params.iloc[0]),
        pendiente=float(modelo.params.iloc[1]),
        r_cuadrado=float(r2_score(Y, predicho)),
        nombre_x=columna_x,
        nombre_y=columna_y,
    )
