"""Selección de variables: estacionariedad (prueba ADF) y correlación."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResultadoADF:
    variable: str
    estadistico: float
    p_valor: float
    retardos: int


def prueba_adf(serie: pd.Series) -> ResultadoADF:
    """Prueba de Dickey-Fuller aumentada con los valores por defecto de statsmodels."""
    from statsmodels.tsa.stattools import adfuller  # importación diferida: es pesada

    try:
        r = adfuller(serie, result_object=False)  # conserva la tupla y evita el aviso de futura versión
    except TypeError:  # versiones de statsmodels sin ese parámetro
        r = adfuller(serie)
    return ResultadoADF(str(serie.name), float(r[0]), float(r[1]), int(r[2]))


def calcular_adf(variables: pd.DataFrame) -> pd.DataFrame:
    """Tabla con estadístico, p-valor y retardos de cada variable (sin aplicar umbral)."""
    filas = [prueba_adf(variables[c]) for c in variables.columns]
    return pd.DataFrame(
        {
            "variable": [f.variable for f in filas],
            "estadistico": [f.estadistico for f in filas],
            "p_valor": [f.p_valor for f in filas],
            "retardos": [f.retardos for f in filas],
        }
    )


def clasificar_adf(tabla: pd.DataFrame, umbral: float) -> pd.DataFrame:
    salida = tabla.copy()
    salida["estacionaria"] = salida["p_valor"] < umbral
    return salida


def pares_correlacionados(variables: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Pares de variables con correlación absoluta mayor que el umbral, de mayor a menor."""
    cols = list(variables.columns)
    corr = variables.corr().abs()
    filas = [
        {"variable_a": a, "variable_b": b, "correlacion": float(corr.loc[a, b])}
        for i, a in enumerate(cols)
        for b in cols[i + 1 :]
        if corr.loc[a, b] > umbral
    ]
    tabla = pd.DataFrame(filas, columns=["variable_a", "variable_b", "correlacion"])
    return tabla.sort_values("correlacion", ascending=False, ignore_index=True)


def sugerir_descartes(pares: pd.DataFrame) -> list[str]:
    """Regla: de cada par, se descarta la variable que aparece después en el orden de columnas.

    Los pares se recorren de mayor a menor correlación, y un par no se evalúa si
    una de sus variables ya se descartó.
    """
    descartadas: list[str] = []
    for fila in pares.itertuples(index=False):
        if fila.variable_a in descartadas or fila.variable_b in descartadas:
            continue
        descartadas.append(fila.variable_b)
    return descartadas
