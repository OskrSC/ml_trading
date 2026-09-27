"""Construcción de la variable objetivo y de las variables candidatas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicadores
from .config import ConfigVariables

PRECIOS = ("open", "high", "low", "close")
CANDIDATAS = (
    "open", "high", "low", "close",
    "pct_change", "pct_change2", "pct_change5",
    "rsi", "adx", "sma", "corr", "volatility", "volatility2",
)


@dataclass
class Candidatas:
    """Variables candidatas y objetivo ya alineados, sin filas incompletas."""

    datos: pd.DataFrame  # CANDIDATAS + future_returns + signal
    filas_iniciales: int
    filas_por_ventana: int  # filas del comienzo sin valor por los periodos de cálculo
    filas_por_objetivo: int  # filas del final sin retorno futuro (la última barra)

    @property
    def filas(self) -> int:
        return len(self.datos)

    @property
    def objetivo(self) -> pd.Series:
        return self.datos["signal"]

    @property
    def variables(self) -> pd.DataFrame:
        return self.datos[list(CANDIDATAS)]


def descripciones(barras: int) -> dict[str, str]:
    return {
        "open": "Precio de apertura de la barra.",
        "high": "Precio máximo de la barra.",
        "low": "Precio mínimo de la barra.",
        "close": "Precio de cierre de la barra.",
        "pct_change": "Cambio porcentual del cierre frente a la barra anterior (15 minutos).",
        "pct_change2": "Cambio porcentual del cierre frente a 2 barras atrás (30 minutos).",
        "pct_change5": "Cambio porcentual del cierre frente a 5 barras atrás (75 minutos).",
        "rsi": f"Índice de fuerza relativa con periodo de {barras} barras.",
        "adx": f"Índice direccional promedio, que mide la fuerza de la tendencia, con periodo de {barras} barras.",
        "sma": f"Media móvil simple del cierre de {barras} barras.",
        "corr": f"Correlación móvil de {barras} barras entre el cierre y su media móvil.",
        "volatility": f"Desviación estándar móvil de {barras} barras del cambio porcentual, en porcentaje.",
        "volatility2": f"Igual que volatility, con una ventana doble de {2 * barras} barras.",
    }


def construir_candidatas(ohlcv: pd.DataFrame, config: ConfigVariables) -> Candidatas:
    """Calcula las 13 variables candidatas y el objetivo.

    La señal vale 1 si el cierre de la barra siguiente es mayor que el actual y
    0 en caso contrario. Se descartan las filas con algún valor faltante.
    """
    faltan = [c for c in PRECIOS if c not in ohlcv.columns]
    if faltan:
        raise ValueError("Faltan columnas de precio: " + ", ".join(faltan) + ".")
    n = config.barras_por_dia
    d = ohlcv.copy()
    cierre = d["close"]

    d["pct_change"] = cierre.pct_change()
    d["pct_change2"] = cierre.pct_change(2)
    d["pct_change5"] = cierre.pct_change(5)
    d["rsi"] = indicadores.rsi(cierre.values, n, config.motor)
    d["adx"] = indicadores.adx(d["high"].values, d["low"].values, cierre.values, n, config.motor)
    d["sma"] = cierre.rolling(window=n).mean()
    d["corr"] = cierre.rolling(window=n).corr(d["sma"])
    d["volatility"] = d["pct_change"].rolling(n, min_periods=n).std() * 100
    d["volatility2"] = d["pct_change"].rolling(2 * n, min_periods=2 * n).std() * 100

    d["future_returns"] = cierre.pct_change().shift(-1)
    d["signal"] = np.where(d["future_returns"] > 0, 1, 0)

    columnas = list(CANDIDATAS) + ["future_returns", "signal"]
    trabajo = d[columnas]
    sin_objetivo = trabajo.drop(columns=["future_returns"]).iloc[:-1]
    completas = trabajo.dropna()
    por_objetivo = 1 if len(trabajo) and pd.isna(trabajo["future_returns"].iloc[-1]) else 0
    por_ventana = len(trabajo) - len(completas) - por_objetivo
    del sin_objetivo
    completas = completas.copy()
    completas["signal"] = completas["signal"].astype(int)
    return Candidatas(
        datos=completas,
        filas_iniciales=len(ohlcv),
        filas_por_ventana=por_ventana,
        filas_por_objetivo=por_objetivo,
    )
