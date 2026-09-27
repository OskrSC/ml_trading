"""Retornos de la estrategia, curva de capital, drawdown y métricas de riesgo."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ConfigBacktest


@dataclass
class ResultadoRetornos:
    datos: pd.DataFrame  # close, pct_change, predicted_signal, strategy_returns, cumulative_returns, benchmark_cumulative_returns, drawdown
    retorno_acumulado_pct: float
    retorno_anualizado_pct: float
    volatilidad_anualizada_pct: float
    sharpe: float
    drawdown_maximo_pct: float
    costo_total_pct: float


def _costo_por_barra(senal: pd.Series, costo_bp: float) -> pd.Series:
    """Costo aplicado en cada barra donde la posición cambia (entrada o salida)."""
    if costo_bp == 0:
        return pd.Series(0.0, index=senal.index)
    cambios = senal.diff().abs().fillna(senal.abs())
    return cambios * (costo_bp / 10_000)


def calcular_retornos(close: pd.Series, senal_predicha: pd.Series, config: ConfigBacktest) -> ResultadoRetornos:
    """Replica el cálculo del capítulo 7: la señal se desplaza un periodo (no se opera con datos futuros)."""
    d = pd.DataFrame({"close": close}).join(senal_predicha.rename("predicted_signal"), how="inner").dropna()
    if d.empty:
        raise ValueError("No hay filas en común entre el precio y la señal predicha.")
    d["pct_change"] = d["close"].pct_change()
    posicion_anterior = d["predicted_signal"].shift(1)
    bruto = posicion_anterior * d["pct_change"]
    costo = _costo_por_barra(posicion_anterior.fillna(0.0), config.costo_bp)
    d["strategy_returns"] = bruto - costo
    d = d.dropna()

    d["cumulative_returns"] = (1 + d["strategy_returns"]).cumprod()
    d["benchmark_cumulative_returns"] = (1 + d["pct_change"]).cumprod()

    maximo_acumulado = np.maximum.accumulate(d["cumulative_returns"])
    maximo_acumulado = maximo_acumulado.clip(lower=1.0)
    d["drawdown"] = (d["cumulative_returns"] / maximo_acumulado - 1) * 100

    k = config.factor_anualizacion
    ret_acum = float((d["cumulative_returns"].iloc[-1] - 1) * 100)
    ret_anual = float((d["cumulative_returns"].iloc[-1] ** (k / len(d)) - 1) * 100)
    vol_anual = float(d["strategy_returns"].std() * np.sqrt(k) * 100)
    desviacion = d["strategy_returns"].std()
    sharpe = float(d["strategy_returns"].mean() / desviacion * np.sqrt(k)) if desviacion else float("nan")
    dd_max = float(d["drawdown"].min())
    costo_total = float((posicion_anterior.fillna(0.0).diff().abs().fillna(posicion_anterior.abs()) * (config.costo_bp / 10_000)).reindex(d.index).sum() * 100) if config.costo_bp else 0.0

    return ResultadoRetornos(
        datos=d, retorno_acumulado_pct=ret_acum, retorno_anualizado_pct=ret_anual,
        volatilidad_anualizada_pct=vol_anual, sharpe=sharpe, drawdown_maximo_pct=dd_max,
        costo_total_pct=costo_total,
    )
