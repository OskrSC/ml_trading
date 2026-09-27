"""Indicadores técnicos con dos motores: TA-Lib y un respaldo en numpy.

El respaldo reproduce el resultado de TA-Lib (diferencias del orden de 1e-14),
de modo que la aplicación funciona igual si TA-Lib no se puede instalar.

Nota sobre el ADX: el valor no depende del precio de cierre. El rango verdadero
aparece en el numerador y el denominador del cociente de los índices
direccionales y se cancela. Se mantiene el parámetro `cierre` por compatibilidad
con la firma habitual del indicador.
"""
from __future__ import annotations

import numpy as np

from .config import MOTOR_AUTO, MOTOR_PANDAS, MOTOR_TALIB

try:  # TA-Lib es opcional
    import talib as _talib
except Exception:  # noqa: BLE001 - puede faltar la librería o su binario
    _talib = None


class MotorNoDisponible(RuntimeError):
    pass


def talib_disponible() -> bool:
    return _talib is not None


def resolver_motor(motor: str) -> str:
    if motor == MOTOR_AUTO:
        return MOTOR_TALIB if talib_disponible() else MOTOR_PANDAS
    if motor == MOTOR_TALIB and not talib_disponible():
        raise MotorNoDisponible("TA-Lib no está instalado en este entorno. Elija el motor Automático o Pandas.")
    return motor


def rsi_respaldo(cierre: np.ndarray, periodo: int) -> np.ndarray:
    """RSI con suavizado de Wilder, inicializado con el promedio simple de los primeros cambios."""
    c = np.asarray(cierre, dtype=float)
    salida = np.full(len(c), np.nan)
    if len(c) <= periodo:
        return salida
    dif = np.diff(c)
    sube = np.where(dif > 0, dif, 0.0)
    baja = np.where(dif < 0, -dif, 0.0)
    prom_sube, prom_baja = sube[:periodo].mean(), baja[:periodo].mean()

    def valor(a: float, b: float) -> float:
        return 100.0 * a / (a + b) if (a + b) != 0 else 0.0

    salida[periodo] = valor(prom_sube, prom_baja)
    for i in range(periodo, len(dif)):
        prom_sube = (prom_sube * (periodo - 1) + sube[i]) / periodo
        prom_baja = (prom_baja * (periodo - 1) + baja[i]) / periodo
        salida[i + 1] = valor(prom_sube, prom_baja)
    return salida


def adx_respaldo(alto: np.ndarray, bajo: np.ndarray, periodo: int) -> np.ndarray:
    """ADX de Wilder. Primer valor en el índice 2 x periodo - 1, igual que TA-Lib."""
    h, l = np.asarray(alto, dtype=float), np.asarray(bajo, dtype=float)
    salida = np.full(len(h), np.nan)
    if len(h) < 2 * periodo:
        return salida
    sube, baja = h[1:] - h[:-1], l[:-1] - l[1:]
    mas = np.where((sube > baja) & (sube > 0), sube, 0.0)
    menos = np.where((baja > sube) & (baja > 0), baja, 0.0)
    s_mas, s_menos = mas[: periodo - 1].sum(), menos[: periodo - 1].sum()
    dxs: list[float] = []
    adx: float | None = None
    for i in range(periodo - 1, len(mas)):
        s_mas = s_mas - s_mas / periodo + mas[i]
        s_menos = s_menos - s_menos / periodo + menos[i]
        total = s_mas + s_menos
        dx = 100.0 * abs(s_mas - s_menos) / total if total != 0 else 0.0
        if adx is None:
            dxs.append(dx)
            if len(dxs) == periodo:
                adx = float(np.mean(dxs))
                salida[i + 1] = adx
        else:
            adx = (adx * (periodo - 1) + dx) / periodo
            salida[i + 1] = adx
    return salida


def rsi(cierre: np.ndarray, periodo: int, motor: str = MOTOR_AUTO) -> np.ndarray:
    if resolver_motor(motor) == MOTOR_TALIB:
        return _talib.RSI(np.asarray(cierre, dtype=float), timeperiod=periodo)
    return rsi_respaldo(cierre, periodo)


def adx(alto: np.ndarray, bajo: np.ndarray, cierre: np.ndarray, periodo: int, motor: str = MOTOR_AUTO) -> np.ndarray:
    if resolver_motor(motor) == MOTOR_TALIB:
        return _talib.ADX(
            np.asarray(alto, dtype=float), np.asarray(bajo, dtype=float), np.asarray(cierre, dtype=float),
            timeperiod=periodo,
        )
    return adx_respaldo(alto, bajo, periodo)


__all__ = ["MOTOR_PANDAS", "MotorNoDisponible", "adx", "resolver_motor", "rsi", "talib_disponible"]
