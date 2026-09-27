"""Registro de operaciones (capítulo 7), calculado de forma vectorizada."""
from __future__ import annotations

import numpy as np
import pandas as pd


def construir_registro(datos: pd.DataFrame, columna_cierre: str = "close",
                       columna_senal: str = "predicted_signal") -> pd.DataFrame:
    """Cada fila es una operación: se abre cuando la señal deja de ser 0 y se cierra cuando cambia de nuevo.

    Es equivalente al bucle `get_trades` del libro, reescrito con operaciones
    vectorizadas de pandas en lugar de `DataFrame.append` en un bucle (retirado
    de pandas).
    """
    senal = datos[columna_senal]
    anterior = senal.shift(1).fillna(0)
    cambia = senal != anterior
    si_cambia = datos.loc[cambia, [columna_senal, columna_cierre]].copy()
    si_cambia["posicion_anterior"] = anterior.loc[cambia]

    aperturas = si_cambia[si_cambia[columna_senal] != 0].copy()
    aperturas["orden"] = range(len(aperturas))
    cierres_posibles = si_cambia[si_cambia["posicion_anterior"] != 0].copy()

    filas = []
    puntero = 0
    lista_cierres = list(cierres_posibles.index)
    for tiempo_entrada, fila in aperturas.iterrows():
        while puntero < len(lista_cierres) and lista_cierres[puntero] <= tiempo_entrada:
            puntero += 1
        if puntero >= len(lista_cierres):
            break
        tiempo_salida = lista_cierres[puntero]
        filas.append(
            {
                "Posición": float(fila[columna_senal]),
                "Entrada": tiempo_entrada,
                "Precio de entrada": float(fila[columna_cierre]),
                "Salida": tiempo_salida,
                "Precio de salida": float(datos.loc[tiempo_salida, columna_cierre]),
            }
        )
        puntero += 1
    registro = pd.DataFrame(filas, columns=["Posición", "Entrada", "Precio de entrada", "Salida", "Precio de salida"])
    if not registro.empty:
        registro["PnL"] = (registro["Precio de salida"] - registro["Precio de entrada"]) * registro["Posición"]
        registro["Duración (barras)"] = (
            registro["Salida"].searchsorted(registro["Salida"]) - registro["Entrada"].map(lambda t: datos.index.get_loc(t))
        )
    return registro
