"""K-Means sobre ROE y Beta (capítulo 17).

El código de origen ajusta el modelo que se visualiza con los datos escalados,
pero calcula la curva del codo con los datos sin escalar (`model.fit(df)`, no
`df_values`). Esta aplicación reproduce esa diferencia: `ajustar` siempre
escala, y `curva_codo` solo escala si se pide explícitamente.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .escalado import escalar

K_PREDETERMINADO = 2
K_MAXIMO_CODO = 9
# El código de origen no fija una semilla para K-Means, así que las etiquetas 0 y 1
# pueden intercambiarse entre ejecuciones (la inercia no cambia, el agrupamiento
# tampoco). Se fija una semilla para que el resultado predeterminado de esta
# aplicación sea exactamente reproducible.
SEMILLA_PREDETERMINADA = 42


@dataclass
class ResultadoKMeans:
    etiquetas: pd.Series  # índice: el de `datos`, valores: número de clúster
    centros: np.ndarray
    inercia: float
    k: int


def ajustar(datos: pd.DataFrame, k: int, semilla: int = SEMILLA_PREDETERMINADA) -> ResultadoKMeans:
    from sklearn.cluster import KMeans  # importación diferida

    if not 1 <= k <= len(datos):
        raise ValueError(f"k debe estar entre 1 y el número de filas ({len(datos)}).")
    modelo = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(escalar(datos))
    etiquetas = pd.Series(modelo.labels_, index=datos.index, name="cluster")
    return ResultadoKMeans(etiquetas=etiquetas, centros=modelo.cluster_centers_, inercia=float(modelo.inertia_), k=k)


def curva_codo(
    datos: pd.DataFrame, k_maximo: int = K_MAXIMO_CODO, escalar_datos: bool = False,
    semilla: int = SEMILLA_PREDETERMINADA,
) -> pd.DataFrame:
    """Inercia para k de 1 a `k_maximo`. `escalar_datos=False` reproduce el cálculo del código de origen."""
    from sklearn.cluster import KMeans  # importación diferida

    valores = escalar(datos) if escalar_datos else datos.values
    filas = []
    for k in range(1, min(k_maximo, len(datos)) + 1):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(valores)
        filas.append({"k": k, "inercia": float(modelo.inertia_)})
    return pd.DataFrame(filas)
