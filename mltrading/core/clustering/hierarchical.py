"""Clustering jerárquico (capítulo 18).

El código de origen dibuja el dendrograma con los datos sin escalar
(`sc.linkage(df, ...)`) pero ajusta `AgglomerativeClustering` con los datos
escalados. Esta aplicación reproduce esa diferencia igual que en `kmeans.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .escalado import escalar

METODO_PREDETERMINADO = "ward"
N_CLUSTERS_PREDETERMINADO = 2
METODOS = ("ward", "complete", "average", "single")


@dataclass
class ResultadoJerarquico:
    etiquetas: pd.Series
    n_clusters: int
    metodo: str


def matriz_enlace(datos: pd.DataFrame, metodo: str = METODO_PREDETERMINADO, escalar_datos: bool = False) -> np.ndarray:
    """Matriz de enlace para el dendrograma. `escalar_datos=False` reproduce el código de origen."""
    from scipy.cluster.hierarchy import linkage  # importación diferida

    valores = escalar(datos) if escalar_datos else datos.values
    return linkage(valores, method=metodo)


def ajustar(datos: pd.DataFrame, n_clusters: int, metodo: str = METODO_PREDETERMINADO) -> ResultadoJerarquico:
    from sklearn.cluster import AgglomerativeClustering  # importación diferida

    if not 1 <= n_clusters <= len(datos):
        raise ValueError(f"n_clusters debe estar entre 1 y el número de filas ({len(datos)}).")
    modelo = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage=metodo)
    modelo.fit(escalar(datos))
    etiquetas = pd.Series(modelo.labels_, index=datos.index, name="cluster")
    return ResultadoJerarquico(etiquetas=etiquetas, n_clusters=n_clusters, metodo=metodo)
