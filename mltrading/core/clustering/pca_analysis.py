"""PCA con K-Means sobre las cargas y visualización con t-SNE (capítulo 19)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_COMPONENTES_PREDETERMINADO = 18
K_PREDETERMINADO = 4
SEMILLA_PREDETERMINADA = 7  # la usa el propio código de origen para KMeans
SEMILLA_TSNE_PREDETERMINADA = 1337  # la usa el propio código de origen
PERPLEJIDAD_PREDETERMINADA = 25  # se ajusta a la baja si hay pocas acciones; ver `perplejidad_efectiva`


def retornos_diarios(precios: pd.DataFrame) -> pd.DataFrame:
    return precios.pct_change().dropna()


@dataclass
class ResultadoPCA:
    modelo: object  # PCA de scikit-learn, ya ajustado
    varianza_explicada: pd.Series  # índice: 1..n_componentes
    varianza_explicada_acumulada: float
    cargas_escaladas: np.ndarray  # (n_acciones, n_componentes), listas para K-Means


def ajustar_pca(retornos: pd.DataFrame, n_componentes: int = N_COMPONENTES_PREDETERMINADO) -> ResultadoPCA:
    from sklearn.decomposition import PCA  # importación diferida
    from sklearn.preprocessing import StandardScaler

    maximo = min(retornos.shape)
    if not 1 <= n_componentes <= maximo:
        raise ValueError(f"El número de componentes debe estar entre 1 y {maximo}.")
    modelo = PCA(n_components=n_componentes).fit(retornos)
    cargas = modelo.components_.T  # (n_acciones, n_componentes)
    cargas_escaladas = StandardScaler().fit_transform(cargas)
    varianza = pd.Series(modelo.explained_variance_ratio_, index=range(1, n_componentes + 1), name="varianza")
    return ResultadoPCA(
        modelo=modelo, varianza_explicada=varianza,
        varianza_explicada_acumulada=float(varianza.sum()), cargas_escaladas=cargas_escaladas,
    )


@dataclass
class ResultadoClustersPCA:
    etiquetas: pd.Series  # índice: los tickers, en el orden de `retornos.columns`
    inercia: float
    k: int


def agrupar(
    resultado_pca: ResultadoPCA, tickers: pd.Index, k: int = K_PREDETERMINADO,
    semilla: int = SEMILLA_PREDETERMINADA,
) -> ResultadoClustersPCA:
    from sklearn.cluster import KMeans  # importación diferida

    if not 1 <= k <= len(tickers):
        raise ValueError(f"k debe estar entre 1 y el número de acciones ({len(tickers)}).")
    modelo = KMeans(n_clusters=k, init="k-means++", max_iter=30, n_init=10, random_state=semilla)
    modelo.fit(resultado_pca.cargas_escaladas)
    etiquetas = pd.Series(modelo.labels_, index=tickers, name="cluster")
    return ResultadoClustersPCA(etiquetas=etiquetas, inercia=float(modelo.inertia_), k=k)


def perplejidad_efectiva(n_muestras: int, solicitada: int = PERPLEJIDAD_PREDETERMINADA) -> int:
    """t-SNE exige perplejidad menor que el número de muestras. Se ajusta a la baja si hace falta."""
    return min(solicitada, max(1, n_muestras - 1))


@dataclass
class ResultadoTSNE:
    coordenadas: np.ndarray  # (n_acciones, 2)
    perplejidad_solicitada: int
    perplejidad_usada: int

    @property
    def ajustada(self) -> bool:
        return self.perplejidad_usada != self.perplejidad_solicitada


def proyectar_tsne(
    resultado_pca: ResultadoPCA, perplejidad: int = PERPLEJIDAD_PREDETERMINADA,
    tasa_aprendizaje: int = 1000, semilla: int = SEMILLA_TSNE_PREDETERMINADA,
) -> ResultadoTSNE:
    from sklearn.manifold import TSNE  # importación diferida

    n = len(resultado_pca.cargas_escaladas)
    usada = perplejidad_efectiva(n, perplejidad)
    modelo = TSNE(learning_rate=tasa_aprendizaje, perplexity=usada, random_state=semilla)
    coordenadas = modelo.fit_transform(resultado_pca.cargas_escaladas)
    return ResultadoTSNE(coordenadas=coordenadas, perplejidad_solicitada=perplejidad, perplejidad_usada=usada)
