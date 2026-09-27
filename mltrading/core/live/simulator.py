"""Simulador de operación en vivo, día a día (capítulo 8: retos del trading en vivo).

El capítulo 8 es conceptual, sin código: explica cómo guardar y cargar un
modelo, cómo ampliar los datos con la jornada más reciente, y quthree criterios
para decidir cuándo reentrenar (por caída de exactitud, por pérdida de
capital, o de forma periódica). Este módulo implementa esos tres criterios.

Por simplicidad, el simulador se limita a los modelos que no necesitan
escalado (Random Forest, Naive Bayes, árbol de decisión): así el estimador
reentrenado se puede aplicar de inmediato a la siguiente barra, sin mantener
por separado un escalador ajustado de forma incremental.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

MODELOS_SIMULABLES = ("random_forest", "naive_bayes", "decision_tree")

NUNCA = "Nunca"
PERIODICO = "Periódico"
POR_EXACTITUD = "Por caída de exactitud"
POR_PERDIDA = "Por pérdida de capital"
CALENDARIOS = (NUNCA, PERIODICO, POR_EXACTITUD, POR_PERDIDA)


@dataclass(frozen=True)
class ConfigSimulacion:
    calendario: str = PERIODICO
    periodo_barras: int = 26  # una jornada de 15 minutos: 6,5 horas x 4
    ventana_exactitud: int = 26
    umbral_exactitud: float = 0.55  # el capítulo 8 usa 55 % como ejemplo
    umbral_perdida_pct: float = -5.0  # retroceso desde el máximo de capital, en porcentaje

    def __post_init__(self) -> None:
        if self.calendario not in CALENDARIOS:
            raise ValueError(f"calendario debe ser uno de {CALENDARIOS}.")
        if self.periodo_barras < 1:
            raise ValueError("periodo_barras debe ser al menos 1.")
        if self.ventana_exactitud < 2:
            raise ValueError("ventana_exactitud debe ser al menos 2.")


CONFIG_PREDETERMINADA = ConfigSimulacion()


@dataclass
class Prediccion:
    indice: Any
    y_real: int
    y_predicho: int
    reentrenado: bool
    retorno: float  # cambio porcentual del precio en esa barra (para el capital)


@dataclass
class EstadoSimulacion:
    id_modelo: str
    parametros: dict
    config: ConfigSimulacion
    X_base: pd.DataFrame
    y_base: pd.Series
    X_restante: pd.DataFrame
    y_restante: pd.Series
    retornos_restantes: pd.Series  # cambio porcentual de precio alineado con X_restante
    estimador: Any = None
    predicciones: list[Prediccion] = field(default_factory=list)
    barras_desde_reentreno: int = 0
    capital: float = 1.0
    capital_maximo: float = 1.0
    tiempos_reentrenamiento: list[float] = field(default_factory=list)

    @property
    def barras_simuladas(self) -> int:
        return len(self.predicciones)

    @property
    def barras_pendientes(self) -> int:
        return len(self.X_restante)

    @property
    def terminada(self) -> bool:
        return self.barras_pendientes == 0

    @property
    def exactitud_global(self) -> float:
        if not self.predicciones:
            return float("nan")
        aciertos = sum(p.y_real == p.y_predicho for p in self.predicciones)
        return aciertos / len(self.predicciones)

    def exactitud_ventana(self, ventana: int) -> float:
        recientes = self.predicciones[-ventana:]
        if not recientes:
            return float("nan")
        aciertos = sum(p.y_real == p.y_predicho for p in recientes)
        return aciertos / len(recientes)

    def a_tabla(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fecha": p.indice, "real": p.y_real, "predicho": p.y_predicho,
                    "acierto": p.y_real == p.y_predicho, "reentrenado": p.reentrenado,
                }
                for p in self.predicciones
            ]
        ).set_index("fecha") if self.predicciones else pd.DataFrame(
            columns=["real", "predicho", "acierto", "reentrenado"]
        )


def _construir_estimador(id_modelo: str, parametros: dict, n_variables: int):
    from mltrading.core.models import decision_tree, naive_bayes, random_forest

    if id_modelo == "random_forest":
        modelo, _ = random_forest.construir_estimador(parametros, n_variables)
        return modelo
    if id_modelo == "naive_bayes":
        return naive_bayes.construir_estimador(parametros)
    if id_modelo == "decision_tree":
        return decision_tree.construir_estimador(parametros)
    raise ValueError(f"'{id_modelo}' no está entre los modelos simulables: {MODELOS_SIMULABLES}.")


def _reentrenar(estado: EstadoSimulacion, X_entrenamiento: pd.DataFrame, y_entrenamiento: pd.Series) -> None:
    t0 = time.perf_counter()
    modelo = _construir_estimador(estado.id_modelo, estado.parametros, X_entrenamiento.shape[1])
    modelo.fit(X_entrenamiento, y_entrenamiento)
    estado.estimador = modelo
    estado.tiempos_reentrenamiento.append(time.perf_counter() - t0)
    estado.barras_desde_reentreno = 0


def _debe_reentrenar(estado: EstadoSimulacion) -> bool:
    config = estado.config
    if config.calendario == NUNCA:
        return False
    if config.calendario == PERIODICO:
        return estado.barras_desde_reentreno >= config.periodo_barras
    if config.calendario == POR_EXACTITUD:
        if estado.barras_simuladas < config.ventana_exactitud:
            return False
        return estado.exactitud_ventana(config.ventana_exactitud) < config.umbral_exactitud
    if config.calendario == POR_PERDIDA:
        if estado.capital_maximo <= 0:
            return False
        retroceso_pct = (estado.capital / estado.capital_maximo - 1) * 100
        return retroceso_pct < config.umbral_perdida_pct
    return False


def iniciar(
    id_modelo: str, parametros: dict, X_base: pd.DataFrame, y_base: pd.Series,
    X_restante: pd.DataFrame, y_restante: pd.Series, retornos_restantes: pd.Series,
    config: ConfigSimulacion = CONFIG_PREDETERMINADA,
) -> EstadoSimulacion:
    if id_modelo not in MODELOS_SIMULABLES:
        raise ValueError(f"'{id_modelo}' no está entre los modelos simulables: {MODELOS_SIMULABLES}.")
    if not X_restante.index.equals(y_restante.index) or not X_restante.index.equals(retornos_restantes.index):
        raise ValueError("X_restante, y_restante y retornos_restantes deben compartir el mismo índice.")
    estado = EstadoSimulacion(
        id_modelo=id_modelo, parametros=dict(parametros), config=config,
        X_base=X_base, y_base=y_base, X_restante=X_restante, y_restante=y_restante,
        retornos_restantes=retornos_restantes,
    )
    _reentrenar(estado, X_base, y_base)
    return estado


def avanzar(estado: EstadoSimulacion, pasos: int) -> EstadoSimulacion:
    """Revela hasta `pasos` barras más, prediciendo y reentrenando según el calendario."""
    if pasos < 1:
        raise ValueError("pasos debe ser al menos 1.")
    pasos = min(pasos, estado.barras_pendientes)
    for _ in range(pasos):
        if estado.terminada:
            break
        indice = estado.X_restante.index[0]
        x_fila = estado.X_restante.iloc[[0]]
        y_real = int(estado.y_restante.iloc[0])
        retorno = float(estado.retornos_restantes.iloc[0])

        y_predicho = int(estado.estimador.predict(x_fila)[0])
        estado.capital *= 1 + (retorno if y_predicho == 1 else 0.0)
        estado.capital_maximo = max(estado.capital_maximo, estado.capital)
        estado.predicciones.append(
            Prediccion(indice=indice, y_real=y_real, y_predicho=y_predicho, reentrenado=False, retorno=retorno)
        )
        estado.barras_desde_reentreno += 1

        # la barra recién revelada pasa a formar parte del entrenamiento acumulado
        estado.X_base = pd.concat([estado.X_base, x_fila])
        estado.y_base = pd.concat([estado.y_base, estado.y_restante.iloc[[0]]])
        estado.X_restante = estado.X_restante.iloc[1:]
        estado.y_restante = estado.y_restante.iloc[1:]
        estado.retornos_restantes = estado.retornos_restantes.iloc[1:]

        if _debe_reentrenar(estado):
            _reentrenar(estado, estado.X_base, estado.y_base)
            estado.predicciones[-1] = Prediccion(
                indice=indice, y_real=y_real, y_predicho=y_predicho, reentrenado=True, retorno=retorno
            )
    return estado
