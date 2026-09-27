"""División de los datos en entrenamiento y prueba."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

CRONOLOGICA = "Cronológica"
ALEATORIA = "Aleatoria"


@dataclass
class Division:
    tipo: str
    proporcion: float
    X_entrenamiento: pd.DataFrame
    X_prueba: pd.DataFrame
    y_entrenamiento: pd.Series
    y_prueba: pd.Series

    @property
    def filas_entrenamiento(self) -> int:
        return len(self.X_entrenamiento)

    @property
    def filas_prueba(self) -> int:
        return len(self.X_prueba)


def _validar(X: pd.DataFrame, y: pd.Series, proporcion: float) -> None:
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X e y deben tener el mismo índice.")
    if not 0.05 <= proporcion <= 0.95:
        raise ValueError("La proporción de entrenamiento debe estar entre 0,05 y 0,95.")


def dividir_cronologica(X: pd.DataFrame, y: pd.Series, proporcion: float = 0.8) -> Division:
    """Las primeras filas entrenan y las últimas prueban. Nunca se usa el futuro para predecir el pasado."""
    _validar(X, y, proporcion)
    corte = math.floor(proporcion * len(X))
    return Division(
        tipo=CRONOLOGICA,
        proporcion=proporcion,
        X_entrenamiento=X.iloc[:corte],
        X_prueba=X.iloc[corte:],
        y_entrenamiento=y.iloc[:corte],
        y_prueba=y.iloc[corte:],
    )


def dividir_aleatoria(X: pd.DataFrame, y: pd.Series, proporcion: float = 0.8, semilla: int = 42) -> Division:
    """Reparto aleatorio. Se ofrece solo para mostrar por qué no conviene con series de tiempo.

    Usa numpy en lugar de scikit-learn para no cargar esa librería en una página informativa.
    """
    _validar(X, y, proporcion)
    orden = np.random.default_rng(semilla).permutation(len(X))
    corte = math.floor(proporcion * len(X))
    entrenamiento, prueba = np.sort(orden[:corte]), np.sort(orden[corte:])
    return Division(
        tipo=ALEATORIA,
        proporcion=proporcion,
        X_entrenamiento=X.iloc[entrenamiento],
        X_prueba=X.iloc[prueba],
        y_entrenamiento=y.iloc[entrenamiento],
        y_prueba=y.iloc[prueba],
    )


def fraccion_prueba_anterior(division: Division) -> float:
    """Fracción de filas de prueba con fecha anterior al último dato de entrenamiento.

    Con una división cronológica vale 0. Con una aleatoria, un valor alto indica
    que el modelo se entrena con datos posteriores a los que luego debe predecir.
    """
    ultimo = division.X_entrenamiento.index.max()
    return float((division.X_prueba.index < ultimo).mean())


def resumen(division: Division) -> pd.DataFrame:
    filas = []
    for nombre, X, y in (
        ("Entrenamiento", division.X_entrenamiento, division.y_entrenamiento),
        ("Prueba", division.X_prueba, division.y_prueba),
    ):
        filas.append(
            {
                "Conjunto": nombre,
                "Filas": len(X),
                "Inicio": X.index.min(),
                "Fin": X.index.max(),
                "Señal 1 (%)": float(y.mean() * 100),
            }
        )
    return pd.DataFrame(filas)
