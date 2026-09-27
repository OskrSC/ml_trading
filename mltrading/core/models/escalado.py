"""Escalado de variables, ajustado solo con el tramo de entrenamiento."""
from __future__ import annotations

import pandas as pd


def escalar(X_entrenamiento: pd.DataFrame, X_prueba: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.preprocessing import StandardScaler  # importación diferida

    escalador = StandardScaler()
    entrenamiento = pd.DataFrame(
        escalador.fit_transform(X_entrenamiento), index=X_entrenamiento.index, columns=X_entrenamiento.columns
    )
    prueba = pd.DataFrame(escalador.transform(X_prueba), index=X_prueba.index, columns=X_prueba.columns)
    return entrenamiento, prueba
