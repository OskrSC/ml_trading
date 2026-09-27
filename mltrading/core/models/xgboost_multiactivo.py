"""XGBoost sobre el conjunto multiactivo (capítulo 14).

A diferencia de los demás modelos del registro, este no usa las variables de
JPMorgan de la página Variables y objetivo: construye su propio conjunto a
partir de varios activos (ver `multiasset.py`). Por eso no se integra en el
registro genérico de `registry.py`, sino que la página de XGBoost lo usa
directamente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .multiasset import ConjuntoMultiactivo
from .params import ParametroEntero

PARAMETROS = (
    ParametroEntero("n_estimators", "Número de árboles", 30, 1, 300,
                    "El capítulo 14 usa 30 árboles."),
    ParametroEntero("max_depth", "Profundidad máxima", 2, 1, 20, "Profundidad máxima de cada árbol."),
)


def entrenar_xgboost(conjunto: ConjuntoMultiactivo, parametros: dict) -> "ResultadoXGBoost":
    from xgboost import XGBClassifier  # importación diferida

    y_entrenamiento_01 = (conjunto.y_entrenamiento == 1).astype(int)
    modelo = XGBClassifier(
        n_estimators=int(parametros["n_estimators"]), max_depth=int(parametros["max_depth"])
    )
    modelo.fit(conjunto.X_entrenamiento, y_entrenamiento_01)
    pred_01 = modelo.predict(conjunto.X_prueba)
    pred = pd.Series(np.where(pred_01 == 1, 1, -1), index=conjunto.X_prueba.index, name="signal")
    importancias = pd.DataFrame(
        {"variable": conjunto.X_entrenamiento.columns, "importancia": modelo.feature_importances_}
    ).sort_values("importancia", ascending=False, ignore_index=True)
    return ResultadoXGBoost(estimador=modelo, y_pred_prueba=pred, interpretacion=importancias)


class ResultadoXGBoost:
    def __init__(self, estimador, y_pred_prueba: pd.Series, interpretacion: pd.DataFrame):
        self.estimador = estimador
        self.y_pred_prueba = y_pred_prueba
        self.interpretacion = interpretacion
