"""Árbol de decisión de clasificación (capítulo 12)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .base import EspecificacionModelo, ResultadoEntrenamiento
from .params import ParametroCategorico, ParametroEntero

PARAMETROS = (
    ParametroEntero("max_depth", "Profundidad máxima", 3, 1, 20, "El capítulo 12 usa una profundidad de 3."),
    ParametroEntero("min_samples_leaf", "Mínimo de casos por hoja", 5, 1, 500,
                    "Evita hojas que memoricen unos pocos casos."),
    ParametroCategorico("criterion", "Criterio de división", "gini", ("gini", "entropy", "log_loss")),
)


def construir_estimador(parametros: dict[str, Any]):
    from sklearn.tree import DecisionTreeClassifier  # importación diferida

    return DecisionTreeClassifier(
        criterion=str(parametros["criterion"]),
        max_depth=int(parametros["max_depth"]),
        min_samples_leaf=int(parametros["min_samples_leaf"]),
    )


def entrenar(X_entrenamiento: pd.DataFrame, y_entrenamiento: pd.Series, X_prueba: pd.DataFrame,
             parametros: dict[str, Any]) -> ResultadoEntrenamiento:
    modelo = construir_estimador(parametros)
    modelo.fit(X_entrenamiento, y_entrenamiento)
    pred = pd.Series(modelo.predict(X_prueba), index=X_prueba.index, name="signal")
    proba = modelo.predict_proba(X_prueba)[:, 1]
    importancias = pd.DataFrame(
        {"variable": X_entrenamiento.columns, "importancia": modelo.feature_importances_}
    ).sort_values("importancia", ascending=False, ignore_index=True)
    return ResultadoEntrenamiento(
        estimador=modelo, y_pred_prueba=pred, probabilidades_prueba=proba, interpretacion=importancias
    )


ESPECIFICACION = EspecificacionModelo(
    id="decision_tree",
    nombre="Árbol de decisión",
    capitulo="Capítulo 12",
    familia="Árboles",
    descripcion=(
        "Divide los datos en preguntas sucesivas sobre las variables hasta llegar a una predicción. "
        "Es fácil de interpretar, aunque un solo árbol tiende a sobreajustar más que un conjunto de árboles."
    ),
    parametros=PARAMETROS,
    entrenar=entrenar,
    necesita_escalado=False,
)
