"""Naive Bayes de Bernoulli (capítulo 11)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .base import EspecificacionModelo, ResultadoEntrenamiento

PARAMETROS: tuple = ()  # el capítulo 11 usa BernoulliNB con sus valores por defecto


def construir_estimador(parametros: dict[str, Any]):
    from sklearn.naive_bayes import BernoulliNB  # importación diferida

    return BernoulliNB()


def entrenar(X_entrenamiento: pd.DataFrame, y_entrenamiento: pd.Series, X_prueba: pd.DataFrame,
             parametros: dict[str, Any]) -> ResultadoEntrenamiento:
    modelo = construir_estimador(parametros)
    modelo.fit(X_entrenamiento, y_entrenamiento)
    pred = pd.Series(modelo.predict(X_prueba), index=X_prueba.index, name="signal")
    proba = modelo.predict_proba(X_prueba)[:, 1]
    return ResultadoEntrenamiento(estimador=modelo, y_pred_prueba=pred, probabilidades_prueba=proba)


ESPECIFICACION = EspecificacionModelo(
    id="naive_bayes",
    nombre="Naive Bayes",
    capitulo="Capítulo 11",
    familia="Modelos probabilísticos",
    descripcion=(
        "Estima la probabilidad de cada señal a partir de las variables tratadas como independientes entre sí. "
        "Es un modelo simple y rápido de entrenar, útil como referencia."
    ),
    parametros=PARAMETROS,
    entrenar=entrenar,
    necesita_escalado=False,
)
