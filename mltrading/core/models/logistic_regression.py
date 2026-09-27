"""Regresión logística (capítulo 10)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .base import EspecificacionModelo, ResultadoEntrenamiento
from .params import ParametroFlotante, ParametroEntero

PARAMETROS = (
    ParametroFlotante("C", "Regularización inversa (C)", 1.0, 0.01, 10.0, paso=0.01, formato="%.2f",
                      ayuda="Valores más bajos regularizan más el modelo. 1,0 es el valor por defecto de la librería."),
    ParametroEntero("max_iter", "Iteraciones máximas", 100, 50, 2000, paso=50,
                    ayuda="Si el modelo no converge, aumente este valor."),
)


def entrenar(X_entrenamiento: pd.DataFrame, y_entrenamiento: pd.Series, X_prueba: pd.DataFrame,
             parametros: dict[str, Any]) -> ResultadoEntrenamiento:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression  # importación diferida
    import warnings

    modelo = LogisticRegression(C=float(parametros["C"]), max_iter=int(parametros["max_iter"]))
    notas: tuple[str, ...] = ()
    with warnings.catch_warnings(record=True) as capturadas:
        warnings.simplefilter("always", ConvergenceWarning)
        modelo.fit(X_entrenamiento, y_entrenamiento)
        if any(issubclass(w.category, ConvergenceWarning) for w in capturadas):
            notas = ("El modelo no convergió con las iteraciones indicadas. Aumente las iteraciones máximas.",)

    pred = pd.Series(modelo.predict(X_prueba), index=X_prueba.index, name="signal")
    proba = modelo.predict_proba(X_prueba)[:, 1]
    coeficientes = pd.DataFrame({"variable": X_entrenamiento.columns, "coeficiente": modelo.coef_[0]})
    coeficientes["importancia"] = coeficientes["coeficiente"].abs()
    coeficientes = coeficientes.sort_values("importancia", ascending=False, ignore_index=True)
    return ResultadoEntrenamiento(
        estimador=modelo, y_pred_prueba=pred, probabilidades_prueba=proba, interpretacion=coeficientes, notas=notas
    )


ESPECIFICACION = EspecificacionModelo(
    id="logistic_regression",
    nombre="Regresión logística",
    capitulo="Capítulo 10",
    familia="Modelos lineales",
    descripcion=(
        "Ajusta una frontera lineal en el espacio de las variables, escaladas, para estimar la probabilidad "
        "de que la señal sea 1. Es el modelo más simple del conjunto y sirve como referencia."
    ),
    parametros=PARAMETROS,
    entrenar=entrenar,
    necesita_escalado=True,
)
