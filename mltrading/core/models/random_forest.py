"""Random Forest (capítulos 5 y 13)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .base import EspecificacionModelo, ResultadoEntrenamiento
from .params import ParametroEntero

PARAMETROS = (
    ParametroEntero("n_estimators", "Número de árboles", 3, 1, 300,
                    "El capítulo 5 usa 3 árboles, para poder inspeccionarlos, y el capítulo 13 usa 100 "
                    "(el valor por defecto de la librería)."),
    ParametroEntero("max_depth", "Profundidad máxima", 2, 1, 20, "Profundidad máxima de cada árbol."),
    ParametroEntero("max_features", "Variables por división", 3, 1, 13, "Variables candidatas en cada división."),
    ParametroEntero("random_state", "Semilla", 4, 0, 9999, "Controla el azar del muestreo y de las divisiones."),
)


def construir_estimador(parametros: dict[str, Any], n_variables: int):
    """Devuelve el estimador sin ajustar. Lo usa `entrenar` y también el simulador de
    operación en vivo, que necesita reentrenar sin pasar por la interfaz de ajuste+predicción."""
    from sklearn.ensemble import RandomForestClassifier  # importación diferida

    max_features = min(int(parametros["max_features"]), n_variables)
    return RandomForestClassifier(
        n_estimators=int(parametros["n_estimators"]),
        max_depth=int(parametros["max_depth"]),
        max_features=max_features,
        random_state=int(parametros["random_state"]),
    ), max_features


def entrenar(X_entrenamiento: pd.DataFrame, y_entrenamiento: pd.Series, X_prueba: pd.DataFrame,
             parametros: dict[str, Any]) -> ResultadoEntrenamiento:
    modelo, max_features = construir_estimador(parametros, X_entrenamiento.shape[1])
    modelo.fit(X_entrenamiento, y_entrenamiento)
    pred = pd.Series(modelo.predict(X_prueba), index=X_prueba.index, name="signal")
    proba = modelo.predict_proba(X_prueba)[:, 1]
    importancias = pd.DataFrame(
        {"variable": X_entrenamiento.columns, "importancia": modelo.feature_importances_}
    ).sort_values("importancia", ascending=False, ignore_index=True)
    notas = ()
    if max_features != int(parametros["max_features"]):
        notas = (f"Variables por división ajustado a {max_features}, el máximo disponible.",)
    return ResultadoEntrenamiento(
        estimador=modelo, y_pred_prueba=pred, probabilidades_prueba=proba, interpretacion=importancias, notas=notas
    )


ESPECIFICACION = EspecificacionModelo(
    id="random_forest",
    nombre="Random Forest",
    capitulo="Capítulos 5 y 13",
    familia="Árboles combinados",
    descripcion=(
        "Combina muchos árboles de decisión entrenados sobre muestras distintas y promedia sus votos. "
        "El resultado predeterminado usa solo 3 árboles, para que se puedan inspeccionar uno a uno."
    ),
    parametros=PARAMETROS,
    entrenar=entrenar,
    necesita_escalado=False,
)
