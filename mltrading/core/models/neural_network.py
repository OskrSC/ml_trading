"""Red neuronal (capítulo 15), con una variante de scikit-learn.

El capítulo 15 construye una red con Keras: dos capas ocultas de 128 neuronas
con activación ReLU, una capa de salida sigmoide, optimizador Adam, 7 épocas y
lotes de 20 casos. El perfil de despliegue elegido para esta aplicación es
ligero y no incluye TensorFlow ni Keras (ver docs/decisiones.md, punto 11), así
que aquí se usa `MLPClassifier` de scikit-learn con una arquitectura
equivalente. No es una réplica exacta: usa entropía cruzada en lugar del error
cuadrático medio que emplea el código de origen. El código de origen tampoco
fija una semilla para Keras o numpy (solo para el módulo `random` de Python),
así que sus resultados no son exactamente reproducibles entre ejecuciones; los
de esta variante sí lo son.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .base import EspecificacionModelo, ResultadoEntrenamiento
from .params import ParametroEntero

PARAMETROS = (
    ParametroEntero("hidden_units", "Neuronas por capa oculta", 128, 4, 256,
                    "El capítulo 15 usa dos capas ocultas de 128 neuronas."),
    ParametroEntero("max_iter", "Épocas", 7, 1, 200, "El capítulo 15 entrena durante 7 épocas."),
    ParametroEntero("batch_size", "Tamaño de lote", 20, 1, 512, "El capítulo 15 usa lotes de 20 casos."),
    ParametroEntero("random_state", "Semilla", 42, 0, 9999),
)


def entrenar(X_entrenamiento: pd.DataFrame, y_entrenamiento: pd.Series, X_prueba: pd.DataFrame,
             parametros: dict[str, Any]) -> ResultadoEntrenamiento:
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier  # importación diferida

    unidades = int(parametros["hidden_units"])
    modelo = MLPClassifier(
        hidden_layer_sizes=(unidades, unidades), activation="relu", solver="adam",
        max_iter=int(parametros["max_iter"]), batch_size=int(parametros["batch_size"]),
        random_state=int(parametros["random_state"]),
    )
    notas: tuple[str, ...] = ()
    with warnings.catch_warnings(record=True) as capturadas:
        warnings.simplefilter("always", ConvergenceWarning)
        modelo.fit(X_entrenamiento, y_entrenamiento)
        if any(issubclass(w.category, ConvergenceWarning) for w in capturadas):
            notas = (
                "El entrenamiento se detuvo en el número de épocas indicado sin converger del todo. "
                "Esto también ocurre con la red del capítulo 15, entrenada con pocas épocas a propósito.",
            )
    pred = pd.Series(modelo.predict(X_prueba), index=X_prueba.index, name="signal")
    proba = modelo.predict_proba(X_prueba)[:, 1]
    return ResultadoEntrenamiento(estimador=modelo, y_pred_prueba=pred, probabilidades_prueba=proba, notas=notas)


ESPECIFICACION = EspecificacionModelo(
    id="neural_network",
    nombre="Red neuronal (variante scikit-learn)",
    capitulo="Capítulo 15",
    familia="Redes neuronales",
    descripcion=(
        "Variante con scikit-learn del capítulo 15: dos capas ocultas de 128 neuronas con activación ReLU. "
        "El capítulo original usa Keras; esta aplicación usa un perfil de despliegue sin TensorFlow."
    ),
    parametros=PARAMETROS,
    entrenar=entrenar,
    necesita_escalado=True,
)
