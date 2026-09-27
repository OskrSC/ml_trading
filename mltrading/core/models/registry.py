"""Registro de modelos disponibles."""
from __future__ import annotations

from . import decision_tree, logistic_regression, naive_bayes, neural_network, random_forest
from .base import EspecificacionModelo

MODELOS: tuple[EspecificacionModelo, ...] = (
    random_forest.ESPECIFICACION,
    logistic_regression.ESPECIFICACION,
    naive_bayes.ESPECIFICACION,
    decision_tree.ESPECIFICACION,
    neural_network.ESPECIFICACION,
)
POR_ID = {m.id: m for m in MODELOS}


def obtener(id_modelo: str) -> EspecificacionModelo:
    try:
        return POR_ID[id_modelo]
    except KeyError as exc:
        raise KeyError(f"No existe el modelo '{id_modelo}' en el registro.") from exc
