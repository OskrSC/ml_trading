"""Orquesta el entrenamiento de un modelo del registro, con escalado si hace falta."""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import escalado
from .base import EspecificacionModelo, ResultadoEntrenamiento
from .params import valores_predeterminados


def es_configuracion_predeterminada(spec: EspecificacionModelo, parametros: dict[str, Any]) -> bool:
    return dict(parametros) == valores_predeterminados(spec.parametros)


def entrenar_modelo(
    spec: EspecificacionModelo,
    X_entrenamiento: pd.DataFrame,
    y_entrenamiento: pd.Series,
    X_prueba: pd.DataFrame,
    parametros: dict[str, Any] | None = None,
) -> ResultadoEntrenamiento:
    parametros = dict(parametros) if parametros is not None else valores_predeterminados(spec.parametros)
    if spec.necesita_escalado:
        X_entrenamiento, X_prueba = escalado.escalar(X_entrenamiento, X_prueba)
    return spec.entrenar(X_entrenamiento, y_entrenamiento, X_prueba, parametros)
