"""Reconstruye variables, división y modelo a partir de una receta."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mltrading.core.features.config import ConfigVariables
from mltrading.core.features.division import Division, dividir_cronologica
from mltrading.core.features.pipeline import ResultadoVariables, preparar_variables
from mltrading.core.models.base import ResultadoEntrenamiento
from mltrading.core.models.entrenamiento import entrenar_modelo
from mltrading.core.models.registry import obtener as obtener_modelo

from .schema import RecetaModelo


@dataclass
class ResultadoReceta:
    resultado_variables: ResultadoVariables
    division: Division
    resultado_entrenamiento: ResultadoEntrenamiento


def aplicar_receta(receta: RecetaModelo, ohlcv: pd.DataFrame) -> ResultadoReceta:
    """Recalcula todo desde los datos crudos: nunca se deserializa un modelo guardado."""
    spec = obtener_modelo(receta.id_modelo)  # KeyError si no existe, con mensaje claro
    config_variables = ConfigVariables(
        barras_por_dia=receta.barras_por_dia, umbral_adf=receta.umbral_adf,
        umbral_correlacion=receta.umbral_correlacion,
    )
    resultado_variables = preparar_variables(ohlcv, config_variables)
    resultado_variables = resultado_variables.con_descartes(receta.descartes_correlacion)
    division = dividir_cronologica(resultado_variables.X, resultado_variables.y, receta.proporcion_entrenamiento)
    resultado_entrenamiento = entrenar_modelo(
        spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba, receta.parametros_modelo
    )
    return ResultadoReceta(
        resultado_variables=resultado_variables, division=division, resultado_entrenamiento=resultado_entrenamiento
    )
