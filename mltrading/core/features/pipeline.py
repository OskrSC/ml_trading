"""Preparación completa de variables: candidatas, estacionariedad, correlación y selección."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import seleccion
from .config import ConfigVariables
from .construccion import Candidatas, construir_candidatas


@dataclass
class ResultadoVariables:
    config: ConfigVariables
    candidatas: Candidatas
    tabla_adf: pd.DataFrame  # variable, estadistico, p_valor, retardos, estacionaria
    descartadas_adf: list[str]
    pares: pd.DataFrame  # pares por encima del umbral de correlación
    sugeridas_correlacion: list[str]
    descartadas_correlacion: list[str] = field(default_factory=list)

    @property
    def estacionarias(self) -> list[str]:
        return list(self.tabla_adf.loc[self.tabla_adf["estacionaria"], "variable"])

    @property
    def variables_finales(self) -> list[str]:
        return [c for c in self.estacionarias if c not in self.descartadas_correlacion]

    @property
    def X(self) -> pd.DataFrame:
        return self.candidatas.variables[self.variables_finales]

    @property
    def y(self) -> pd.Series:
        return self.candidatas.objetivo

    def con_descartes(self, descartes: list[str]) -> "ResultadoVariables":
        """Copia del resultado con otra elección de variables descartadas por correlación."""
        validos = [d for d in descartes if d in self.estacionarias]
        return ResultadoVariables(
            config=self.config,
            candidatas=self.candidatas,
            tabla_adf=self.tabla_adf,
            descartadas_adf=self.descartadas_adf,
            pares=self.pares,
            sugeridas_correlacion=self.sugeridas_correlacion,
            descartadas_correlacion=validos,
        )


def preparar_variables(
    ohlcv: pd.DataFrame,
    config: ConfigVariables,
    tabla_adf: pd.DataFrame | None = None,
) -> ResultadoVariables:
    """Ejecuta el flujo completo.

    `tabla_adf` permite reutilizar una prueba ADF ya calculada (la prueba tarda
    unos 13 segundos con los datos de JPMorgan). Debe corresponder a las mismas
    variables candidatas.
    """
    candidatas = construir_candidatas(ohlcv, config)
    if tabla_adf is None:
        tabla_adf = seleccion.calcular_adf(candidatas.variables)
    tabla = seleccion.clasificar_adf(tabla_adf, config.umbral_adf)
    estacionarias = list(tabla.loc[tabla["estacionaria"], "variable"])
    descartadas_adf = [v for v in tabla["variable"] if v not in estacionarias]

    pares = seleccion.pares_correlacionados(candidatas.variables[estacionarias], config.umbral_correlacion)
    sugeridas = seleccion.sugerir_descartes(pares)
    return ResultadoVariables(
        config=config,
        candidatas=candidatas,
        tabla_adf=tabla,
        descartadas_adf=descartadas_adf,
        pares=pares,
        sugeridas_correlacion=sugeridas,
        descartadas_correlacion=list(sugeridas),
    )
