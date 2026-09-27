"""Validaciones de calidad sobre un conjunto de datos ya leído."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .catalog import Especificacion
from .loaders import InformeCarga

ERROR = "Error"
AVISO = "Aviso"
INFO = "Información"


@dataclass(frozen=True)
class Incidencia:
    nivel: str
    codigo: str
    mensaje: str


def validar(df: pd.DataFrame, spec: Especificacion, informe: InformeCarga) -> list[Incidencia]:
    incidencias: list[Incidencia] = []

    if informe.filas == 0:
        return [Incidencia(ERROR, "vacio", "El conjunto de datos no contiene filas.")]

    esperadas = set(spec.columnas)
    presentes = set(informe.columnas_presentes)
    faltantes = sorted(esperadas - presentes)
    sobrantes = sorted(presentes - esperadas)
    if faltantes:
        incidencias.append(
            Incidencia(ERROR, "columnas_faltantes", "Faltan columnas esperadas: " + ", ".join(faltantes) + ".")
        )
    if sobrantes:
        incidencias.append(
            Incidencia(AVISO, "columnas_extra", "Hay columnas no previstas en el catálogo: " + ", ".join(sobrantes) + ".")
        )

    if spec.tiene_fechas and not informe.indice_ordenado:
        incidencias.append(
            Incidencia(AVISO, "fechas_desordenadas", "Las fechas venían desordenadas y se ordenaron al cargar.")
        )
    if informe.indice_duplicados:
        incidencias.append(
            Incidencia(AVISO, "indice_duplicado", f"Hay {informe.indice_duplicados} valores de índice repetidos.")
        )

    if informe.filas_con_nulos:
        detalle = ""
        if spec.tiene_fechas:
            filas_nulas = df.index[df.isna().any(axis=1)]
            if len(filas_nulas):
                detalle = f" entre {filas_nulas.min():%Y-%m-%d} y {filas_nulas.max():%Y-%m-%d}"
        incidencias.append(
            Incidencia(
                AVISO,
                "nulos",
                f"{informe.filas_con_nulos} filas contienen valores nulos{detalle}. "
                "Se conservan al cargar y se tratan de forma explícita en cada modelo.",
            )
        )

    if not incidencias:
        incidencias.append(Incidencia(INFO, "sin_incidencias", "No se detectaron incidencias."))
    return incidencias
