"""Lectura y normalización de los conjuntos de datos del catálogo.

Normalizaciones que aplica siempre:
- Las fechas se interpretan con el formato declarado en el catálogo, se llevan
  a UTC y se descarta la zona horaria (queda la hora de reloj original).
- El índice de fechas queda ordenado de forma ascendente y con nombre "fecha".
- Se descartan las columnas declaradas como sobrantes (por ejemplo, un número
  de fila guardado por error en el CSV).

La lectura no elimina filas con nulos salvo que se pida de forma explícita.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from mltrading.config import settings

from .catalog import Especificacion, obtener


@dataclass
class InformeCarga:
    archivo: str
    filas: int
    columnas: int
    filas_con_nulos: int
    nulos_por_columna: dict[str, int]
    indice_ordenado: bool
    indice_duplicados: int
    inicio: pd.Timestamp | None
    fin: pd.Timestamp | None
    filas_descartadas: int = 0
    segundos: float = 0.0
    columnas_presentes: tuple[str, ...] = field(default_factory=tuple)


def _fechas(valores, formato: str | None) -> pd.DatetimeIndex:
    convertidas = pd.to_datetime(valores, format=formato, utc=True)
    return pd.DatetimeIndex(convertidas).tz_localize(None)


def _leer_crudo(spec: Especificacion) -> pd.DataFrame:
    ruta = settings.DIRECTORIO_DATOS / spec.archivo
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta}")
    if spec.origen_fecha in {"indice", "etiquetas"}:
        df = pd.read_csv(ruta, index_col=0)
    else:
        df = pd.read_csv(ruta)
    if spec.columnas_descartadas:
        df = df.drop(columns=list(spec.columnas_descartadas), errors="ignore")
    return df


def cargar(archivo: str, descartar_nulos: bool = False) -> tuple[pd.DataFrame, InformeCarga]:
    """Devuelve el DataFrame normalizado y un informe de la lectura."""
    spec = obtener(archivo)
    t0 = time.perf_counter()
    df = _leer_crudo(spec)

    if spec.origen_fecha == "columna":
        df.index = _fechas(df.pop(spec.col_fecha), spec.formato_fecha)
        df.index.name = "fecha"
    elif spec.origen_fecha == "indice":
        df.index = _fechas(df.index, spec.formato_fecha)
        df.index.name = "fecha"
    elif spec.origen_fecha == "etiquetas" and spec.nombre_indice:
        df.index.name = spec.nombre_indice

    ordenado = bool(df.index.is_monotonic_increasing) if spec.tiene_fechas else True
    if spec.tiene_fechas and not ordenado:
        df = df.sort_index()
    duplicados = int(df.index.duplicated().sum()) if spec.origen_fecha != "ninguno" else 0

    filas_leidas = len(df)
    con_nulos = df.isna().any(axis=1)
    nulos_col = {c: int(n) for c, n in df.isna().sum().items() if n > 0}
    n_con_nulos = int(con_nulos.sum())
    descartadas = 0
    if descartar_nulos and n_con_nulos:
        df = df.loc[~con_nulos]
        descartadas = filas_leidas - len(df)

    informe = InformeCarga(
        archivo=archivo,
        filas=filas_leidas,
        columnas=df.shape[1],
        filas_con_nulos=n_con_nulos,
        nulos_por_columna=nulos_col,
        indice_ordenado=ordenado,
        indice_duplicados=duplicados,
        inicio=df.index.min() if spec.tiene_fechas and len(df) else None,
        fin=df.index.max() if spec.tiene_fechas and len(df) else None,
        filas_descartadas=descartadas,
        segundos=time.perf_counter() - t0,
        columnas_presentes=tuple(str(c) for c in df.columns),
    )
    return df, informe
