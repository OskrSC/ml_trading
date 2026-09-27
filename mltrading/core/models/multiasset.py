"""Conjunto de datos multiactivo para XGBoost (capítulo 14).

El código de origen descarga cinco acciones estadounidenses con `yfinance`
(AAPL, AMZN, NFLX, WMT, MSFT). Esta aplicación no descarga datos en tiempo de
ejecución por defecto (ver fase 0), así que el universo predeterminado se arma
solo con archivos de `data_modules/`: dos series OHLCV completas y tres
columnas de precio de `pca.csv`. La lógica de variables, objetivo y división
reproduce la del código de origen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mltrading.config import settings

VENTANAS = tuple(range(10, 60, 5))  # 10, 15, ..., 55: igual que range(10, 60, 5) del código de origen


@dataclass(frozen=True)
class EspecificacionActivo:
    id: str
    nombre: str
    archivo: str
    columna: str
    descartar_nulos: bool = False


UNIVERSO_PREDETERMINADO: tuple[EspecificacionActivo, ...] = (
    EspecificacionActivo("RELIANCE", "Reliance Industries", "RELIANCE.NS.csv", "Adj Close", descartar_nulos=True),
    EspecificacionActivo("KO", "Coca-Cola", "coca_cola_price.csv", "Adj Close"),
    EspecificacionActivo("GOOG", "Alphabet (Google)", "pca.csv", "GOOG"),
    EspecificacionActivo("AMZN", "Amazon", "pca.csv", "AMZN"),
    EspecificacionActivo("MMM", "3M", "pca.csv", "MMM"),
)


def nombres_columnas() -> list[str]:
    columnas = []
    for r in VENTANAS:
        columnas += [f"pct_change_{r}", f"std_{r}"]
    return columnas


def _precio(spec: EspecificacionActivo) -> pd.Series:
    from mltrading.core.data.loaders import cargar  # importación diferida: evita ciclos

    df, _ = cargar(spec.archivo, descartar_nulos=spec.descartar_nulos)
    return df[spec.columna].rename(spec.id)


@dataclass
class DatosActivo:
    id: str
    nombre: str
    variables: pd.DataFrame  # columnas de nombres_columnas()
    objetivo: pd.Series  # 1 o -1
    retorno_diario_siguiente: pd.Series
    filas_iniciales: int


def construir_variables_activo(spec: EspecificacionActivo) -> DatosActivo:
    """Reproduce el bloque `for stock_name in stock_list` del código de origen para un activo."""
    precio = _precio(spec)
    cambio = precio.pct_change()
    d = pd.DataFrame(index=precio.index)
    for r in VENTANAS:
        d[f"pct_change_{r}"] = cambio.rolling(r).sum()
        d[f"std_{r}"] = cambio.rolling(r).std()
    retorno_siguiente = cambio.shift(-1)
    objetivo = pd.Series(np.where(retorno_siguiente > 0, 1, -1), index=d.index, name="signal")

    completo = pd.concat([d, retorno_siguiente.rename("retorno_siguiente"), objetivo], axis=1).dropna()
    return DatosActivo(
        id=spec.id,
        nombre=spec.nombre,
        variables=completo[nombres_columnas()],
        objetivo=completo["signal"].astype(int),
        retorno_diario_siguiente=completo["retorno_siguiente"],
        filas_iniciales=len(precio),
    )


@dataclass
class ConjuntoMultiactivo:
    datos_por_activo: dict[str, DatosActivo]
    X_entrenamiento: pd.DataFrame
    X_prueba: pd.DataFrame
    y_entrenamiento: pd.Series
    y_prueba: pd.Series
    activo_entrenamiento: pd.Series  # id del activo, mismo índice posicional que X_entrenamiento
    activo_prueba: pd.Series
    proporcion: float


def construir_conjunto(activos: tuple[EspecificacionActivo, ...], proporcion: float = 0.8) -> ConjuntoMultiactivo:
    """División por activo (80/20 cronológico dentro de cada uno) y luego concatenación, en el orden dado."""
    if not activos:
        raise ValueError("Debe indicarse al menos un activo.")
    if not 0.05 <= proporcion <= 0.95:
        raise ValueError("La proporción de entrenamiento debe estar entre 0,05 y 0,95.")

    datos_por_activo: dict[str, DatosActivo] = {}
    partes_X_tr, partes_X_te, partes_y_tr, partes_y_te = [], [], [], []
    id_tr, id_te = [], []
    for spec in activos:
        datos = construir_variables_activo(spec)
        datos_por_activo[spec.id] = datos
        corte = int(len(datos.variables) * proporcion)
        partes_X_tr.append(datos.variables.iloc[:corte])
        partes_X_te.append(datos.variables.iloc[corte:])
        partes_y_tr.append(datos.objetivo.iloc[:corte])
        partes_y_te.append(datos.objetivo.iloc[corte:])
        id_tr += [spec.id] * corte
        id_te += [spec.id] * (len(datos.variables) - corte)

    X_entrenamiento = pd.concat(partes_X_tr)
    X_prueba = pd.concat(partes_X_te)
    return ConjuntoMultiactivo(
        datos_por_activo=datos_por_activo,
        X_entrenamiento=X_entrenamiento,
        X_prueba=X_prueba,
        y_entrenamiento=pd.concat(partes_y_tr),
        y_prueba=pd.concat(partes_y_te),
        activo_entrenamiento=pd.Series(id_tr, index=X_entrenamiento.index),
        activo_prueba=pd.Series(id_te, index=X_prueba.index),
        proporcion=proporcion,
    )
