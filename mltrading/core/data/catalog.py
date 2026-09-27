"""Catálogo declarativo de los 16 conjuntos de datos de data_modules.

Cada entrada describe cómo leer y validar un archivo, y en qué páginas se
usará. La lectura y la validación se apoyan solo en este catálogo, de modo que
añadir o corregir un conjunto de datos no exige tocar el resto del código.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ORIGINAL = "Datos originales"
PREDETERMINADO = "Artefacto predeterminado"

FMT_DIA_PRIMERO = "%d-%m-%Y"


@dataclass(frozen=True)
class Especificacion:
    archivo: str
    titulo: str
    tipo: str
    descripcion: str
    frecuencia: str
    columnas: tuple[str, ...]
    # "indice": la primera columna es el índice de fechas. "columna": la fecha
    # está en la columna `col_fecha`. "etiquetas": el índice son etiquetas de
    # texto (por ejemplo, tickers). "ninguno": sin índice significativo.
    origen_fecha: str = "indice"
    col_fecha: str | None = None
    formato_fecha: str | None = None
    columnas_descartadas: tuple[str, ...] = ()
    nombre_indice: str | None = None
    columna_grafico: str | None = None
    paginas: tuple[str, ...] = field(default_factory=tuple)

    @property
    def tiene_fechas(self) -> bool:
        return self.origen_fecha in {"indice", "columna"}


_VARIABLES = ("pct_change", "pct_change2", "pct_change5", "rsi", "adx", "corr", "volatility")
_OHLCV_MIN = ("open", "high", "low", "close", "volume")
_OHLCV_MAY = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
_TICKERS_PCA = (
    "GOOG", "GOOGL", "AMZN", "MA", "BA", "C", "ABT", "CRM", "COST", "ACN",
    "AVGO", "MMM", "CVS", "FIS", "SYK", "MDLZ", "CI", "CME", "ISRG", "COP",
)

CATALOGO: tuple[Especificacion, ...] = (
    Especificacion(
        archivo="JPM_2017_2019.csv",
        titulo="JPMorgan, barras de 15 minutos",
        tipo=ORIGINAL,
        descripcion="Apertura, máximo, mínimo, cierre y volumen de JPMorgan entre 2017 y 2019.",
        frecuencia="15 minutos",
        columnas=_OHLCV_MIN,
        columna_grafico="close",
        paginas=("Datos", "Variables y objetivo", "Backtesting", "Operación en vivo"),
    ),
    Especificacion(
        archivo="JPM_features_2017_2019.csv",
        titulo="JPMorgan, variables (completo)",
        tipo=PREDETERMINADO,
        descripcion="Las siete variables predeterminadas calculadas sobre todo el periodo.",
        frecuencia="15 minutos",
        columnas=_VARIABLES,
        columna_grafico="rsi",
        paginas=("Variables y objetivo", "División de datos"),
    ),
    Especificacion(
        archivo="JPM_features_training_2017_2019.csv",
        titulo="JPMorgan, variables (entrenamiento)",
        tipo=PREDETERMINADO,
        descripcion="Tramo de entrenamiento de las variables (primer 80 % del periodo).",
        frecuencia="15 minutos",
        columnas=_VARIABLES,
        columna_grafico="rsi",
        paginas=("División de datos", "Modelos supervisados"),
    ),
    Especificacion(
        archivo="JPM_features_testing_2017_2019.csv",
        titulo="JPMorgan, variables (prueba)",
        tipo=PREDETERMINADO,
        descripcion="Tramo de prueba de las variables (último 20 % del periodo).",
        frecuencia="15 minutos",
        columnas=_VARIABLES,
        columna_grafico="rsi",
        paginas=("División de datos", "Modelos supervisados", "Evaluación"),
    ),
    Especificacion(
        archivo="JPM_target_2017_2019.csv",
        titulo="JPMorgan, objetivo (completo)",
        tipo=PREDETERMINADO,
        descripcion="Señal binaria: 1 si el cierre siguiente es mayor, 0 en caso contrario.",
        frecuencia="15 minutos",
        columnas=("signal",),
        paginas=("Variables y objetivo",),
    ),
    Especificacion(
        archivo="JPM_target_training_2017_2019.csv",
        titulo="JPMorgan, objetivo (entrenamiento)",
        tipo=PREDETERMINADO,
        descripcion="Señal objetivo del tramo de entrenamiento.",
        frecuencia="15 minutos",
        columnas=("signal",),
        paginas=("División de datos", "Modelos supervisados"),
    ),
    Especificacion(
        archivo="JPM_target_testing_2017_2019.csv",
        titulo="JPMorgan, objetivo (prueba)",
        tipo=PREDETERMINADO,
        descripcion="Señal objetivo del tramo de prueba.",
        frecuencia="15 minutos",
        columnas=("signal",),
        paginas=("División de datos", "Evaluación"),
    ),
    Especificacion(
        archivo="JPM_predicted_2017_2019.csv",
        titulo="JPMorgan, señales predichas",
        tipo=PREDETERMINADO,
        descripcion="Señales que produce el Random Forest predeterminado sobre el tramo de prueba.",
        frecuencia="15 minutos",
        columnas=("signal",),
        paginas=("Evaluación", "Backtesting"),
    ),
    Especificacion(
        archivo="RELIANCE.NS.csv",
        titulo="Reliance Industries, diario",
        tipo=ORIGINAL,
        descripcion="Precios diarios de Reliance Industries (1996 a 2018). Contiene filas con valores nulos.",
        frecuencia="Diaria",
        columnas=_OHLCV_MAY,
        origen_fecha="columna",
        col_fecha="Date",
        columna_grafico="Adj Close",
        paginas=("XGBoost",),
    ),
    Especificacion(
        archivo="coca_cola_price.csv",
        titulo="Coca-Cola, diario",
        tipo=ORIGINAL,
        descripcion="Precios diarios de Coca-Cola entre 2019 y 2021.",
        frecuencia="Diaria",
        columnas=_OHLCV_MAY,
        origen_fecha="columna",
        col_fecha="Date",
        columna_grafico="Adj Close",
        paginas=("XGBoost",),
    ),
    Especificacion(
        archivo="jpm_and_bac_price.csv",
        titulo="JPMorgan y Bank of America, 2020 a 2021",
        tipo=ORIGINAL,
        descripcion="Cierres diarios de ambos bancos y rendimiento diario de Bank of America.",
        frecuencia="Diaria",
        columnas=("Close_BAC", "Close_JPM", "daily_returns_BAC"),
        origen_fecha="columna",
        col_fecha="Date",
        columna_grafico="Close_JPM",
        paginas=("Regresión lineal",),
    ),
    Especificacion(
        archivo="jpm_and_bac_price_2019.csv",
        titulo="JPMorgan y Bank of America, 2019",
        tipo=ORIGINAL,
        descripcion="Cierre de Bank of America, JPMorgan observado y valor ajustado por la regresión.",
        frecuencia="Diaria",
        columnas=("BAC Close", "Observed JPM", "Predicted JPM_BAC"),
        origen_fecha="columna",
        col_fecha="Date",
        formato_fecha=FMT_DIA_PRIMERO,
        columna_grafico="Observed JPM",
        paginas=("Regresión lineal",),
    ),
    Especificacion(
        archivo="predicted_jpm_and_nestle_price_2019.csv",
        titulo="JPMorgan y Nestlé, 2019",
        tipo=ORIGINAL,
        descripcion="JPMorgan observado, cierre de Nestlé y valor ajustado por la regresión.",
        frecuencia="Diaria",
        columnas=("Observed JPM", "Nestle Close", "Predicted JPM_Nestle"),
        origen_fecha="columna",
        col_fecha="Date",
        formato_fecha=FMT_DIA_PRIMERO,
        columna_grafico="Nestle Close",
        paginas=("Regresión lineal",),
    ),
    Especificacion(
        archivo="pca.csv",
        titulo="Veinte acciones, cierres diarios",
        tipo=ORIGINAL,
        descripcion="Precios de cierre diarios de 20 acciones de Estados Unidos entre 2018 y 2019.",
        frecuencia="Diaria",
        columnas=_TICKERS_PCA,
        origen_fecha="columna",
        col_fecha="Date",
        columna_grafico="GOOG",
        paginas=("PCA y t-SNE", "XGBoost"),
    ),
    Especificacion(
        archivo="sample_stocks.csv",
        titulo="Doce acciones, ROE y Beta",
        tipo=ORIGINAL,
        descripcion="Rentabilidad sobre el patrimonio (ROE) y Beta de 12 acciones.",
        frecuencia="Sin fechas",
        columnas=("ROE(%)", "Beta"),
        origen_fecha="etiquetas",
        nombre_indice="ticker",
        paginas=("K-Means", "Clustering jerárquico"),
    ),
    Especificacion(
        archivo="stock_list.csv",
        titulo="Lista de símbolos y capitalización",
        tipo=ORIGINAL,
        descripcion="89 símbolos con su capitalización de mercado. Se usa como referencia visual.",
        frecuencia="Sin fechas",
        columnas=("Symbols", "marketcap"),
        origen_fecha="ninguno",
        columnas_descartadas=("0",),
        paginas=("PCA y t-SNE",),
    ),
)

POR_ARCHIVO = {e.archivo: e for e in CATALOGO}


def obtener(archivo: str) -> Especificacion:
    try:
        return POR_ARCHIVO[archivo]
    except KeyError as exc:
        raise KeyError(f"El archivo '{archivo}' no está en el catálogo de datos.") from exc
