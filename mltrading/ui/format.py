"""Formato numérico y de fechas con convenciones en español."""
from __future__ import annotations

import pandas as pd


def entero(valor: int | float) -> str:
    """19370 -> '19.370'."""
    return f"{int(valor):,}".replace(",", ".")


def decimal(valor: float, decimales: int = 2) -> str:
    """1234.5 -> '1.234,50'."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def fecha(valor: pd.Timestamp | None) -> str:
    return "" if valor is None or pd.isna(valor) else f"{valor:%d/%m/%Y}"


def periodo(inicio: pd.Timestamp | None, fin: pd.Timestamp | None) -> str:
    if inicio is None or fin is None:
        return "Sin fechas"
    return f"{fecha(inicio)} a {fecha(fin)}"
