"""Artefactos predeterminados de la preparación de variables.

La prueba ADF sobre los datos de JPMorgan tarda unos 13 segundos, así que su
resultado con la configuración predeterminada se guarda como artefacto. Una
prueba comprueba que el artefacto coincide con recalcularlo.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mltrading.config import settings

ARCHIVO_ADF = "adf_predeterminado.json"
COLUMNAS = ["variable", "estadistico", "p_valor", "retardos"]


def ruta_adf() -> Path:
    return settings.DIRECTORIO_ARTEFACTOS / ARCHIVO_ADF


def guardar_adf(tabla: pd.DataFrame, barras_por_dia: int) -> Path:
    ruta = ruta_adf()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    carga = {"barras_por_dia": barras_por_dia, "filas": tabla[COLUMNAS].to_dict(orient="records")}
    ruta.write_text(json.dumps(carga, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def cargar_adf() -> pd.DataFrame | None:
    ruta = ruta_adf()
    if not ruta.exists():
        return None
    carga = json.loads(ruta.read_text(encoding="utf-8"))
    return pd.DataFrame(carga["filas"], columns=COLUMNAS)
