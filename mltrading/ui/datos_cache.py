"""Envolturas con caché de la capa de datos (solo lectura, compartidas entre sesiones)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.core.data import loaders, summary
from mltrading.core.data.validation import Incidencia, validar
from mltrading.core.data.catalog import obtener


_SEIS_HORAS = 6 * 3600

# El catálogo tiene 16 archivos y `descartar_nulos` es booleano: como mucho 32
# combinaciones. El límite es una red de seguridad, no algo que deba alcanzarse
# en uso normal; el ttl evita mantener los datos en memoria indefinidamente en
# un proceso de servidor de muy larga duración.
@st.cache_data(show_spinner=False, max_entries=32, ttl=_SEIS_HORAS)
def cargar(archivo: str, descartar_nulos: bool = False):
    return loaders.cargar(archivo, descartar_nulos=descartar_nulos)


@st.cache_data(show_spinner=False, max_entries=2, ttl=_SEIS_HORAS)
def resumen_catalogo() -> pd.DataFrame:
    return summary.resumen_catalogo()


def incidencias(archivo: str) -> list[Incidencia]:
    df, informe = cargar(archivo)
    return validar(df, obtener(archivo), informe)
