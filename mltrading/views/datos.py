"""Catálogo de datos: resumen de los 16 conjuntos y detalle de cada uno."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.config import strings as S
from mltrading.core.data.catalog import ORIGINAL, PREDETERMINADO, obtener
from mltrading.core.data.validation import AVISO, ERROR
from mltrading.ui import charts
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import format as f

_ETIQUETA_COLUMNAS = "Descripción de las columnas"


def _resumen_columnas(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for nombre in df.columns:
        serie = df[nombre]
        numerica = pd.api.types.is_numeric_dtype(serie)
        filas.append(
            {
                "Columna": str(nombre),
                "Tipo": str(serie.dtype),
                "Nulos": int(serie.isna().sum()),
                "Mínimo": serie.min() if numerica else None,
                "Máximo": serie.max() if numerica else None,
            }
        )
    return pd.DataFrame(filas)


def _detalle(archivo: str) -> None:
    spec = obtener(archivo)
    df, informe = dc.cargar(archivo)

    st.markdown(f"**{spec.titulo}**")
    st.caption(f"{spec.descripcion} Frecuencia: {spec.frecuencia.lower()}. Periodo: {f.periodo(informe.inicio, informe.fin)}.")

    numericas = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if spec.tiene_fechas and numericas:
        por_defecto = numericas.index(spec.columna_grafico) if spec.columna_grafico in numericas else 0
        elegida = st.selectbox(S.DATOS_GRAFICO_SERIE, numericas, index=por_defecto, key=f"serie_{archivo}")
        charts.mostrar(charts.linea({elegida: df[elegida]}, altura=300), clave=f"grafico_{archivo}")
    else:
        st.caption(S.DATOS_SIN_GRAFICO)

    tab_vista, tab_columnas, tab_calidad = st.tabs([S.DATOS_TAB_VISTA, S.DATOS_TAB_COLUMNAS, S.DATOS_TAB_CALIDAD])
    with tab_vista:
        st.caption("Primeras 10 filas")
        st.dataframe(df.head(10), width="stretch")
        st.caption("Últimas 5 filas")
        st.dataframe(df.tail(5), width="stretch")
    with tab_columnas:
        st.dataframe(_resumen_columnas(df), hide_index=True, width="stretch")
    with tab_calidad:
        for incidencia in dc.incidencias(archivo):
            if incidencia.nivel == ERROR:
                st.error(incidencia.mensaje, icon=":material/error:")
            elif incidencia.nivel == AVISO:
                st.warning(incidencia.mensaje, icon=":material/warning:")
            else:
                st.success(incidencia.mensaje, icon=":material/check_circle:")
        st.caption(f"Lectura y normalización: {f.decimal(informe.segundos * 1000, 1)} ms.")


def render() -> None:
    c.encabezado(S.DATOS_TITULO, S.DATOS_LEAD)

    resumen = dc.resumen_catalogo()
    opciones = [S.DATOS_FILTRO_TODOS, ORIGINAL, PREDETERMINADO]
    filtro = st.segmented_control(S.DATOS_FILTRO_TIPO, opciones, default=opciones[0], key="datos_tipo")
    filtro = filtro or S.DATOS_FILTRO_TODOS
    vista = resumen if filtro == S.DATOS_FILTRO_TODOS else resumen[resumen["Tipo"] == filtro]

    st.dataframe(
        vista,
        hide_index=True,
        width="stretch",
        column_config={
            "Filas": st.column_config.NumberColumn(format="%d"),
            "Columnas": st.column_config.NumberColumn(format="%d"),
            "Filas con nulos": st.column_config.NumberColumn(format="%d"),
            "Inicio": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Fin": st.column_config.DateColumn(format="DD/MM/YYYY"),
        },
    )

    st.divider()
    archivo = st.selectbox(S.DATOS_SELECTOR, list(vista["Archivo"]), key="datos_archivo")
    _detalle(archivo)
