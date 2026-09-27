"""Página de inicio."""
from __future__ import annotations

import streamlit as st

from mltrading.config import atribucion, roadmap
from mltrading.config import strings as S
from mltrading.core.data.catalog import CATALOGO
from mltrading.ui import charts
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import format as f

_INSIGNIA = {
    roadmap.COMPLETADA: ":green-badge[Completada]",
    roadmap.EN_CURSO: ":blue-badge[En curso]",
    roadmap.PENDIENTE: ":gray-badge[Pendiente]",
}


def render() -> None:
    c.encabezado(S.APP_NOMBRE, S.INICIO_LEAD)

    jpm, informe_jpm = dc.cargar("JPM_2017_2019.csv")
    _, informe_pca = dc.cargar("pca.csv")

    c.seccion(S.INICIO_GRAFICO_TITULO, S.INICIO_GRAFICO_AYUDA)
    charts.mostrar(
        charts.linea({"Cierre": jpm["close"]}, altura=380, selector_rango=True, titulo_y="Precio (USD)"),
        clave="inicio_jpm",
    )

    with st.container(key="indicadores"):
        columnas = st.columns(4)
        with columnas[0]:
            c.indicador(S.INICIO_KPI_CONJUNTOS, str(len(CATALOGO)))
        with columnas[1]:
            c.indicador(S.INICIO_KPI_BARRAS, f.entero(informe_jpm.filas))
        with columnas[2]:
            c.indicador(S.INICIO_KPI_ACCIONES, str(informe_pca.columnas))
        with columnas[3]:
            c.indicador(S.INICIO_KPI_PERIODO, f"{informe_jpm.inicio:%Y} a {informe_jpm.fin:%Y}")

    izquierda, derecha = st.columns([3, 2], gap="large")
    with izquierda:
        c.seccion(S.INICIO_ESTADO_TITULO, S.INICIO_ESTADO_AYUDA)
        for fase in roadmap.FASES:
            with st.container(border=True):
                st.markdown(f"**Fase {fase.numero}. {fase.nombre}**  {_INSIGNIA[fase.estado]}")
                st.caption(fase.alcance)
    with derecha:
        c.seccion(S.INICIO_APARIENCIA_TITULO)
        st.write(S.INICIO_APARIENCIA_TEXTO)
        c.aviso_educativo()

    st.divider()
    st.caption(atribucion.TEXTO)
