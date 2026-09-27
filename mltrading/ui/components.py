"""Componentes reutilizables de la interfaz."""
from __future__ import annotations

from html import escape

import streamlit as st

from mltrading.config import strings as S


def encabezado(titulo: str, descripcion: str | None = None, referencia: str | None = None) -> None:
    st.title(titulo)
    if descripcion:
        st.markdown(f'<p class="mlt-lead">{escape(descripcion)}</p>', unsafe_allow_html=True)
    if referencia:
        st.caption(referencia)


def seccion(titulo: str, ayuda: str | None = None) -> None:
    st.subheader(titulo)
    if ayuda:
        st.caption(ayuda)


def indicador(etiqueta: str, valor: str, ayuda: str | None = None) -> None:
    with st.container(border=True):
        st.metric(etiqueta, valor, help=ayuda)


def aviso_educativo() -> None:
    st.info(S.AVISO_EDUCATIVO, icon=":material/info:")


def muestra_color(nombre: str, valor: str, nota: str = "") -> str:
    """HTML de una muestra de color (el color se escribe como dato, no como estilo global)."""
    return (
        '<div class="mlt-muestra">'
        f'<div style="background:{escape(valor)}"></div>'
        f"<div><b>{escape(nombre)}</b>{escape(valor)}"
        + (f"<br>{escape(nota)}" if nota else "")
        + "</div></div>"
    )
