"""Configuración de página y estilos base.

Los estilos propios se limitan a lo que la configuración de temas de Streamlit
no cubre (ancho de lectura, cifras tabulares, muestras de color). Solo usan
selectores estables y heredan los colores del tema activo.
"""
from __future__ import annotations

import streamlit as st

from mltrading.config import strings as S

_CSS = """
<style>
:root { font-variant-numeric: tabular-nums; }
[data-testid="stMainBlockContainer"] {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}
.mlt-lead {
    font-size: 1.125rem;
    line-height: 1.6;
    max-width: 68ch;
    margin: 0 0 1.25rem 0;
    opacity: 0.85;
}
.mlt-muestras { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.5rem 0 1rem 0; }
.mlt-muestra {
    width: 8.5rem;
    border: 1px solid rgba(128, 138, 158, 0.45);
    border-radius: 0.5rem;
    overflow: hidden;
    font-size: 0.8125rem;
    line-height: 1.35;
}
.mlt-muestra > div:first-child { height: 3rem; }
.mlt-muestra > div:last-child { padding: 0.4rem 0.6rem; }
.mlt-muestra b { display: block; font-weight: 600; }
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding-top: 1.5rem; padding-left: 1rem; padding-right: 1rem; }
    [data-testid="stMainBlockContainer"] h1 { font-size: 1.85rem; }
    /* Indicadores en dos columnas en pantallas estrechas */
    .st-key-indicadores [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.75rem; }
    .st-key-indicadores [data-testid="stColumn"] { min-width: calc(50% - 0.5rem); flex: 1 1 calc(50% - 0.5rem); }
    .mlt-lead { font-size: 1.0625rem; }
    .mlt-muestra { width: calc(50% - 0.4rem); }
}
</style>
"""


def configurar_pagina() -> None:
    st.set_page_config(
        page_title=S.APP_NOMBRE,
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="auto"  # abierta en escritorio, cerrada en móvil,
    )


def inyectar_estilos() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
