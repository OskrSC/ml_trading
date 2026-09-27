"""Detección del tema activo.

Streamlit cambia el tema en el navegador de inmediato, pero Python solo se
entera en la siguiente ejecución del script (cualquier interacción o recarga).
Por eso ningún resultado depende de esta función: solo sirve para informar.
"""
from __future__ import annotations

import streamlit as st

CLARO = "claro"
OSCURO = "oscuro"


def tema_actual() -> str:
    try:
        tipo = st.context.theme.type
    except Exception:  # noqa: BLE001 - versiones o contextos sin información de tema
        tipo = None
    return OSCURO if tipo == "dark" else CLARO
