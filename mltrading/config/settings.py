"""Configuración global de la aplicación (rutas y banderas de ejecución)."""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

# Carpeta con los conjuntos de datos. Se puede cambiar con MLT_DATA_DIR.
DIRECTORIO_DATOS = Path(os.environ.get("MLT_DATA_DIR", RAIZ / "data_modules"))

# Resultados precalculados que sirven de configuración predeterminada.
DIRECTORIO_ARTEFACTOS = Path(os.environ.get("MLT_ARTEFACTOS_DIR", RAIZ / "artefactos"))

_VERDADERO = {"1", "true", "si", "sí", "yes", "on"}


def _secreto(nombre: str) -> str | None:
    """Lee un valor de los secretos de Streamlit si existen (no falla si no hay)."""
    try:
        import streamlit as st

        valor = st.secrets.get(nombre)
        return None if valor is None else str(valor)
    except Exception:  # noqa: BLE001 - sin archivo de secretos o fuera de Streamlit
        return None


def _bandera(nombre: str, por_defecto: bool) -> bool:
    """Orden de prioridad: variable de entorno, secretos de Streamlit, valor por defecto."""
    valor = os.environ.get(nombre)
    if valor is None:
        valor = _secreto(nombre)
    if valor is None:
        return por_defecto
    return valor.strip().lower() in _VERDADERO


# Perfil de despliegue. "ligero" no incluye TensorFlow: la red neuronal usa la
# alternativa de scikit-learn. Perfil elegido para streamlit.app. "completo"
# solo tiene sentido si se instala TensorFlow (ver docs/despliegue_streamlit_cloud.md).
PERFILES = ("ligero", "completo")
PERFIL = os.environ.get("MLT_PERFIL") or _secreto("MLT_PERFIL") or "ligero"
if PERFIL not in PERFILES:
    PERFIL = "ligero"

# Página interna de diagnóstico. Activa durante la fase 0; en el lanzamiento se
# desactiva con MLT_MOSTRAR_DIAGNOSTICO=0 (variable de entorno o secreto del servicio).
MOSTRAR_DIAGNOSTICO = _bandera("MLT_MOSTRAR_DIAGNOSTICO", True)
