"""Diagnóstico del entorno (fase 0): mide el servidor para decidir el perfil de despliegue."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from mltrading.config import strings as S
from mltrading.core.env import diagnostics as d
from mltrading.core.observabilidad import limpiar_buffer, ultimas_entradas
from mltrading.ui import components as c
from mltrading.ui import format as f
from mltrading.ui.tema import tema_actual

_MODULOS_POR_DEFECTO = ["sklearn", "statsmodels", "xgboost", "talib", "tensorflow"]
_MODULOS_MEDIBLES = ["sklearn", "scipy", "statsmodels", "xgboost", "talib", "plotly", "tensorflow", "keras", "yfinance"]
_CLAVE_CARGAS = "diag_cargas"
_CLAVE_IMPORTACIONES = "diag_importaciones"


def _mb(valor: float | None) -> str:
    return "No disponible" if valor is None else f"{f.decimal(valor, 0)} MB"


def _entorno() -> None:
    c.seccion("Entorno")
    info = d.informacion_entorno()
    tabla = pd.DataFrame(
        [
            ("Perfil de despliegue", info["perfil"]),
            ("Python", info["python"]),
            ("Streamlit", info["streamlit"]),
            ("Sistema", info["sistema"]),
            ("Arquitectura", info["arquitectura"]),
            ("Núcleos visibles", str(info["nucleos"])),
            ("Tema detectado", tema_actual()),
        ],
        columns=["Dato", "Valor"],
    )
    st.dataframe(tabla, hide_index=True, width="stretch")


def _recursos() -> None:
    c.seccion("Recursos", "Se actualizan al pulsar el botón. El límite del contenedor es el que aplica el servicio.")
    r = d.instantanea_recursos()
    columnas = st.columns(4)
    with columnas[0]:
        c.indicador("Memoria del proceso", _mb(r["memoria_proceso_mb"]))
    with columnas[1]:
        c.indicador("Memoria del contenedor", _mb(r["memoria_contenedor_mb"]))
    with columnas[2]:
        limite = r["limite_contenedor_mb"]
        c.indicador("Límite del contenedor", "Sin límite detectado" if limite is None else _mb(limite))
    with columnas[3]:
        activo = r["segundos_activo"]
        c.indicador("Tiempo activo", "No disponible" if activo is None else f"{f.decimal(activo / 60, 1)} min")
    st.button("Actualizar mediciones", key="diag_actualizar", icon=":material/refresh:")


def _paquetes() -> None:
    c.seccion("Librerías", "Los paquetes opcionales pueden figurar como no instalados sin que sea un problema.")
    st.dataframe(pd.DataFrame(d.estado_paquetes()), hide_index=True, width="stretch")


def _carga_datos() -> None:
    c.seccion("Carga de los conjuntos de datos", "Lee los 16 archivos y anota el tiempo y el cambio aproximado de memoria.")
    if st.button("Medir la carga de datos", key="diag_medir_cargas", icon=":material/speed:"):
        with st.spinner("Leyendo los conjuntos de datos"):
            st.session_state[_CLAVE_CARGAS] = d.medir_carga_datasets()
    filas = st.session_state.get(_CLAVE_CARGAS)
    if filas:
        tabla = pd.DataFrame(filas).rename(
            columns={
                "archivo": "Archivo",
                "filas": "Filas",
                "segundos": "Segundos",
                "memoria_df_mb": "Tamaño en memoria (MB)",
                "delta_rss_mb": "Cambio de memoria (MB)",
            }
        )
        st.dataframe(tabla, hide_index=True, width="stretch")
        st.caption(f"Tiempo total de lectura: {f.decimal(sum(x['segundos'] for x in filas), 2)} s.")


def _importaciones() -> None:
    c.seccion(
        "Importación de librerías",
        "Cada librería se importa en un subproceso aparte. Si el servicio lo termina por falta de memoria, "
        "la aplicación sigue funcionando y el resultado queda registrado.",
    )
    elegidos = st.multiselect(
        "Librerías a medir", _MODULOS_MEDIBLES, default=_MODULOS_POR_DEFECTO, key="diag_modulos"
    )
    if st.button("Medir importación", key="diag_medir_import", icon=":material/timer:", disabled=not elegidos):
        resultados = []
        avance = st.progress(0.0, text="Midiendo")
        for i, modulo in enumerate(elegidos, start=1):
            avance.progress((i - 1) / len(elegidos), text=f"Midiendo {modulo}")
            resultados.append(d.medir_importacion(modulo))
        avance.empty()
        st.session_state[_CLAVE_IMPORTACIONES] = resultados
    filas = st.session_state.get(_CLAVE_IMPORTACIONES)
    if filas:
        tabla = pd.DataFrame(filas).rename(
            columns={"modulo": "Librería", "estado": "Estado", "segundos": "Segundos", "memoria_mb": "Memoria añadida (MB)", "detalle": "Detalle"}
        )
        st.dataframe(tabla, hide_index=True, width="stretch")


def _informe() -> None:
    c.seccion("Informe", "Descargue las mediciones para registrarlas en docs/informe_fase0.md.")
    informe = d.construir_informe(
        st.session_state.get(_CLAVE_CARGAS), st.session_state.get(_CLAVE_IMPORTACIONES), tema_actual()
    )
    st.download_button(
        "Descargar informe (JSON)",
        data=json.dumps(informe, ensure_ascii=False, indent=2, default=str),
        file_name="diagnostico_fase0.json",
        mime="application/json",
        icon=":material/download:",
        key="diag_descargar",
    )


def _registros() -> None:
    c.seccion(
        "Registros recientes",
        "Las últimas entradas registradas en este proceso del servidor (errores de entrenamiento, recetas "
        "rechazadas, etc.). Se comparten entre las sesiones que atiende este proceso; no es un registro por "
        "persona.",
    )
    entradas = ultimas_entradas(50)
    if not entradas:
        st.caption("Sin entradas registradas todavía en este proceso.")
    else:
        tabla = pd.DataFrame(
            [{"Momento (UTC)": e.marca_tiempo, "Nivel": e.nivel, "Origen": e.origen, "Mensaje": e.mensaje}
             for e in entradas]
        )
        st.dataframe(tabla, hide_index=True, width="stretch")
    st.button("Limpiar registros de este proceso", key="diag_limpiar_registros", on_click=limpiar_buffer,
             icon=":material/delete_sweep:")


def render() -> None:
    c.encabezado(S.DIAG_TITULO, S.DIAG_LEAD)
    _entorno()
    _recursos()
    _paquetes()
    _carga_datos()
    _importaciones()
    _registros()
    _informe()
