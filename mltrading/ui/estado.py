"""Estado compartido entre páginas y envolturas con caché de la preparación de datos.

Streamlit descarta el estado de un widget cuando la página que lo muestra deja de
ejecutarse. Por eso los valores que otras páginas necesitan se guardan aparte,
en claves que empiezan con guion bajo, y los widgets se inicializan desde ellas.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.core.features import predeterminados
from mltrading.core.features.config import CONFIG_PREDETERMINADA, ConfigVariables, es_predeterminada
from mltrading.core.features.division import CRONOLOGICA, Division, dividir_cronologica
from mltrading.core.models.base import EspecificacionModelo, ResultadoEntrenamiento
from mltrading.core.models.entrenamiento import entrenar_modelo, es_configuracion_predeterminada as _modelo_es_predeterminado
from mltrading.core.models.params import valores_predeterminados
from mltrading.core.models.registry import obtener as obtener_modelo
from mltrading.core.features.pipeline import ResultadoVariables, preparar_variables
from mltrading.ui import datos_cache as dc

MODO_PREDETERMINADO = "Predeterminado"
MODO_PERSONALIZADO = "Personalizado"
_MODOS = [MODO_PREDETERMINADO, MODO_PERSONALIZADO]

PROPORCION_PREDETERMINADA = 80  # porcentaje de filas para entrenamiento
SEMILLA_PREDETERMINADA = 42


# ------------------------------------------------------------------- modo ---
def modo_actual() -> str:
    return st.session_state.get("_modo", MODO_PREDETERMINADO)


def selector_modo(clave: str) -> str:
    """Control Predeterminado/Personalizado que se mantiene al cambiar de página."""
    elegido = st.segmented_control(
        "Modo",
        _MODOS,
        default=modo_actual(),
        key=clave,
        help="Predeterminado usa los valores de referencia. Personalizado permite modificarlos.",
    )
    if elegido:
        st.session_state["_modo"] = elegido
    return modo_actual()


# --------------------------------------------------- variables (con caché) ---
@st.cache_data(show_spinner=False, max_entries=6, ttl=3600)
def _resultado_variables(config: ConfigVariables) -> ResultadoVariables:
    ohlcv, _ = dc.cargar("JPM_2017_2019.csv")
    adf = predeterminados.cargar_adf() if config.barras_por_dia == CONFIG_PREDETERMINADA.barras_por_dia else None
    return preparar_variables(ohlcv, config, tabla_adf=adf)


def config_variables_actual() -> ConfigVariables:
    if modo_actual() == MODO_PREDETERMINADO:
        return CONFIG_PREDETERMINADA
    return ConfigVariables(
        barras_por_dia=int(st.session_state.get("_barras", CONFIG_PREDETERMINADA.barras_por_dia)),
        umbral_adf=float(st.session_state.get("_umbral_adf", CONFIG_PREDETERMINADA.umbral_adf)),
        umbral_correlacion=float(st.session_state.get("_umbral_corr", CONFIG_PREDETERMINADA.umbral_correlacion)),
        motor=str(st.session_state.get("_motor", CONFIG_PREDETERMINADA.motor)),
    )


def resultado_base() -> ResultadoVariables:
    """Resultado con la configuración vigente y los descartes sugeridos por la regla de correlación."""
    return _resultado_variables(config_variables_actual())


def resultado_variables() -> ResultadoVariables:
    """Resultado de la preparación de variables con la configuración vigente y los descartes elegidos."""
    config = config_variables_actual()
    resultado = _resultado_variables(config)
    if modo_actual() == MODO_PERSONALIZADO:
        firma = _firma(config)
        descartes = st.session_state.get("_descartes")
        if descartes is not None and st.session_state.get("_firma_descartes") == firma:
            return resultado.con_descartes(list(descartes))
    return resultado


def _firma(config: ConfigVariables) -> tuple:
    return (config.barras_por_dia, config.umbral_adf, config.umbral_correlacion)


def recordar_descartes(config: ConfigVariables, descartes: list[str]) -> None:
    st.session_state["_descartes"] = list(descartes)
    st.session_state["_firma_descartes"] = _firma(config)


def firma_descartes_vigente(config: ConfigVariables) -> bool:
    return st.session_state.get("_firma_descartes") == _firma(config)


def es_configuracion_predeterminada(resultado: ResultadoVariables) -> bool:
    """Verdadero si se usan los parámetros y los descartes predeterminados."""
    return es_predeterminada(resultado.config) and sorted(resultado.descartadas_correlacion) == sorted(
        resultado.sugeridas_correlacion
    )


# ----------------------------------------------------------------- división ---
def proporcion_actual() -> float:
    if modo_actual() == MODO_PREDETERMINADO:
        return PROPORCION_PREDETERMINADA / 100
    return float(st.session_state.get("_proporcion", PROPORCION_PREDETERMINADA)) / 100


def division_actual(resultado: ResultadoVariables | None = None) -> Division:
    resultado = resultado or resultado_variables()
    return dividir_cronologica(resultado.X, resultado.y, proporcion_actual())


def es_division_predeterminada(division: Division) -> bool:
    return division.tipo == CRONOLOGICA and round(division.proporcion * 100) == PROPORCION_PREDETERMINADA


def a_csv(datos: pd.DataFrame | pd.Series) -> bytes:
    return datos.to_csv().encode("utf-8")


# ------------------------------------------------------------------ modelo ---
CLAVE_ENTRENAMIENTO = "_entrenamiento"


def parametros_modelo_actual(spec: EspecificacionModelo) -> dict:
    if modo_actual() == MODO_PREDETERMINADO:
        return valores_predeterminados(spec.parametros)
    guardados = st.session_state.get(f"_parametros_{spec.id}", {})
    predeterminados = valores_predeterminados(spec.parametros)
    return {nombre: guardados.get(nombre, valor) for nombre, valor in predeterminados.items()}


@st.cache_data(show_spinner=False, max_entries=12, ttl=3600)
def _entrenar_cacheado(id_modelo: str, parametros: dict, division: Division) -> ResultadoEntrenamiento:
    spec = obtener_modelo(id_modelo)
    return entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba, parametros)


def entrenar_y_recordar(id_modelo: str, division: Division, resultado_variables=None) -> ResultadoEntrenamiento:
    spec = obtener_modelo(id_modelo)
    parametros = parametros_modelo_actual(spec)
    resultado = _entrenar_cacheado(id_modelo, parametros, division)
    st.session_state[CLAVE_ENTRENAMIENTO] = {
        "id_modelo": id_modelo,
        "parametros": parametros,
        "predeterminado": _modelo_es_predeterminado(spec, parametros) and modo_actual() == MODO_PREDETERMINADO,
        "division": division,
        "resultado": resultado,
        "variables_finales": list(division.X_entrenamiento.columns),
        "config_variables": resultado_variables.config if resultado_variables is not None else None,
        "descartes_correlacion": (
            list(resultado_variables.descartadas_correlacion) if resultado_variables is not None else None
        ),
        "proporcion_entrenamiento": division.proporcion,
        "origen": "entrenamiento",
    }
    return resultado


def recordar_desde_receta(id_modelo: str, parametros: dict, division: Division, resultado, receta) -> None:
    """Guarda en el estado un entrenamiento que vino de aplicar una receta, no de entrenar en la página."""
    st.session_state[CLAVE_ENTRENAMIENTO] = {
        "id_modelo": id_modelo,
        "parametros": dict(parametros),
        "predeterminado": False,  # una receta cargada siempre se marca como configuración explícita
        "division": division,
        "resultado": resultado,
        "variables_finales": list(division.X_entrenamiento.columns),
        "config_variables": None,
        "descartes_correlacion": list(receta.descartes_correlacion),
        "proporcion_entrenamiento": division.proporcion,
        "origen": "receta",
    }


def entrenamiento_actual() -> dict | None:
    return st.session_state.get(CLAVE_ENTRENAMIENTO)


def hay_entrenamiento() -> bool:
    return CLAVE_ENTRENAMIENTO in st.session_state
