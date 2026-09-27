"""Comparador: entrena varios modelos del registro con los mismos datos y los compara."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.core.backtest.config import CONFIG_PREDETERMINADA
from mltrading.core.backtest.returns import calcular_retornos
from mltrading.core.evaluation.metrics import evaluar
from mltrading.core.features.indicadores import MotorNoDisponible
from mltrading.core.models.entrenamiento import entrenar_modelo
from mltrading.core.models.registry import MODELOS
from mltrading.core.observabilidad import obtener_logger
from mltrading.ui import charts
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import estado
from mltrading.ui import format as f

TITULO = "Comparador de modelos"
LEAD = (
    "Entrena varios modelos con exactamente las mismas variables y la misma división, para que la comparación "
    "sea justa. Es una extensión de esta aplicación: los capítulos del libro presentan cada modelo por separado."
)
_CLAVE = "_comparador"


def render() -> None:
    c.encabezado(TITULO, LEAD)

    etiquetas = {m.id: f"{m.nombre} ({m.capitulo})" for m in MODELOS}
    elegidos = st.multiselect(
        "Modelos a comparar", list(etiquetas), default=[m.id for m in MODELOS], format_func=lambda x: etiquetas[x],
        key="w_comparador_modelos",
    )
    if len(elegidos) < 2:
        st.info("Elija al menos dos modelos para comparar.", icon=":material/info:")
        return

    try:
        with st.spinner("Preparando las variables y la división"):
            resultado_vars = estado.resultado_variables()
            division = estado.division_actual(resultado_vars)
    except (MotorNoDisponible, ValueError) as error:
        st.error(f"No se pudo preparar los datos: {error}", icon=":material/error:")
        return
    st.caption(
        f"Todos los modelos usan las {len(resultado_vars.variables_finales)} variables finales y la división de "
        "las páginas anteriores, con sus valores predeterminados."
    )

    if st.button("Entrenar y comparar", type="primary", key="entrenar_comparador", icon=":material/play_arrow:"):
        ohlcv, _ = dc.cargar("JPM_2017_2019.csv")
        filas = []
        curvas = {}
        progreso = st.progress(0.0, text="Entrenando")
        fallidos = []
        for i, id_modelo in enumerate(elegidos, start=1):
            spec = next(m for m in MODELOS if m.id == id_modelo)
            progreso.progress((i - 1) / len(elegidos), text=f"Entrenando {spec.nombre}")
            try:
                resultado = entrenar_modelo(
                    spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba
                )
                ev = evaluar(division.y_prueba, resultado.y_pred_prueba)
                bt = calcular_retornos(ohlcv["close"], resultado.y_pred_prueba, CONFIG_PREDETERMINADA)
            except Exception as error:  # noqa: BLE001 - un modelo fallido no debe tumbar la comparación
                obtener_logger("comparador").exception("Fallo al entrenar %s: %s", id_modelo, error)
                fallidos.append(spec.nombre)
                continue
            filas.append(
                {
                    "Modelo": spec.nombre, "Capítulo": spec.capitulo, "Exactitud": ev.exactitud,
                    "Retorno acumulado (%)": bt.retorno_acumulado_pct, "Sharpe": bt.sharpe,
                    "Drawdown máximo (%)": bt.drawdown_maximo_pct,
                }
            )
            curvas[spec.nombre] = bt.datos["cumulative_returns"].reset_index(drop=True)
        progreso.empty()
        if fallidos:
            st.warning(
                "No se pudo entrenar: " + ", ".join(fallidos) + ". El resto de resultados sigue siendo válido.",
                icon=":material/warning:",
            )
        if filas:
            st.session_state[_CLAVE] = {"tabla": pd.DataFrame(filas), "curvas": curvas}

    guardado = st.session_state.get(_CLAVE)
    if not guardado:
        st.info("Pulse «Entrenar y comparar» para ver los resultados.", icon=":material/info:")
        return

    tabla = guardado["tabla"]
    c.seccion("Métricas")
    vista = tabla.copy()
    vista["Exactitud"] = vista["Exactitud"].map(lambda x: f"{f.decimal(x * 100, 2)} %")
    vista["Retorno acumulado (%)"] = vista["Retorno acumulado (%)"].map(lambda x: f.decimal(x, 2))
    vista["Sharpe"] = vista["Sharpe"].map(lambda x: f.decimal(x, 2))
    vista["Drawdown máximo (%)"] = vista["Drawdown máximo (%)"].map(lambda x: f.decimal(x, 2))
    mejor = tabla.loc[tabla["Sharpe"].idxmax(), "Modelo"]
    st.dataframe(vista, hide_index=True, width="stretch")
    st.caption(f"Mejor Sharpe en este conjunto: {mejor}. No es garantía de resultados futuros.")

    c.seccion("Curvas de capital", "Estrategia de cada modelo sobre el mismo tramo de prueba, con costo de transacción en 0.")
    charts.mostrar(charts.curvas_multiples(guardado["curvas"]), clave="curvas_comparador")

    st.download_button(
        "Descargar comparación (CSV)", estado.a_csv(tabla.set_index("Modelo")), "comparacion_modelos.csv", "text/csv",
        icon=":material/download:", key="descarga_comparacion",
    )
