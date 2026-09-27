"""Backtesting de la estrategia (capítulo 7)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.core.backtest.analytics import analitica
from mltrading.core.backtest.config import CONFIG_PREDETERMINADA, ConfigBacktest, es_predeterminada
from mltrading.core.backtest.returns import calcular_retornos
from mltrading.core.backtest.trades import construir_registro
from mltrading.ui import charts, estado
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import format as f

TITULO = "Backtesting"
LEAD = (
    "Convierte las señales del modelo en una estrategia y mide su resultado histórico: retorno, riesgo y el "
    "detalle de cada operación. No incluye costos de transacción salvo que los indique."
)
REFERENCIA = "Capítulo 7"
_INSIGNIA_PRED = ":green-badge[Resultado predeterminado]"
_INSIGNIA_PERS = ":orange-badge[Configuración personalizada]"


def _controles() -> ConfigBacktest:
    with st.container(border=True):
        costo = st.slider(
            "Costo de transacción (puntos básicos por operación)", 0.0, 20.0,
            float(st.session_state.get("_costo_bp", 0.0)), step=0.5, key="w_costo_bp",
            help="0 reproduce el resultado predeterminado, que no incluye costos de transacción ni deslizamiento.",
        )
        st.session_state["_costo_bp"] = costo
        if costo == 0.0:
            st.caption("Sin costos de transacción: es el cálculo predeterminado.")
        else:
            st.warning(
                "Con costos de transacción, el resultado deja de ser el predeterminado. Esta es una extensión: "
                "la configuración predeterminada no los incluye.",
                icon=":material/warning:",
            )
    return ConfigBacktest(costo_bp=costo)


def render() -> None:
    c.encabezado(TITULO, LEAD, REFERENCIA)

    if not estado.hay_entrenamiento():
        st.info(
            "Todavía no hay un modelo entrenado. Vaya a Modelos supervisados, elija uno y pulse «Entrenar».",
            icon=":material/info:",
        )
        return

    entrenamiento = estado.entrenamiento_actual()
    resultado = entrenamiento["resultado"]
    st.caption(f"Modelo usado: {entrenamiento['id_modelo'].replace('_', ' ').title()}.")

    config = _controles()
    ohlcv, _ = dc.cargar("JPM_2017_2019.csv")
    try:
        r = calcular_retornos(ohlcv["close"], resultado.y_pred_prueba, config)
    except ValueError as error:
        st.error(str(error), icon=":material/error:")
        return

    predeterminado = es_predeterminada(config) and entrenamiento["predeterminado"]
    st.markdown(_INSIGNIA_PRED if predeterminado else _INSIGNIA_PERS)

    columnas = st.columns(5)
    with columnas[0]:
        c.indicador("Retorno acumulado", f"{f.decimal(r.retorno_acumulado_pct, 2)} %")
    with columnas[1]:
        c.indicador("Retorno anualizado", f"{f.decimal(r.retorno_anualizado_pct, 2)} %")
    with columnas[2]:
        c.indicador("Volatilidad anualizada", f"{f.decimal(r.volatilidad_anualizada_pct, 2)} %")
    with columnas[3]:
        c.indicador("Sharpe", f.decimal(r.sharpe, 2))
    with columnas[4]:
        c.indicador("Drawdown máximo", f"{f.decimal(r.drawdown_maximo_pct, 2)} %")

    c.seccion("Curva de capital", "Estrategia contra comprar y mantener, con capital inicial de 1.")
    charts.mostrar(charts.curva_capital(r.datos), clave="curva_capital")
    c.seccion("Drawdown de la estrategia")
    charts.mostrar(charts.drawdown(r.datos), clave="drawdown_estrategia")

    with st.spinner("Construyendo el registro de operaciones"):
        registro = construir_registro(r.datos)
    c.seccion("Registro de operaciones", f"{f.entero(len(registro))} operaciones en el periodo evaluado.")
    if registro.empty:
        st.info("El modelo no generó ninguna operación en el tramo de prueba.", icon=":material/info:")
    else:
        vista = registro.copy()
        vista["Posición"] = vista["Posición"].map({1: "Larga", -1: "Corta"})
        for col in ("Entrada", "Salida"):
            vista[col] = vista[col].map(lambda x: f"{x:%d/%m/%Y %H:%M}")
        for col in ("Precio de entrada", "Precio de salida", "PnL"):
            vista[col] = vista[col].map(lambda x: f.decimal(x, 2))
        st.dataframe(vista, hide_index=True, width="stretch")

        c.seccion("Analítica de las operaciones")
        a = analitica(registro).iloc[0]
        columnas = st.columns(4)
        with columnas[0]:
            c.indicador("Total de operaciones", f.entero(a["Total de operaciones"]))
        with columnas[1]:
            c.indicador("Porcentaje de aciertos", f"{f.decimal(a['Porcentaje de aciertos'], 1)} %")
        with columnas[2]:
            c.indicador("Ganancia bruta", f.decimal(a["Ganancia bruta"], 2))
        with columnas[3]:
            c.indicador("Pérdida bruta", f.decimal(a["Pérdida bruta"], 2))
        columnas = st.columns(3)
        with columnas[0]:
            c.indicador("Resultado neto", f.decimal(a["Resultado neto"], 2))
        with columnas[1]:
            c.indicador("Resultado medio, ganadoras", f.decimal(a["Resultado medio (ganadoras)"], 2))
        with columnas[2]:
            c.indicador("Resultado medio, perdedoras", f.decimal(a["Resultado medio (perdedoras)"], 2))

        st.download_button(
            "Descargar operaciones (CSV)", estado.a_csv(registro), "operaciones.csv", "text/csv",
            icon=":material/download:", key="descarga_operaciones",
        )

    if config.costo_bp:
        st.caption(f"Costo total descontado del retorno acumulado: {f.decimal(r.costo_total_pct, 2)} puntos porcentuales.")
    st.download_button(
        "Descargar retornos de la estrategia (CSV)",
        estado.a_csv(r.datos[["close", "predicted_signal", "strategy_returns", "cumulative_returns"]]),
        "retornos_estrategia.csv", "text/csv", icon=":material/download:", key="descarga_retornos",
    )
