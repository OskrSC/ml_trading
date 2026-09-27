"""Operación en vivo, simulada día a día (capítulo 8)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.core.live import simulator as sim
from mltrading.core.models.registry import obtener as obtener_modelo
from mltrading.ui import charts
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import estado
from mltrading.ui import format as f

TITULO = "Operación en vivo"
LEAD = (
    "El capítulo 8 explica, sin código, cómo ampliar los datos con la jornada más reciente y qué criterios "
    "seguir para decidir cuándo reentrenar un modelo: por caída de exactitud, por pérdida de capital, o de "
    "forma periódica. Esta página implementa esos tres criterios y deja avanzar la simulación barra a barra."
)
REFERENCIA = "Capítulo 8"
_CLAVE = "_simulacion"
_ETIQUETAS_MODELO = {"random_forest": "Random Forest", "naive_bayes": "Naive Bayes", "decision_tree": "Árbol de decisión"}
_ETIQUETAS_CALENDARIO = {
    sim.PERIODICO: "Periódico", sim.POR_EXACTITUD: "Por caída de exactitud",
    sim.POR_PERDIDA: "Por pérdida de capital", sim.NUNCA: "Nunca",
}
_INSIGNIA_PRED = ":green-badge[Resultado predeterminado]"
_INSIGNIA_PERS = ":orange-badge[Configuración personalizada]"


def _config_predeterminada() -> tuple[str, dict, sim.ConfigSimulacion]:
    spec = obtener_modelo("random_forest")
    from mltrading.core.models.params import valores_predeterminados

    return "random_forest", valores_predeterminados(spec.parametros), sim.ConfigSimulacion()


def _panel_configuracion(modo: str) -> tuple[str, dict, sim.ConfigSimulacion]:
    id_pred, parametros_pred, config_pred = _config_predeterminada()
    with st.container(border=True):
        if modo == estado.MODO_PREDETERMINADO:
            st.caption(
                f"Valores predeterminados: {_ETIQUETAS_MODELO[id_pred]}, calendario "
                f"«{_ETIQUETAS_CALENDARIO[config_pred.calendario]}» cada {config_pred.periodo_barras} barras "
                "(una jornada de 15 minutos)."
            )
            return id_pred, parametros_pred, config_pred

        columnas = st.columns(2)
        with columnas[0]:
            id_modelo = st.selectbox(
                "Modelo", list(_ETIQUETAS_MODELO), format_func=lambda x: _ETIQUETAS_MODELO[x], key="w_sim_modelo",
                help="El simulador se limita a los modelos sin escalado, para reentrenar sin mantener un "
                     "escalador aparte.",
            )
        with columnas[1]:
            calendario = st.selectbox(
                "Calendario de reentrenamiento", list(_ETIQUETAS_CALENDARIO),
                format_func=lambda x: _ETIQUETAS_CALENDARIO[x], key="w_sim_calendario",
            )
        if calendario == sim.PERIODICO:
            periodo = st.slider("Reentrenar cada (barras)", 5, 260,
                                int(st.session_state.get("_sim_periodo", 26)), key="w_sim_periodo")
            config = sim.ConfigSimulacion(calendario=calendario, periodo_barras=periodo)
        elif calendario == sim.POR_EXACTITUD:
            columnas2 = st.columns(2)
            with columnas2[0]:
                ventana = st.slider("Ventana de exactitud (barras)", 5, 260, 26, key="w_sim_ventana")
            with columnas2[1]:
                umbral = st.slider("Umbral de exactitud (%)", 30, 70, 55, key="w_sim_umbral_acc")
            config = sim.ConfigSimulacion(calendario=calendario, ventana_exactitud=ventana, umbral_exactitud=umbral / 100)
        elif calendario == sim.POR_PERDIDA:
            umbral = st.slider("Retroceso de capital que dispara el reentrenamiento (%)", -20.0, -0.5, -5.0,
                               step=0.5, key="w_sim_umbral_perdida")
            config = sim.ConfigSimulacion(calendario=calendario, umbral_perdida_pct=umbral)
        else:
            config = sim.ConfigSimulacion(calendario=calendario)
        parametros = valores_predeterminados_de(id_modelo)
        return id_modelo, parametros, config


def valores_predeterminados_de(id_modelo: str) -> dict:
    from mltrading.core.models.params import valores_predeterminados

    return valores_predeterminados(obtener_modelo(id_modelo).parametros)


def _reiniciar() -> None:
    st.session_state.pop(_CLAVE, None)


def render() -> None:
    c.encabezado(TITULO, LEAD, REFERENCIA)
    modo = estado.selector_modo("modo_en_vivo")
    id_modelo, parametros, config = _panel_configuracion(modo)

    try:
        with st.spinner("Preparando las variables y la división"):
            resultado_vars = estado.resultado_variables()
            division = estado.division_actual(resultado_vars)
    except Exception as error:  # noqa: BLE001
        st.error(f"No se pudo preparar los datos: {error}", icon=":material/error:")
        return

    firma = (id_modelo, tuple(sorted(parametros.items())), config)
    simulacion = st.session_state.get(_CLAVE)
    if simulacion is None or simulacion["firma"] != firma:
        ohlcv, _ = dc.cargar("JPM_2017_2019.csv")
        retornos = ohlcv["close"].pct_change().reindex(division.X_prueba.index)
        try:
            estado_sim = sim.iniciar(
                id_modelo, parametros, division.X_entrenamiento, division.y_entrenamiento,
                division.X_prueba, division.y_prueba, retornos, config,
            )
        except ValueError as error:
            st.error(str(error), icon=":material/error:")
            return
        simulacion = {"firma": firma, "estado": estado_sim}
        st.session_state[_CLAVE] = simulacion

    predeterminado = modo == estado.MODO_PREDETERMINADO
    st.markdown(_INSIGNIA_PRED if predeterminado else _INSIGNIA_PERS)

    estado_sim = simulacion["estado"]
    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        pasos = st.select_slider("Barras a avanzar", options=[1, 5, 26, 130, 500], value=26, key="w_sim_pasos")
    with derecha:
        col_a, col_b = st.columns(2)
        with col_a:
            avanzar_click = st.button("Avanzar", type="primary", key="sim_avanzar", icon=":material/play_arrow:",
                                      disabled=estado_sim.terminada)
        with col_b:
            st.button("Reiniciar", key="sim_reiniciar", on_click=_reiniciar, icon=":material/restart_alt:")

    if avanzar_click and not estado_sim.terminada:
        with st.spinner(f"Simulando {pasos} barras"):
            simulacion["estado"] = sim.avanzar(estado_sim, pasos)
        estado_sim = simulacion["estado"]

    if estado_sim.terminada and estado_sim.barras_simuladas > 0:
        st.success("La simulación llegó al final del tramo de prueba.", icon=":material/check_circle:")

    total_barras = estado_sim.barras_simuladas + estado_sim.barras_pendientes
    tiempo_medio = (sum(estado_sim.tiempos_reentrenamiento) / len(estado_sim.tiempos_reentrenamiento) * 1000
                    if estado_sim.tiempos_reentrenamiento else 0)
    with st.container(key="indicadores"):
        columnas = st.columns(4)
        with columnas[0]:
            c.indicador("Barras simuladas", f"{f.entero(estado_sim.barras_simuladas)} / {f.entero(total_barras)}")
        with columnas[1]:
            exact = estado_sim.exactitud_global
            c.indicador("Exactitud acumulada", "—" if pd.isna(exact) else f"{f.decimal(exact * 100, 1)} %")
        with columnas[2]:
            c.indicador("Capital simulado", f.decimal(estado_sim.capital, 4),
                        "Capital de 1 al inicio; sube cuando se predice señal 1 y el precio sube.")
        with columnas[3]:
            c.indicador("Reentrenamientos", f.entero(len(estado_sim.tiempos_reentrenamiento)),
                        f"Tiempo medio por reentrenamiento: {f.decimal(tiempo_medio, 1)} ms.")

    if estado_sim.barras_simuladas == 0:
        st.info("Pulse «Avanzar» para empezar a revelar barras del tramo de prueba.", icon=":material/info:")
        return

    tabla = estado_sim.a_tabla()
    tabs = st.tabs(["Evolución", "Detalle", "Descargar"])
    with tabs[0]:
        c.seccion("Exactitud en una ventana móvil de 26 barras")
        movil = tabla["acierto"].rolling(26, min_periods=5).mean()
        charts.mostrar(charts.linea({"Exactitud móvil": movil}, altura=260), clave="sim_exactitud_movil")
        c.seccion("Capital simulado")
        capitales = []
        cap = 1.0
        for p in estado_sim.predicciones:
            cap *= 1 + (p.retorno if p.y_predicho == 1 else 0.0)
            capitales.append(cap)
        serie_capital = pd.Series(capitales, index=tabla.index)
        charts.mostrar(charts.linea({"Capital simulado": serie_capital}, altura=260), clave="sim_capital")
    with tabs[1]:
        vista = tabla.copy()
        vista["acierto"] = vista["acierto"].map({True: "Sí", False: "No"})
        vista["reentrenado"] = vista["reentrenado"].map({True: "Sí", False: "No"})
        st.dataframe(vista, width="stretch")
    with tabs[2]:
        st.download_button(
            "Descargar simulación (CSV)", estado.a_csv(tabla), "simulacion_en_vivo.csv", "text/csv",
            icon=":material/download:", key="descarga_simulacion",
        )
