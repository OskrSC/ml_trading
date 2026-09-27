"""Variables y objetivo: construcción de la señal, indicadores, estacionariedad y correlación."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.core.features import indicadores
from mltrading.core.features.config import CONFIG_PREDETERMINADA, MOTOR_AUTO, MOTOR_PANDAS, MOTOR_TALIB
from mltrading.core.features.construccion import CANDIDATAS, descripciones
from mltrading.core.features.indicadores import MotorNoDisponible
from mltrading.ui import charts, estado
from mltrading.ui import components as c
from mltrading.ui import format as f

TITULO = "Variables y objetivo"
LEAD = (
    "Antes de entrenar un modelo hay que decidir qué se quiere predecir y con qué información. "
    "Aquí se construye la señal, se calculan las variables y se descartan las que no aportan."
)
REFERENCIA = "Capítulos 2 y 3"
_MOTORES = {"Automático": MOTOR_AUTO, "TA-Lib": MOTOR_TALIB, "Pandas y numpy": MOTOR_PANDAS}

_INSIGNIA_PRED = ":green-badge[Resultado predeterminado]"
_INSIGNIA_PERS = ":orange-badge[Configuración personalizada]"


def _restablecer() -> None:
    for clave in ("_barras", "_umbral_adf", "_umbral_corr", "_motor", "_descartes", "_firma_descartes",
                  "w_barras", "w_umbral_adf", "w_umbral_corr", "w_motor", "w_descartes"):
        st.session_state.pop(clave, None)
    st.session_state["_modo"] = estado.MODO_PREDETERMINADO
    st.session_state["modo_variables"] = estado.MODO_PREDETERMINADO


def _panel_configuracion(modo: str) -> None:
    pred = CONFIG_PREDETERMINADA
    with st.container(border=True):
        if modo == estado.MODO_PREDETERMINADO:
            st.caption(
                f"Valores predeterminados: {pred.barras_por_dia} barras por día (6,5 horas de 15 minutos), "
                f"umbral de la prueba ADF de {f.decimal(pred.umbral_adf, 2)} y umbral de correlación de "
                f"{f.decimal(pred.umbral_correlacion, 2)}."
            )
            return
        izq, der = st.columns(2)
        with izq:
            barras = st.number_input(
                "Barras por día",
                min_value=10, max_value=60, step=1,
                value=int(st.session_state.get("_barras", pred.barras_por_dia)),
                key="w_barras",
                help="Define las ventanas de RSI, ADX, media móvil, correlación y volatilidad. 26 barras de 15 minutos son 6,5 horas.",
            )
            umbral_adf = st.number_input(
                "Umbral de la prueba ADF",
                min_value=0.01, max_value=0.10, step=0.01, format="%.2f",
                value=float(st.session_state.get("_umbral_adf", pred.umbral_adf)),
                key="w_umbral_adf",
                help="Una variable se considera estacionaria si su p-valor es menor que este umbral.",
            )
        with der:
            umbral_corr = st.number_input(
                "Umbral de correlación",
                min_value=0.30, max_value=0.95, step=0.05, format="%.2f",
                value=float(st.session_state.get("_umbral_corr", pred.umbral_correlacion)),
                key="w_umbral_corr",
                help="Dos variables se consideran redundantes si su correlación absoluta supera este valor.",
            )
            etiquetas = list(_MOTORES)
            actual = next((k for k, v in _MOTORES.items() if v == st.session_state.get("_motor", MOTOR_AUTO)), etiquetas[0])
            motor = st.selectbox("Motor de indicadores", etiquetas, index=etiquetas.index(actual), key="w_motor",
                                 help="El resultado es el mismo con cualquier motor. Automático usa TA-Lib si está disponible.")
        st.session_state.update(_barras=int(barras), _umbral_adf=float(umbral_adf),
                                _umbral_corr=float(umbral_corr), _motor=_MOTORES[motor])
        st.button("Restablecer valores predeterminados", key="restablecer_variables", on_click=_restablecer,
                  icon=":material/restart_alt:")


def _p_valor(p: float) -> str:
    return "menor que 0,0001" if p < 1e-4 else f.decimal(p, 4)


def _tab_objetivo(res) -> None:
    datos = res.candidatas.datos
    y = res.y
    st.write(
        "La señal vale 1 cuando el cierre de la barra siguiente es mayor que el de la barra actual, "
        "y 0 en caso contrario. El modelo intentará anticipar esa señal."
    )
    uno = int(y.sum())
    cero = len(y) - uno
    izq, der = st.columns(2)
    with izq:
        c.indicador("Barras con señal 1", f"{f.entero(uno)} ({f.decimal(uno / len(y) * 100, 1)} %)")
    with der:
        c.indicador("Barras con señal 0", f"{f.entero(cero)} ({f.decimal(cero / len(y) * 100, 1)} %)")
    st.caption("Primeras filas: el retorno futuro y la señal que produce")
    muestra = datos[["close", "future_returns", "signal"]].head(8).rename(
        columns={"close": "Cierre", "future_returns": "Retorno futuro", "signal": "Señal"}
    )
    st.dataframe(muestra, width="stretch", column_config={"Retorno futuro": st.column_config.NumberColumn(format="%.5f")})


def _tab_variables(res) -> None:
    barras = res.config.barras_por_dia
    desc = descripciones(barras)
    elegida = st.selectbox("Variable a graficar", list(CANDIDATAS), index=list(CANDIDATAS).index("rsi"), key="w_var_grafico")
    st.caption(desc[elegida])
    charts.mostrar(charts.linea({elegida: res.candidatas.datos[elegida]}, altura=300), clave="grafico_variable")
    tabla = pd.DataFrame({"Variable": list(CANDIDATAS), "Qué mide": [desc[v] for v in CANDIDATAS]})
    st.dataframe(tabla, hide_index=True, width="stretch")


def _tab_estacionariedad(res) -> None:
    st.write(
        "Muchos modelos suponen que las variables mantienen sus propiedades estadísticas en el tiempo. "
        "La prueba ADF comprueba esa condición: si el p-valor es menor que "
        f"{f.decimal(res.config.umbral_adf, 2)}, la variable es estacionaria y se conserva."
    )
    t = res.tabla_adf
    salida = pd.DataFrame(
        {
            "Variable": t["variable"],
            "Estadístico ADF": [f.decimal(x, 2) for x in t["estadistico"]],
            "p-valor": [_p_valor(x) for x in t["p_valor"]],
            "Resultado": ["Estacionaria" if e else "No estacionaria" for e in t["estacionaria"]],
            "Decisión": ["Se conserva" if e else "Se descarta" for e in t["estacionaria"]],
        }
    )
    st.dataframe(salida, hide_index=True, width="stretch")
    if res.descartadas_adf:
        st.caption("Se descartan por no ser estacionarias: " + ", ".join(res.descartadas_adf) + ".")


def _tab_correlacion(res, modo: str):
    estacionarias = res.estacionarias
    st.write(
        "Dos variables muy correlacionadas aportan la misma información. Entre los pares que superan el umbral "
        f"de {f.decimal(res.config.umbral_correlacion, 2)}, la regla descarta la variable que aparece después."
    )
    matriz = res.candidatas.variables[estacionarias].corr()
    charts.mostrar(charts.mapa_calor(matriz, altura=380), clave="mapa_correlacion")
    if res.pares.empty:
        st.success("Ningún par supera el umbral de correlación.", icon=":material/check_circle:")
    else:
        pares = pd.DataFrame(
            {
                "Variable A": res.pares["variable_a"],
                "Variable B": res.pares["variable_b"],
                "Correlación absoluta": [f.decimal(x, 3) for x in res.pares["correlacion"]],
                "Sugerencia": ["Descartar " + v for v in res.pares["variable_b"]],
            }
        )
        st.dataframe(pares, hide_index=True, width="stretch")

    config = res.config
    if modo == estado.MODO_PERSONALIZADO and not estado.firma_descartes_vigente(config):
        st.session_state["w_descartes"] = list(res.sugeridas_correlacion)
        estado.recordar_descartes(config, res.sugeridas_correlacion)
    if modo == estado.MODO_PERSONALIZADO:
        elegidas = st.multiselect("Variables a descartar por correlación", estacionarias, key="w_descartes")
        estado.recordar_descartes(config, elegidas)
        return elegidas
    st.multiselect("Variables a descartar por correlación", estacionarias, default=res.sugeridas_correlacion,
                   disabled=True, key="w_descartes_fijo")
    return list(res.sugeridas_correlacion)


def _tab_resultado(res) -> None:
    ca = res.candidatas
    pasos = pd.DataFrame(
        [
            {"Paso": "Variables candidatas", "Cantidad": len(CANDIDATAS), "Detalle": ", ".join(CANDIDATAS)},
            {"Paso": "Descartadas por no ser estacionarias", "Cantidad": len(res.descartadas_adf),
             "Detalle": ", ".join(res.descartadas_adf) or "Ninguna"},
            {"Paso": "Descartadas por correlación", "Cantidad": len(res.descartadas_correlacion),
             "Detalle": ", ".join(res.descartadas_correlacion) or "Ninguna"},
            {"Paso": "Variables finales", "Cantidad": len(res.variables_finales), "Detalle": ", ".join(res.variables_finales)},
        ]
    )
    st.dataframe(pasos, hide_index=True, width="stretch")
    st.markdown(" ".join(f":blue-badge[{v}]" for v in res.variables_finales))
    st.caption(
        f"Filas utilizables: {f.entero(ca.filas)} de {f.entero(ca.filas_iniciales)}. Se descartan "
        f"{f.entero(ca.filas_por_ventana)} filas iniciales que no tienen valor en todas las variables por sus "
        f"ventanas de cálculo, y {ca.filas_por_objetivo} al final, porque la última barra no tiene retorno futuro."
    )
    st.dataframe(res.X.head(8), width="stretch")
    izq, der = st.columns(2)
    with izq:
        st.download_button("Descargar variables (CSV)", estado.a_csv(res.X), "variables.csv", "text/csv",
                           icon=":material/download:", key="descarga_variables")
    with der:
        st.download_button("Descargar señal (CSV)", estado.a_csv(res.y.rename("signal")), "senal.csv", "text/csv",
                           icon=":material/download:", key="descarga_senal")


def render() -> None:
    c.encabezado(TITULO, LEAD, REFERENCIA)
    modo = estado.selector_modo("modo_variables")
    _panel_configuracion(modo)
    superior = st.container(key="indicadores")

    try:
        with st.spinner("Calculando variables y la prueba de estacionariedad. Con valores personalizados puede tardar unos 15 segundos."):
            base = estado.resultado_base()
    except MotorNoDisponible as error:
        st.error(str(error), icon=":material/error:")
        return
    except ValueError as error:
        st.error(f"La configuración no es válida: {error}", icon=":material/error:")
        return

    etiquetas = ["Objetivo", "Variables", "Estacionariedad", "Correlación", "Resultado"]
    t_obj, t_var, t_est, t_cor, t_res = st.tabs(etiquetas)
    with t_obj:
        _tab_objetivo(base)
    with t_var:
        _tab_variables(base)
    with t_est:
        _tab_estacionariedad(base)
    with t_cor:
        descartes = _tab_correlacion(base, modo)
    final = base.con_descartes(descartes)
    with t_res:
        _tab_resultado(final)

    with superior:
        columnas = st.columns(4)
        with columnas[0]:
            c.indicador("Filas utilizables", f.entero(final.candidatas.filas))
        with columnas[1]:
            c.indicador("Variables candidatas", str(len(CANDIDATAS)))
        with columnas[2]:
            c.indicador("Variables finales", str(len(final.variables_finales)))
        with columnas[3]:
            c.indicador("Señal 1", f"{f.decimal(final.y.mean() * 100, 1)} %")
        st.markdown(_INSIGNIA_PRED if estado.es_configuracion_predeterminada(final) else _INSIGNIA_PERS)
    st.caption(f"Motor de indicadores en uso: {'TA-Lib' if indicadores.resolver_motor(final.config.motor) == MOTOR_TALIB else 'Pandas y numpy'}.")
