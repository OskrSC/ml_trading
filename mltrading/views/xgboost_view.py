"""XGBoost sobre un conjunto multiactivo (capítulo 14)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.core.models.multiasset import UNIVERSO_PREDETERMINADO, construir_conjunto
from mltrading.core.models.params import valores_predeterminados
from mltrading.core.models.xgboost_multiactivo import PARAMETROS, entrenar_xgboost
from mltrading.core.observabilidad import obtener_logger
from mltrading.ui import charts
from mltrading.ui import components as c
from mltrading.ui import estado
from mltrading.ui import format as f

TITULO = "XGBoost multiactivo"
LEAD = (
    "El capítulo 14 entrena un solo modelo sobre varias acciones a la vez, con variables de retorno y "
    "volatilidad a distintas ventanas. El código de origen descarga cinco acciones estadounidenses en el "
    "momento; esta aplicación no descarga datos por defecto (ver la página Diagnóstico del entorno), así que "
    "el conjunto predeterminado usa cinco activos ya presentes en el catálogo de datos."
)
REFERENCIA = "Capítulo 14"
_INSIGNIA_PRED = ":green-badge[Resultado predeterminado]"
_INSIGNIA_PERS = ":orange-badge[Configuración personalizada]"
_CLAVE = "_xgboost_entrenamiento"


def _seleccion_activos(modo: str) -> tuple:
    etiquetas = {a.id: f"{a.nombre} ({a.archivo})" for a in UNIVERSO_PREDETERMINADO}
    if modo == estado.MODO_PREDETERMINADO:
        st.caption("Activos predeterminados: " + ", ".join(etiquetas.values()) + ".")
        return UNIVERSO_PREDETERMINADO
    elegidos = st.multiselect(
        "Activos a incluir", list(etiquetas), default=list(etiquetas), format_func=lambda x: etiquetas[x],
        key="w_xgb_activos",
    )
    return tuple(a for a in UNIVERSO_PREDETERMINADO if a.id in elegidos)


def render() -> None:
    c.encabezado(TITULO, LEAD, REFERENCIA)
    modo = estado.selector_modo("modo_xgboost")
    activos = _seleccion_activos(modo)
    if not activos:
        st.warning("Elija al menos un activo.", icon=":material/warning:")
        return

    parametros = valores_predeterminados(PARAMETROS)
    with st.container(border=True):
        if modo == estado.MODO_PREDETERMINADO:
            st.caption(
                f"Valores predeterminados: número de árboles = {parametros['n_estimators']}, "
                f"profundidad máxima = {parametros['max_depth']}."
            )
        else:
            columnas = st.columns(2)
            with columnas[0]:
                parametros["n_estimators"] = st.number_input(
                    "Número de árboles", min_value=1, max_value=300, step=1,
                    value=int(st.session_state.get("_xgb_n_estimators", parametros["n_estimators"])),
                    key="w_xgb_n_estimators",
                )
            with columnas[1]:
                parametros["max_depth"] = st.number_input(
                    "Profundidad máxima", min_value=1, max_value=20, step=1,
                    value=int(st.session_state.get("_xgb_max_depth", parametros["max_depth"])),
                    key="w_xgb_max_depth",
                )
            st.session_state["_xgb_n_estimators"] = parametros["n_estimators"]
            st.session_state["_xgb_max_depth"] = parametros["max_depth"]

    if st.button("Entrenar XGBoost", type="primary", key="entrenar_xgboost", icon=":material/play_arrow:"):
        try:
            with st.spinner("Construyendo las variables de cada activo y entrenando"):
                conjunto = construir_conjunto(activos, proporcion=0.8)
                resultado = entrenar_xgboost(conjunto, parametros)
        except Exception as error:  # noqa: BLE001 - se muestra el motivo a la persona
            obtener_logger("xgboost").exception("Fallo al entrenar XGBoost multiactivo: %s", error)
            st.error(f"No se pudo entrenar el modelo: {error}", icon=":material/error:")
            return
        predeterminado = (
            modo == estado.MODO_PREDETERMINADO
            and set(a.id for a in activos) == set(a.id for a in UNIVERSO_PREDETERMINADO)
            and parametros == valores_predeterminados(PARAMETROS)
        )
        st.session_state[_CLAVE] = {"conjunto": conjunto, "resultado": resultado, "predeterminado": predeterminado}

    guardado = st.session_state.get(_CLAVE)
    if not guardado:
        st.info("Pulse «Entrenar XGBoost» para ver los resultados.", icon=":material/info:")
        return

    conjunto = guardado["conjunto"]
    resultado = guardado["resultado"]
    st.markdown(_INSIGNIA_PRED if guardado["predeterminado"] else _INSIGNIA_PERS)

    accuracy_global = (resultado.y_pred_prueba.values == conjunto.y_prueba.values).mean()
    columnas = st.columns(3)
    with columnas[0]:
        c.indicador("Exactitud global en prueba", f"{f.decimal(accuracy_global * 100, 2)} %")
    with columnas[1]:
        c.indicador("Filas de entrenamiento", f.entero(len(conjunto.X_entrenamiento)))
    with columnas[2]:
        c.indicador("Filas de prueba", f.entero(len(conjunto.X_prueba)))

    tabs = st.tabs(["Resumen por activo", "Interpretación", "Portafolio", "Predicciones"])
    with tabs[0]:
        filas = []
        for aid, datos in conjunto.datos_por_activo.items():
            m = conjunto.activo_prueba == aid
            if m.sum() == 0:
                continue
            acc = (resultado.y_pred_prueba[m.values].values == conjunto.y_prueba[m].values).mean()
            filas.append({"Activo": datos.nombre, "Filas de prueba": int(m.sum()), "Exactitud": acc})
        tabla = pd.DataFrame(filas)
        tabla["Exactitud"] = tabla["Exactitud"].map(lambda x: f"{f.decimal(x * 100, 1)} %")
        st.dataframe(tabla, hide_index=True, width="stretch")

    with tabs[1]:
        st.write("La importancia mide cuánto contribuye cada variable a reducir el error del modelo.")
        tabla_imp = resultado.interpretacion.head(10).copy()
        tabla_imp["importancia"] = tabla_imp["importancia"].round(4)
        charts.mostrar(charts.barras_horizontales(tabla_imp, "variable", "importancia"), clave="importancias_xgb")

    with tabs[2]:
        st.write(
            "Retorno de cada activo si se hubiera seguido la señal predicha en su tramo de prueba, y el "
            "portafolio con la misma ponderación para todos. Los activos no comparten calendario, así que las "
            "curvas se alinean por posición dentro de cada tramo de prueba, no por fecha."
        )
        curvas = {}
        por_posicion = []
        for aid, datos in conjunto.datos_por_activo.items():
            m = conjunto.activo_prueba == aid
            if m.sum() == 0:
                continue
            retorno_estrategia = resultado.y_pred_prueba[m.values].values * datos.retorno_diario_siguiente[
                conjunto.X_prueba.index[m.values]
            ].values
            curva = (1 + pd.Series(retorno_estrategia)).cumprod()
            curvas[datos.nombre] = curva
            por_posicion.append(pd.Series(retorno_estrategia))
        charts.mostrar(charts.curvas_multiples(curvas), clave="curvas_xgb")

        largo_minimo = min(len(s) for s in por_posicion)
        portafolio = pd.concat([s.iloc[:largo_minimo].reset_index(drop=True) for s in por_posicion], axis=1).mean(axis=1)
        curva_portafolio = (1 + portafolio).cumprod()
        c.indicador(
            "Retorno del portafolio (ponderación igual)",
            f"{f.decimal((curva_portafolio.iloc[-1] - 1) * 100, 2)} %",
            "Cada activo pesa lo mismo. Alineado por posición dentro del tramo de prueba, no por fecha.",
        )
        charts.mostrar(charts.linea({"Portafolio": curva_portafolio}, altura=280), clave="curva_portafolio_xgb")

    with tabs[3]:
        salida = pd.DataFrame({"Activo": conjunto.activo_prueba.values, "Señal predicha": resultado.y_pred_prueba.values},
                              index=conjunto.X_prueba.index)
        st.download_button(
            "Descargar predicciones (CSV)", estado.a_csv(salida), "predicciones_xgboost.csv", "text/csv",
            icon=":material/download:", key="descarga_predicciones_xgb",
        )
        st.dataframe(salida.head(20), width="stretch")
