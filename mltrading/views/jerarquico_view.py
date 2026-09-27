"""Clustering jerárquico (capítulo 18)."""
from __future__ import annotations

import streamlit as st

from mltrading.core.clustering import hierarchical
from mltrading.ui import charts, estado
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import format as f

TITULO = "Clustering jerárquico"
LEAD = (
    "A diferencia de K-Means, no hace falta decidir el número de clústeres por adelantado: el dendrograma "
    "muestra cómo se van agrupando las acciones a distintos niveles de similitud."
)
REFERENCIA = "Capítulo 18"
_INSIGNIA_PRED = ":green-badge[Resultado predeterminado]"
_INSIGNIA_PERS = ":orange-badge[Configuración personalizada]"


def render() -> None:
    c.encabezado(TITULO, LEAD, REFERENCIA)
    datos, _ = dc.cargar("sample_stocks.csv")

    modo = estado.selector_modo("modo_jerarquico")
    with st.container(border=True):
        if modo == estado.MODO_PREDETERMINADO:
            n_clusters = hierarchical.N_CLUSTERS_PREDETERMINADO
            metodo = hierarchical.METODO_PREDETERMINADO
            st.caption(f"Valores predeterminados: {n_clusters} clústeres, método de enlace «{metodo}».")
        else:
            columnas = st.columns(2)
            with columnas[0]:
                n_clusters = st.slider("Número de clústeres", 1, len(datos),
                                       int(st.session_state.get("_jerarquico_n", 2)), key="w_jerarquico_n")
            with columnas[1]:
                metodo = st.selectbox("Método de enlace", hierarchical.METODOS,
                                      index=hierarchical.METODOS.index(
                                          st.session_state.get("_jerarquico_metodo", "ward")),
                                      key="w_jerarquico_metodo")
            st.session_state["_jerarquico_n"] = n_clusters
            st.session_state["_jerarquico_metodo"] = metodo

    try:
        resultado = hierarchical.ajustar(datos, n_clusters, metodo)
    except ValueError as error:
        st.error(str(error), icon=":material/error:")
        return

    predeterminado = (
        modo == estado.MODO_PREDETERMINADO and n_clusters == hierarchical.N_CLUSTERS_PREDETERMINADO
        and metodo == hierarchical.METODO_PREDETERMINADO
    )
    st.markdown(_INSIGNIA_PRED if predeterminado else _INSIGNIA_PERS)

    columnas = st.columns(2)
    with columnas[0]:
        c.indicador("Acciones", f.entero(len(datos)))
    with columnas[1]:
        c.indicador("Clústeres", str(n_clusters))

    tabs = st.tabs(["Dendrograma", "Clústeres", "Datos"])
    with tabs[0]:
        st.write(
            "Cada unión representa dos grupos que se combinan a esa distancia. Cortar el árbol a una altura "
            "determinada da un número de clústeres. El código de origen dibuja el dendrograma sin escalar los "
            "datos; aquí se reproduce así por defecto, con la opción de escalarlos."
        )
        escalar_dendro = st.checkbox("Escalar los datos del dendrograma", value=False, key="w_jerarquico_escalar")
        enlace = hierarchical.matriz_enlace(datos, metodo, escalar_datos=escalar_dendro)
        charts.mostrar(charts.dendrograma(enlace, list(datos.index)), clave="dendrograma")
    with tabs[1]:
        charts.mostrar(
            charts.dispersion_clusters(datos, "Beta", "ROE(%)", resultado.etiquetas), clave="dispersion_jerarquico"
        )
    with tabs[2]:
        tabla = datos.copy()
        tabla["Clúster"] = resultado.etiquetas
        st.dataframe(tabla, width="stretch")
        st.download_button(
            "Descargar clústeres (CSV)", estado.a_csv(tabla), "jerarquico_clusters.csv", "text/csv",
            icon=":material/download:", key="descarga_jerarquico",
        )
