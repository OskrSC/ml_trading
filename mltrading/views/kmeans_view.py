"""K-Means sobre ROE y Beta (capítulo 17)."""
from __future__ import annotations

import streamlit as st

from mltrading.core.clustering import kmeans
from mltrading.ui import charts, estado
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import format as f

TITULO = "K-Means"
LEAD = (
    "Agrupa las acciones por rentabilidad sobre el patrimonio (ROE) y Beta, sin usar ninguna etiqueta previa: "
    "el algoritmo descubre los grupos a partir de la similitud entre los datos."
)
REFERENCIA = "Capítulo 17"
_INSIGNIA_PRED = ":green-badge[Resultado predeterminado]"
_INSIGNIA_PERS = ":orange-badge[Configuración personalizada]"


def render() -> None:
    c.encabezado(TITULO, LEAD, REFERENCIA)
    datos, _ = dc.cargar("sample_stocks.csv")

    modo = estado.selector_modo("modo_kmeans")
    with st.container(border=True):
        if modo == estado.MODO_PREDETERMINADO:
            k = kmeans.K_PREDETERMINADO
            st.caption(f"Valor predeterminado: k = {k}. El ajuste del modelo siempre escala los datos.")
        else:
            k = st.slider("Número de clústeres (k)", 1, len(datos), int(st.session_state.get("_kmeans_k", 2)),
                          key="w_kmeans_k")
            st.session_state["_kmeans_k"] = k

    try:
        resultado = kmeans.ajustar(datos, k)
    except ValueError as error:
        st.error(str(error), icon=":material/error:")
        return

    predeterminado = modo == estado.MODO_PREDETERMINADO and k == kmeans.K_PREDETERMINADO
    st.markdown(_INSIGNIA_PRED if predeterminado else _INSIGNIA_PERS)

    columnas = st.columns(3)
    with columnas[0]:
        c.indicador("Acciones", f.entero(len(datos)))
    with columnas[1]:
        c.indicador("Clústeres", str(k))
    with columnas[2]:
        c.indicador("Inercia", f.decimal(resultado.inercia, 2), "Suma de distancias al centro de cada clúster: cuanto más baja, más compactos son los grupos.")

    tabs = st.tabs(["Clústeres", "Curva del codo", "Datos"])
    with tabs[0]:
        charts.mostrar(
            charts.dispersion_clusters(datos, "Beta", "ROE(%)", resultado.etiquetas), clave="dispersion_kmeans"
        )
    with tabs[1]:
        st.write(
            "La curva del codo ayuda a elegir k: el punto donde deja de bajar con fuerza sugiere un número "
            "razonable de clústeres. El código de origen la calcula sin escalar los datos; aquí se reproduce "
            "así por defecto, con la opción de escalarlos."
        )
        escalar_codo = st.checkbox("Escalar los datos de la curva del codo", value=False, key="w_kmeans_escalar_codo")
        tabla_codo = kmeans.curva_codo(datos, escalar_datos=escalar_codo)
        charts.mostrar(charts.curva_codo(tabla_codo), clave="curva_codo_kmeans")
    with tabs[2]:
        tabla = datos.copy()
        tabla["Clúster"] = resultado.etiquetas
        st.dataframe(tabla, width="stretch")
        st.download_button(
            "Descargar clústeres (CSV)", estado.a_csv(tabla), "kmeans_clusters.csv", "text/csv",
            icon=":material/download:", key="descarga_kmeans",
        )
