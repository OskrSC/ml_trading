"""Reducción de dimensiones con PCA, agrupamiento y visualización con t-SNE (capítulo 19)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.core.clustering import pca_analysis as pca
from mltrading.ui import charts, estado
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import format as f

TITULO = "PCA y t-SNE"
LEAD = (
    "Veinte acciones y muchos días de retornos son demasiadas dimensiones para ver de un vistazo. El análisis "
    "de componentes principales las resume, K-Means agrupa las acciones parecidas sobre ese resumen, y t-SNE "
    "las dibuja en dos dimensiones para poder verlas."
)
REFERENCIA = "Capítulo 19"
_INSIGNIA_PRED = ":green-badge[Resultado predeterminado]"
_INSIGNIA_PERS = ":orange-badge[Configuración personalizada]"


def render() -> None:
    c.encabezado(TITULO, LEAD, REFERENCIA)
    precios, _ = dc.cargar("pca.csv")
    capitalizacion, _ = dc.cargar("stock_list.csv")
    capitalizacion = capitalizacion.set_index("Symbols")["marketcap"]

    retornos = pca.retornos_diarios(precios)

    modo = estado.selector_modo("modo_pca")
    with st.container(border=True):
        if modo == estado.MODO_PREDETERMINADO:
            n_componentes = pca.N_COMPONENTES_PREDETERMINADO
            k = pca.K_PREDETERMINADO
            perplejidad_solicitada = pca.PERPLEJIDAD_PREDETERMINADA
            st.caption(
                f"Valores predeterminados: {n_componentes} componentes principales, k = {k} para K-Means, "
                f"perplejidad de t-SNE = {perplejidad_solicitada}."
            )
        else:
            columnas = st.columns(3)
            with columnas[0]:
                n_componentes = st.slider("Componentes principales", 2, min(retornos.shape),
                                          int(st.session_state.get("_pca_n", 18)), key="w_pca_n")
            with columnas[1]:
                k = st.slider("k para K-Means", 1, retornos.shape[1],
                              int(st.session_state.get("_pca_k", 4)), key="w_pca_k")
            with columnas[2]:
                perplejidad_solicitada = st.slider("Perplejidad de t-SNE", 2, 50,
                                                    int(st.session_state.get("_pca_perp", 25)), key="w_pca_perp")
            st.session_state.update(_pca_n=n_componentes, _pca_k=k, _pca_perp=perplejidad_solicitada)

    try:
        resultado_pca = pca.ajustar_pca(retornos, n_componentes)
        clusters = pca.agrupar(resultado_pca, retornos.columns, k)
    except ValueError as error:
        st.error(str(error), icon=":material/error:")
        return

    predeterminado = (
        modo == estado.MODO_PREDETERMINADO and n_componentes == pca.N_COMPONENTES_PREDETERMINADO
        and k == pca.K_PREDETERMINADO and perplejidad_solicitada == pca.PERPLEJIDAD_PREDETERMINADA
    )
    st.markdown(_INSIGNIA_PRED if predeterminado else _INSIGNIA_PERS)

    columnas = st.columns(4)
    with columnas[0]:
        c.indicador("Acciones", f.entero(len(retornos.columns)))
    with columnas[1]:
        c.indicador("Días de retornos", f.entero(len(retornos)))
    with columnas[2]:
        c.indicador("Varianza explicada", f"{f.decimal(resultado_pca.varianza_explicada_acumulada * 100, 1)} %")
    with columnas[3]:
        c.indicador("Clústeres", str(k))

    tabs = st.tabs(["Varianza explicada", "Clústeres (t-SNE)", "Datos"])
    with tabs[0]:
        st.write(
            f"Cada componente resume una parte de cómo se mueven juntas las 20 acciones. Con "
            f"{n_componentes} componentes se conserva el {f.decimal(resultado_pca.varianza_explicada_acumulada * 100, 1)} % "
            "de la variación original."
        )
        charts.mostrar(charts.varianza_explicada(resultado_pca.varianza_explicada), clave="varianza_pca")
    with tabs[1]:
        with st.spinner("Calculando la proyección t-SNE"):
            resultado_tsne = pca.proyectar_tsne(resultado_pca, perplejidad_solicitada)
        if resultado_tsne.ajustada:
            st.info(
                f"La perplejidad se ajustó de {resultado_tsne.perplejidad_solicitada} a "
                f"{resultado_tsne.perplejidad_usada}, porque debe ser menor que el número de acciones "
                f"({len(retornos.columns)}).", icon=":material/info:",
            )
        st.write("El tamaño de cada punto refleja la capitalización de mercado de la acción.")
        tabla_tsne = pd.DataFrame(
            resultado_tsne.coordenadas, index=retornos.columns,
            columns=["Componente t-SNE 1", "Componente t-SNE 2"],
        )
        charts.mostrar(
            charts.dispersion_clusters(
                tabla_tsne, "Componente t-SNE 1", "Componente t-SNE 2", clusters.etiquetas, tamanos=capitalizacion
            ),
            clave="tsne_clusters",
        )
    with tabs[2]:
        tabla = pd.DataFrame({"Clúster": clusters.etiquetas, "Capitalización": capitalizacion.reindex(retornos.columns)})
        tabla["Capitalización"] = tabla["Capitalización"].map(lambda x: f.entero(x) if pd.notna(x) else "")
        st.dataframe(tabla, width="stretch")
        st.download_button(
            "Descargar clústeres (CSV)", estado.a_csv(tabla), "pca_clusters.csv", "text/csv",
            icon=":material/download:", key="descarga_pca",
        )
