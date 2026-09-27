"""Punto de entrada de la aplicación.

En Streamlit Community Cloud, este es el archivo principal del despliegue.
"""
import streamlit as st

from mltrading.config import settings
from mltrading.config import strings as S
from mltrading.ui.layout import configurar_pagina, inyectar_estilos
from mltrading.views import (
    backtesting, comparador, datos, diagnostico, diseno, division, en_vivo, evaluacion, guia, inicio,
    jerarquico_view, kmeans_view, modelos, pca_view, regresion, variables, xgboost_view,
)

configurar_pagina()
inyectar_estilos()

paginas = {
    S.NAV_SECCION_APLICACION: [
        st.Page(inicio.render, title=S.NAV_INICIO, icon=":material/home:", default=True),
        st.Page(datos.render, title=S.NAV_DATOS, icon=":material/database:", url_path="datos"),
    ],
    S.NAV_SECCION_PREPARACION: [
        st.Page(variables.render, title=S.NAV_VARIABLES, icon=":material/tune:", url_path="variables"),
        st.Page(division.render, title=S.NAV_DIVISION, icon=":material/call_split:", url_path="division"),
    ],
    S.NAV_SECCION_MODELADO: [
        st.Page(modelos.render, title=S.NAV_MODELOS, icon=":material/psychology:", url_path="modelos"),
        st.Page(regresion.render, title=S.NAV_REGRESION, icon=":material/show_chart:", url_path="regresion"),
        st.Page(xgboost_view.render, title=S.NAV_XGBOOST, icon=":material/bolt:", url_path="xgboost"),
        st.Page(evaluacion.render, title=S.NAV_EVALUACION, icon=":material/rule:", url_path="evaluacion"),
        st.Page(backtesting.render, title=S.NAV_BACKTESTING, icon=":material/candlestick_chart:", url_path="backtesting"),
        st.Page(comparador.render, title=S.NAV_COMPARADOR, icon=":material/compare_arrows:", url_path="comparador"),
    ],
    S.NAV_SECCION_NO_SUPERVISADO: [
        st.Page(kmeans_view.render, title=S.NAV_KMEANS, icon=":material/scatter_plot:", url_path="kmeans"),
        st.Page(jerarquico_view.render, title=S.NAV_JERARQUICO, icon=":material/account_tree:", url_path="jerarquico"),
        st.Page(pca_view.render, title=S.NAV_PCA, icon=":material/blur_on:", url_path="pca"),
    ],
    S.NAV_SECCION_OPERACION: [
        st.Page(en_vivo.render, title=S.NAV_EN_VIVO, icon=":material/sensors:", url_path="en-vivo"),
        st.Page(guia.render, title=S.NAV_GUIA, icon=":material/menu_book:", url_path="guia"),
    ],
    S.NAV_SECCION_SISTEMA: [
        st.Page(diseno.render, title=S.NAV_DISENO, icon=":material/palette:", url_path="diseno"),
    ],
}
if settings.MOSTRAR_DIAGNOSTICO:
    paginas[S.NAV_SECCION_SISTEMA].append(
        st.Page(
            diagnostico.render,
            title=S.NAV_DIAGNOSTICO,
            icon=":material/monitor_heart:",
            url_path="diagnostico",
        )
    )

st.navigation(paginas).run()
