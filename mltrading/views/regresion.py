"""Modelos de regresión: regresión lineal (capítulo 9) y árbol de regresión (capítulo 12)."""
from __future__ import annotations

import streamlit as st

from mltrading.core.features.indicadores import MotorNoDisponible
from mltrading.core.models import decision_tree_regressor as arbol
from mltrading.core.models import linear_regression as lineal
from mltrading.core.models.params import valores_predeterminados
from mltrading.ui import charts, estado
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import format as f

TITULO = "Regresión"
LEAD = (
    "A diferencia de los modelos anteriores, aquí no se predice una señal de compra o venta: se predice un "
    "número. La regresión lineal estima el precio de una acción a partir de otra; el árbol de regresión "
    "estima el retorno de la barra siguiente."
)
REFERENCIA = "Capítulos 9 y 12"
_INSIGNIA_PRED = ":green-badge[Resultado predeterminado]"
_INSIGNIA_PERS = ":orange-badge[Configuración personalizada]"


def _tab_lineal() -> None:
    st.write(
        "Se ajusta una recta que explica el precio de JPMorgan a partir del precio de otra acción, por mínimos "
        "cuadrados ordinarios. El R² indica qué proporción de la variación de JPMorgan queda explicada por esa "
        "otra acción."
    )
    opciones = {
        "Bank of America (2019)": ("jpm_and_bac_price_2019.csv", "BAC Close", "Observed JPM"),
        "Nestlé (2019)": ("predicted_jpm_and_nestle_price_2019.csv", "Nestle Close", "Observed JPM"),
    }
    eleccion = st.selectbox("Par de acciones", list(opciones), key="w_lineal_par")
    archivo, columna_x, columna_y = opciones[eleccion]
    datos, _ = dc.cargar(archivo)
    resultado = lineal.ajustar_ols(datos, columna_x, columna_y)

    st.markdown(_INSIGNIA_PRED)
    columnas = st.columns(3)
    with columnas[0]:
        c.indicador("R²", f.decimal(resultado.r_cuadrado, 2))
    with columnas[1]:
        c.indicador("Pendiente", f.decimal(resultado.pendiente, 4))
    with columnas[2]:
        c.indicador("Intersección", f.decimal(resultado.interseccion, 2))
    st.caption(
        f"JPM ≈ {f.decimal(resultado.interseccion, 2)} + {f.decimal(resultado.pendiente, 4)} × "
        f"{columna_x.replace(' Close', '')}."
    )
    charts.mostrar(charts.dispersion_regresion(resultado.datos, columna_x, columna_y), clave="dispersion_ols")
    if resultado.r_cuadrado >= 0.6:
        st.success(
            f"Un R² de {f.decimal(resultado.r_cuadrado, 2)} indica que buena parte de la variación de JPMorgan "
            "queda explicada por esta acción.", icon=":material/check_circle:",
        )
    else:
        st.warning(
            f"Un R² de {f.decimal(resultado.r_cuadrado, 2)} es bajo: la mayor parte de la variación de JPMorgan "
            "no queda explicada por esta acción.", icon=":material/warning:",
        )


def _tab_arbol() -> None:
    st.write(
        "Se entrena sobre las mismas variables y la misma división que las páginas anteriores, pero el objetivo "
        "cambia: en lugar de la señal, predice el cambio porcentual de la barra siguiente."
    )
    modo = estado.selector_modo("modo_arbol_regresion")
    parametros = valores_predeterminados(arbol.PARAMETROS)
    with st.container(border=True):
        if modo == estado.MODO_PREDETERMINADO:
            st.caption(f"Valor predeterminado: mínimo de casos por hoja = {parametros['min_samples_leaf']}.")
        else:
            parametros["min_samples_leaf"] = st.number_input(
                "Mínimo de casos por hoja", min_value=5, max_value=2000, step=5,
                value=int(st.session_state.get("_arbol_hoja", parametros["min_samples_leaf"])),
                key="w_arbol_hoja",
            )
            st.session_state["_arbol_hoja"] = parametros["min_samples_leaf"]

    try:
        with st.spinner("Preparando las variables"):
            resultado_vars = estado.resultado_variables()
    except (MotorNoDisponible, ValueError) as error:
        st.error(f"No se pudo preparar los datos: {error}", icon=":material/error:")
        return
    if "pct_change" not in resultado_vars.variables_finales:
        st.warning(
            "El árbol de regresión necesita la variable «pct_change» entre las variables finales. "
            "Revísela en la página Variables y objetivo.", icon=":material/warning:",
        )
        return

    division = estado.division_actual(resultado_vars)
    objetivo = arbol.construir_objetivo(resultado_vars.candidatas.variables)
    datos = division.X_entrenamiento.join(objetivo, how="left")
    y_entrenamiento = datos["objetivo_regresion"]
    datos_prueba = division.X_prueba.join(objetivo, how="left")
    y_prueba = datos_prueba["objetivo_regresion"]
    validos_tr = y_entrenamiento.notna()
    validos_te = y_prueba.notna()

    if st.button("Entrenar árbol de regresión", type="primary", key="entrenar_arbol_regresion",
                icon=":material/play_arrow:"):
        with st.spinner("Entrenando"):
            resultado = arbol.entrenar(
                division.X_entrenamiento[validos_tr], y_entrenamiento[validos_tr],
                division.X_prueba[validos_te], y_prueba[validos_te], parametros,
            )
            st.session_state["_arbol_regresion"] = {
                "resultado": resultado, "y_prueba": y_prueba[validos_te],
                "predeterminado": modo == estado.MODO_PREDETERMINADO and parametros["min_samples_leaf"] == 200,
            }

    guardado = st.session_state.get("_arbol_regresion")
    if not guardado:
        st.info("Pulse «Entrenar árbol de regresión» para ver los resultados.", icon=":material/info:")
        return

    resultado = guardado["resultado"]
    st.markdown(_INSIGNIA_PRED if guardado["predeterminado"] else _INSIGNIA_PERS)
    columnas = st.columns(2)
    with columnas[0]:
        c.indicador("R² en prueba", f.decimal(resultado.r_cuadrado, 4),
                    "Puede ser negativo: significa que el modelo predice peor que usar siempre el promedio.")
    with columnas[1]:
        c.indicador("Error cuadrático medio", f"{resultado.error_cuadratico_medio:.2e}")
    charts.mostrar(
        charts.dispersion_prediccion(guardado["y_prueba"], resultado.y_pred_prueba), clave="dispersion_arbol"
    )
    c.seccion("Importancia de las variables")
    tabla = resultado.interpretacion.copy()
    tabla["importancia"] = tabla["importancia"].round(4)
    charts.mostrar(charts.barras_horizontales(tabla, "variable", "importancia"), clave="importancias_arbol")


def render() -> None:
    c.encabezado(TITULO, LEAD, REFERENCIA)
    tabs = st.tabs(["Regresión lineal", "Árbol de regresión"])
    with tabs[0]:
        _tab_lineal()
    with tabs[1]:
        _tab_arbol()
