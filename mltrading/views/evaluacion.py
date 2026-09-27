"""Evaluación del modelo entrenado (capítulo 6)."""
from __future__ import annotations

import streamlit as st

from mltrading.core.evaluation.metrics import evaluar
from mltrading.ui import charts, estado
from mltrading.ui import components as c
from mltrading.ui import format as f

TITULO = "Evaluación"
LEAD = (
    "La exactitud por sí sola puede engañar. La matriz de confusión y el informe de clasificación muestran "
    "en qué se equivoca el modelo y con qué frecuencia."
)
REFERENCIA = "Capítulo 6"
_ETIQUETAS = ("Sin posición (0)", "Posición larga (1)")


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
    division = entrenamiento["division"]
    st.caption(f"Modelo evaluado: {entrenamiento['id_modelo'].replace('_', ' ').title()}.")

    ev = evaluar(division.y_prueba, resultado.y_pred_prueba)

    columnas = st.columns(4)
    with columnas[0]:
        c.indicador("Exactitud", f"{f.decimal(ev.exactitud * 100, 2)} %")
    with columnas[1]:
        aciertos_clase_1 = ev.reporte.loc[ev.reporte["Clase"] == "Posición larga (1)", "Sensibilidad"].iloc[0]
        c.indicador("Sensibilidad, posición larga", f"{f.decimal(aciertos_clase_1 * 100, 1)} %")
    with columnas[2]:
        aciertos_clase_0 = ev.reporte.loc[ev.reporte["Clase"] == "Sin posición (0)", "Sensibilidad"].iloc[0]
        c.indicador("Sensibilidad, sin posición", f"{f.decimal(aciertos_clase_0 * 100, 1)} %")
    with columnas[3]:
        c.indicador("Casos evaluados", f.entero(len(ev.y_real)))

    c.seccion("Precisión a lo largo del tiempo", "Cada punto es una barra de prueba; en rojo, donde el modelo falló.")
    charts.mostrar(charts.precision_temporal(ev.aciertos), clave="precision_temporal")

    izquierda, derecha = st.columns([2, 3], gap="large")
    with izquierda:
        c.seccion("Matriz de confusión")
        charts.mostrar(charts.matriz_confusion(ev.matriz_confusion, _ETIQUETAS), clave="matriz_confusion")
    with derecha:
        c.seccion("Informe de clasificación")
        tabla = ev.reporte.copy()
        for col in ("Precisión", "Sensibilidad", "F1"):
            tabla[col] = tabla[col].map(lambda x: f.decimal(x * 100, 1) + " %")
        tabla["Soporte"] = tabla["Soporte"].map(f.entero)
        st.dataframe(tabla, hide_index=True, width="stretch")
        st.caption(
            "Precisión: de lo que el modelo marcó como esa clase, cuánto acertó. "
            "Sensibilidad: de lo que realmente era esa clase, cuánto detectó el modelo."
        )

    st.download_button(
        "Descargar evaluación (CSV)", estado.a_csv(ev.reporte.set_index("Clase")), "evaluacion.csv", "text/csv",
        icon=":material/download:", key="descarga_evaluacion",
    )
