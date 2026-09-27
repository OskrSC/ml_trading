"""Sistema de diseño: paleta, tipografía, componentes y gráficos en el tema activo."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.config import strings as S
from mltrading.config import tokens as t
from mltrading.ui import charts
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import format as f
from mltrading.ui.color import contraste
from mltrading.ui.tema import tema_actual

_UMBRAL_TEXTO = 4.5
_UMBRAL_GRAFICO = 3.0


def _filas_contraste() -> pd.DataFrame:
    filas = []
    for nombre, p in (("claro", t.CLARO), ("oscuro", t.OSCURO)):
        pares = [
            (f"Texto sobre fondo ({nombre})", p.texto, p.fondo, _UMBRAL_TEXTO),
            (f"Texto sobre fondo secundario ({nombre})", p.texto, p.fondo_secundario, _UMBRAL_TEXTO),
            (f"Texto blanco sobre primario ({nombre})", "#FFFFFF", p.primario, _UMBRAL_TEXTO),
            (f"Primario sobre fondo ({nombre})", p.primario, p.fondo, _UMBRAL_GRAFICO),
        ]
        peor_serie = min(t.SERIES + (t.COLOR_POSITIVO, t.COLOR_NEGATIVO), key=lambda col: contraste(col, p.fondo))
        pares.append((f"Series y semánticos, el más bajo ({nombre})", peor_serie, p.fondo, _UMBRAL_GRAFICO))
        for etiqueta, a, b, umbral in pares:
            razon = contraste(a, b)
            filas.append(
                {
                    "Combinación": etiqueta,
                    "Contraste": f.decimal(razon, 2),
                    "Mínimo": f.decimal(umbral, 1),
                    "Resultado": "Cumple" if razon >= umbral else "No cumple",
                }
            )
    return pd.DataFrame(filas)


def _paleta() -> None:
    c.seccion("Paleta", "Los colores de las series son idénticos en ambos temas.")
    for titulo, paleta in (("Tema claro", t.CLARO), ("Tema oscuro", t.OSCURO)):
        st.caption(titulo)
        html = "".join(
            c.muestra_color(nombre, valor)
            for nombre, valor in (
                ("Fondo", paleta.fondo),
                ("Fondo secundario", paleta.fondo_secundario),
                ("Texto", paleta.texto),
                ("Primario", paleta.primario),
                ("Borde", paleta.borde),
            )
        )
        st.markdown(f'<div class="mlt-muestras">{html}</div>', unsafe_allow_html=True)
    st.caption("Series y semánticos")
    nombres = ("Serie 1, estrategia", "Serie 2, referencia", "Serie 3", "Serie 4", "Serie 5", "Serie 6")
    html = "".join(c.muestra_color(n, v) for n, v in zip(nombres, t.SERIES))
    html += c.muestra_color("Positivo", t.COLOR_POSITIVO) + c.muestra_color("Negativo", t.COLOR_NEGATIVO)
    st.markdown(f'<div class="mlt-muestras">{html}</div>', unsafe_allow_html=True)
    st.caption("Contraste medido según WCAG 2.1")
    st.dataframe(_filas_contraste(), hide_index=True, width="stretch")


def _tipografia() -> None:
    c.seccion("Tipografía", "Títulos en una serif de texto y cuerpo en una sans humanista con cifras tabulares.")
    st.header("Título de sección")
    st.subheader("Título de bloque")
    st.write(
        "El texto de cuerpo mantiene una línea de lectura corta para que los párrafos "
        "se lean sin esfuerzo, incluso cuando se explica un modelo con varios pasos."
    )
    st.caption("Texto secundario para aclaraciones y fuentes.")
    st.code("cierre.pct_change().shift(-1) > 0", language="python")
    st.markdown("Cifras alineadas en columna:  \n`19.370`  \n`15.453`  \n`3.864`")


def _componentes() -> None:
    c.seccion("Componentes")
    columnas = st.columns(3)
    with columnas[0]:
        c.indicador("Filas", "19.370", "Barras de 15 minutos")
    with columnas[1]:
        c.indicador("Columnas", "5")
    with columnas[2]:
        c.indicador("Nulos", "0")

    st.success("Operación completada.", icon=":material/check_circle:")
    st.info("Mensaje informativo con una aclaración breve.", icon=":material/info:")
    st.warning("Advertencia sobre un dato que requiere atención.", icon=":material/warning:")
    st.error("Error con la causa y la forma de corregirlo.", icon=":material/error:")

    izq, der = st.columns(2)
    with izq:
        st.button("Acción principal", type="primary", key="diseno_primario")
        st.button("Acción secundaria", key="diseno_secundario")
        st.segmented_control("Modo", ["Predeterminado", "Personalizado"], default="Predeterminado", key="diseno_modo")
    with der:
        st.selectbox("Selector", ["Opción A", "Opción B", "Opción C"], key="diseno_selector")
        st.slider("Deslizador", 0, 100, 80, key="diseno_deslizador")
    with st.expander("Panel desplegable"):
        st.write("Contenido secundario que se muestra bajo demanda.")


def _graficos() -> None:
    c.seccion("Gráficos", "Calculados con los datos reales de JPMorgan y de las 20 acciones.")
    jpm, _ = dc.cargar("JPM_2017_2019.csv")
    diario = jpm["close"].resample("D").last().dropna()
    charts.mostrar(
        charts.linea(
            {"Cierre diario": diario, "Media móvil de 20 días": diario.rolling(20).mean()},
            altura=320,
            titulo_y="Precio (USD)",
        ),
        clave="diseno_linea",
    )
    st.caption("Rendimiento mensual de JPMorgan")
    mensual = jpm["close"].resample("ME").last().pct_change().dropna().rename("Rendimiento mensual")
    charts.mostrar(charts.barras_con_signo(mensual), clave="diseno_barras")
    st.caption("Correlación de rendimientos diarios entre diez acciones")
    pca, _ = dc.cargar("pca.csv")
    matriz = pca.iloc[:, :10].pct_change().dropna().corr()
    charts.mostrar(charts.mapa_calor(matriz), clave="diseno_calor")


def render() -> None:
    c.encabezado(S.DISENO_TITULO, S.DISENO_LEAD)
    st.caption(
        f"Tema detectado por la aplicación: {tema_actual()}. "
        "Este dato se actualiza en la siguiente interacción; los colores de la interfaz cambian al instante."
    )
    _paleta()
    _tipografia()
    _componentes()
    _graficos()
