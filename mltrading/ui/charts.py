"""Gráficos con Plotly.

Los colores de series y los semánticos vienen de los tokens y son iguales en
ambos temas. El fondo, la rejilla y el texto los aplica Streamlit según el tema.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mltrading.config import tokens as t

_UMBRAL_WEBGL = 5000


def _base(fig: go.Figure, altura: int) -> go.Figure:
    fig.update_layout(
        height=altura,
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        colorway=list(t.SERIES),
    )
    return fig


def linea(
    series: dict[str, pd.Series],
    altura: int = 360,
    selector_rango: bool = False,
    titulo_y: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    for i, (nombre, serie) in enumerate(series.items()):
        # El selector de rango no dibuja trazas WebGL, así que con él se usa SVG.
        clase = go.Scattergl if len(serie) > _UMBRAL_WEBGL and not selector_rango else go.Scatter
        fig.add_trace(
            clase(
                x=serie.index,
                y=serie.values,
                name=nombre,
                mode="lines",
                line=dict(color=t.SERIES[i % len(t.SERIES)], width=1.6),
            )
        )
    fig.update_yaxes(title_text=titulo_y)
    if selector_rango:
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08))
    return _base(fig, altura)


def barras_con_signo(serie: pd.Series, altura: int = 300, formato_y: str = ".1%") -> go.Figure:
    colores = [t.COLOR_POSITIVO if v >= 0 else t.COLOR_NEGATIVO for v in serie.values]
    fig = go.Figure(go.Bar(x=serie.index, y=serie.values, marker_color=colores, name=serie.name or ""))
    fig.update_yaxes(tickformat=formato_y, zeroline=True)
    fig.update_layout(showlegend=False)
    return _base(fig, altura)


def mapa_calor(matriz: pd.DataFrame, altura: int = 420) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matriz.values,
            x=list(matriz.columns),
            y=list(matriz.index),
            zmin=-1,
            zmax=1,
            colorscale=[list(par) for par in t.ESCALA_DIVERGENTE],
            colorbar=dict(title="Correlación", thickness=12),
            hovertemplate="%{y} y %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(hovermode="closest")
    return _base(fig, altura)


def precio_division(cierre: pd.Series, entrenamiento_fin: pd.Timestamp, altura: int = 320) -> go.Figure:
    """Precio de cierre con el tramo de entrenamiento y el de prueba en colores distintos."""
    fig = go.Figure()
    for i, (nombre, tramo) in enumerate(
        (("Entrenamiento", cierre.loc[:entrenamiento_fin]), ("Prueba", cierre.loc[entrenamiento_fin:]))
    ):
        fig.add_trace(
            go.Scatter(x=tramo.index, y=tramo.values, name=nombre, mode="lines",
                       line=dict(color=t.SERIES[i], width=1.6))
        )
    fig.add_shape(type="line", x0=entrenamiento_fin, x1=entrenamiento_fin, y0=0, y1=1, yref="paper",
                  line=dict(color=t.SERIES[5], width=1, dash="dash"))
    fig.update_yaxes(title_text="Precio (USD)")
    return _base(fig, altura)


def dispersion_division(entrenamiento: pd.Series, prueba: pd.Series, altura: int = 300, titulo_y: str = "") -> go.Figure:
    """Puntos de una variable, coloreados según pertenezcan a entrenamiento o a prueba."""
    fig = go.Figure()
    for i, (nombre, serie) in enumerate((("Entrenamiento", entrenamiento), ("Prueba", prueba))):
        fig.add_trace(
            go.Scattergl(x=serie.index, y=serie.values, name=nombre, mode="markers",
                         marker=dict(color=t.SERIES[i], size=3, opacity=0.8))
        )
    fig.update_yaxes(title_text=titulo_y)
    fig.update_layout(hovermode="closest")
    return _base(fig, altura)


def curva_capital(datos, altura: int = 340):
    """Curva de capital de la estrategia contra comprar y mantener."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos.index, y=datos["cumulative_returns"], name="Estrategia",
                             mode="lines", line=dict(color=t.COLOR_ESTRATEGIA, width=1.8)))
    fig.add_trace(go.Scatter(x=datos.index, y=datos["benchmark_cumulative_returns"], name="Comprar y mantener",
                             mode="lines", line=dict(color=t.COLOR_REFERENCIA, width=1.4, dash="dot")))
    fig.update_yaxes(title_text="Capital acumulado (base 1)")
    return _base(fig, altura)


def drawdown(datos, altura: int = 220):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos.index, y=datos["drawdown"], mode="lines", name="Drawdown",
                             line=dict(color=t.COLOR_NEGATIVO, width=1.2), fill="tozeroy",
                             fillcolor="rgba(210,85,75,0.18)"))
    fig.update_yaxes(title_text="Drawdown (%)")
    fig.update_layout(showlegend=False)
    return _base(fig, altura)


def matriz_confusion(matriz, etiquetas=("Sin posición", "Posición larga"), altura: int = 360):
    z = matriz
    texto = [[str(v) for v in fila] for fila in z]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(etiquetas), y=list(etiquetas), text=texto, texttemplate="%{text}",
            colorscale=[[0, "#EBEEF4"], [1, t.SERIES[0]]], showscale=False,
            hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Casos: %{z}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Predicho")
    fig.update_yaxes(title_text="Real", autorange="reversed")
    return _base(fig, altura)


def precision_temporal(aciertos, altura: int = 160):
    """Línea de puntos: acierto y fallo se distinguen por color, símbolo y posición.

    El color por sí solo no basta para una persona con daltonismo: los aciertos
    también usan un círculo relleno en la fila superior y los fallos una cruz en
    la inferior, con su propia etiqueta en el texto emergente.
    """
    fig = go.Figure()
    for correcto, nombre, color, simbolo, y in (
        (True, "Acierto", t.COLOR_POSITIVO, "circle", 1.15),
        (False, "Fallo", t.COLOR_NEGATIVO, "x", 0.85),
    ):
        m = aciertos.values == correcto
        if not m.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=aciertos.index[m], y=[y] * int(m.sum()), mode="markers", name=nombre,
                marker=dict(color=color, size=6, opacity=0.75, symbol=simbolo, line=dict(width=1, color=color)),
                hovertemplate=f"{nombre}<br>%{{x}}<extra></extra>",
            )
        )
    fig.update_yaxes(visible=False, range=[0.6, 1.4])
    fig.update_layout(
        showlegend=True, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return _base(fig, altura)


def barras_horizontales(tabla: pd.DataFrame, columna_etiqueta: str, columna_valor: str, altura: int = 320,
                        colorear_signo: bool = False):
    orden = tabla.sort_values(columna_valor)
    colores = (
        [t.COLOR_POSITIVO if v >= 0 else t.COLOR_NEGATIVO for v in orden[columna_valor]]
        if colorear_signo else t.SERIES[0]
    )
    fig = go.Figure(go.Bar(x=orden[columna_valor], y=orden[columna_etiqueta], orientation="h", marker_color=colores))
    fig.update_layout(showlegend=False)
    return _base(fig, altura)


def dispersion_regresion(datos: pd.DataFrame, columna_x: str, columna_y: str, altura: int = 380) -> go.Figure:
    """Puntos observados y la recta ajustada, ordenada por x para dibujarla sin zigzags."""
    ordenado = datos.sort_values(columna_x)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=datos[columna_x], y=datos[columna_y], mode="markers", name="Observado",
                    marker=dict(color=t.SERIES[0], size=5, opacity=0.6))
    )
    fig.add_trace(
        go.Scatter(x=ordenado[columna_x], y=ordenado["predicho"], mode="lines", name="Ajuste",
                   line=dict(color=t.SERIES[1], width=2.2))
    )
    fig.update_xaxes(title_text=columna_x)
    fig.update_yaxes(title_text=columna_y)
    return _base(fig, altura)


def dispersion_prediccion(y_real: pd.Series, y_predicho: pd.Series, altura: int = 340) -> go.Figure:
    """Predicho contra real, con la diagonal de un ajuste perfecto como referencia."""
    minimo, maximo = float(min(y_real.min(), y_predicho.min())), float(max(y_real.max(), y_predicho.max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=y_real, y=y_predicho, mode="markers", name="Casos",
                    marker=dict(color=t.SERIES[0], size=4, opacity=0.5))
    )
    fig.add_trace(
        go.Scatter(x=[minimo, maximo], y=[minimo, maximo], mode="lines", name="Ajuste perfecto",
                   line=dict(color=t.SERIES[5], width=1.4, dash="dash"))
    )
    fig.update_xaxes(title_text="Real")
    fig.update_yaxes(title_text="Predicho")
    return _base(fig, altura)


def curvas_multiples(series: dict, altura: int = 360, titulo_y: str = "Capital acumulado (base 1)") -> go.Figure:
    """Varias curvas superpuestas (comparador de modelos o portafolio multiactivo)."""
    fig = go.Figure()
    for i, (nombre, serie) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(x=list(range(len(serie))), y=serie.values, name=nombre, mode="lines",
                      line=dict(color=t.SERIES[i % len(t.SERIES)], width=1.6))
        )
    fig.update_yaxes(title_text=titulo_y)
    fig.update_xaxes(title_text="Barras de prueba, en orden")
    return _base(fig, altura)


def dispersion_clusters(
    datos: pd.DataFrame, columna_x: str, columna_y: str, etiquetas: pd.Series, altura: int = 420,
    tamanos: pd.Series | None = None,
) -> go.Figure:
    """Dispersión con un color por clúster y el nombre de cada punto."""
    fig = go.Figure()
    for cluster in sorted(etiquetas.unique()):
        m = etiquetas == cluster
        indice = etiquetas.index[m]
        marker = dict(color=t.SERIES[int(cluster) % len(t.SERIES)], size=10)
        if tamanos is not None:
            escala = tamanos.reindex(indice).fillna(tamanos.median())
            marker["size"] = 8 + 22 * (escala - escala.min()) / max(escala.max() - escala.min(), 1e-9)
        fig.add_trace(
            go.Scatter(
                x=datos.loc[indice, columna_x], y=datos.loc[indice, columna_y], mode="markers+text",
                text=list(indice), textposition="top center", textfont=dict(size=10),
                name=f"Clúster {cluster}", marker=marker,
            )
        )
    fig.update_xaxes(title_text=columna_x)
    fig.update_yaxes(title_text=columna_y)
    return _base(fig, altura)


def curva_codo(tabla, altura: int = 320) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=tabla["k"], y=tabla["inercia"], mode="lines+markers",
                  line=dict(color=t.SERIES[0], width=1.8), marker=dict(size=7))
    )
    fig.update_xaxes(title_text="k", dtick=1)
    fig.update_yaxes(title_text="Inercia")
    fig.update_layout(showlegend=False)
    return _base(fig, altura)


def varianza_explicada(serie: pd.Series, altura: int = 320) -> go.Figure:
    acumulada = serie.cumsum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=serie.index, y=serie.values, name="Por componente", marker_color=t.SERIES[0]))
    fig.add_trace(
        go.Scatter(x=acumulada.index, y=acumulada.values, name="Acumulada", mode="lines+markers",
                  line=dict(color=t.SERIES[1], width=1.8), yaxis="y2")
    )
    fig.update_layout(
        yaxis=dict(title="Varianza explicada", tickformat=".0%"),
        yaxis2=dict(title="Acumulada", tickformat=".0%", overlaying="y", side="right", range=[0, 1.02]),
        xaxis=dict(title="Componente", dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return _base(fig, altura)


def dendrograma(matriz_enlace, etiquetas: list[str], altura: int = 420) -> go.Figure:
    """Dendrograma dibujado a mano a partir de una matriz de enlace de scipy.

    No usa `plotly.figure_factory.create_dendrogram`: esa función interna llama a
    `scipy.array`, retirado de scipy hace tiempo, y rompe con versiones de plotly
    anteriores a la que fija `requirements.txt` (ver docs/solucion_problemas.md).
    Aquí se toman las coordenadas de `scipy.cluster.hierarchy.dendrogram` con
    `no_plot=True` (scipy no dibuja nada, solo calcula) y se dibujan como líneas
    de Plotly, con control total sobre el color y el tema.
    """
    from scipy.cluster.hierarchy import dendrogram as _dendrogram_scipy

    calculado = _dendrogram_scipy(matriz_enlace, labels=etiquetas, no_plot=True)
    fig = go.Figure()
    for icoord, dcoord in zip(calculado["icoord"], calculado["dcoord"]):
        fig.add_trace(
            go.Scatter(
                x=icoord, y=dcoord, mode="lines", line=dict(color=t.SERIES[0], width=1.4),
                hoverinfo="skip", showlegend=False,
            )
        )
    hojas = calculado["ivl"]
    posiciones = [5 + 10 * i for i in range(len(hojas))]
    fig.update_xaxes(
        tickmode="array", tickvals=posiciones, ticktext=hojas, range=[-2, 10 * len(hojas) + 2]
    )
    fig.update_yaxes(title_text="Distancia euclidiana", rangemode="tozero")
    fig.update_layout(showlegend=False)
    return _base(fig, altura)


def mostrar(fig: go.Figure, clave: str | None = None) -> None:
    st.plotly_chart(
        fig,
        theme="streamlit",
        width="stretch",
        key=clave,
        config={"displaylogo": False, "responsive": True},
    )
