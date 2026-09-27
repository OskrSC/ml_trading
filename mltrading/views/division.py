"""División de datos: entrenamiento y prueba, y por qué el orden temporal importa."""
from __future__ import annotations

import streamlit as st

from mltrading.core.features import division as div
from mltrading.core.features.indicadores import MotorNoDisponible
from mltrading.ui import charts, estado
from mltrading.ui import components as c
from mltrading.ui import datos_cache as dc
from mltrading.ui import format as f

TITULO = "División de datos"
LEAD = (
    "El modelo se entrena con una parte de los datos y se evalúa con otra que no ha visto. "
    "En series de tiempo, esa separación debe respetar el orden cronológico."
)
REFERENCIA = "Capítulo 4"
_INSIGNIA_PRED = ":green-badge[Resultado predeterminado]"
_INSIGNIA_PERS = ":orange-badge[Configuración personalizada]"


def _controles(modo: str) -> int:
    with st.container(border=True):
        if modo == estado.MODO_PREDETERMINADO:
            st.caption(
                f"Valores predeterminados: {estado.PROPORCION_PREDETERMINADA} % de las filas para entrenamiento, "
                f"el resto para prueba, en orden cronológico. Semilla del ejemplo aleatorio: {estado.SEMILLA_PREDETERMINADA}."
            )
            return estado.SEMILLA_PREDETERMINADA
        izq, der = st.columns(2)
        with izq:
            proporcion = st.slider("Proporción de entrenamiento (%)", 50, 95,
                                   int(st.session_state.get("_proporcion", estado.PROPORCION_PREDETERMINADA)),
                                   key="w_proporcion")
        with der:
            semilla = st.number_input("Semilla del ejemplo aleatorio", min_value=0, max_value=9999, step=1,
                                      value=int(st.session_state.get("_semilla", estado.SEMILLA_PREDETERMINADA)),
                                      key="w_semilla")
        st.session_state.update(_proporcion=int(proporcion), _semilla=int(semilla))
        return int(semilla)


def _tabla_resumen(division: div.Division) -> None:
    resumen = div.resumen(division)
    resumen["Inicio"] = resumen["Inicio"].map(lambda x: f"{x:%d/%m/%Y %H:%M}")
    resumen["Fin"] = resumen["Fin"].map(lambda x: f"{x:%d/%m/%Y %H:%M}")
    resumen["Filas"] = resumen["Filas"].map(f.entero)
    resumen["Señal 1 (%)"] = resumen["Señal 1 (%)"].map(lambda x: f.decimal(x, 1))
    st.dataframe(resumen, hide_index=True, width="stretch")


def render() -> None:
    c.encabezado(TITULO, LEAD, REFERENCIA)
    modo = estado.selector_modo("modo_division")
    semilla = _controles(modo)

    try:
        with st.spinner("Preparando las variables"):
            resultado = estado.resultado_variables()
    except (MotorNoDisponible, ValueError) as error:
        st.error(f"No se pudieron preparar las variables: {error}", icon=":material/error:")
        return
    division = estado.division_actual(resultado)
    predeterminado = estado.es_configuracion_predeterminada(resultado) and estado.es_division_predeterminada(division)
    st.markdown(_INSIGNIA_PRED if predeterminado else _INSIGNIA_PERS)
    st.caption(
        f"Se usan las {len(resultado.variables_finales)} variables finales de la página Variables y objetivo. "
        "Para cambiarlas, modifique esa página."
    )

    with st.container(key="indicadores"):
        columnas = st.columns(4)
        with columnas[0]:
            c.indicador("Filas de entrenamiento", f.entero(division.filas_entrenamiento))
        with columnas[1]:
            c.indicador("Filas de prueba", f.entero(division.filas_prueba))
        with columnas[2]:
            c.indicador("Primer dato de prueba", f"{division.X_prueba.index.min():%d/%m/%Y}")
        with columnas[3]:
            c.indicador("Señal 1 en prueba", f"{f.decimal(division.y_prueba.mean() * 100, 1)} %")

    c.seccion("Reparto cronológico", "El tramo de entrenamiento precede siempre al de prueba.")
    _tabla_resumen(division)
    ohlcv, _ = dc.cargar("JPM_2017_2019.csv")
    cierre = ohlcv["close"].loc[division.X_entrenamiento.index.min():]
    charts.mostrar(charts.precio_division(cierre, division.X_entrenamiento.index.max()), clave="precio_division")

    c.seccion(
        "Por qué no se mezclan los datos",
        "Con un reparto aleatorio, el modelo entrena con datos posteriores a los que luego debe predecir.",
    )
    variable = "pct_change" if "pct_change" in division.X_entrenamiento.columns else division.X_entrenamiento.columns[0]
    aleatoria = div.dividir_aleatoria(resultado.X, resultado.y, division.proporcion, semilla)
    izq, der = st.columns(2)
    for columna, particion, titulo in ((izq, division, "Cronológica"), (der, aleatoria, "Aleatoria")):
        with columna:
            st.markdown(f"**{titulo}**")
            charts.mostrar(
                charts.dispersion_division(particion.X_entrenamiento[variable], particion.X_prueba[variable],
                                           altura=260, titulo_y=variable),
                clave=f"dispersion_{titulo}",
            )
            fraccion = div.fraccion_prueba_anterior(particion)
            c.indicador("Prueba anterior al último dato de entrenamiento", f"{f.decimal(fraccion * 100, 1)} %",
                        "Fracción de filas de prueba con fecha anterior al último dato de entrenamiento.")
    st.info(
        "En el reparto cronológico ninguna fila de prueba es anterior al entrenamiento. En el aleatorio, la mayoría "
        "lo es: el modelo vería el futuro durante el entrenamiento y sus resultados resultarían demasiado optimistas.",
        icon=":material/info:",
    )
