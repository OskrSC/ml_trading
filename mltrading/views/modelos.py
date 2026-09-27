"""Modelos supervisados: elegir, configurar, entrenar e interpretar."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from mltrading.core.features.indicadores import MotorNoDisponible
from mltrading.core.observabilidad import obtener_logger
from mltrading.core.models.params import ParametroCategorico, ParametroEntero, ParametroFlotante
from mltrading.core.models.registry import MODELOS, obtener as obtener_modelo
from mltrading.ui import charts, estado
from mltrading.ui import components as c
from mltrading.ui import format as f

TITULO = "Modelos supervisados"
LEAD = (
    "Entrene un modelo de clasificación sobre las variables preparadas y véalo predecir la señal en el tramo "
    "de prueba. El resultado alimenta las páginas de Evaluación y Backtesting."
)
_INSIGNIA_PRED = ":green-badge[Resultado predeterminado]"
_INSIGNIA_PERS = ":orange-badge[Configuración personalizada]"
_TAMANO_MAXIMO_RECETA = 64 * 1024  # 64 KB: una receta es un JSON pequeño, no un modelo serializado


def _control(param, valores: dict) -> None:
    actual = valores.get(param.nombre, param.valor_predeterminado)
    if isinstance(param, ParametroEntero):
        valores[param.nombre] = st.number_input(
            param.etiqueta, min_value=param.minimo, max_value=param.maximo, step=param.paso,
            value=int(actual), help=param.ayuda or None, key=f"w_modelo_{param.nombre}",
        )
    elif isinstance(param, ParametroFlotante):
        valores[param.nombre] = st.number_input(
            param.etiqueta, min_value=param.minimo, max_value=param.maximo, step=param.paso, format=param.formato,
            value=float(actual), help=param.ayuda or None, key=f"w_modelo_{param.nombre}",
        )
    elif isinstance(param, ParametroCategorico):
        valores[param.nombre] = st.selectbox(
            param.etiqueta, param.opciones, index=param.opciones.index(actual), help=param.ayuda or None,
            key=f"w_modelo_{param.nombre}",
        )


def _restablecer(id_modelo: str) -> None:
    spec = obtener_modelo(id_modelo)
    for p in spec.parametros:
        st.session_state.pop(f"w_modelo_{p.nombre}", None)
    st.session_state.pop(f"_parametros_{id_modelo}", None)


def _panel_configuracion(spec, modo: str) -> None:
    with st.container(border=True):
        if modo == estado.MODO_PREDETERMINADO:
            resumen = ", ".join(f"{p.etiqueta.lower()} = {p.valor_predeterminado}" for p in spec.parametros)
            st.caption(f"Valores predeterminados: {resumen}.")
            return
        valores = dict(st.session_state.get(f"_parametros_{spec.id}", {}))
        columnas = st.columns(2)
        for i, param in enumerate(spec.parametros):
            with columnas[i % 2]:
                _control(param, valores)
        st.session_state[f"_parametros_{spec.id}"] = valores
        st.button("Restablecer valores predeterminados", key=f"restablecer_{spec.id}",
                  on_click=_restablecer, args=(spec.id,), icon=":material/restart_alt:")


def _tab_resumen(entrenamiento: dict) -> None:
    resultado = entrenamiento["resultado"]
    division = entrenamiento["division"]
    aciertos = (division.y_prueba.values == resultado.y_pred_prueba.values)
    columnas = st.columns(3)
    with columnas[0]:
        c.indicador("Exactitud en prueba", f"{f.decimal(aciertos.mean() * 100, 2)} %")
    with columnas[1]:
        c.indicador("Señal 1 predicha", f"{f.decimal(resultado.y_pred_prueba.mean() * 100, 1)} %")
    with columnas[2]:
        c.indicador("Filas de prueba", f.entero(len(resultado.y_pred_prueba)))
    for nota in resultado.notas:
        st.warning(nota, icon=":material/warning:")
    st.caption("Primeras predicciones sobre el tramo de prueba")
    muestra = pd.DataFrame(
        {"Real": division.y_prueba, "Predicho": resultado.y_pred_prueba}
    )
    if resultado.probabilidades_prueba is not None:
        muestra["Probabilidad de señal 1"] = resultado.probabilidades_prueba
    st.dataframe(muestra.head(8), width="stretch")


def _tab_interpretacion(spec, entrenamiento: dict) -> None:
    resultado = entrenamiento["resultado"]
    if resultado.interpretacion is None:
        st.caption("Este modelo no ofrece una interpretación adicional.")
        return
    tabla = resultado.interpretacion.copy()
    if spec.id == "logistic_regression":
        st.write("Un coeficiente positivo empuja la predicción hacia la señal 1; uno negativo, hacia la señal 0.")
        tabla["coeficiente"] = tabla["coeficiente"].round(4)
        charts.mostrar(
            charts.barras_horizontales(tabla, "variable", "coeficiente", colorear_signo=True),
            clave="coeficientes_modelo",
        )
        st.dataframe(tabla[["variable", "coeficiente"]], hide_index=True, width="stretch")
    else:
        st.write("La importancia mide cuánto contribuye cada variable a reducir el error del modelo.")
        tabla["importancia"] = tabla["importancia"].round(4)
        charts.mostrar(
            charts.barras_horizontales(tabla, "variable", "importancia"), clave="importancias_modelo"
        )
        st.dataframe(tabla, hide_index=True, width="stretch")


def _tab_predicciones(entrenamiento: dict) -> None:
    resultado = entrenamiento["resultado"]
    st.download_button(
        "Descargar predicciones (CSV)", estado.a_csv(resultado.y_pred_prueba.rename("signal")),
        "predicciones.csv", "text/csv", icon=":material/download:", key="descarga_predicciones",
    )
    st.dataframe(resultado.y_pred_prueba.rename("Señal predicha"), width="stretch")


def _cargar_receta_subida() -> bool:
    """Muestra el cargador de recetas. Devuelve True si se aplicó una receta en esta ejecución."""
    from mltrading.core.recipes.aplicar import aplicar_receta
    from mltrading.core.recipes.schema import RecetaInvalida, desde_json

    with st.expander("Cargar una receta (JSON)", icon=":material/upload_file:"):
        st.caption(
            "Una receta describe la configuración completa (modelo, variables y división), no un modelo ya "
            "entrenado. Al cargarla, esta aplicación recalcula todo desde los datos originales; nunca deserializa "
            "un objeto guardado con pickle o joblib."
        )
        archivo = st.file_uploader("Archivo de receta", type=["json"], key="w_receta_archivo")
        if archivo is None:
            return False
        if archivo.size > _TAMANO_MAXIMO_RECETA:
            st.error(
                f"El archivo pesa {archivo.size // 1024} KB; una receta no debería superar "
                f"{_TAMANO_MAXIMO_RECETA // 1024} KB. ¿Seguro que es una receta y no un modelo serializado?",
                icon=":material/error:",
            )
            return False
        try:
            receta = desde_json(archivo.getvalue().decode("utf-8"))
        except (RecetaInvalida, UnicodeDecodeError) as error:
            obtener_logger("recetas").warning("Receta rechazada: %s", error)
            st.error(f"La receta no es válida: {error}", icon=":material/error:")
            return False
        try:
            with st.spinner("Aplicando la receta: recalculando variables, división y modelo"):
                from mltrading.ui import datos_cache as dc

                ohlcv, _ = dc.cargar("JPM_2017_2019.csv")
                resultado = aplicar_receta(receta, ohlcv)
        except (KeyError, MotorNoDisponible, ValueError) as error:
            obtener_logger("recetas").exception("Fallo al aplicar una receta: %s", error)
            st.error(f"No se pudo aplicar la receta: {error}", icon=":material/error:")
            return False
        estado.recordar_desde_receta(
            receta.id_modelo, receta.parametros_modelo, resultado.division, resultado.resultado_entrenamiento, receta
        )
        st.session_state["w_modelo_elegido"] = receta.id_modelo
        st.success("Receta aplicada. Vea los resultados más abajo.", icon=":material/check_circle:")
        return True


def _tab_receta(entrenamiento: dict) -> None:
    from mltrading.core.recipes.schema import TIPO_MODELO_SUPERVISADO, VERSION_ESQUEMA, RecetaModelo

    config_vars = entrenamiento.get("config_variables")
    descartes = entrenamiento.get("descartes_correlacion")
    if config_vars is None or descartes is None:
        st.info(
            "Este entrenamiento no guardó la configuración de variables (por ejemplo, vino de una receta "
            "cargada). Entrene desde esta página para poder descargar su receta.", icon=":material/info:",
        )
        return
    receta = RecetaModelo(
        tipo=TIPO_MODELO_SUPERVISADO, version=VERSION_ESQUEMA, id_modelo=entrenamiento["id_modelo"],
        parametros_modelo=entrenamiento["parametros"], barras_por_dia=config_vars.barras_por_dia,
        umbral_adf=config_vars.umbral_adf, umbral_correlacion=config_vars.umbral_correlacion,
        descartes_correlacion=descartes, proporcion_entrenamiento=entrenamiento["proporcion_entrenamiento"],
    )
    st.write(
        "Una receta guarda la configuración, no el modelo entrenado. Al cargarla, se recalcula todo desde los "
        "datos originales, de forma determinista."
    )
    st.code(receta.a_json(), language="json")
    st.download_button(
        "Descargar receta (JSON)", receta.a_json(), f"receta_{entrenamiento['id_modelo']}.json",
        "application/json", icon=":material/download:", key="descarga_receta",
    )


def render() -> None:
    c.encabezado(TITULO, LEAD)
    _cargar_receta_subida()
    modo = estado.selector_modo("modo_modelos")

    etiquetas = {m.id: f"{m.nombre} ({m.capitulo})" for m in MODELOS}
    id_modelo = st.selectbox("Modelo", list(etiquetas), format_func=lambda x: etiquetas[x], key="w_modelo_elegido")
    spec = obtener_modelo(id_modelo)
    st.caption(spec.descripcion)
    _panel_configuracion(spec, modo)

    try:
        with st.spinner("Preparando las variables y la división"):
            resultado_vars = estado.resultado_variables()
            division = estado.division_actual(resultado_vars)
    except (MotorNoDisponible, ValueError) as error:
        st.error(f"No se pudo preparar los datos: {error}", icon=":material/error:")
        return

    st.caption(
        f"Se entrena con las {len(resultado_vars.variables_finales)} variables finales y la división de la página "
        "anterior. Para cambiarlas, use las páginas Variables y objetivo o División de datos."
    )
    if spec.necesita_escalado:
        st.caption("Este modelo necesita variables escaladas. El escalador se ajusta solo con el tramo de entrenamiento.")

    entrenar = st.button(f"Entrenar {spec.nombre}", type="primary", key="entrenar_modelo", icon=":material/play_arrow:")
    if entrenar:
        with st.spinner(f"Entrenando {spec.nombre}"):
            try:
                estado.entrenar_y_recordar(id_modelo, division, resultado_vars)
            except Exception as error:  # noqa: BLE001 - se muestra el motivo a la persona
                obtener_logger("modelos").exception("Fallo al entrenar %s: %s", id_modelo, error)
                st.error(f"No se pudo entrenar el modelo: {error}", icon=":material/error:")
                return

    entrenamiento = estado.entrenamiento_actual()
    if not entrenamiento or entrenamiento["id_modelo"] != id_modelo:
        st.info("Pulse «Entrenar» para ver los resultados de este modelo.", icon=":material/info:")
        return

    st.markdown(_INSIGNIA_PRED if entrenamiento["predeterminado"] else _INSIGNIA_PERS)
    tabs = st.tabs(["Resumen", "Interpretación", "Predicciones", "Receta"])
    with tabs[0]:
        _tab_resumen(entrenamiento)
    with tabs[1]:
        _tab_interpretacion(spec, entrenamiento)
    with tabs[2]:
        _tab_predicciones(entrenamiento)
    with tabs[3]:
        _tab_receta(entrenamiento)

    st.divider()
    st.caption("Con este modelo entrenado, continúe en Evaluación o en Backtesting.")
