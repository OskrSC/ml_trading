"""Pruebas de humo: cada página se ejecuta completa sin lanzar excepciones."""
import pytest
from streamlit.testing.v1 import AppTest

from reglas_texto import RAIZ

PAGINAS = [
    "inicio", "datos", "variables", "division", "modelos", "regresion", "xgboost_view", "evaluacion",
    "backtesting", "comparador", "kmeans_view", "jerarquico_view", "pca_view", "en_vivo", "guia", "diseno",
    "diagnostico",
]


def _app(pagina: str) -> AppTest:
    return AppTest.from_string(f"from mltrading.views.{pagina} import render\nrender()", default_timeout=90)


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_se_ejecuta_sin_errores(pagina):
    at = _app(pagina).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1


def test_inicio_muestra_los_indicadores():
    at = _app("inicio").run()
    etiquetas = [m.label for m in at.metric]
    assert "Conjuntos de datos" in etiquetas
    assert [m.value for m in at.metric if m.label == "Conjuntos de datos"] == ["16"]
    assert [m.value for m in at.metric if m.label == "Barras de JPMorgan"] == ["19.370"]


def test_catalogo_permite_cambiar_de_conjunto():
    at = _app("datos").run()
    selector = at.selectbox(key="datos_archivo")
    assert len(selector.options) == 16
    selector.select("RELIANCE.NS.csv").run()
    assert not at.exception
    assert any("nulos" in w.value for w in at.warning)


def test_catalogo_filtra_por_tipo():
    at = _app("datos").run()
    at.get("button_group")[0].set_value("Artefacto predeterminado").run()
    assert not at.exception
    assert len(at.selectbox(key="datos_archivo").options) == 7


def test_diagnostico_ejecuta_las_mediciones():
    at = _app("diagnostico").run()
    at.button(key="diag_medir_cargas").click().run()
    assert not at.exception
    at.button(key="diag_medir_import").click().run()
    assert not at.exception


def test_punto_de_entrada_con_navegacion():
    at = AppTest.from_file(str(RAIZ / "streamlit_app.py"), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == ["Aprendizaje automático en trading"]


def _boton_modo(at, clave):
    return next(b for b in at.get("button_group") if b.key == clave)


def test_variables_muestra_el_resultado_predeterminado():
    at = _app("variables").run()
    assert not at.exception
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Filas utilizables"] == "19.317"
    assert metricas["Variables finales"] == "7"
    assert any("Resultado predeterminado" in m.value for m in at.markdown)


def test_variables_personalizado_cambia_la_marca_y_las_filas():
    at = _app("variables").run()
    _boton_modo(at, "modo_variables").set_value("Personalizado").run()
    assert not at.exception
    at.number_input(key="w_barras").set_value(20).run()
    assert not at.exception, [e.value for e in at.exception]
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Filas utilizables"] == "19.329"  # 19370 - 40 - 1
    assert any("Configuración personalizada" in m.value for m in at.markdown)


def test_variables_restablecer_vuelve_al_predeterminado():
    at = _app("variables").run()
    _boton_modo(at, "modo_variables").set_value("Personalizado").run()
    at.number_input(key="w_umbral_corr").set_value(0.9).run()
    assert any("Configuración personalizada" in m.value for m in at.markdown)
    at.button(key="restablecer_variables").click().run()
    assert not at.exception
    assert any("Resultado predeterminado" in m.value for m in at.markdown)


def test_division_predeterminada_y_personalizada():
    at = _app("division").run()
    assert not at.exception
    metricas = {m.label: m.value for m in at.metric}
    assert (metricas["Filas de entrenamiento"], metricas["Filas de prueba"]) == ("15.453", "3.864")
    assert any("Resultado predeterminado" in m.value for m in at.markdown)
    _boton_modo(at, "modo_division").set_value("Personalizado").run()
    at.slider(key="w_proporcion").set_value(70).run()
    assert not at.exception
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Filas de entrenamiento"] == "13.521"
    assert any("Configuración personalizada" in m.value for m in at.markdown)


def test_evaluacion_y_backtesting_piden_entrenar_primero():
    for pagina in ("evaluacion", "backtesting"):
        at = _app(pagina).run()
        assert not at.exception
        assert any("entrenar" in i.value.lower() for i in at.info)


def test_flujo_completo_random_forest_predeterminado():
    at = _app("modelos").run()
    at.button(key="entrenar_modelo").click().run()
    assert not at.exception, [e.value for e in at.exception]
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Exactitud en prueba"] == "51,55 %"
    assert any("Resultado predeterminado" in m.value for m in at.markdown)
    entrenamiento = at.session_state["_entrenamiento"]

    at_ev = _app("evaluacion").run()
    at_ev.session_state["_entrenamiento"] = entrenamiento
    at_ev.run()
    assert not at_ev.exception, [e.value for e in at_ev.exception]
    assert {m.label: m.value for m in at_ev.metric}["Exactitud"] == "51,55 %"

    at_bt = _app("backtesting").run()
    at_bt.session_state["_entrenamiento"] = entrenamiento
    at_bt.run()
    assert not at_bt.exception, [e.value for e in at_bt.exception]
    metricas_bt = {m.label: m.value for m in at_bt.metric}
    assert metricas_bt["Retorno acumulado"] == "28,10 %"
    assert metricas_bt["Sharpe"] == "2,89"
    assert metricas_bt["Total de operaciones"] == "169"
    assert any("Resultado predeterminado" in m.value for m in at_bt.markdown)


def test_flujo_completo_regresion_logistica_predeterminada():
    at = _app("modelos").run()
    at.selectbox(key="w_modelo_elegido").select("logistic_regression").run()
    at.button(key="entrenar_modelo").click().run()
    assert not at.exception, [e.value for e in at.exception]
    entrenamiento = at.session_state["_entrenamiento"]

    at_bt = _app("backtesting").run()
    at_bt.session_state["_entrenamiento"] = entrenamiento
    at_bt.run()
    assert not at_bt.exception
    metricas_bt = {m.label: m.value for m in at_bt.metric}
    assert metricas_bt["Sharpe"] == "2,75"


def test_modelo_personalizado_marca_configuracion_personalizada():
    at = _app("modelos").run()
    _boton_modo(at, "modo_modelos").set_value("Personalizado").run()
    assert not at.exception
    at.number_input(key="w_modelo_n_estimators").set_value(20).run()
    at.button(key="entrenar_modelo").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Configuración personalizada" in m.value for m in at.markdown)


def test_backtesting_con_costo_deja_de_ser_predeterminado():
    at = _app("modelos").run()
    at.button(key="entrenar_modelo").click().run()
    entrenamiento = at.session_state["_entrenamiento"]

    at_bt = _app("backtesting").run()
    at_bt.session_state["_entrenamiento"] = entrenamiento
    at_bt.run()
    at_bt.slider(key="w_costo_bp").set_value(5.0).run()
    assert not at_bt.exception, [e.value for e in at_bt.exception]
    assert any("Configuración personalizada" in m.value for m in at_bt.markdown)
    metricas_bt = {m.label: m.value for m in at_bt.metric}
    assert metricas_bt["Retorno acumulado"] != "28,10 %"


def test_modelos_registro_completo_entrena_sin_excepciones():
    at = _app("modelos").run()
    for id_modelo in ("naive_bayes", "decision_tree", "neural_network"):
        at.selectbox(key="w_modelo_elegido").select(id_modelo).run()
        at.button(key="entrenar_modelo").click().run()
        assert not at.exception, (id_modelo, [e.value for e in at.exception])


def test_regresion_lineal_y_arbol_dan_los_valores_predeterminados():
    at = _app("regresion").run()
    assert not at.exception
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["R²"] == "0,82"
    at.tabs[1].button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["R² en prueba"] == "-0,0172"


def test_xgboost_multiactivo_predeterminado():
    at = _app("xgboost_view").run()
    at.button(key="entrenar_xgboost").click().run()
    assert not at.exception, [e.value for e in at.exception]
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Exactitud global en prueba"] == "49,97 %"
    assert any("Resultado predeterminado" in m.value for m in at.markdown)


def test_xgboost_personalizado_sin_activos_avisa():
    at = _app("xgboost_view").run()
    _boton_modo(at, "modo_xgboost").set_value("Personalizado").run()
    assert not at.exception
    at.multiselect(key="w_xgb_activos").set_value([]).run()
    assert not at.exception
    assert any("al menos un activo" in w.value.lower() for w in at.warning)


def test_comparador_entrena_varios_modelos():
    at = _app("comparador").run()
    at.button(key="entrenar_comparador").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.dataframe) >= 1


def test_comparador_exige_al_menos_dos_modelos():
    at = _app("comparador").run()
    at.multiselect(key="w_comparador_modelos").set_value(["random_forest"]).run()
    assert not at.exception
    assert any("al menos dos" in i.value.lower() for i in at.info)


def test_kmeans_predeterminado_y_personalizado():
    at = _app("kmeans_view").run()
    assert not at.exception
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Inercia"] == "7,70"
    assert any("Resultado predeterminado" in m.value for m in at.markdown)
    _boton_modo(at, "modo_kmeans").set_value("Personalizado").run()
    at.slider(key="w_kmeans_k").set_value(4).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Configuración personalizada" in m.value for m in at.markdown)


def test_kmeans_codo_escalado_no_rompe_la_pagina():
    at = _app("kmeans_view").run()
    at.tabs[1].checkbox[0].set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]


def test_jerarquico_predeterminado_y_metodo_personalizado():
    at = _app("jerarquico_view").run()
    assert not at.exception
    assert any("Resultado predeterminado" in m.value for m in at.markdown)
    _boton_modo(at, "modo_jerarquico").set_value("Personalizado").run()
    at.selectbox(key="w_jerarquico_metodo").select("complete").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Configuración personalizada" in m.value for m in at.markdown)


def test_pca_predeterminado_avisa_ajuste_de_perplejidad():
    at = _app("pca_view").run()
    assert not at.exception, [e.value for e in at.exception]
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Varianza explicada"] == "99,2 %"
    assert any("Resultado predeterminado" in m.value for m in at.markdown)
    assert any("perplejidad se ajustó" in i.value for i in at.info)


def test_pca_personalizado_cambia_componentes():
    at = _app("pca_view").run()
    _boton_modo(at, "modo_pca").set_value("Personalizado").run()
    at.slider(key="w_pca_n").set_value(5).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Configuración personalizada" in m.value for m in at.markdown)


def test_receta_se_puede_descargar_tras_entrenar():
    at = _app("modelos").run()
    at.button(key="entrenar_modelo").click().run()
    assert not at.exception
    codigos = [c.value for c in at.tabs[3].code]
    assert codigos and "random_forest" in codigos[0]


def test_cargar_receta_reproduce_el_modelo_sin_entrenar_manualmente():
    from mltrading.core.recipes.schema import TIPO_MODELO_SUPERVISADO, VERSION_ESQUEMA, RecetaModelo

    receta = RecetaModelo(
        tipo=TIPO_MODELO_SUPERVISADO, version=VERSION_ESQUEMA, id_modelo="decision_tree",
        parametros_modelo={"max_depth": 3, "min_samples_leaf": 5, "criterion": "gini"},
        barras_por_dia=26, umbral_adf=0.05, umbral_correlacion=0.7,
        descartes_correlacion=["volatility2"], proporcion_entrenamiento=0.8,
    )
    at = _app("modelos").run()
    uploader = at.get("file_uploader")[0]
    uploader.upload("receta.json", receta.a_json().encode("utf-8"), "application/json")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="w_modelo_elegido").value == "decision_tree"
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Exactitud en prueba"] == "51,04 %"
    assert any("Configuración personalizada" in m.value for m in at.markdown)


def test_receta_invalida_muestra_error():
    at = _app("modelos").run()
    uploader = at.get("file_uploader")[0]
    uploader.upload("receta.json", b"esto no es json valido", "application/json")
    at.run()
    assert not at.exception
    assert any("no es válida" in e.value for e in at.error)


def test_en_vivo_predeterminado_avanza_y_reproduce_la_exactitud_conocida():
    at = _app("en_vivo").run()
    assert not at.exception
    at.button(key="sim_avanzar").click().run()
    assert not at.exception, [e.value for e in at.exception]
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Barras simuladas"] == "26 / 3.864"
    assert any("Resultado predeterminado" in m.value for m in at.markdown)


def test_en_vivo_calendario_nunca_reproduce_51_55():
    at = _app("en_vivo").run()
    _boton_modo(at, "modo_en_vivo").set_value("Personalizado").run()
    at.selectbox(key="w_sim_calendario").select("Nunca").run()
    at.select_slider(key="w_sim_pasos").set_value(500).run()
    for _ in range(8):
        if at.button(key="sim_avanzar").disabled:
            break
        at.button(key="sim_avanzar").click().run()
    assert not at.exception, [e.value for e in at.exception]
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Reentrenamientos"] == "1"
    assert metricas["Barras simuladas"] == "3.864 / 3.864"
    assert metricas["Exactitud acumulada"] == "51,6 %"  # redondeo a 1 decimal de 51,55 %


def test_en_vivo_reiniciar_vuelve_al_principio():
    at = _app("en_vivo").run()
    at.button(key="sim_avanzar").click().run()
    at.button(key="sim_reiniciar").click().run()
    assert not at.exception
    metricas = {m.label: m.value for m in at.metric}
    assert metricas["Barras simuladas"] == "0 / 3.864"


def test_guia_muestra_las_seis_pestanas():
    at = _app("guia").run()
    assert not at.exception
    assert len(at.tabs) == 6


def test_guia_glosario_tiene_contenido():
    at = _app("guia").run()
    assert len(at.tabs[1].markdown) > 15


def test_guia_avisa_que_nlp_y_refuerzo_no_estan_implementados():
    at = _app("guia").run()
    textos_nlp = [i.value for i in at.tabs[3].info]
    textos_rl = [i.value for i in at.tabs[4].info]
    assert any("no implementa" in t for t in textos_nlp)
    assert any("no implementa" in t for t in textos_rl)


def test_guia_muestra_el_aviso_educativo():
    at = _app("guia").run()
    avisos = [w.value for w in at.tabs[5].warning]
    assert any("fines educativos" in a for a in avisos)


def test_receta_demasiado_grande_se_rechaza_sin_leerla():
    at = _app("modelos").run()
    uploader = at.get("file_uploader")[0]
    contenido_grande = b"{" + b'"relleno": "' + b"x" * (70 * 1024) + b'"}'
    uploader.upload("receta_enorme.json", contenido_grande, "application/json")
    at.run()
    assert not at.exception
    assert any("KB" in e.value for e in at.error)
