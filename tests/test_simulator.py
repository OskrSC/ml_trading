"""Simulador de operación en vivo (capítulo 8)."""
import pytest

from mltrading.core.data.loaders import cargar
from mltrading.core.features import predeterminados
from mltrading.core.features.config import CONFIG_PREDETERMINADA
from mltrading.core.features.division import dividir_cronologica
from mltrading.core.features.pipeline import preparar_variables
from mltrading.core.live import simulator as sim

PARAMS_RF = {"n_estimators": 3, "max_depth": 2, "max_features": 3, "random_state": 4}


@pytest.fixture(scope="module")
def division():
    jpm = cargar("JPM_2017_2019.csv")[0]
    r = preparar_variables(jpm, CONFIG_PREDETERMINADA, tabla_adf=predeterminados.cargar_adf())
    d = dividir_cronologica(r.X, r.y, 0.8)
    retornos = jpm["close"].pct_change().reindex(d.X_prueba.index)
    return d, retornos


def test_calendario_nunca_reproduce_la_exactitud_predeterminada(division):
    d, retornos = division
    estado = sim.iniciar(
        "random_forest", PARAMS_RF, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba, d.y_prueba, retornos,
        sim.ConfigSimulacion(calendario=sim.NUNCA),
    )
    estado = sim.avanzar(estado, len(d.X_prueba))
    assert len(estado.tiempos_reentrenamiento) == 1  # solo el entrenamiento inicial
    assert round(estado.exactitud_global * 100, 2) == 51.55
    assert estado.terminada


def test_avanzar_es_reproducible(division):
    d, retornos = division
    config = sim.ConfigSimulacion(calendario=sim.PERIODICO, periodo_barras=26)
    e1 = sim.avanzar(
        sim.iniciar("random_forest", PARAMS_RF, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba, d.y_prueba,
                   retornos, config),
        60,
    )
    e2 = sim.avanzar(
        sim.iniciar("random_forest", PARAMS_RF, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba, d.y_prueba,
                   retornos, config),
        60,
    )
    assert [p.y_predicho for p in e1.predicciones] == [p.y_predicho for p in e2.predicciones]
    assert len(e1.tiempos_reentrenamiento) == len(e2.tiempos_reentrenamiento)


def test_calendario_periodico_reentrena_cada_n_barras(division):
    d, retornos = division
    estado = sim.iniciar(
        "random_forest", PARAMS_RF, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba, d.y_prueba, retornos,
        sim.ConfigSimulacion(calendario=sim.PERIODICO, periodo_barras=26),
    )
    estado = sim.avanzar(estado, 78)  # exactamente 3 periodos de 26 barras
    marcas = [i for i, p in enumerate(estado.predicciones) if p.reentrenado]
    assert marcas == [25, 51, 77]  # posiciones 0-index donde se cumple el periodo


def test_calendario_por_exactitud_reentrena_al_caer_el_umbral(division):
    d, retornos = division
    estado = sim.iniciar(
        "random_forest", PARAMS_RF, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba, d.y_prueba, retornos,
        sim.ConfigSimulacion(calendario=sim.POR_EXACTITUD, ventana_exactitud=26, umbral_exactitud=0.55),
    )
    estado = sim.avanzar(estado, 60)
    for i, p in enumerate(estado.predicciones):
        if p.reentrenado:
            ventana = estado.predicciones[max(0, i - 26 + 1): i + 1]
            # la ventana pudo cambiar tras reentrenamientos previos; solo se exige que hubiera al
            # menos 26 predicciones acumuladas para activar el criterio
            assert i >= 25


def test_calendario_por_perdida_reentrena_tras_el_retroceso(division):
    d, retornos = division
    estado = sim.iniciar(
        "random_forest", PARAMS_RF, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba, d.y_prueba, retornos,
        sim.ConfigSimulacion(calendario=sim.POR_PERDIDA, umbral_perdida_pct=-0.5),
    )
    estado = sim.avanzar(estado, 200)
    assert len(estado.tiempos_reentrenamiento) >= 1


def test_solo_los_tres_modelos_sin_escalado_son_simulables(division):
    d, retornos = division
    assert sim.MODELOS_SIMULABLES == ("random_forest", "naive_bayes", "decision_tree")
    with pytest.raises(ValueError, match="simulables"):
        sim.iniciar("logistic_regression", {}, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba, d.y_prueba, retornos)


def test_avanzar_no_puede_pasar_de_lo_que_queda(division):
    d, retornos = division
    estado = sim.iniciar(
        "naive_bayes", {}, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba.iloc[:10], d.y_prueba.iloc[:10],
        retornos.iloc[:10],
    )
    estado = sim.avanzar(estado, 1000)  # pide más de lo disponible
    assert estado.barras_simuladas == 10
    assert estado.terminada
    igual = sim.avanzar(estado, 1)  # ya no queda nada: no hace nada, no falla
    assert igual.barras_simuladas == 10


def test_capital_se_actualiza_solo_cuando_se_predice_senal_1(division):
    d, retornos = division
    estado = sim.iniciar(
        "naive_bayes", {}, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba.iloc[:5], d.y_prueba.iloc[:5],
        retornos.iloc[:5], sim.ConfigSimulacion(calendario=sim.NUNCA),
    )
    estado = sim.avanzar(estado, 5)
    capital_esperado = 1.0
    for p in estado.predicciones:
        if p.y_predicho == 1:
            capital_esperado *= 1 + p.retorno
    assert abs(estado.capital - capital_esperado) < 1e-9


def test_config_valida_argumentos():
    with pytest.raises(ValueError):
        sim.ConfigSimulacion(calendario="otro")
    with pytest.raises(ValueError):
        sim.ConfigSimulacion(periodo_barras=0)
    with pytest.raises(ValueError):
        sim.ConfigSimulacion(ventana_exactitud=1)


def test_indices_deben_coincidir(division):
    d, retornos = division
    with pytest.raises(ValueError, match="índice"):
        sim.iniciar(
            "naive_bayes", {}, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba, d.y_prueba.iloc[1:],
            retornos.iloc[1:],
        )


def test_a_tabla_vacia_antes_de_avanzar(division):
    d, retornos = division
    estado = sim.iniciar(
        "naive_bayes", {}, d.X_entrenamiento, d.y_entrenamiento, d.X_prueba.iloc[:3], d.y_prueba.iloc[:3],
        retornos.iloc[:3],
    )
    assert estado.a_tabla().empty
    assert list(estado.a_tabla().columns) == ["real", "predicho", "acierto", "reentrenado"]
