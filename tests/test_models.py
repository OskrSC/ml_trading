"""Resultados predeterminados de los modelos supervisados."""
import numpy as np
import pandas as pd
import pytest

from mltrading.core.data.loaders import cargar
from mltrading.core.features import predeterminados
from mltrading.core.features.config import CONFIG_PREDETERMINADA
from mltrading.core.features.division import dividir_cronologica
from mltrading.core.features.pipeline import preparar_variables
from mltrading.core.models.entrenamiento import entrenar_modelo, es_configuracion_predeterminada
from mltrading.core.models.params import valores_predeterminados
from mltrading.core.models.registry import MODELOS, obtener


@pytest.fixture(scope="module")
def division():
    jpm = cargar("JPM_2017_2019.csv")[0]
    r = preparar_variables(jpm, CONFIG_PREDETERMINADA, tabla_adf=predeterminados.cargar_adf())
    return dividir_cronologica(r.X, r.y, 0.8)


@pytest.mark.parametrize("spec", MODELOS, ids=lambda m: m.id)
def test_cada_modelo_entrena_con_sus_valores_predeterminados(spec, division):
    resultado = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    assert resultado.y_pred_prueba.index.equals(division.X_prueba.index)
    assert set(resultado.y_pred_prueba.unique()) <= {0, 1}
    assert es_configuracion_predeterminada(spec, valores_predeterminados(spec.parametros))


def test_random_forest_predeterminado_reproduce_el_artefacto(division):
    spec = obtener("random_forest")
    resultado = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    esperado = cargar("JPM_predicted_2017_2019.csv")[0]["signal"]
    assert resultado.y_pred_prueba.index.equals(esperado.index)
    assert (resultado.y_pred_prueba.values == esperado.values).all()
    accuracy = (resultado.y_pred_prueba.values == division.y_prueba.values).mean()
    assert round(accuracy * 100, 2) == 51.55


def test_random_forest_interpretacion_son_importancias(division):
    spec = obtener("random_forest")
    resultado = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    tabla = resultado.interpretacion
    assert set(tabla["variable"]) == set(division.X_entrenamiento.columns)
    assert np.isclose(tabla["importancia"].sum(), 1.0)
    assert (tabla["importancia"] >= 0).all()


def test_logistic_regression_predeterminada_reproduce_las_metricas_del_backtest(division):
    spec = obtener("logistic_regression")
    resultado = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    accuracy = (resultado.y_pred_prueba.values == division.y_prueba.values).mean()
    assert round(accuracy * 100, 2) == 50.96
    conteo = resultado.y_pred_prueba.value_counts()
    assert conteo.get(0, 0) == 2973 and conteo.get(1, 0) == 891


def test_logistic_regression_interpretacion_son_coeficientes(division):
    spec = obtener("logistic_regression")
    resultado = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    tabla = resultado.interpretacion.set_index("variable")
    esperado = {
        "pct_change": 0.0154, "pct_change2": -0.0803, "pct_change5": 0.0243, "rsi": 0.0053,
        "adx": 0.0135, "corr": -0.0016, "volatility": 0.0163,
    }
    for variable, valor in esperado.items():
        assert round(float(tabla.loc[variable, "coeficiente"]), 4) == valor


def test_el_escalador_se_ajusta_solo_con_entrenamiento(division):
    spec = obtener("logistic_regression")
    assert spec.necesita_escalado
    from mltrading.core.models import escalado

    entrenamiento_escalado, prueba_escalada = escalado.escalar(division.X_entrenamiento, division.X_prueba)
    np.testing.assert_allclose(entrenamiento_escalado.mean().values, 0, atol=1e-9)
    np.testing.assert_allclose(entrenamiento_escalado.std(ddof=0).values, 1, atol=1e-9)
    # el conjunto de prueba no tiene por qué quedar centrado, porque usa la media del entrenamiento
    assert not np.allclose(prueba_escalada.mean().values, 0, atol=1e-2)


def test_max_features_se_ajusta_si_hay_menos_variables(division):
    spec = obtener("random_forest")
    parametros = dict(valores_predeterminados(spec.parametros))
    parametros["max_features"] = 50
    resultado = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba, parametros)
    assert resultado.notas and "ajustado a" in resultado.notas[0]


def test_configuracion_personalizada_no_es_predeterminada():
    spec = obtener("random_forest")
    parametros = dict(valores_predeterminados(spec.parametros))
    parametros["n_estimators"] = 50
    assert not es_configuracion_predeterminada(spec, parametros)
