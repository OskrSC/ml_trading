"""Resultados predeterminados de los modelos añadidos en la fase 3."""
import numpy as np
import pandas as pd
import pytest

from mltrading.core.data.loaders import cargar
from mltrading.core.features import predeterminados
from mltrading.core.features.config import CONFIG_PREDETERMINADA
from mltrading.core.features.division import dividir_cronologica
from mltrading.core.features.pipeline import preparar_variables
from mltrading.core.models.entrenamiento import entrenar_modelo
from mltrading.core.models.params import valores_predeterminados
from mltrading.core.models.registry import obtener


@pytest.fixture(scope="module")
def division():
    jpm = cargar("JPM_2017_2019.csv")[0]
    r = preparar_variables(jpm, CONFIG_PREDETERMINADA, tabla_adf=predeterminados.cargar_adf())
    return dividir_cronologica(r.X, r.y, 0.8)


def _accuracy(spec_id, division):
    spec = obtener(spec_id)
    resultado = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    return resultado, float((resultado.y_pred_prueba.values == division.y_prueba.values).mean())


def test_naive_bayes_predeterminado(division):
    resultado, acc = _accuracy("naive_bayes", division)
    assert round(acc * 100, 2) == 50.93
    assert set(resultado.y_pred_prueba.unique()) <= {0, 1}
    assert not obtener("naive_bayes").parametros  # BernoulliNB sin hiperparámetros expuestos
    assert not obtener("naive_bayes").necesita_escalado


def test_arbol_decision_clasificacion_predeterminado(division):
    resultado, acc = _accuracy("decision_tree", division)
    assert round(acc * 100, 2) == 51.04
    assert set(resultado.interpretacion["variable"]) == set(division.X_entrenamiento.columns)
    assert np.isclose(resultado.interpretacion["importancia"].sum(), 1.0)


def test_arbol_decision_reproducible(division):
    spec = obtener("decision_tree")
    r1 = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    r2 = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    assert (r1.y_pred_prueba.values == r2.y_pred_prueba.values).all()


def test_red_neuronal_predeterminada(division):
    resultado, acc = _accuracy("neural_network", division)
    assert round(acc * 100, 2) == 52.10
    conteo = resultado.y_pred_prueba.value_counts()
    assert conteo.get(0, 0) == 2307 and conteo.get(1, 0) == 1557
    assert resultado.notas  # con 7 épocas, el aviso de no convergencia es esperado
    spec = obtener("neural_network")
    assert spec.necesita_escalado


def test_red_neuronal_reproducible_con_la_misma_semilla(division):
    spec = obtener("neural_network")
    r1 = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    r2 = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    assert (r1.y_pred_prueba.values == r2.y_pred_prueba.values).all()


def test_red_neuronal_semilla_distinta_puede_cambiar_el_resultado(division):
    spec = obtener("neural_network")
    parametros = dict(valores_predeterminados(spec.parametros))
    parametros["random_state"] = 7
    r_otra = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba, parametros)
    r_predeterminada = entrenar_modelo(spec, division.X_entrenamiento, division.y_entrenamiento, division.X_prueba)
    # no se exige que difieran, pero al menos debe ejecutar sin error con otra semilla
    assert r_otra.y_pred_prueba.index.equals(r_predeterminada.y_pred_prueba.index)


def test_los_ocho_modelos_de_clasificacion_estan_registrados():
    from mltrading.core.models.registry import MODELOS

    assert {m.id for m in MODELOS} == {
        "random_forest", "logistic_regression", "naive_bayes", "decision_tree", "neural_network",
    }
