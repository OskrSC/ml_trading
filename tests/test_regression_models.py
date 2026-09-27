"""Regresión lineal y árbol de regresión."""
import numpy as np
import pytest

from mltrading.core.data.loaders import cargar
from mltrading.core.features import predeterminados
from mltrading.core.features.config import CONFIG_PREDETERMINADA
from mltrading.core.features.division import dividir_cronologica
from mltrading.core.features.pipeline import preparar_variables
from mltrading.core.models import decision_tree_regressor as arbol
from mltrading.core.models import linear_regression as lineal


def test_ols_jpm_bac_reproduce_el_r_cuadrado_predeterminado():
    datos = cargar("jpm_and_bac_price_2019.csv")[0]
    r = lineal.ajustar_ols(datos, "BAC Close", "Observed JPM")
    assert round(r.r_cuadrado, 2) == 0.82
    assert round(r.pendiente, 4) == 4.2167
    assert round(r.interseccion, 2) == -10.86


def test_ols_jpm_nestle_reproduce_el_r_cuadrado_predeterminado():
    datos = cargar("predicted_jpm_and_nestle_price_2019.csv")[0]
    r = lineal.ajustar_ols(datos, "Nestle Close", "Observed JPM")
    assert round(r.r_cuadrado, 2) == 0.35


def test_ols_propio_coincide_con_la_columna_predicha_del_archivo():
    from sklearn.metrics import r2_score

    datos = cargar("jpm_and_bac_price_2019.csv")[0]
    r = lineal.ajustar_ols(datos, "BAC Close", "Observed JPM")
    r2_archivo = r2_score(datos["Observed JPM"], datos["Predicted JPM_BAC"])
    assert round(r.r_cuadrado, 4) == round(r2_archivo, 4)


@pytest.fixture(scope="module")
def division_jpm():
    jpm = cargar("JPM_2017_2019.csv")[0]
    r = preparar_variables(jpm, CONFIG_PREDETERMINADA, tabla_adf=predeterminados.cargar_adf())
    return r, dividir_cronologica(r.X, r.y, 0.8)


def test_arbol_de_regresion_predeterminado(division_jpm):
    resultado_vars, division = division_jpm
    objetivo = arbol.construir_objetivo(resultado_vars.candidatas.variables)
    y_tr = objetivo.reindex(division.X_entrenamiento.index)
    y_te = objetivo.reindex(division.X_prueba.index)
    v_tr, v_te = y_tr.notna(), y_te.notna()
    r = arbol.entrenar(
        division.X_entrenamiento[v_tr], y_tr[v_tr], division.X_prueba[v_te], y_te[v_te],
        {"min_samples_leaf": 200},
    )
    assert round(r.r_cuadrado, 4) == -0.0172
    assert v_te.sum() == 3863  # la última barra de prueba tampoco tiene retorno futuro
    assert set(r.interpretacion["variable"]) == set(division.X_entrenamiento.columns)


def test_objetivo_de_regresion_es_el_cambio_porcentual_siguiente(division_jpm):
    resultado_vars, _ = division_jpm
    objetivo = arbol.construir_objetivo(resultado_vars.candidatas.variables)
    variables = resultado_vars.candidatas.variables
    assert np.isclose(objetivo.iloc[0], variables["pct_change"].iloc[1])
    assert np.isnan(objetivo.iloc[-1])
