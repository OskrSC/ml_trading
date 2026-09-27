import numpy as np
import pandas as pd
import pytest

from mltrading.core.data.loaders import cargar
from mltrading.core.features import division as div
from mltrading.core.features import predeterminados
from mltrading.core.features.config import CONFIG_PREDETERMINADA
from mltrading.core.features.pipeline import preparar_variables


@pytest.fixture(scope="module")
def resultado():
    jpm = cargar("JPM_2017_2019.csv")[0]
    return preparar_variables(jpm, CONFIG_PREDETERMINADA, tabla_adf=predeterminados.cargar_adf())


def test_division_predeterminada_tiene_los_tamanos_esperados(resultado):
    d = div.dividir_cronologica(resultado.X, resultado.y, 0.8)
    assert (d.filas_entrenamiento, d.filas_prueba) == (15453, 3864)
    assert d.tipo == div.CRONOLOGICA


def test_division_identica_a_los_artefactos_predeterminados(resultado):
    d = div.dividir_cronologica(resultado.X, resultado.y, 0.8)
    for nuestro, archivo in (
        (d.X_entrenamiento, "JPM_features_training_2017_2019.csv"),
        (d.X_prueba, "JPM_features_testing_2017_2019.csv"),
    ):
        esperado = cargar(archivo)[0]
        assert nuestro.index.equals(esperado.index)
        np.testing.assert_allclose(nuestro.values, esperado[list(nuestro.columns)].values, rtol=1e-9, atol=1e-12)
    for nuestro, archivo in (
        (d.y_entrenamiento, "JPM_target_training_2017_2019.csv"),
        (d.y_prueba, "JPM_target_testing_2017_2019.csv"),
    ):
        assert (nuestro.values == cargar(archivo)[0]["signal"].values).all()


def test_la_division_cronologica_no_mira_el_futuro(resultado):
    d = div.dividir_cronologica(resultado.X, resultado.y, 0.8)
    assert d.X_entrenamiento.index.max() < d.X_prueba.index.min()
    assert div.fraccion_prueba_anterior(d) == 0.0


def test_la_division_aleatoria_si_mira_el_futuro(resultado):
    a = div.dividir_aleatoria(resultado.X, resultado.y, 0.8, semilla=42)
    assert a.filas_entrenamiento == 15453 and a.filas_prueba == 3864
    assert div.fraccion_prueba_anterior(a) > 0.9
    assert set(a.X_entrenamiento.index).isdisjoint(a.X_prueba.index)


def test_la_semilla_controla_el_reparto_aleatorio(resultado):
    a1 = div.dividir_aleatoria(resultado.X, resultado.y, 0.8, semilla=1)
    a2 = div.dividir_aleatoria(resultado.X, resultado.y, 0.8, semilla=1)
    a3 = div.dividir_aleatoria(resultado.X, resultado.y, 0.8, semilla=2)
    assert a1.X_prueba.index.equals(a2.X_prueba.index)
    assert not a1.X_prueba.index.equals(a3.X_prueba.index)


@pytest.mark.parametrize("proporcion,entrenamiento", [(0.5, 9658), (0.7, 13521), (0.9, 17385)])
def test_tamanos_con_otras_proporciones(resultado, proporcion, entrenamiento):
    d = div.dividir_cronologica(resultado.X, resultado.y, proporcion)
    assert d.filas_entrenamiento == entrenamiento  # parte entera de proporcion x 19317
    assert d.filas_entrenamiento + d.filas_prueba == 19317


def test_validaciones(resultado):
    with pytest.raises(ValueError, match="proporción"):
        div.dividir_cronologica(resultado.X, resultado.y, 1.0)
    with pytest.raises(ValueError, match="mismo índice"):
        div.dividir_cronologica(resultado.X, resultado.y.iloc[1:], 0.8)


def test_resumen_de_la_division(resultado):
    r = div.resumen(div.dividir_cronologica(resultado.X, resultado.y, 0.8))
    assert list(r["Conjunto"]) == ["Entrenamiento", "Prueba"]
    assert list(r["Filas"]) == [15453, 3864]
    assert round(r.loc[1, "Señal 1 (%)"], 1) == 49.9
    assert r.loc[0, "Fin"] < r.loc[1, "Inicio"]
