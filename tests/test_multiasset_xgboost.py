"""Conjunto multiactivo y XGBoost (capítulo 14)."""
import numpy as np
import pytest

from mltrading.core.models.multiasset import (
    UNIVERSO_PREDETERMINADO, VENTANAS, construir_conjunto, construir_variables_activo, nombres_columnas,
)
from mltrading.core.models.params import valores_predeterminados
from mltrading.core.models.xgboost_multiactivo import PARAMETROS, entrenar_xgboost


def test_ventanas_y_columnas():
    assert VENTANAS == (10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
    assert nombres_columnas() == [
        "pct_change_10", "std_10", "pct_change_15", "std_15", "pct_change_20", "std_20",
        "pct_change_25", "std_25", "pct_change_30", "std_30", "pct_change_35", "std_35",
        "pct_change_40", "std_40", "pct_change_45", "std_45", "pct_change_50", "std_50",
        "pct_change_55", "std_55",
    ]


def test_universo_predeterminado_tiene_cinco_activos():
    assert len(UNIVERSO_PREDETERMINADO) == 5
    assert {a.id for a in UNIVERSO_PREDETERMINADO} == {"RELIANCE", "KO", "GOOG", "AMZN", "MMM"}


@pytest.fixture(scope="module")
def conjunto():
    return construir_conjunto(UNIVERSO_PREDETERMINADO, proporcion=0.8)


def test_filas_por_activo(conjunto):
    esperado = {"RELIANCE": 5462, "KO": 509, "GOOG": 446, "AMZN": 446, "MMM": 446}
    for aid, filas in esperado.items():
        assert len(conjunto.datos_por_activo[aid].variables) == filas


def test_division_80_20_por_activo(conjunto):
    assert len(conjunto.X_entrenamiento) == 5844
    assert len(conjunto.X_prueba) == 1465
    assert len(conjunto.X_entrenamiento) + len(conjunto.X_prueba) == sum(
        len(d.variables) for d in conjunto.datos_por_activo.values()
    )


def test_objetivo_es_mas_uno_o_menos_uno(conjunto):
    assert set(conjunto.y_entrenamiento.unique()) == {1, -1}
    assert set(conjunto.y_prueba.unique()) == {1, -1}


def test_construir_variables_de_un_solo_activo():
    spec = UNIVERSO_PREDETERMINADO[1]  # Coca-Cola
    datos = construir_variables_activo(spec)
    assert list(datos.variables.columns) == nombres_columnas()
    assert datos.filas_iniciales == 565
    assert len(datos.variables) == 509  # 565 - 55 (ventana máxima) - 1 (objetivo)


def test_xgboost_predeterminado_es_reproducible(conjunto):
    parametros = valores_predeterminados(PARAMETROS)
    assert parametros == {"n_estimators": 30, "max_depth": 2}
    r1 = entrenar_xgboost(conjunto, parametros)
    r2 = entrenar_xgboost(conjunto, parametros)
    assert (r1.y_pred_prueba.values == r2.y_pred_prueba.values).all()
    assert set(r1.y_pred_prueba.unique()) <= {1, -1}


def test_xgboost_exactitud_predeterminada(conjunto):
    parametros = valores_predeterminados(PARAMETROS)
    r = entrenar_xgboost(conjunto, parametros)
    acc = (r.y_pred_prueba.values == conjunto.y_prueba.values).mean()
    assert round(acc * 100, 2) == 49.97


def test_xgboost_importancias_suman_uno(conjunto):
    r = entrenar_xgboost(conjunto, valores_predeterminados(PARAMETROS))
    assert set(r.interpretacion["variable"]) == set(nombres_columnas())
    assert np.isclose(r.interpretacion["importancia"].sum(), 1.0, atol=1e-4)


def test_construir_conjunto_valida_argumentos():
    with pytest.raises(ValueError):
        construir_conjunto(())
    with pytest.raises(ValueError):
        construir_conjunto(UNIVERSO_PREDETERMINADO, proporcion=1.5)


def test_reliance_descarta_las_filas_con_nulos():
    spec = next(a for a in UNIVERSO_PREDETERMINADO if a.id == "RELIANCE")
    assert spec.descartar_nulos
    datos = construir_variables_activo(spec)
    assert datos.filas_iniciales == 5518  # ya sin las 120 filas con nulos
    assert not datos.variables.isna().any().any()
