import numpy as np
import pandas as pd
import pytest

from mltrading.core.data.loaders import cargar
from mltrading.core.evaluation.metrics import evaluar


@pytest.fixture(scope="module")
def real_y_predicho():
    y = cargar("JPM_target_testing_2017_2019.csv")[0]["signal"]
    pred = cargar("JPM_predicted_2017_2019.csv")[0]["signal"]
    return y, pred


def test_matriz_de_confusion_coincide_con_el_calculo_manual(real_y_predicho):
    y, pred = real_y_predicho
    ev = evaluar(y, pred)
    assert ev.matriz_confusion == [[985, 951], [921, 1007]]


def test_exactitud_y_reporte(real_y_predicho):
    y, pred = real_y_predicho
    ev = evaluar(y, pred)
    assert round(ev.exactitud * 100, 2) == 51.55
    fila_1 = ev.reporte.set_index("Clase").loc["Posición larga (1)"]
    assert round(fila_1["Precisión"], 4) == 0.5143
    assert round(fila_1["Sensibilidad"], 4) == 0.5223
    assert int(fila_1["Soporte"]) == 1928


def test_columna_de_aciertos(real_y_predicho):
    y, pred = real_y_predicho
    ev = evaluar(y, pred)
    assert ev.aciertos.sum() == 985 + 1007


def test_indices_deben_coincidir():
    y = pd.Series([0, 1, 0], index=[0, 1, 2])
    pred = pd.Series([0, 1, 0], index=[0, 1, 3])
    with pytest.raises(ValueError):
        evaluar(y, pred)
