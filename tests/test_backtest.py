import numpy as np
import pandas as pd
import pytest

from mltrading.core.backtest.analytics import analitica
from mltrading.core.backtest.config import CONFIG_PREDETERMINADA, ConfigBacktest, es_predeterminada
from mltrading.core.backtest.returns import calcular_retornos
from mltrading.core.backtest.trades import construir_registro
from mltrading.core.data.loaders import cargar


@pytest.fixture(scope="module")
def precio_y_senal():
    close = cargar("JPM_2017_2019.csv")[0]["close"]
    senal = cargar("JPM_predicted_2017_2019.csv")[0]["signal"]
    return close, senal


@pytest.fixture(scope="module")
def resultado_predeterminado(precio_y_senal):
    close, senal = precio_y_senal
    return calcular_retornos(close, senal, CONFIG_PREDETERMINADA)


def test_metricas_identicas_al_libro(resultado_predeterminado):
    r = resultado_predeterminado
    assert round(r.retorno_acumulado_pct, 2) == 28.10
    assert round(r.retorno_anualizado_pct, 2) == 52.20
    assert round(r.volatilidad_anualizada_pct, 2) == 14.90
    assert round(r.sharpe, 2) == 2.89
    assert round(r.drawdown_maximo_pct, 2) == -7.94
    assert len(r.datos) == 3863


def test_registro_de_operaciones_coincide_con_el_libro(resultado_predeterminado):
    registro = construir_registro(resultado_predeterminado.datos)
    assert len(registro) == 169
    assert (registro["Posición"] == 1).all()  # la señal del libro es 0/1: nunca hay posiciones cortas
    assert round(registro.loc[registro.PnL > 0, "PnL"].sum(), 2) == 59.21
    assert round(registro.loc[registro.PnL < 0, "PnL"].sum(), 2) == -29.49
    assert round(registro["PnL"].sum(), 2) == 29.72


def test_analitica_de_las_operaciones(resultado_predeterminado):
    registro = construir_registro(resultado_predeterminado.datos)
    a = analitica(registro).iloc[0]
    assert a["Total de operaciones"] == 169
    assert a["Ganadoras"] == 102 and a["Perdedoras"] == 67
    assert round(a["Porcentaje de aciertos"], 2) == round(100 * 102 / 169, 2)


def test_analitica_con_registro_vacio():
    vacio = pd.DataFrame(columns=["Posición", "Entrada", "Precio de entrada", "Salida", "Precio de salida", "PnL"])
    a = analitica(vacio).iloc[0]
    assert a["Total de operaciones"] == 0
    assert np.isnan(a["Porcentaje de aciertos"])


def test_costo_de_transaccion_reduce_el_retorno_y_deja_de_ser_predeterminado(precio_y_senal):
    close, senal = precio_y_senal
    config = ConfigBacktest(costo_bp=5.0)
    assert not es_predeterminada(config)
    r = calcular_retornos(close, senal, config)
    r0 = calcular_retornos(close, senal, CONFIG_PREDETERMINADA)
    assert r.retorno_acumulado_pct < r0.retorno_acumulado_pct
    assert r.costo_total_pct > 0


def test_config_predeterminada_se_reconoce():
    assert es_predeterminada(CONFIG_PREDETERMINADA)
    assert es_predeterminada(ConfigBacktest())
    assert not es_predeterminada(ConfigBacktest(costo_bp=1.0))


def test_error_si_no_hay_filas_en_comun():
    close = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="D"))
    senal = pd.Series([1, 0], index=pd.date_range("2021-01-01", periods=2, freq="D"))
    with pytest.raises(ValueError):
        calcular_retornos(close, senal, CONFIG_PREDETERMINADA)


def test_construir_registro_sin_operaciones():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    datos = pd.DataFrame({"close": [1, 2, 3, 4, 5], "predicted_signal": [0, 0, 0, 0, 0]}, index=idx)
    registro = construir_registro(datos)
    assert registro.empty


def test_factor_de_anualizacion_afecta_al_resultado(precio_y_senal):
    close, senal = precio_y_senal
    r_15min = calcular_retornos(close, senal, CONFIG_PREDETERMINADA)
    r_diario = calcular_retornos(close, senal, ConfigBacktest(factor_anualizacion=252))
    assert r_15min.retorno_anualizado_pct != r_diario.retorno_anualizado_pct
    assert r_15min.retorno_acumulado_pct == r_diario.retorno_acumulado_pct  # no depende de la anualización
