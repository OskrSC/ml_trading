"""Resultados predeterminados de la preparación de variables y del objetivo."""
import numpy as np
import pandas as pd
import pytest

from mltrading.config import roadmap
from mltrading.core.data.loaders import cargar
from mltrading.core.features import indicadores, predeterminados, seleccion
from mltrading.core.features.config import CONFIG_PREDETERMINADA, ConfigVariables, es_predeterminada
from mltrading.core.features.construccion import CANDIDATAS, construir_candidatas
from mltrading.core.features.indicadores import MotorNoDisponible
from mltrading.core.features.pipeline import preparar_variables

FINALES = ["pct_change", "pct_change2", "pct_change5", "rsi", "adx", "corr", "volatility"]
NO_ESTACIONARIAS = {"open", "high", "low", "close", "sma"}

requiere_talib = pytest.mark.skipif(not indicadores.talib_disponible(), reason="TA-Lib no está instalado")


@pytest.fixture(scope="module")
def jpm():
    return cargar("JPM_2017_2019.csv")[0]


@pytest.fixture(scope="module")
def adf_predeterminada():
    tabla = predeterminados.cargar_adf()
    assert tabla is not None, "Falta el artefacto: python scripts/generar_artefactos.py"
    return tabla


@pytest.fixture(scope="module")
def resultado(jpm, adf_predeterminada):
    return preparar_variables(jpm, CONFIG_PREDETERMINADA, tabla_adf=adf_predeterminada)


# ------------------------------------------------------------ predeterminado ---
def test_variables_finales_son_las_siete_predeterminadas(resultado):
    assert resultado.variables_finales == FINALES
    assert list(resultado.X.columns) == FINALES


def test_variables_identicas_al_artefacto_predeterminado(resultado):
    esperado = cargar("JPM_features_2017_2019.csv")[0]
    assert resultado.X.index.equals(esperado.index)
    assert len(resultado.X) == 19317
    np.testing.assert_allclose(resultado.X.values, esperado[FINALES].values, rtol=0, atol=1e-12)


def test_objetivo_identico_al_artefacto_predeterminado(resultado):
    esperado = cargar("JPM_target_2017_2019.csv")[0]["signal"]
    assert resultado.y.index.equals(esperado.index)
    assert (resultado.y.values == esperado.values).all()
    assert set(resultado.y.unique()) == {0, 1}


def test_filas_descartadas_por_ventana_y_por_objetivo(resultado):
    ca = resultado.candidatas
    assert ca.filas_iniciales == 19370
    assert ca.filas_por_ventana == 52  # volatility2 necesita 52 cambios porcentuales
    assert ca.filas_por_objetivo == 1  # la última barra no tiene retorno futuro
    assert ca.filas == 19317


def test_estacionariedad_descarta_precios_y_media_movil(resultado):
    assert set(resultado.descartadas_adf) == NO_ESTACIONARIAS
    assert len(resultado.estacionarias) == 8


def test_un_unico_par_correlacionado_y_se_descarta_volatility2(resultado):
    assert len(resultado.pares) == 1
    fila = resultado.pares.iloc[0]
    assert (fila.variable_a, fila.variable_b) == ("volatility", "volatility2")
    assert round(fila.correlacion, 3) == 0.832
    assert resultado.sugeridas_correlacion == ["volatility2"]
    assert resultado.descartadas_correlacion == ["volatility2"]


def test_artefacto_adf_coincide_con_recalcularlo(jpm):
    candidatas = construir_candidatas(jpm, CONFIG_PREDETERMINADA)
    recalculada = seleccion.calcular_adf(candidatas.variables)
    guardada = predeterminados.cargar_adf()
    assert list(recalculada["variable"]) == list(guardada["variable"]) == list(CANDIDATAS)
    np.testing.assert_allclose(recalculada["estadistico"], guardada["estadistico"], rtol=1e-9)
    np.testing.assert_allclose(recalculada["p_valor"], guardada["p_valor"], rtol=1e-6, atol=1e-12)
    assert list(recalculada["retardos"]) == list(guardada["retardos"])


# ---------------------------------------------------------------- motores ---
@requiere_talib
@pytest.mark.parametrize("periodo", [10, 14, 26, 40])
def test_respaldo_reproduce_a_talib(jpm, periodo):
    import talib

    cierre, alto, bajo = jpm["close"].values, jpm["high"].values, jpm["low"].values
    np.testing.assert_allclose(indicadores.rsi_respaldo(cierre, periodo), talib.RSI(cierre, periodo), atol=1e-9)
    np.testing.assert_allclose(indicadores.adx_respaldo(alto, bajo, periodo), talib.ADX(alto, bajo, cierre, periodo), atol=1e-9)


@requiere_talib
def test_el_adx_no_depende_del_cierre(jpm):
    import talib

    a = talib.ADX(jpm["high"].values, jpm["low"].values, jpm["close"].values, 26)
    b = talib.ADX(jpm["high"].values, jpm["low"].values, jpm["open"].values, 26)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_resultado_con_motor_pandas_es_el_mismo(jpm, adf_predeterminada, resultado):
    r = preparar_variables(jpm, ConfigVariables(motor="pandas"), tabla_adf=adf_predeterminada)
    assert r.variables_finales == resultado.variables_finales
    np.testing.assert_allclose(r.X.values, resultado.X.values, rtol=0, atol=1e-9)
    assert (r.y.values == resultado.y.values).all()


def test_sin_talib_el_modo_automatico_usa_el_respaldo(monkeypatch):
    monkeypatch.setattr(indicadores, "_talib", None)
    assert indicadores.resolver_motor("auto") == "pandas"
    with pytest.raises(MotorNoDisponible):
        indicadores.resolver_motor("talib")


def test_rsi_y_adx_con_series_cortas_devuelven_nan():
    corta = np.arange(1.0, 10.0)
    assert np.isnan(indicadores.rsi_respaldo(corta, 26)).all()
    assert np.isnan(indicadores.adx_respaldo(corta, corta, 26)).all()


# ------------------------------------------------------------ personalizado ---
def test_ventana_personalizada_cambia_las_filas_descartadas(jpm, adf_predeterminada):
    r = preparar_variables(jpm, ConfigVariables(barras_por_dia=20), tabla_adf=adf_predeterminada)
    assert r.candidatas.filas_por_ventana == 40
    assert r.candidatas.filas == 19370 - 40 - 1
    assert not es_predeterminada(r.config)


def test_con_descartes_admite_otra_eleccion(resultado):
    sin_descartes = resultado.con_descartes([])
    assert len(sin_descartes.variables_finales) == 8
    assert resultado.con_descartes(["open", "no_existe"]).variables_finales == resultado.estacionarias
    assert len(resultado.variables_finales) == 7  # el original no cambia


def test_umbral_de_correlacion_alto_no_descarta_nada(jpm, adf_predeterminada):
    r = preparar_variables(jpm, ConfigVariables(umbral_correlacion=0.9), tabla_adf=adf_predeterminada)
    assert r.pares.empty and r.descartadas_correlacion == []
    assert len(r.variables_finales) == 8


def test_configuracion_valida():
    assert es_predeterminada(ConfigVariables(motor="pandas"))
    assert not es_predeterminada(ConfigVariables(umbral_adf=0.01))
    for argumentos in ({"barras_por_dia": 2}, {"umbral_adf": 0}, {"umbral_correlacion": 1.5}, {"motor": "x"}):
        with pytest.raises(ValueError):
            ConfigVariables(**argumentos)


def test_faltan_columnas_de_precio(jpm):
    with pytest.raises(ValueError, match="Faltan columnas"):
        construir_candidatas(jpm.drop(columns=["high"]), CONFIG_PREDETERMINADA)


def test_hoja_de_ruta_refleja_el_avance():
    estados = {f.numero: f.estado for f in roadmap.FASES}
    assert all(estado == roadmap.COMPLETADA for estado in estados.values())
    assert len(roadmap.FASES) == 7


def test_sugerir_descartes_recorre_pares_por_correlacion():
    pares = pd.DataFrame(
        [("a", "b", 0.95), ("b", "c", 0.9), ("c", "d", 0.8)], columns=["variable_a", "variable_b", "correlacion"]
    )
    assert seleccion.sugerir_descartes(pares) == ["b", "d"]  # el par (b, c) se omite porque b ya se descartó
