"""El fallo de un modelo no debe tumbar páginas que entrenan varios a la vez."""
import pytest

from mltrading.core.models import entrenamiento as entrenamiento_mod
from mltrading.core.observabilidad import limpiar_buffer, ultimas_entradas


def test_comparador_sigue_con_los_demas_modelos_si_uno_falla(monkeypatch):
    from streamlit.testing.v1 import AppTest

    original = entrenamiento_mod.entrenar_modelo
    llamadas = {"n": 0}

    def falla_la_primera_vez(spec, *args, **kwargs):
        llamadas["n"] += 1
        if llamadas["n"] == 1:
            raise RuntimeError("fallo simulado para la prueba")
        return original(spec, *args, **kwargs)

    monkeypatch.setattr("mltrading.views.comparador.entrenar_modelo", falla_la_primera_vez)
    limpiar_buffer()

    at = AppTest.from_string(
        "from mltrading.views.comparador import render\nrender()", default_timeout=120
    ).run()
    at.button(key="entrenar_comparador").click().run()

    assert not at.exception, [e.value for e in at.exception]
    assert any("No se pudo entrenar" in w.value for w in at.warning)
    assert len(at.dataframe) >= 1  # el resto de modelos sí produjo resultados
    assert any("fallo simulado" in e.mensaje for e in ultimas_entradas())


def test_xgboost_muestra_error_si_falla_el_entrenamiento(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def siempre_falla(*args, **kwargs):
        raise RuntimeError("fallo simulado de xgboost")

    monkeypatch.setattr("mltrading.views.xgboost_view.entrenar_xgboost", siempre_falla)
    limpiar_buffer()

    at = AppTest.from_string(
        "from mltrading.views.xgboost_view import render\nrender()", default_timeout=120
    ).run()
    at.button(key="entrenar_xgboost").click().run()

    assert not at.exception, [e.value for e in at.exception]
    assert any("No se pudo entrenar" in e.value for e in at.error)
    assert any("fallo simulado de xgboost" in e.mensaje for e in ultimas_entradas())
