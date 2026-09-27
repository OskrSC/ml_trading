"""Recetas de modelos: guardar y cargar configuración sin serializar objetos."""
import pytest

from mltrading.core.data.loaders import cargar
from mltrading.core.recipes.aplicar import aplicar_receta
from mltrading.core.recipes.schema import (
    TIPO_MODELO_SUPERVISADO, VERSION_ESQUEMA, RecetaInvalida, RecetaModelo, desde_json,
)

RECETA_PREDETERMINADA = RecetaModelo(
    tipo=TIPO_MODELO_SUPERVISADO, version=VERSION_ESQUEMA, id_modelo="random_forest",
    parametros_modelo={"n_estimators": 3, "max_depth": 2, "max_features": 3, "random_state": 4},
    barras_por_dia=26, umbral_adf=0.05, umbral_correlacion=0.7,
    descartes_correlacion=["volatility2"], proporcion_entrenamiento=0.8,
)


def test_ida_y_vuelta_por_json():
    texto = RECETA_PREDETERMINADA.a_json()
    recuperada = desde_json(texto)
    assert recuperada == RECETA_PREDETERMINADA


def test_aplicar_receta_predeterminada_reproduce_el_resultado_conocido():
    jpm = cargar("JPM_2017_2019.csv")[0]
    resultado = aplicar_receta(RECETA_PREDETERMINADA, jpm)
    acc = (resultado.resultado_entrenamiento.y_pred_prueba.values == resultado.division.y_prueba.values).mean()
    assert round(acc * 100, 2) == 51.55
    assert resultado.division.filas_entrenamiento == 15453
    assert resultado.division.filas_prueba == 3864


@pytest.mark.parametrize(
    "texto,fragmento",
    [
        ("no es json", "JSON válido"),
        ("[]", "objeto JSON"),
        ("{}", "Faltan campos"),
        ('{"tipo":"x"}', "Faltan campos"),
    ],
)
def test_json_invalido_o_incompleto(texto, fragmento):
    with pytest.raises(RecetaInvalida, match=fragmento):
        desde_json(texto)


def test_tipo_no_admitido():
    datos = RECETA_PREDETERMINADA.a_json().replace('"modelo_supervisado"', '"otro_tipo"')
    with pytest.raises(RecetaInvalida, match="Tipo de receta"):
        desde_json(datos)


def test_version_no_admitida():
    datos = RECETA_PREDETERMINADA.a_json().replace('"version": 1', '"version": 99')
    with pytest.raises(RecetaInvalida, match="Versión de receta"):
        desde_json(datos)


def test_campo_sobrante_se_rechaza():
    import json

    datos = json.loads(RECETA_PREDETERMINADA.a_json())
    datos["campo_extra"] = "algo"
    with pytest.raises(RecetaInvalida, match="no reconocidos"):
        desde_json(json.dumps(datos))


def test_proporcion_fuera_de_rango_se_rechaza():
    import json

    datos = json.loads(RECETA_PREDETERMINADA.a_json())
    datos["proporcion_entrenamiento"] = 1.5
    with pytest.raises(RecetaInvalida, match="proporcion_entrenamiento"):
        desde_json(json.dumps(datos))


def test_tipo_de_campo_incorrecto_se_rechaza():
    import json

    datos = json.loads(RECETA_PREDETERMINADA.a_json())
    datos["barras_por_dia"] = "veintiséis"
    with pytest.raises(RecetaInvalida, match="barras_por_dia"):
        desde_json(json.dumps(datos))


def test_modelo_desconocido_falla_al_aplicar():
    import dataclasses

    receta = dataclasses.replace(RECETA_PREDETERMINADA, id_modelo="modelo_que_no_existe")
    jpm = cargar("JPM_2017_2019.csv")[0]
    with pytest.raises(KeyError):
        aplicar_receta(receta, jpm)


def test_ninguna_receta_ejecuta_codigo_arbitrario():
    """Confirma que desde_json solo usa json.loads, nunca eval, exec ni pickle."""
    import inspect

    from mltrading.core.recipes import schema

    codigo = inspect.getsource(schema)
    for prohibido in ("eval(", "exec(", "pickle.", "__import__", "compile("):
        assert prohibido not in codigo


def test_limite_de_tamano_del_archivo_de_receta_esta_documentado():
    """La página rechaza archivos mayores a 64 KB antes de intentar interpretarlos como JSON.

    Cubierto de forma end-to-end en test_app_smoke.py; aquí solo se fija el umbral esperado.
    """
    from mltrading.views.modelos import _TAMANO_MAXIMO_RECETA

    assert _TAMANO_MAXIMO_RECETA == 64 * 1024
