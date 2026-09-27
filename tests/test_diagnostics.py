import json

import pytest

from mltrading.core.env import diagnostics as d


def test_estado_de_paquetes_incluye_lo_esencial():
    filas = {f["Paquete"]: f for f in d.estado_paquetes()}
    assert filas["streamlit"]["Estado"] == "Instalado"
    assert filas["streamlit"]["Requerido"] == "Sí"
    assert "graphviz (programa dot)" in filas
    assert filas["tensorflow"]["Requerido"] == "Opcional"


def test_medir_importacion_en_subproceso():
    r = d.medir_importacion("json")
    assert r["estado"] == "Correcto"
    assert r["segundos"] is not None and r["segundos"] >= 0


def test_modulo_inexistente_se_informa_sin_fallar():
    r = d.medir_importacion("paquete_que_no_existe_xyz")
    assert r["estado"] == "No instalado"


@pytest.mark.parametrize("nombre", ["os; import sys", "a b", "../x", "", "x-y"])
def test_nombres_de_modulo_no_validos_se_rechazan(nombre):
    with pytest.raises(ValueError):
        d.medir_importacion(nombre)


def test_carga_de_datos_mide_los_16_archivos():
    filas = d.medir_carga_datasets()
    assert len(filas) == 16
    assert all(f["segundos"] >= 0 and f["filas"] > 0 for f in filas)


def test_recursos_tienen_forma_valida():
    r = d.instantanea_recursos()
    assert r["memoria_proceso_mb"] is None or r["memoria_proceso_mb"] > 0
    limite = r["limite_contenedor_mb"]
    assert limite is None or limite > 0


def test_el_informe_se_puede_serializar():
    informe = d.construir_informe(d.medir_carga_datasets(), [d.medir_importacion("json")], "claro")
    texto = json.dumps(informe, ensure_ascii=False, default=str)
    assert json.loads(texto)["tema_detectado"] == "claro"
