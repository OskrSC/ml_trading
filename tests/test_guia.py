"""Glosario y contenido de la página de guía."""
from mltrading.config.glosario import GLOSARIO, por_seccion


def test_glosario_no_tiene_terminos_repetidos():
    nombres = [t.nombre for t in GLOSARIO]
    assert len(nombres) == len(set(nombres))


def test_glosario_cubre_las_secciones_principales():
    secciones = {t.seccion for t in GLOSARIO}
    esperadas = {
        "Fundamentos", "Preparación de datos", "Modelado", "Evaluación", "Backtesting",
        "No supervisado", "Operación en vivo",
    }
    assert esperadas <= secciones


def test_por_seccion_agrupa_todos_los_terminos():
    agrupados = por_seccion()
    total = sum(len(v) for v in agrupados.values())
    assert total == len(GLOSARIO)


def test_cada_termino_tiene_definicion_no_vacia():
    for t in GLOSARIO:
        assert len(t.definicion) > 20
        assert len(t.nombre) > 0
        assert len(t.nombre_original) > 0
