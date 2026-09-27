import re

import pytest

from reglas_texto import ARCHIVOS_INTERFAZ, EXPRESIONES_RETIRADAS, PERMITE_ATRIBUCION

REVISADOS = [p for p in ARCHIVOS_INTERFAZ if p not in PERMITE_ATRIBUCION]


@pytest.mark.parametrize("ruta", REVISADOS, ids=lambda p: str(p.name))
def test_sin_expresiones_retiradas(ruta):
    texto = ruta.read_text(encoding="utf-8").lower()
    halladas = [e for e in EXPRESIONES_RETIRADAS if re.search(e, texto)]
    assert not halladas, f"{ruta.name} usa expresiones retiradas: {halladas}"


def test_la_regla_detecta_las_expresiones():
    ejemplo = "Resultado que coincide con el libro y pruebas de paridad".lower()
    assert sum(bool(re.search(e, ejemplo)) for e in EXPRESIONES_RETIRADAS) >= 2
