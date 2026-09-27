import pytest

from reglas_texto import ARCHIVOS_SIN_EMOJIS, RANGOS_EMOJI


@pytest.mark.parametrize("ruta", ARCHIVOS_SIN_EMOJIS, ids=lambda p: str(p.name))
def test_sin_emojis(ruta):
    texto = ruta.read_text(encoding="utf-8")
    encontrados = sorted(set(RANGOS_EMOJI.findall(texto)))
    assert not encontrados, f"{ruta.name} contiene emojis: {encontrados}"


def test_la_regla_detecta_emojis():
    assert RANGOS_EMOJI.search("resultado \u2705")
    assert RANGOS_EMOJI.search("aviso \u26a0\ufe0f")
    assert not RANGOS_EMOJI.search("Cotización, rendimiento ≥ 3 → correcto")
