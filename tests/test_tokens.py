import re
import tomllib

from mltrading.config import tokens as t
from mltrading.config.theme_config import generar_config_toml
from mltrading.ui.color import contraste
from reglas_texto import RAIZ

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def test_config_toml_esta_sincronizado_con_los_tokens():
    actual = (RAIZ / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert actual == generar_config_toml(), "Ejecute: python scripts/generate_theme.py"


def test_config_toml_define_ambos_temas():
    datos = tomllib.loads((RAIZ / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    for nombre, paleta in (("light", t.CLARO), ("dark", t.OSCURO)):
        tema = datos["theme"][nombre]
        assert tema["primaryColor"] == paleta.primario
        assert tema["backgroundColor"] == paleta.fondo
        assert tema["textColor"] == paleta.texto
    assert datos["theme"]["chartCategoricalColors"] == list(t.SERIES)


def test_todos_los_colores_son_hexadecimales():
    colores = [*t.SERIES, t.COLOR_POSITIVO, t.COLOR_NEGATIVO]
    for paleta in (t.CLARO, t.OSCURO):
        colores += [paleta.fondo, paleta.fondo_secundario, paleta.texto, paleta.primario, paleta.borde]
    assert all(HEX.match(c) for c in colores)
    assert len(set(t.SERIES)) == len(t.SERIES)


def test_contraste_de_texto_y_botones():
    for p in (t.CLARO, t.OSCURO):
        assert contraste(p.texto, p.fondo) >= 4.5
        assert contraste(p.texto, p.fondo_secundario) >= 4.5
        assert contraste("#FFFFFF", p.primario) >= 4.5  # texto de los botones primarios
        assert contraste(p.primario, p.fondo) >= 3.0


def test_series_y_semanticos_legibles_en_ambos_fondos():
    for p in (t.CLARO, t.OSCURO):
        for color in (*t.SERIES, t.COLOR_POSITIVO, t.COLOR_NEGATIVO):
            assert contraste(color, p.fondo) >= 3.0, (color, p.fondo)


def test_calculo_de_contraste_conocido():
    assert round(contraste("#000000", "#FFFFFF"), 1) == 21.0
    assert round(contraste("#777777", "#FFFFFF"), 2) == 4.48
