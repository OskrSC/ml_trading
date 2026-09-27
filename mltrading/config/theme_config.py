"""Genera el bloque de temas de .streamlit/config.toml a partir de los tokens."""
from __future__ import annotations

from . import tokens as t

ENCABEZADO = (
    "# ARCHIVO GENERADO por scripts/generate_theme.py.\n"
    "# No editar a mano: modifique mltrading/config/tokens.py y vuelva a generarlo.\n"
)


def _fuente(nombre: str, url: str) -> str:
    return f"'{nombre}':{url}"


def _lista(valores) -> str:
    return "[" + ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in valores) + "]"


def _bloque_paleta(nombre: str, p: t.Paleta) -> str:
    return (
        f"[theme.{nombre}]\n"
        f'primaryColor = "{p.primario}"\n'
        f'backgroundColor = "{p.fondo}"\n'
        f'secondaryBackgroundColor = "{p.fondo_secundario}"\n'
        f'textColor = "{p.texto}"\n'
        f'borderColor = "{p.borde}"\n'
    )


def generar_config_toml() -> str:
    comun = (
        "[theme]\n"
        f'baseRadius = "{t.RADIO_BASE}"\n'
        "showSidebarBorder = true\n"
        f'font = "{_fuente(*t.FUENTE_TEXTO)}"\n'
        f'headingFont = "{_fuente(*t.FUENTE_TITULOS)}"\n'
        f"headingFontSizes = {_lista(t.TAMANOS_TITULOS)}\n"
        f"headingFontWeights = {_lista(t.PESOS_TITULOS)}\n"
        f"chartCategoricalColors = {_lista(t.SERIES)}\n"
    )
    otros = (
        "[browser]\n"
        "gatherUsageStats = false\n\n"
        "[client]\n"
        '# "type" en producción: se ve el tipo de error, no la traza completa ni rutas de archivo.\n'
        '# Para depurar en local con la traza completa, sin tocar este archivo, ejecute:\n'
        '# streamlit run streamlit_app.py --client.showErrorDetails=full\n'
        'showErrorDetails = "type"\n'
    )
    return "\n".join(
        [
            ENCABEZADO,
            comun,
            _bloque_paleta("light", t.CLARO),
            _bloque_paleta("dark", t.OSCURO),
            otros,
        ]
    )
