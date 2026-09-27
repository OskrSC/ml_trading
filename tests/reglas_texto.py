"""Reglas de redacción que se comprueban automáticamente.

Este archivo vive en tests/ y por eso queda fuera de la revisión de texto.
"""
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Código y textos de la interfaz que se revisan.
ARCHIVOS_INTERFAZ = [RAIZ / "streamlit_app.py", *sorted((RAIZ / "mltrading").rglob("*.py"))]

# Además de la interfaz, estos archivos tampoco pueden contener emojis.
ARCHIVOS_SIN_EMOJIS = [
    *ARCHIVOS_INTERFAZ,
    RAIZ / "README.md",
    RAIZ / "NOTICE",
    RAIZ / "LICENSE",
    RAIZ / ".streamlit" / "config.toml",
    *sorted((RAIZ / "docs").glob("*.md")),
    *sorted((RAIZ / ".github").rglob("*.yml")),
]

# Módulos donde sí se puede citar el material de origen (atribución exigida por su licencia).
PERMITE_ATRIBUCION = {
    RAIZ / "mltrading" / "config" / "atribucion.py",
}

# Expresiones retiradas: lo que sirve de referencia se llama "predeterminado".
EXPRESIONES_RETIRADAS = [
    r"coincide con el libro",
    r"reproducir el libro",
    r"valores? del libro",
    r"resultados? del libro",
    r"pruebas? de paridad",
    r"paridad con",
    r"verificaci[oó]n contra",
    r"\bel libro\b",
]

RANGOS_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # pictogramas, emoticonos, símbolos suplementarios
    "\u2600-\u27BF"  # símbolos varios y dingbats
    "\u2B00-\u2BFF"  # flechas y símbolos varios
    "\uFE0F\u200D"  # selector de variación y unión de emojis
    "]"
)
