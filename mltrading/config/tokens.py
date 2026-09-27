"""Tokens de diseño: única fuente de verdad para color, tipografía y forma.

El archivo .streamlit/config.toml se genera a partir de este módulo con
scripts/generate_theme.py, y una prueba comprueba que ambos coinciden.

Decisiones de diseño:
- Los colores de las series de los gráficos son iguales en el tema claro y en
  el oscuro. Así una serie conserva su identidad al cambiar de tema y, como
  Streamlit cambia el tema en el navegador sin volver a ejecutar el script,
  ningún gráfico queda ilegible durante la transición. Cada color cumple un
  contraste mínimo de 3:1 contra los dos fondos.
- Solo cambian entre temas el fondo, el texto, el borde y el color primario.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Paleta:
    fondo: str
    fondo_secundario: str
    texto: str
    primario: str
    borde: str


CLARO = Paleta(
    fondo="#F8F9FB",
    fondo_secundario="#EBEEF4",
    texto="#15202E",
    primario="#1F4FA3",
    borde="#D3D9E4",
)

OSCURO = Paleta(
    fondo="#0E1522",
    fondo_secundario="#172033",
    texto="#E6EBF4",
    primario="#3D6AD6",
    borde="#2B3650",
)

# Series categóricas de los gráficos (idénticas en ambos temas).
SERIES = ("#2F6FDE", "#C96A12", "#1E9C7E", "#8E63C9", "#D0506A", "#77849A")
COLOR_ESTRATEGIA = SERIES[0]
COLOR_REFERENCIA = SERIES[1]

# Colores semánticos. Nunca se usan como único indicador: el signo también se
# codifica con la posición respecto del eje cero.
COLOR_POSITIVO = "#2E9E7F"
COLOR_NEGATIVO = "#D2554B"

# Escala divergente para correlaciones: negativo, neutro, positivo.
ESCALA_DIVERGENTE = ((0.0, "#C96A12"), (0.5, "#E4E7EE"), (1.0, "#2F6FDE"))

# Tipografía: cuerpo sobre una sans humanista con cifras tabulares y títulos en
# una serif de texto. Streamlit añade por su cuenta la fuente de respaldo
# (Source Sans incluida en la propia librería), así que aquí no se declara:
# si se añadiera tras la URL, se interpretaría como parte de ella.
FUENTE_TEXTO = (
    "Source Sans 3",
    "https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600&display=swap",
)
FUENTE_TITULOS = (
    "Newsreader",
    "https://fonts.googleapis.com/css2?family=Newsreader:wght@500;600&display=swap",
)

RADIO_BASE = "0.5rem"
TAMANOS_TITULOS = ("2.25rem", "1.75rem", "1.375rem", "1.125rem", "1rem", "0.875rem")
PESOS_TITULOS = (600, 600, 600, 600, 600, 600)
