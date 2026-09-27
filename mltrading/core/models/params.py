"""Definición declarativa de hiperparámetros para construir controles genéricos."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParametroEntero:
    nombre: str
    etiqueta: str
    valor_predeterminado: int
    minimo: int
    maximo: int
    ayuda: str = ""
    paso: int = 1


@dataclass(frozen=True)
class ParametroFlotante:
    nombre: str
    etiqueta: str
    valor_predeterminado: float
    minimo: float
    maximo: float
    ayuda: str = ""
    paso: float = 0.01
    formato: str = "%.2f"


@dataclass(frozen=True)
class ParametroCategorico:
    nombre: str
    etiqueta: str
    valor_predeterminado: str
    opciones: tuple[str, ...]
    ayuda: str = ""


Parametro = ParametroEntero | ParametroFlotante | ParametroCategorico


def valores_predeterminados(parametros: tuple[Parametro, ...]) -> dict[str, Any]:
    return {p.nombre: p.valor_predeterminado for p in parametros}
