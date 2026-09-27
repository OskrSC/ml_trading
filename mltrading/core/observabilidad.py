"""Registro estructurado de la aplicación.

Escribe a la salida estándar (donde Streamlit Community Cloud y la mayoría de
servicios de despliegue capturan los registros) y, además, guarda las últimas
entradas en memoria para que la página de Diagnóstico del entorno las pueda
mostrar sin depender de acceso al sistema de archivos del servidor, que en la
nube es efímero.

El buffer en memoria es por proceso, no por sesión: en un servicio con varios
usuarios a la vez, puede mostrar entradas de otras sesiones. Es una ayuda de
diagnóstico, no un sistema de auditoría por persona.
"""
from __future__ import annotations

import logging
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

NOMBRE_LOGGER = "mltrading"
TAMANO_BUFFER = 200

_buffer: deque[dict] = deque(maxlen=TAMANO_BUFFER)


@dataclass(frozen=True)
class EntradaRegistro:
    marca_tiempo: str
    nivel: str
    origen: str
    mensaje: str


class _ManejadorEnMemoria(logging.Handler):
    def emit(self, registro: logging.LogRecord) -> None:
        _buffer.append(
            {
                "marca_tiempo": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "nivel": registro.levelname,
                "origen": registro.name,
                "mensaje": self.format(registro),
            }
        )


_configurado = False


def configurar(nivel: int = logging.INFO) -> logging.Logger:
    """Configura el logger una sola vez por proceso (seguro de llamar varias veces)."""
    global _configurado
    logger = logging.getLogger(NOMBRE_LOGGER)
    if _configurado:
        return logger
    logger.setLevel(nivel)
    formato = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    salida = logging.StreamHandler(stream=sys.stdout)
    salida.setFormatter(formato)
    logger.addHandler(salida)

    memoria = _ManejadorEnMemoria()
    memoria.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(memoria)

    logger.propagate = False
    _configurado = True
    return logger


def obtener_logger(nombre: str) -> logging.Logger:
    """Logger hijo de 'mltrading', p. ej. obtener_logger('modelos')."""
    configurar()
    return logging.getLogger(f"{NOMBRE_LOGGER}.{nombre}")


def ultimas_entradas(n: int = 50) -> list[EntradaRegistro]:
    configurar()
    return [EntradaRegistro(**e) for e in list(_buffer)[-n:]]


def limpiar_buffer() -> None:
    _buffer.clear()
