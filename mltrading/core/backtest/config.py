"""Configuración del backtest y su valor predeterminado."""
from __future__ import annotations

from dataclasses import dataclass

# Barras de 15 minutos en un año de 252 sesiones de 6,5 horas: 252 x 6,5 x 4.
FACTOR_ANUALIZACION_15MIN = 252 * 6.5 * 4


@dataclass(frozen=True)
class ConfigBacktest:
    costo_bp: float = 0.0  # puntos básicos por operación (compra o venta); 0 reproduce el resultado predeterminado
    factor_anualizacion: float = FACTOR_ANUALIZACION_15MIN

    def __post_init__(self) -> None:
        if self.costo_bp < 0:
            raise ValueError("El costo no puede ser negativo.")
        if self.factor_anualizacion <= 0:
            raise ValueError("El factor de anualización debe ser positivo.")


CONFIG_PREDETERMINADA = ConfigBacktest()


def es_predeterminada(config: ConfigBacktest) -> bool:
    return config.costo_bp == 0.0 and config.factor_anualizacion == FACTOR_ANUALIZACION_15MIN
