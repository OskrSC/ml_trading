"""Configuración de la preparación de variables y su valor predeterminado."""
from __future__ import annotations

from dataclasses import dataclass, replace

MOTOR_AUTO = "auto"
MOTOR_TALIB = "talib"
MOTOR_PANDAS = "pandas"
MOTORES = (MOTOR_AUTO, MOTOR_TALIB, MOTOR_PANDAS)


@dataclass(frozen=True)
class ConfigVariables:
    # Barras de 15 minutos en una jornada de 6,5 horas: 6,5 x 4 = 26.
    barras_por_dia: int = 26
    umbral_adf: float = 0.05
    umbral_correlacion: float = 0.7
    # El motor de indicadores no cambia el resultado (se comprueba en las pruebas),
    # por eso no forma parte de la definición de "predeterminada".
    motor: str = MOTOR_AUTO

    def __post_init__(self) -> None:
        if self.barras_por_dia < 5:
            raise ValueError("barras_por_dia debe ser al menos 5.")
        if not 0 < self.umbral_adf < 1:
            raise ValueError("umbral_adf debe estar entre 0 y 1.")
        if not 0 < self.umbral_correlacion < 1:
            raise ValueError("umbral_correlacion debe estar entre 0 y 1.")
        if self.motor not in MOTORES:
            raise ValueError(f"motor debe ser uno de {MOTORES}.")


CONFIG_PREDETERMINADA = ConfigVariables()


def es_predeterminada(config: ConfigVariables) -> bool:
    return replace(config, motor=MOTOR_AUTO) == CONFIG_PREDETERMINADA
