"""Contrato común de los modelos supervisados del registro."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from .params import Parametro


@dataclass
class ResultadoEntrenamiento:
    """Salida uniforme de cualquier modelo del registro."""

    estimador: Any
    y_pred_prueba: pd.Series
    y_pred_entrenamiento: pd.Series | None = None
    probabilidades_prueba: np.ndarray | None = None  # columna de la clase 1, si aplica
    interpretacion: pd.DataFrame | None = None  # coeficientes o importancias, según el modelo
    notas: tuple[str, ...] = field(default_factory=tuple)


class FuncionEntrenamiento(Protocol):
    def __call__(
        self, X_entrenamiento: pd.DataFrame, y_entrenamiento: pd.Series, X_prueba: pd.DataFrame,
        parametros: dict[str, Any],
    ) -> ResultadoEntrenamiento: ...


@dataclass(frozen=True)
class EspecificacionModelo:
    id: str
    nombre: str
    capitulo: str
    familia: str
    descripcion: str
    parametros: tuple[Parametro, ...]
    entrenar: FuncionEntrenamiento
    necesita_escalado: bool = False
    admite_probabilidad: bool = True
