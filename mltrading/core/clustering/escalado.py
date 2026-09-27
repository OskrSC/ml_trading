"""Escalado compartido por los módulos de clustering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def escalar(datos: pd.DataFrame) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler  # importación diferida

    return StandardScaler().fit_transform(datos.values)
