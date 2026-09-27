"""Analítica agregada del registro de operaciones (capítulo 7)."""
from __future__ import annotations

import pandas as pd


def analitica(registro: pd.DataFrame) -> pd.DataFrame:
    if registro.empty:
        vacio = {
            "Operaciones largas": 0, "Operaciones cortas": 0, "Total de operaciones": 0,
            "Ganancia bruta": 0.0, "Pérdida bruta": 0.0, "Resultado neto": 0.0,
            "Ganadoras": 0, "Perdedoras": 0, "Porcentaje de aciertos": float("nan"),
            "Resultado medio (ganadoras)": float("nan"), "Resultado medio (perdedoras)": float("nan"),
        }
        return pd.DataFrame([vacio])

    largas = int((registro["Posición"] == 1).sum())
    cortas = int((registro["Posición"] == -1).sum())
    ganadoras = registro.loc[registro["PnL"] > 0, "PnL"]
    perdedoras = registro.loc[registro["PnL"] <= 0, "PnL"]
    total = len(registro)
    datos = {
        "Operaciones largas": largas,
        "Operaciones cortas": cortas,
        "Total de operaciones": total,
        "Ganancia bruta": float(ganadoras.sum()),
        "Pérdida bruta": float(perdedoras.sum()),
        "Resultado neto": float(registro["PnL"].sum()),
        "Ganadoras": len(ganadoras),
        "Perdedoras": len(perdedoras),
        "Porcentaje de aciertos": 100 * len(ganadoras) / total if total else float("nan"),
        "Resultado medio (ganadoras)": float(ganadoras.mean()) if len(ganadoras) else float("nan"),
        "Resultado medio (perdedoras)": float(perdedoras.mean()) if len(perdedoras) else float("nan"),
    }
    return pd.DataFrame([datos])
