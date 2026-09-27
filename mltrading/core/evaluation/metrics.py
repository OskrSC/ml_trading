"""Métricas de clasificación (capítulo 6)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResultadoEvaluacion:
    y_real: pd.Series
    y_predicho: pd.Series
    matriz_confusion: "list[list[int]]"
    exactitud: float
    reporte: pd.DataFrame  # filas: 0, 1, accuracy, macro avg, weighted avg

    @property
    def aciertos(self) -> pd.Series:
        return self.y_real == self.y_predicho


def evaluar(y_real: pd.Series, y_predicho: pd.Series) -> ResultadoEvaluacion:
    from sklearn.metrics import classification_report, confusion_matrix  # importación diferida

    if not y_real.index.equals(y_predicho.index):
        raise ValueError("y_real e y_predicho deben compartir el mismo índice.")
    matriz = confusion_matrix(y_real, y_predicho).tolist()
    exactitud = float((y_real.values == y_predicho.values).mean())
    reporte_dic = classification_report(y_real, y_predicho, output_dict=True, zero_division=0)
    filas = []
    for clave in ("0", "1"):
        if clave in reporte_dic:
            r = reporte_dic[clave]
            filas.append({"Clase": "Sin posición (0)" if clave == "0" else "Posición larga (1)",
                          "Precisión": r["precision"], "Sensibilidad": r["recall"],
                          "F1": r["f1-score"], "Soporte": int(r["support"])})
    for clave, etiqueta in (("macro avg", "Promedio simple"), ("weighted avg", "Promedio ponderado")):
        r = reporte_dic[clave]
        filas.append({"Clase": etiqueta, "Precisión": r["precision"], "Sensibilidad": r["recall"],
                      "F1": r["f1-score"], "Soporte": int(r["support"])})
    return ResultadoEvaluacion(
        y_real=y_real, y_predicho=y_predicho, matriz_confusion=matriz, exactitud=exactitud,
        reporte=pd.DataFrame(filas),
    )
