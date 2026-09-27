"""Resumen tabular del catálogo (una fila por conjunto de datos)."""
from __future__ import annotations

import pandas as pd

from .catalog import CATALOGO
from .loaders import cargar
from .validation import AVISO, ERROR, validar


def resumen_catalogo() -> pd.DataFrame:
    filas = []
    for spec in CATALOGO:
        df, informe = cargar(spec.archivo)
        incidencias = validar(df, spec, informe)
        n_problemas = sum(1 for i in incidencias if i.nivel in {ERROR, AVISO})
        filas.append(
            {
                "Archivo": spec.archivo,
                "Tipo": spec.tipo,
                "Frecuencia": spec.frecuencia,
                "Filas": informe.filas,
                "Columnas": informe.columnas,
                "Inicio": informe.inicio,
                "Fin": informe.fin,
                "Filas con nulos": informe.filas_con_nulos,
                "Incidencias": n_problemas,
                "Se usará en": ", ".join(spec.paginas),
            }
        )
    return pd.DataFrame(filas)
