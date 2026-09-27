"""Genera los artefactos predeterminados que se guardan en artefactos/.

Uso:  python scripts/generar_artefactos.py
Vuelva a ejecutarlo si cambian los datos, la lista de variables candidatas o
las versiones de statsmodels y numpy.
"""
from pathlib import Path
import sys
import time

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from mltrading.core.data.loaders import cargar  # noqa: E402
from mltrading.core.features import predeterminados, seleccion  # noqa: E402
from mltrading.core.features.config import CONFIG_PREDETERMINADA  # noqa: E402
from mltrading.core.features.construccion import construir_candidatas  # noqa: E402

datos, _ = cargar("JPM_2017_2019.csv")
candidatas = construir_candidatas(datos, CONFIG_PREDETERMINADA)
t0 = time.perf_counter()
tabla = seleccion.calcular_adf(candidatas.variables)
ruta = predeterminados.guardar_adf(tabla, CONFIG_PREDETERMINADA.barras_por_dia)
print(f"ADF calculada en {time.perf_counter() - t0:.1f} s. Guardada en {ruta}")
