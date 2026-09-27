"""Mediciones del entorno de ejecución para decidir el perfil de despliegue.

Todo lo que pueda consumir memoria de forma apreciable (importar TensorFlow,
por ejemplo) se mide en un subproceso aislado: si el servicio lo termina por
falta de memoria, la aplicación principal sigue viva y el resultado se informa.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from mltrading.core.data.catalog import CATALOGO
from mltrading.core.data.loaders import cargar


@dataclass(frozen=True)
class Paquete:
    modulo: str
    distribucion: str
    rol: str
    requerido: bool


PAQUETES: tuple[Paquete, ...] = (
    Paquete("streamlit", "streamlit", "Interfaz", True),
    Paquete("pandas", "pandas", "Datos tabulares", True),
    Paquete("numpy", "numpy", "Cálculo numérico", True),
    Paquete("plotly", "plotly", "Gráficos interactivos", True),
    Paquete("psutil", "psutil", "Medición de recursos", True),
    Paquete("sklearn", "scikit-learn", "Modelos clásicos y clustering", True),
    Paquete("scipy", "scipy", "Dendrograma y estadística", True),
    Paquete("statsmodels", "statsmodels", "Regresión lineal y prueba ADF", True),
    Paquete("xgboost", "xgboost", "Modelo XGBoost", True),
    Paquete("talib", "TA-Lib", "Indicadores técnicos", True),
    Paquete("tensorflow", "tensorflow", "Red neuronal con Keras", False),
    Paquete("keras", "keras", "Red neuronal con Keras", False),
    Paquete("yfinance", "yfinance", "Descarga de precios en vivo", False),
    Paquete("graphviz", "graphviz", "Dibujo del árbol de decisión", False),
)

_MODULO_VALIDO = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


# ---------------------------------------------------------------- entorno ---
def informacion_entorno() -> dict:
    import streamlit as st

    from mltrading.config import settings

    return {
        "perfil": settings.PERFIL,
        "python": platform.python_version(),
        "implementacion": platform.python_implementation(),
        "sistema": platform.platform(),
        "arquitectura": platform.machine(),
        "nucleos": os.cpu_count(),
        "streamlit": st.__version__,
        "directorio_datos": str(Path(os.environ.get("MLT_DATA_DIR", "data_modules"))),
    }


def estado_paquetes() -> list[dict]:
    filas = []
    for p in PAQUETES:
        instalado = find_spec(p.modulo) is not None
        try:
            version = metadata.version(p.distribucion) if instalado else ""
        except metadata.PackageNotFoundError:
            version = "sin metadatos"
        filas.append(
            {
                "Paquete": p.distribucion,
                "Uso": p.rol,
                "Requerido": "Sí" if p.requerido else "Opcional",
                "Estado": "Instalado" if instalado else "No instalado",
                "Versión": version,
            }
        )
    ruta_dot = shutil.which("dot")
    filas.append(
        {
            "Paquete": "graphviz (programa dot)",
            "Uso": "Dibujo del árbol de decisión",
            "Requerido": "Opcional",
            "Estado": "Instalado" if ruta_dot else "No instalado",
            "Versión": ruta_dot or "",
        }
    )
    return filas


# ---------------------------------------------------------------- memoria ---
def _rss_desde_proc() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for linea in f:
                if linea.startswith("VmRSS:"):
                    return int(linea.split()[1]) / 1024
    except OSError:
        pass
    return None


def memoria_proceso_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except Exception:  # noqa: BLE001
        return _rss_desde_proc()


def _leer_entero(ruta: str) -> int | None:
    try:
        texto = Path(ruta).read_text(encoding="utf-8").strip()
        return None if texto == "max" else int(texto)
    except (OSError, ValueError):
        return None


def limite_memoria_contenedor_mb() -> float | None:
    """Límite de memoria del contenedor (cgroup v2 o v1), o None si no hay."""
    for ruta in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        valor = _leer_entero(ruta)
        if valor is not None and valor < 1 << 50:  # valores enormes equivalen a "sin límite"
            return valor / 1024**2
    return None


def uso_memoria_contenedor_mb() -> float | None:
    for ruta in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        valor = _leer_entero(ruta)
        if valor is not None:
            return valor / 1024**2
    return None


def instantanea_recursos() -> dict:
    try:
        import psutil

        proceso = psutil.Process()
        segundos_activo = time.time() - proceso.create_time()
        total = psutil.virtual_memory().total / 1024**2
    except Exception:  # noqa: BLE001
        segundos_activo, total = None, None
    return {
        "memoria_proceso_mb": memoria_proceso_mb(),
        "memoria_contenedor_mb": uso_memoria_contenedor_mb(),
        "limite_contenedor_mb": limite_memoria_contenedor_mb(),
        "memoria_total_servidor_mb": total,
        "segundos_activo": segundos_activo,
    }


# ------------------------------------------------------------- mediciones ---
def medir_carga_datasets() -> list[dict]:
    """Lee cada conjunto de datos y anota tiempo y cambio aproximado de memoria."""
    filas = []
    for spec in CATALOGO:
        antes = memoria_proceso_mb()
        t0 = time.perf_counter()
        df, informe = cargar(spec.archivo)
        segundos = time.perf_counter() - t0
        despues = memoria_proceso_mb()
        filas.append(
            {
                "archivo": spec.archivo,
                "filas": informe.filas,
                "segundos": round(segundos, 4),
                "memoria_df_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
                "delta_rss_mb": None if antes is None or despues is None else round(despues - antes, 2),
            }
        )
        del df
    return filas


_CODIGO_HIJO = r"""
import importlib, json, sys, time
def rss():
    try:
        with open('/proc/self/status') as f:
            for linea in f:
                if linea.startswith('VmRSS:'):
                    return int(linea.split()[1]) / 1024
    except OSError:
        return None
modulo = sys.argv[1]
r0 = rss(); t0 = time.perf_counter()
try:
    importlib.import_module(modulo); ok, error = True, None
except BaseException as exc:
    ok, error = False, repr(exc)[:300]
print(json.dumps({'ok': ok, 'segundos': time.perf_counter() - t0, 'rss0': r0, 'rss1': rss(), 'error': error}))
"""


def medir_importacion(modulo: str, tiempo_maximo: int = 180) -> dict:
    """Importa `modulo` en un subproceso y mide tiempo y memoria añadida."""
    if not _MODULO_VALIDO.match(modulo):
        raise ValueError(f"Nombre de módulo no válido: {modulo!r}")
    if find_spec(modulo.split(".")[0]) is None:
        return {"modulo": modulo, "estado": "No instalado", "segundos": None, "memoria_mb": None, "detalle": ""}
    try:
        proceso = subprocess.run(
            [sys.executable, "-c", _CODIGO_HIJO, modulo],
            capture_output=True,
            text=True,
            timeout=tiempo_maximo,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"modulo": modulo, "estado": "Tiempo agotado", "segundos": None, "memoria_mb": None,
                "detalle": f"Más de {tiempo_maximo} s"}
    if proceso.returncode != 0:
        return {"modulo": modulo, "estado": "Terminado por el sistema", "segundos": None, "memoria_mb": None,
                "detalle": f"Código de salida {proceso.returncode} (probable falta de memoria)"}
    try:
        datos = json.loads(proceso.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {"modulo": modulo, "estado": "Sin respuesta", "segundos": None, "memoria_mb": None,
                "detalle": proceso.stderr[-200:]}
    memoria = None
    if datos.get("rss0") is not None and datos.get("rss1") is not None:
        memoria = round(datos["rss1"] - datos["rss0"], 1)
    return {
        "modulo": modulo,
        "estado": "Correcto" if datos["ok"] else "Falló",
        "segundos": round(datos["segundos"], 2),
        "memoria_mb": memoria,
        "detalle": datos.get("error") or "",
    }


# ---------------------------------------------------------------- informe ---
def construir_informe(cargas: list[dict] | None, importaciones: list[dict] | None, tema: str) -> dict:
    return {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entorno": informacion_entorno(),
        "paquetes": estado_paquetes(),
        "recursos": instantanea_recursos(),
        "tema_detectado": tema,
        "carga_de_datos": cargas,
        "importacion_de_librerias": importaciones,
    }
