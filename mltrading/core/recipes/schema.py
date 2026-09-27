"""Esquema de una receta: una configuración completa, sin ningún objeto serializado.

Guardar un modelo entrenado con `pickle` exige después cargarlo con `pickle`,
lo que ejecuta código arbitrario al abrir el archivo. Esta aplicación no
acepta esa clase de carga (ver docs/decisiones.md, punto sobre la fase 0). Una
receta es una alternativa seria: un JSON con los parámetros necesarios para
reconstruir exactamente el mismo modelo a partir de los datos de
`data_modules/`, validado campo a campo antes de usarse.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

VERSION_ESQUEMA = 1
TIPO_MODELO_SUPERVISADO = "modelo_supervisado"


class RecetaInvalida(ValueError):
    pass


@dataclass(frozen=True)
class RecetaModelo:
    tipo: str
    version: int
    id_modelo: str
    parametros_modelo: dict
    barras_por_dia: int
    umbral_adf: float
    umbral_correlacion: float
    descartes_correlacion: list
    proporcion_entrenamiento: float

    def a_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=False)


_CAMPOS_REQUERIDOS = {
    "tipo": str, "version": int, "id_modelo": str, "parametros_modelo": dict,
    "barras_por_dia": int, "umbral_adf": (int, float), "umbral_correlacion": (int, float),
    "descartes_correlacion": list, "proporcion_entrenamiento": (int, float),
}


def _validar_tipos(datos: dict) -> None:
    faltantes = sorted(set(_CAMPOS_REQUERIDOS) - set(datos))
    if faltantes:
        raise RecetaInvalida("Faltan campos en la receta: " + ", ".join(faltantes) + ".")
    sobrantes = sorted(set(datos) - set(_CAMPOS_REQUERIDOS))
    if sobrantes:
        raise RecetaInvalida("La receta tiene campos no reconocidos: " + ", ".join(sobrantes) + ".")
    for campo, tipo in _CAMPOS_REQUERIDOS.items():
        if not isinstance(datos[campo], tipo):
            raise RecetaInvalida(f"El campo '{campo}' debe ser de tipo {tipo}.")


def desde_json(texto: str) -> RecetaModelo:
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError as error:
        raise RecetaInvalida(f"El archivo no es un JSON válido: {error}.") from error
    if not isinstance(datos, dict):
        raise RecetaInvalida("La receta debe ser un objeto JSON, no una lista ni un valor suelto.")
    _validar_tipos(datos)

    if datos["tipo"] != TIPO_MODELO_SUPERVISADO:
        raise RecetaInvalida(f"Tipo de receta no admitido: '{datos['tipo']}'.")
    if datos["version"] != VERSION_ESQUEMA:
        raise RecetaInvalida(f"Versión de receta no admitida: {datos['version']}.")
    if not 0 < datos["proporcion_entrenamiento"] < 1:
        raise RecetaInvalida("proporcion_entrenamiento debe estar entre 0 y 1.")
    if not all(isinstance(v, str) for v in datos["descartes_correlacion"]):
        raise RecetaInvalida("descartes_correlacion debe ser una lista de nombres de variable.")
    if not isinstance(datos["parametros_modelo"], dict) or not all(
        isinstance(k, str) for k in datos["parametros_modelo"]
    ):
        raise RecetaInvalida("parametros_modelo debe ser un objeto con nombres de parámetro como claves.")

    return RecetaModelo(
        tipo=datos["tipo"], version=datos["version"], id_modelo=datos["id_modelo"],
        parametros_modelo=dict(datos["parametros_modelo"]), barras_por_dia=int(datos["barras_por_dia"]),
        umbral_adf=float(datos["umbral_adf"]), umbral_correlacion=float(datos["umbral_correlacion"]),
        descartes_correlacion=list(datos["descartes_correlacion"]),
        proporcion_entrenamiento=float(datos["proporcion_entrenamiento"]),
    )
