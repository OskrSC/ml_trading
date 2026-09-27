"""Fases de implementación y su estado, mostradas en la página de inicio."""
from __future__ import annotations

from dataclasses import dataclass

EN_CURSO = "En curso"
PENDIENTE = "Pendiente"
COMPLETADA = "Completada"


@dataclass(frozen=True)
class Fase:
    numero: int
    nombre: str
    alcance: str
    estado: str


FASES: tuple[Fase, ...] = (
    Fase(0, "Cimientos y entorno en la nube", "Tema, catálogo de datos, diagnóstico y despliegue base", COMPLETADA),
    Fase(1, "Datos, variables y división", "Indicadores técnicos, estacionariedad, correlación y división temporal", COMPLETADA),
    Fase(2, "Primer flujo completo", "Random Forest y regresión logística con evaluación y backtesting", COMPLETADA),
    Fase(3, "Resto de modelos supervisados", "Naive Bayes, árboles, regresión lineal, XGBoost, red neuronal y comparador", COMPLETADA),
    Fase(4, "Aprendizaje no supervisado", "K-Means, clustering jerárquico, PCA y t-SNE", COMPLETADA),
    Fase(5, "Operación en vivo y guía", "Simulación día a día, recetas de modelos, glosario y referencias", COMPLETADA),
    Fase(6, "Endurecimiento y lanzamiento", "Rendimiento, seguridad, accesibilidad y despliegue continuo", COMPLETADA),
)
