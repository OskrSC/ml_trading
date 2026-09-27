"""Guía y referencias: fundamentos, glosario, temas sin implementación y licencias."""
from __future__ import annotations

import streamlit as st

from mltrading.config import atribucion
from mltrading.config.glosario import por_seccion
from mltrading.ui import components as c

TITULO = "Guía y referencias"
LEAD = (
    "Fundamentos de la aplicación, un glosario de los términos que aparecen en cada página, y las tres "
    "familias de modelos que el material de origen trata en profundidad pero que esta aplicación no implementa, por no "
    "disponer de código ni datos de origen para ellas."
)


def _fundamentos() -> None:
    st.write(
        "Un modelo de aprendizaje automático no memoriza reglas: aprende un patrón a partir de ejemplos. En "
        "esta aplicación, cada ejemplo es una barra de 15 minutos de JPMorgan, y lo que se aprende es a "
        "distinguir cuándo el precio de la siguiente barra tiende a subir de cuándo tiende a bajar."
    )
    st.write(
        "El aprendizaje **supervisado** (páginas Modelos supervisados, Regresión y XGBoost multiactivo) parte "
        "de ejemplos donde ya se conoce la respuesta correcta: para cada barra pasada, se sabe si el precio "
        "subió o bajó después. El modelo aprende a predecir esa respuesta para barras que no ha visto."
    )
    st.write(
        "El aprendizaje **no supervisado** (páginas K-Means, Clustering jerárquico y PCA y t-SNE) no tiene una "
        "respuesta correcta que aprender. En su lugar, busca estructura en los datos: qué acciones se parecen "
        "entre sí, sin que nadie le diga de antemano en qué grupo debería quedar cada una."
    )
    st.write(
        "Antes de entrenar cualquier modelo hace falta decidir dos cosas: qué se quiere predecir (el "
        "**objetivo**) y con qué información (las **variables**). Un objetivo mal definido, o variables que en "
        "la práctica no estarían disponibles en el momento de predecir, invalidan el modelo aunque sus métricas "
        "se vean bien en la pantalla."
    )


def _glosario() -> None:
    st.write("Los términos están agrupados por la sección de la aplicación donde aparecen primero.")
    for seccion, terminos in por_seccion().items():
        st.markdown(f"**{seccion}**")
        for t in terminos:
            st.markdown(f"- **{t.nombre}** ({t.nombre_original}): {t.definicion}")


def _no_supervisado_intro() -> None:
    st.write(
        "Las páginas de esta aplicación ya cubren clustering y PCA con código funcionando. Esta sección resume "
        "la idea general con la que el material de origen presenta el aprendizaje no supervisado, antes de entrar en esos "
        "algoritmos concretos."
    )
    st.write(
        "La diferencia con la clasificación es que, en clustering, el algoritmo no recibe ninguna etiqueta de "
        "compra o no compra: solo recibe las variables, y agrupa los casos parecidos entre sí. Después, quien "
        "usa el modelo tiene que examinar cada grupo y decidir qué significa. Por ejemplo, se le pueden pasar "
        "datos de precio y fundamentales de muchas acciones, y el algoritmo agrupará las que se comportan de "
        "forma parecida, sin saber de antemano que existen sectores como bancos o tecnología: esa etiqueta la "
        "pone la persona después, al mirar qué quedó agrupado con qué."
    )
    st.write(
        "Una limitación importante: no hay control directo sobre qué agrupa el algoritmo. Un grupo puede "
        "resultar útil, y el de al lado puede mezclar casos que no tienen nada en común desde el punto de vista "
        "de quien opera. Interpretar los grupos sigue siendo tarea de una persona, no del algoritmo."
    )


def _nlp() -> None:
    st.write(
        "El procesamiento de lenguaje natural (NLP) convierte texto en información que un modelo pueda usar. "
        "En trading, la aplicación más habitual es leer noticias, publicaciones o comunicados y convertirlos en "
        "una puntuación de sentimiento, que después se combina con indicadores técnicos para generar señales."
    )
    st.write("El material de origen describe el proceso en cinco pasos, sin proporcionar código ni datos para reproducirlo:")
    st.markdown(
        "1. Obtener el texto (noticias, publicaciones, comunicados).\n"
        "2. Limpiarlo: quitar menciones, etiquetas y elementos que no aportan significado.\n"
        "3. Convertirlo en una puntuación numérica de sentimiento.\n"
        "4. Combinar esa puntuación con otras variables para generar una señal.\n"
        "5. Probar el resultado con datos históricos antes de operar con él."
    )
    st.write(
        "Distingue además entre texto **estructurado** (comunicados oficiales, con lenguaje consistente) y "
        "**no estructurado** (publicaciones en redes sociales, con lenguaje variable y mensajes cortos), porque "
        "cada uno necesita un tratamiento distinto para extraer su sentimiento con precisión."
    )
    st.info(
        "Esta aplicación no implementa NLP: el capítulo correspondiente del material de origen no incluye código ni datos "
        "en el repositorio de origen.", icon=":material/info:",
    )


def _refuerzo() -> None:
    st.write(
        "El aprendizaje por refuerzo (RL) no aprende de ejemplos etiquetados, sino de prueba y error: un agente "
        "toma acciones en un entorno, recibe una recompensa o un castigo según el resultado, y ajusta su "
        "comportamiento para maximizar la recompensa a largo plazo. Es el mismo principio con el que se premia "
        "o corrige a un niño: repite lo que le sale bien, evita lo que le sale mal."
    )
    st.write("Sus piezas principales, tal como las describe el material de origen:")
    st.markdown(
        "- **Acciones**: lo que el agente puede hacer (comprar, vender, mantener).\n"
        "- **Estado**: la información que ve el agente antes de decidir (precio, indicadores, sentimiento).\n"
        "- **Recompensa**: el objetivo a maximizar (el beneficio, o el retorno ajustado por riesgo).\n"
        "- **Entorno**: el mercado o el activo sobre el que el agente actúa.\n"
        "- **Agente**: el modelo que observa el estado y decide la acción."
    )
    st.write(
        "La diferencia clave frente a la clasificación es que el aprendizaje por refuerzo puede aceptar una "
        "pérdida a corto plazo si eso lleva a un resultado mejor más adelante, algo que a los modelos de "
        "clasificación, entrenados para acertar en cada instante, les cuesta capturar."
    )
    st.info(
        "Esta aplicación no implementa aprendizaje por refuerzo: el capítulo correspondiente del material de origen no "
        "incluye código ni datos en el repositorio de origen.", icon=":material/info:",
    )


def _licencias() -> None:
    st.write(atribucion.TEXTO)
    st.write(
        "Los datos de `data_modules/` y el código de origen provienen de un repositorio de GitHub, con "
        "licencia Apache 2.0. El contenido conceptual de esta página resume, con palabras propias, ideas y "
        "ejemplos del material de origen, cuya licencia (CC BY-SA 4.0) exige atribución y que cualquier "
        "reutilización se comparta bajo los mismos términos."
    )
    st.warning(
        "Esta aplicación tiene fines educativos. Ningún resultado que muestra —predicciones, métricas de "
        "backtesting o simulaciones— constituye asesoría financiera ni una recomendación de inversión. El "
        "desempeño pasado, real o simulado, no garantiza resultados futuros.", icon=":material/warning:",
    )


def render() -> None:
    c.encabezado(TITULO, LEAD)
    tabs = st.tabs([
        "Fundamentos", "Glosario", "Aprendizaje no supervisado", "Procesamiento de lenguaje natural",
        "Aprendizaje por refuerzo", "Licencias y atribución",
    ])
    with tabs[0]:
        _fundamentos()
    with tabs[1]:
        _glosario()
    with tabs[2]:
        _no_supervisado_intro()
    with tabs[3]:
        _nlp()
    with tabs[4]:
        _refuerzo()
    with tabs[5]:
        _licencias()
