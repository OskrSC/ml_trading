"""Textos de la interfaz, centralizados.

Reglas de redacción (ver docs/terminologia.md):
- Español, tono directo, voz activa y mayúscula solo al inicio de la frase.
- Sin emojis.
- Lo que se toma como referencia se llama "predeterminado".
"""

APP_NOMBRE = "Aprendizaje automático en trading"
APP_SUBTITULO = "Laboratorio interactivo de modelos sobre datos de mercado"

# Navegación
NAV_SECCION_APLICACION = "Aplicación"
NAV_SECCION_PREPARACION = "Preparación de datos"
NAV_SECCION_MODELADO = "Modelado"
NAV_SECCION_NO_SUPERVISADO = "No supervisado"
NAV_SECCION_OPERACION = "Operación"
NAV_SECCION_SISTEMA = "Sistema"
NAV_INICIO = "Inicio"
NAV_DATOS = "Catálogo de datos"
NAV_VARIABLES = "Variables y objetivo"
NAV_DIVISION = "División de datos"
NAV_MODELOS = "Modelos supervisados"
NAV_EVALUACION = "Evaluación"
NAV_REGRESION = "Regresión"
NAV_XGBOOST = "XGBoost multiactivo"
NAV_BACKTESTING = "Backtesting"
NAV_COMPARADOR = "Comparador de modelos"
NAV_KMEANS = "K-Means"
NAV_JERARQUICO = "Clustering jerárquico"
NAV_PCA = "PCA y t-SNE"
NAV_EN_VIVO = "Operación en vivo"
NAV_GUIA = "Guía y referencias"
NAV_DISENO = "Sistema de diseño"
NAV_DIAGNOSTICO = "Diagnóstico del entorno"

# Inicio
INICIO_LEAD = (
    "Explore cómo se entrenan, evalúan y prueban modelos de aprendizaje automático "
    "sobre precios reales de acciones. Cada modelo parte de una configuración "
    "predeterminada que puede modificar y comparar."
)
INICIO_GRAFICO_TITULO = "Punto de partida: JPMorgan en barras de 15 minutos"
INICIO_GRAFICO_AYUDA = "Precio de cierre entre 2017 y 2019. Use el control inferior para acercar un periodo."
INICIO_ESTADO_TITULO = "Estado del desarrollo"
INICIO_ESTADO_AYUDA = "La aplicación se construye por fases."
INICIO_APARIENCIA_TITULO = "Apariencia"
INICIO_APARIENCIA_TEXTO = (
    "Elija el tema claro u oscuro desde el menú de la esquina superior derecha. "
    "La opción Sistema sigue la preferencia de su dispositivo."
)
INICIO_KPI_CONJUNTOS = "Conjuntos de datos"
INICIO_KPI_BARRAS = "Barras de JPMorgan"
INICIO_KPI_ACCIONES = "Acciones multiactivo"
INICIO_KPI_PERIODO = "Periodo de JPMorgan"

AVISO_EDUCATIVO = (
    "Esta aplicación tiene fines educativos. Los resultados se calculan sobre datos "
    "históricos, no incluyen costos de transacción salvo que se indique y no constituyen "
    "una recomendación de inversión."
)

# Catálogo de datos
DATOS_TITULO = "Catálogo de datos"
DATOS_LEAD = (
    "Los 16 conjuntos de datos que alimentan la aplicación, con su calidad medida al cargarlos. "
    "Los artefactos predeterminados son resultados ya calculados que sirven de referencia."
)
DATOS_FILTRO_TIPO = "Tipo"
DATOS_FILTRO_TODOS = "Todos"
DATOS_SELECTOR = "Conjunto de datos"
DATOS_TAB_VISTA = "Vista previa"
DATOS_TAB_COLUMNAS = "Columnas"
DATOS_TAB_CALIDAD = "Calidad"
DATOS_SIN_GRAFICO = "Este conjunto no tiene una serie que graficar."
DATOS_GRAFICO_SERIE = "Serie a graficar"

# Sistema de diseño
DISENO_TITULO = "Sistema de diseño"
DISENO_LEAD = (
    "Vista previa de los componentes en el tema activo. Cambie el tema desde el menú "
    "superior derecho para revisar ambos."
)

# Diagnóstico
DIAG_TITULO = "Diagnóstico del entorno"
DIAG_LEAD = (
    "Página interna de la fase 0. Mide el servidor donde se ejecuta la aplicación para "
    "decidir el perfil de despliegue. Se oculta antes del lanzamiento."
)
