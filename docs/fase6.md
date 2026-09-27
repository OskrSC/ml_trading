# Fase 6: endurecimiento y lanzamiento

## Alcance

Con las seis fases de funcionalidad completas, esta fase revisa la aplicación entera en cinco frentes:
rendimiento, seguridad, accesibilidad, observabilidad y despliegue continuo, y cierra la documentación.

## Rendimiento

- Revisadas todas las cachés (`st.cache_data`): las dos de `mltrading/ui/datos_cache.py` no tenían límite
  explícito. Aunque el catálogo acota el número de combinaciones posibles (16 archivos, como mucho 32 con
  `descartar_nulos`), se añadió `max_entries` y un `ttl` de 6 horas como red de seguridad y para no mantener
  datos en memoria indefinidamente en un proceso de servidor de muy larga duración.
- Las importaciones diferidas (dentro de cada función, no al principio del módulo) ya eran la práctica en
  todo `mltrading/core/`, confirmada de nuevo en esta fase; no hizo falta ningún cambio.

## Seguridad

- Las recetas ahora tienen un límite de tamaño (64 KB) antes de intentar interpretarlas como JSON, con un
  mensaje que sugiere qué pudo haber pasado si alguien intenta subir algo mucho más grande, como un modelo
  serializado.
- `pip-audit -r requirements.txt` no encontró vulnerabilidades conocidas en las dependencias fijadas (revisado
  en este entorno; el nuevo job `seguridad` del flujo de CI lo repite en cada cambio, sin bloquear el resto).
- Se revisó de nuevo que ningún módulo use `eval`, `exec`, `pickle` ni `compile` fuera de lo estrictamente
  necesario (ninguno lo hace); la prueba `test_ninguna_receta_ejecuta_codigo_arbitrario` de la fase 5 ya lo
  cubría para las recetas.
- `showErrorDetails` pasa de `"full"` a `"type"` en `.streamlit/config.toml`: en producción se ve el tipo de
  error, no la traza completa ni rutas de archivo del servidor.

## Accesibilidad

- El gráfico de precisión temporal de la página Evaluación (capítulo 6) solo distinguía acierto de fallo por
  color. Ahora también usa un símbolo distinto (círculo y cruz) y una posición distinta (fila superior e
  inferior), con su etiqueta en el texto emergente: una persona con daltonismo puede leerlo igual de bien.
- El resto de la aplicación ya usaba, desde fases anteriores, contraste verificado (`tests/test_tokens.py`),
  iconos junto al texto en vez de solo color, y controles nativos de Streamlit, accesibles por teclado sin
  ningún HTML ni JavaScript propio que pudiera romper el orden de tabulación.

## Observabilidad

- Módulo nuevo, `mltrading/core/observabilidad.py`: un logger que escribe a la salida estándar (donde la
  mayoría de servicios de despliegue capturan los registros) y guarda además las últimas 200 entradas en
  memoria, sin depender del sistema de archivos del servidor, efímero en la nube.
- La página Diagnóstico del entorno muestra esas últimas entradas, con un botón para limpiarlas.
- Se detectaron y corrigieron dos puntos sin manejo de errores: el entrenamiento en el Comparador de modelos
  (un modelo que falla ya no tumba la comparación completa; se avisa y se sigue con el resto) y en XGBoost
  multiactivo (un fallo ahora muestra un mensaje claro en vez de una traza sin control). Ambos casos quedan
  también registrados por el logger nuevo.

## Despliegue continuo

- Nuevo job `seguridad` en `.github/workflows/ci.yml`, con `pip-audit` contra `requirements.txt`.
- `docs/checklist_lanzamiento.md` consolida en una sola lista los pasos de despliegue y endurecimiento que
  antes estaban repartidos entre varios documentos de fases anteriores.
- `docs/mantenimiento.md` documenta las tareas recurrentes: actualizar una dependencia, añadir un modelo al
  registro, regenerar el tema o las capturas, y diagnosticar un incidente en producción.

## Licencia del código propio

Quedaba pendiente desde la fase 0. Se añadió `LICENSE` (MIT) como una propuesta razonable para un proyecto
educativo, y `NOTICE` la señala explícitamente como sugerencia, a confirmar o sustituir por quien mantenga el
repositorio antes de un lanzamiento público. No es una decisión que correspondiera tomar de forma unilateral.

## Qué no se hizo, y por qué

- **Cola de entrenamientos entre sesiones.** Cada sesión de Streamlit ya es independiente; una cola global
  solo aportaría valor si el servicio mostrara signos reales de saturación con varios usuarios a la vez, algo
  que no se puede medir sin el servicio desplegado. Queda como posible trabajo futuro, con la página de
  Diagnóstico como punto de partida para decidir si hace falta.
- **Entornos de prueba y producción separados.** Streamlit Community Cloud liga una app a un repositorio y una
  rama. Crear un entorno de prueba real implicaría una segunda app apuntando a otra rama; se documentó el
  patrón en `docs/despliegue_streamlit_cloud.md` en vez de darlo por hecho, porque supondría una decisión de
  infraestructura (y de costo) que corresponde a quien despliegue la aplicación.
