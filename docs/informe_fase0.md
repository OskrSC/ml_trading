# Informe de la fase 0

Complete este informe con el archivo JSON de la página de diagnóstico de la app desplegada.

## 1. Despliegue

| Dato | Valor |
|---|---|
| Fecha del despliegue | |
| Versión de Python elegida | |
| Duración del build | |
| TA-Lib, xgboost y statsmodels instalados sin error | |
| Arranque en frío (primera visita tras reposo) | |

## 2. Recursos en la nube

| Medida | Valor |
|---|---|
| Memoria del proceso en reposo | |
| Memoria del contenedor en reposo | |
| Límite del contenedor | |
| Núcleos visibles | |

## 3. Carga de datos

Tiempo total de lectura de los 16 conjuntos: ______ s. Tamaño total en memoria: ______ MB.

## 4. Importación de librerías (subproceso)

| Librería | Estado | Segundos | Memoria añadida (MB) |
|---|---|---|---|
| scikit-learn | | | |
| scipy | | | |
| statsmodels | | | |
| xgboost | | | |
| talib | | | |
| tensorflow (experimento) | | | |

## 5. Comprobaciones manuales

- [ ] El menú superior derecho muestra Sistema, Claro y Oscuro a un visitante.
- [ ] El tema cambia sin recargar y los gráficos siguen el cambio.
- [ ] Las fuentes Source Sans 3 y Newsreader cargan.
- [ ] La barra lateral queda cerrada en móvil y las cuatro páginas se leen bien.
- [ ] Los indicadores de inicio se ven en dos columnas en móvil.

## 6. Decisión de perfil

Criterio propuesto: se elige el perfil completo (con TensorFlow) solo si el build termina sin error y la memoria del contenedor, con TensorFlow importado y la app en reposo, queda por debajo del 60 % del límite observado. En caso contrario, perfil ligero.

**Perfil elegido: ligero** (sin TensorFlow).

Motivo: decisión del equipo, tomada sin las mediciones de este informe. Las secciones 1 a 5 siguen pendientes. No bloquean el desarrollo, pero conviene completarlas para confirmar que el perfil ligero cabe con margen en el servicio y decidir cuántos usuarios simultáneos admite la app. Si las mediciones mostraran un margen amplio, el perfil completo se puede reconsiderar más adelante.

## Mediciones locales de referencia

Tomadas en el entorno de desarrollo, no en la nube. Sirven solo de comparación.

| Medida | Valor |
|---|---|
| Lectura de los 16 conjuntos | 0,23 s y 4,3 MB en memoria |
| Proceso tras importar la interfaz | unos 136 MB |
| Proceso tras cargar los datos | unos 167 MB |
| Importar scikit-learn | 0,9 s, unos 159 MB |
| Importar xgboost | 1,0 s, unos 172 MB |
| Importar TA-Lib | 0,3 s, unos 93 MB |
| Importar scipy | 0,1 s, unos 18 MB |
| Importar plotly | 0,04 s, unos 6 MB |
